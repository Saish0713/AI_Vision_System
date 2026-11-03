"""Compact video-to-English sequence model for the How2Sign prototype."""

import re
from collections import Counter

import torch
from torch import nn


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(texts: list[str], min_frequency: int = 1):
    frequencies = {}
    for text in texts:
        for token in tokenize(text):
            frequencies[token] = frequencies.get(token, 0) + 1
    tokens = [
        token
        for token, frequency in sorted(frequencies.items())
        if frequency >= min_frequency and token not in SPECIAL_TOKENS
    ]
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    vocabulary.update({token: len(vocabulary) + index for index, token in enumerate(tokens)})
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int], max_tokens: int) -> list[int]:
    ids = [vocabulary[BOS_TOKEN]]
    ids.extend(vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokenize(text))
    ids = ids[: max_tokens - 1]
    ids.append(vocabulary[EOS_TOKEN])
    return ids


def train_bpe_tokenizer(texts: list[str], vocab_size: int = 512):
    """Learn a small character-based BPE vocabulary from training text only."""
    sequences = []
    for text in texts:
        sequences.append([
            "▁" + token[0], *token[1:]
        ] for token in tokenize(text))
    sequences = [piece for sentence in sequences for piece in sentence]
    merges = []
    while True:
        symbols = {symbol for sequence in sequences for symbol in sequence}
        if len(symbols) + len(SPECIAL_TOKENS) >= vocab_size:
            break
        pair_counts = Counter(
            pair
            for sequence in sequences
            for pair in zip(sequence, sequence[1:])
        )
        if not pair_counts:
            break
        best_pair, frequency = min(
            pair_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if frequency < 2:
            break
        merged_symbol = "".join(best_pair)
        merges.append(best_pair)
        for index, sequence in enumerate(sequences):
            merged = []
            cursor = 0
            while cursor < len(sequence):
                if (
                    cursor + 1 < len(sequence)
                    and (sequence[cursor], sequence[cursor + 1]) == best_pair
                ):
                    merged.append(merged_symbol)
                    cursor += 2
                else:
                    merged.append(sequence[cursor])
                    cursor += 1
            sequences[index] = merged
    symbols = sorted({symbol for sequence in sequences for symbol in sequence})
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    vocabulary.update({token: len(vocabulary) + index for index, token in enumerate(symbols)})
    return vocabulary, merges


def encode_bpe(text: str, vocabulary: dict[str, int], merges, max_tokens: int) -> list[int]:
    merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
    pieces = []
    for token in tokenize(text):
        word_pieces = ["▁" + token[0], *token[1:]]
        while len(word_pieces) > 1:
            options = [
                (merge_ranks[(word_pieces[i], word_pieces[i + 1])], i)
                for i in range(len(word_pieces) - 1)
                if (word_pieces[i], word_pieces[i + 1]) in merge_ranks
            ]
            if not options:
                break
            _, index = min(options)
            word_pieces[index:index + 2] = [word_pieces[index] + word_pieces[index + 1]]
        pieces.extend(word_pieces)
    ids = [vocabulary[BOS_TOKEN]]
    ids.extend(vocabulary.get(piece, vocabulary[UNK_TOKEN]) for piece in pieces)
    ids = ids[: max_tokens - 1]
    ids.append(vocabulary[EOS_TOKEN])
    return ids


def detokenize_bpe(pieces: list[str]) -> str:
    text = ""
    for piece in pieces:
        starts_word = piece.startswith("▁")
        value = piece[1:] if starts_word else piece
        if starts_word and text:
            text += " "
        text += value
    text = re.sub(r"\s+([.,!?;:%\)\]])", r"\1", text)
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    text = re.sub(r"\s+'\s*", "'", text)
    return text.strip()


class FrameEncoder(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Preserve coarse hand/body layout instead of collapsing each frame
            # to one global average-pooled vector.
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, frames):
        batch_size, time_steps, channels, height, width = frames.shape
        encoded = self.cnn(frames.reshape(batch_size * time_steps, channels, height, width))
        return encoded.reshape(batch_size, time_steps, -1)


class HandLandmarkEncoder(nn.Module):
    """Project pretrained MediaPipe hand coordinates into per-frame features."""

    def __init__(self, output_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * 21 * 3 + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, landmarks, presence):
        batch, time_steps = landmarks.shape[:2]
        features = torch.cat(
            (landmarks.flatten(start_dim=2), presence), dim=2
        )
        return self.network(features.reshape(batch * time_steps, -1)).reshape(
            batch, time_steps, -1
        )


class VideoEncoder(nn.Module):
    def __init__(
        self,
        frame_features: int,
        recurrent_size: int,
        decoder_size: int,
        use_hand_landmarks: bool = False,
    ):
        super().__init__()
        self.use_hand_landmarks = use_hand_landmarks
        self.frame_encoder = FrameEncoder(frame_features)
        self.hand_encoder = HandLandmarkEncoder() if use_hand_landmarks else None
        temporal_input_size = frame_features + (64 if use_hand_landmarks else 0)
        self.temporal = nn.GRU(
            input_size=temporal_input_size,
            hidden_size=recurrent_size,
            batch_first=True,
            bidirectional=True,
        )
        self.initial_state = nn.Linear(recurrent_size * 2, decoder_size)

    def forward(self, frames, hand_landmarks=None, hand_presence=None):
        frame_features = self.frame_encoder(frames)
        if self.use_hand_landmarks:
            if hand_landmarks is None or hand_presence is None:
                raise ValueError("This model requires hand landmark features per frame")
            hand_features = self.hand_encoder(hand_landmarks, hand_presence)
            frame_features = torch.cat((frame_features, hand_features), dim=2)
        encoded, hidden = self.temporal(frame_features)
        final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        decoder_hidden = torch.tanh(self.initial_state(final)).unsqueeze(0)
        return encoded, decoder_hidden


class AttentionTextDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        pad_id: int,
        embedding_size: int,
        encoder_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=pad_id)
        self.encoder_projection = nn.Linear(encoder_size, hidden_size, bias=False)
        self.hidden_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_score = nn.Linear(hidden_size, 1, bias=False)
        self.recurrent = nn.GRU(
            embedding_size + encoder_size,
            hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size + encoder_size, vocabulary_size)

    def step(self, input_ids, hidden, encoder_outputs):
        query = hidden[-1]
        energies = self.attention_score(
            torch.tanh(
                self.encoder_projection(encoder_outputs)
                + self.hidden_projection(query).unsqueeze(1)
            )
        ).squeeze(-1)
        weights = torch.softmax(energies, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        embedded = self.embedding(input_ids)
        output, hidden = self.recurrent(
            torch.cat((embedded, context), dim=1).unsqueeze(1), hidden
        )
        logits = self.output(torch.cat((output.squeeze(1), context), dim=1))
        return logits, hidden

    def forward(self, input_ids, hidden, encoder_outputs):
        logits = []
        for index in range(input_ids.shape[1]):
            step_logits, hidden = self.step(input_ids[:, index], hidden, encoder_outputs)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)


class VideoToTextModel(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        pad_id: int,
        frame_features: int = 192,
        recurrent_size: int = 192,
        embedding_size: int = 192,
        decoder_size: int = 256,
        use_hand_landmarks: bool = False,
    ):
        super().__init__()
        self.encoder = VideoEncoder(
            frame_features,
            recurrent_size,
            decoder_size,
            use_hand_landmarks=use_hand_landmarks,
        )
        self.decoder = AttentionTextDecoder(
            vocabulary_size,
            pad_id,
            embedding_size,
            recurrent_size * 2,
            decoder_size,
        )

    def forward(self, frames, input_ids, hand_landmarks=None, hand_presence=None):
        encoder_outputs, hidden = self.encoder(frames, hand_landmarks, hand_presence)
        return self.decoder(input_ids, hidden, encoder_outputs)


@torch.inference_mode()
def greedy_decode(
    model: VideoToTextModel,
    frames: torch.Tensor,
    id_to_token: list[str],
    max_tokens: int = 48,
    hand_landmarks: torch.Tensor | None = None,
    hand_presence: torch.Tensor | None = None,
) -> str:
    model.eval()
    encoder_outputs, hidden = model.encoder(frames, hand_landmarks, hand_presence)
    bos_id = id_to_token.index(BOS_TOKEN)
    eos_id = id_to_token.index(EOS_TOKEN)
    unk_id = id_to_token.index(UNK_TOKEN)
    current = torch.tensor([bos_id], device=frames.device)
    output_tokens = []
    for _ in range(max_tokens):
        logits, hidden = model.decoder.step(current, hidden, encoder_outputs)
        current = logits.argmax(dim=1)
        token_id = int(current.item())
        if token_id == eos_id:
            break
        if token_id >= len(id_to_token) or token_id < len(SPECIAL_TOKENS):
            token_id = unk_id
        output_tokens.append(id_to_token[token_id])
    return detokenize_bpe(output_tokens)
