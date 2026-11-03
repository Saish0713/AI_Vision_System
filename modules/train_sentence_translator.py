"""Train an experimental How2Sign video-to-English baseline."""

import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from modules.sentence_model import (
    PAD_TOKEN,
    VideoToTextModel,
    encode_bpe,
    greedy_decode,
    tokenize,
    train_bpe_tokenizer,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "how2sign_sequence_manifest.csv"
DEFAULT_OUTPUT = ROOT / "models" / "how2sign_video_to_text_demo.pt"
DEFAULT_PREDICTIONS = ROOT / "data" / "processed" / "sentence_validation_predictions.csv"


class SequenceDataset(Dataset):
    def __init__(self, rows, vocabulary, merges, max_tokens, use_hand_landmarks=False):
        self.rows = rows
        self.vocabulary = vocabulary
        self.merges = merges
        self.max_tokens = max_tokens
        self.use_hand_landmarks = use_hand_landmarks

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = ROOT / row["sequence_path"]
        with np.load(path) as archive:
            frames = archive["frames"].copy()
            if self.use_hand_landmarks:
                hand_landmarks = archive["hand_landmarks"].copy()
                hand_presence = archive["hand_presence"].copy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
        target = torch.tensor(
            encode_bpe(
                row["english_translation"],
                self.vocabulary,
                self.merges,
                self.max_tokens,
            ),
            dtype=torch.long,
        )
        if self.use_hand_landmarks:
            return (
                frames,
                torch.from_numpy(hand_landmarks),
                torch.from_numpy(hand_presence),
                target,
            )
        return frames, target


def collate_batch(items, pad_id):
    if len(items[0]) == 4:
        frames, landmarks, presence, targets = zip(*items)
    else:
        frames, targets = zip(*items)
    max_length = max(target.numel() for target in targets)
    padded = torch.full((len(targets), max_length), pad_id, dtype=torch.long)
    for index, target in enumerate(targets):
        padded[index, : target.numel()] = target
    batch = (torch.stack(frames), padded)
    if len(items[0]) == 4:
        batch = (batch[0], torch.stack(landmarks), torch.stack(presence), batch[1])
    return batch


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Manifest must contain train and validation rows")
    return train, validation


def sentence_metrics(references, hypotheses):
    exact = sum(
        tokenize(reference) == tokenize(hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    ) / max(len(references), 1)
    matches = [0] * 4
    totals = [0] * 4
    reference_length = 0
    hypothesis_length = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens, hyp_tokens = tokenize(reference), tokenize(hypothesis)
        reference_length += len(ref_tokens)
        hypothesis_length += len(hyp_tokens)
        for order in range(1, 5):
            ref_ngrams = Counter(tuple(ref_tokens[i:i + order]) for i in range(max(0, len(ref_tokens) - order + 1)))
            hyp_ngrams = Counter(tuple(hyp_tokens[i:i + order]) for i in range(max(0, len(hyp_tokens) - order + 1)))
            matches[order - 1] += sum((hyp_ngrams & ref_ngrams).values())
            totals[order - 1] += sum(hyp_ngrams.values())
    if not hypothesis_length:
        bleu = 0.0
    else:
        log_precision = sum(
            math.log((matches[i] + 1) / (totals[i] + 1))
            for i in range(4)
        ) / 4
        brevity_penalty = min(1.0, math.exp(1 - reference_length / hypothesis_length))
        bleu = brevity_penalty * math.exp(log_precision)
    return exact, bleu


def run_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    for batch in loader:
        if len(batch) == 4:
            frames, hand_landmarks, hand_presence, targets = [value.to(device) for value in batch]
        else:
            frames, targets = [value.to(device) for value in batch]
            hand_landmarks = hand_presence = None
        decoder_input, expected = targets[:, :-1], targets[:, 1:]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(frames, decoder_input, hand_landmarks, hand_presence)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), expected.reshape(-1))
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
        mask = expected.ne(loss_fn.ignore_index)
        count = int(mask.sum().item())
        total_loss += float(loss.item()) * count
        total_tokens += count
        total_correct += int(((logits.argmax(-1) == expected) & mask).sum().item())
    return total_loss / max(total_tokens, 1), total_correct / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-target-tokens", type=int, default=160,
                        help="Target text limit including BOS/EOS (default: 160); not a GPU hardware limit.")
    parser.add_argument("--vocabulary-size", type=int, default=512)
    parser.add_argument("--use-hand-landmarks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_target_tokens < 3 or args.vocabulary_size < 32:
        parser.error("epochs/batch-size must be positive, max-target-tokens >= 3, vocab-size >= 32")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_rows, val_rows = read_rows(args.manifest)
    vocabulary, merges = train_bpe_tokenizer(
        [row["english_translation"] for row in train_rows],
        vocab_size=args.vocabulary_size,
    )
    id_to_token = [token for token, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
    for split, rows in (("train", train_rows), ("validation", val_rows)):
        lengths = [len(encode_bpe(row["english_translation"], vocabulary, merges,
                                 2 * len(row["english_translation"]) + 2)) for row in rows]
        truncated = sum(length > args.max_target_tokens for length in lengths)
        print(f"{split}: target limit {args.max_target_tokens} including BOS/EOS; "
              f"{truncated}/{len(rows)} targets truncated ({truncated / len(rows):.2%}); "
              f"longest target {max(lengths)} tokens", flush=True)
    pad_id = vocabulary[PAD_TOKEN]
    train_data = SequenceDataset(
        train_rows, vocabulary, merges, args.max_target_tokens, args.use_hand_landmarks
    )
    val_data = SequenceDataset(
        val_rows, vocabulary, merges, args.max_target_tokens, args.use_hand_landmarks
    )
    collate = lambda items: collate_batch(items, pad_id)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoToTextModel(
        len(vocabulary), pad_id, use_hand_landmarks=args.use_hand_landmarks
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print(f"Device: {device}; train: {len(train_rows)}; validation: {len(val_rows)}; BPE vocab: {len(vocabulary)}")

    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, None, loss_fn, device)
        print(
            f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f}, token acc {train_acc:.3f} "
            f"| val loss {val_loss:.4f}, token acc {val_acc:.3f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "vocabulary": vocabulary,
                    "bpe_merges": merges,
                    "tokenizer": "character-BPE",
                    "max_target_tokens": args.max_target_tokens,
                    "architecture": "FrameCNN-4x4-BiGRU-attention-GRU",
                    "use_hand_landmarks": args.use_hand_landmarks,
                    "training_rows": len(train_rows),
                    "validation_rows": len(val_rows),
                    "warning": "Small prototype subset; not a quality ASL translator.",
                },
                args.output,
            )
    print(f"Best checkpoint: {args.output}")

    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Generating full validation predictions for sentence-level metrics...")
    hypotheses = []
    for index in range(len(val_rows)):
        validation_item = val_data[index]
        frames = validation_item[0]
        if args.use_hand_landmarks:
            hand_landmarks = validation_item[1].unsqueeze(0).to(device)
            hand_presence = validation_item[2].unsqueeze(0).to(device)
        else:
            hand_landmarks = hand_presence = None
        hypotheses.append(
            greedy_decode(
                model,
                frames.unsqueeze(0).to(device),
                id_to_token,
                args.max_target_tokens,
                hand_landmarks,
                hand_presence,
            )
        )
    references = [row["english_translation"] for row in val_rows]
    predictions_output = args.predictions_output or (
        DEFAULT_PREDICTIONS.parent / f"{args.output.stem}_validation_predictions.csv"
    )
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_output.open("w", newline="", encoding="utf-8") as predictions_file:
        writer = csv.DictWriter(predictions_file, fieldnames=["reference", "hypothesis"])
        writer.writeheader()
        writer.writerows(
            {"reference": reference, "hypothesis": hypothesis}
            for reference, hypothesis in zip(references, hypotheses)
        )
    print(f"Saved validation predictions: {predictions_output}")
    exact, bleu = sentence_metrics(references, hypotheses)
    print(f"Validation exact sentence match: {exact:.3f}")
    print(f"Validation local BLEU-4 (add-one smoothed): {bleu:.3f}")
    try:
        from sacrebleu.metrics import BLEU

        bleu_metric = BLEU()
        standard_bleu = bleu_metric.corpus_score(hypotheses, [references])
        print(f"SacreBLEU: {standard_bleu.score:.2f}")
        print(f"SacreBLEU signature: {bleu_metric.get_signature()}")
    except ImportError:
        print("SacreBLEU not installed; install requirements.txt to report standard BLEU.")
    print("Validation examples (reference => greedy output):")
    for reference, hypothesis in list(zip(references, hypotheses))[:5]:
        print(f"- {reference} => {hypothesis}")


if __name__ == "__main__":
    main()
