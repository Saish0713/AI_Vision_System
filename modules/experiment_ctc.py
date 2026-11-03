"""CTC small-set experiment, gated by the visual-grounding result.

English BPE targets (not gloss). Blank is an output class, never a target.
Variable actual input lengths are packed through the existing RGB/hand BiGRU.
"""
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from modules.prepare_how2sign_sequences import decode_video_path
from modules.sentence_model import VideoEncoder, detokenize_bpe, encode_bpe
from modules.train_sentence_translator import ROOT
from modules.experiment_visual_grounding import score

OUT = ROOT / 'output/ctc_small'


def required_steps(target):
    return len(target) + sum(a == b for a, b in zip(target, target[1:]))


def collapse_ctc(ids, blank):
    result, previous = [], None
    for token in ids:
        if token != previous and token != blank:
            result.append(token)
        previous = token
    return result


class CTCVideoModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.encoder = VideoEncoder(192, 192, 256, use_hand_landmarks=True)
        self.projection = nn.Linear(384, vocabulary_size + 1)
        self.blank = vocabulary_size

    def forward(self, examples):
        # Encode only real frames: padding never reaches CNN BatchNorm.
        lengths = torch.tensor([len(x[0]) for x in examples], dtype=torch.long)
        rgb = torch.cat([x[0] for x in examples])
        hands = torch.cat([x[1] for x in examples])
        presence = torch.cat([x[2] for x in examples])
        visual = self.encoder.frame_encoder.cnn(rgb)
        landmarks = self.encoder.hand_encoder(hands[None], presence[None])[0]
        features = torch.cat((visual, landmarks), dim=-1)
        packed = pack_padded_sequence(pad_sequence(features.split(lengths.tolist()), batch_first=True),
                                      lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder.temporal(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True)
        logits = self.projection(encoded)
        logits[:, :, [0, 2, 3]] = -10000  # PAD/BOS/EOS are not CTC text classes.
        return logits.log_softmax(-1).transpose(0, 1), lengths


def ctc_loss(log_probs, lengths, targets, blank):
    for length, target in zip(lengths.tolist(), targets):
        if not len(target) or blank in target.tolist() or any(t in (0, 2, 3) for t in target.tolist()):
            raise ValueError('CTC target must contain text only, without blank/PAD/BOS/EOS')
        if required_steps(target.tolist()) > length:
            raise ValueError('Insufficient real input steps for target and adjacent repeats')
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    # Do not hide invalid alignments with zero_infinity=True.
    loss = nn.CTCLoss(blank=blank, zero_infinity=False)(log_probs, torch.cat(targets), lengths, target_lengths)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite CTC loss')
    return loss


def main():
    grounding = json.loads((ROOT / 'output/visual_grounding/report.json').read_text())
    if not grounding.get('complete') or not grounding['fresh']['passed']:
        raise RuntimeError('Fresh visual-grounding experiment must pass before CTC')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ROOT / 'output/visual_grounding/fresh.pt', map_location='cpu', weights_only=False)
    vocab, merges = checkpoint['vocabulary'], checkpoint['bpe_merges']
    token_list = sorted(vocab, key=vocab.get)
    with (ROOT / 'output/visual_grounding/tiny_manifest.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    targets = [torch.tensor(encode_bpe(r['english_translation'], vocab, merges, 100000)[1:-1], dtype=torch.long, device=device) for r in rows]
    report = {'seed': 42, 'clip_ids': [r['clip_name'] for r in rows], 'history': [],
              'blank_id': len(vocab), 'target_type': 'English BPE, no sentence markers',
              'gate': 'At least 29/32 exact; each shuffle >=26 changed predictions and <=16 exact; finite falling loss',
              'max_epochs': 200, 'batch_size': 2, 'learning_rate': 0.001,
              'temporal_sampling': 'up to 96 distinct source frames per clip; not repeated 16-frame cache',
              'initialization': 'fresh encoder and CTC projection, seed 42', 'passed': False}
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / 'models/hand_landmarker.task')),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2)
    examples = []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for row, target in zip(rows, targets):
            source = ROOT / 'data/raw/how2sign/archives/train_rgb_front_clips' / row['clip_name']
            cache = OUT / (Path(row['clip_name']).stem + '.npz')
            if not cache.exists():
                cap = cv2.VideoCapture(str(source))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                count = min(96, total)
                if count < required_steps(target.tolist()):
                    raise ValueError(f'CTC cannot fit target in source: {source}')
                frames, fps, original, hands, presence = decode_video_path(source, count, 112, detector)
                np.savez_compressed(cache, frames=frames, hand_landmarks=hands, hand_presence=presence,
                                    original_frame_count=original, fps=fps)
            with np.load(cache) as data:
                example = (torch.from_numpy(data['frames'].copy()).permute(0, 3, 1, 2).float().div_(255),
                           torch.from_numpy(data['hand_landmarks'].copy()), torch.from_numpy(data['hand_presence'].copy()))
            if len(example[0]) < required_steps(target.tolist()):
                raise ValueError('Cached input is too short')
            examples.append(example)
            print('Prepared', len(examples), '/', len(rows), 'frames', len(example[0]), flush=True)
    report['input_lengths'] = [len(x[0]) for x in examples]
    report['required_lengths'] = [required_steps(t.tolist()) for t in targets]
    model = CTCVideoModel(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)

    def save():
        (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    def gpu(indices):
        return [tuple(t.to(device) for t in examples[i]) for i in indices]

    @torch.inference_mode()
    def evaluate():
        model.eval()
        hyps, losses = [], []
        for i in range(len(rows)):
            probs, lengths = model(gpu([i]))
            losses.append(float(ctc_loss(probs, lengths, [targets[i]], model.blank)))
            ids = collapse_ctc(probs[:lengths[0], 0].argmax(-1).tolist(), model.blank)
            hyps.append(detokenize_bpe([token_list[t] for t in ids]))
        refs = [r['english_translation'] for r in rows]
        result, records = {}, []
        for offset in (0, 1, 7, 13):
            shuffled = [hyps[(i + offset) % len(hyps)] for i in range(len(hyps))]
            result[str(offset)] = dict(**score(refs, shuffled), changed_predictions=sum(a != b for a, b in zip(hyps, shuffled)))
            records.extend(dict(offset=offset, target_clip=r['clip_name'], input_clip=rows[(i + offset) % len(rows)]['clip_name'],
                                reference=refs[i], hypothesis=shuffled[i]) for i, r in enumerate(rows))
        with (OUT / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        return float(np.mean(losses)), result

    initial, _ = evaluate()
    report['initial_loss'] = initial
    save()
    for epoch in range(1, 201):
        model.train()
        order = rng.permutation(len(rows)).tolist()
        loss_total = 0
        for start in range(0, len(order), 2):
            indices = order[start:start + 2]
            optimizer.zero_grad(set_to_none=True)
            probs, lengths = model(gpu(indices))
            loss = ctc_loss(probs, lengths, [targets[i] for i in indices], model.blank)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss_total += float(loss.detach()) * len(indices)
        if epoch % 10 == 0:
            loss, metrics = evaluate()
            report['history'].append(dict(epoch=epoch, train_loss=loss_total / len(rows), eval_loss=loss, metrics=metrics))
            passed = (loss < initial * 0.2 and metrics['0']['exact_matches'] >= 29
                      and all(metrics[str(k)]['changed_predictions'] >= 26 and metrics[str(k)]['exact_matches'] <= 16 for k in (1, 7, 13)))
            report['passed'] = passed
            print('CTC epoch', epoch, 'loss', loss, 'exact', metrics['0']['exact_matches'], 'gate', passed, flush=True)
            save()
            if passed:
                break
    torch.save(dict(state_dict=model.state_dict(), vocabulary=vocab, bpe_merges=merges,
                    architecture='RGB-hands-BiGRU-CTC', blank_id=model.blank,
                    warning='32-clip memorization experiment, not a production translator'), OUT / 'model.pt')
    report['complete'] = True
    save()
    print('CTC complete; scaling gate:', report['passed'], flush=True)


if __name__ == '__main__':
    main()
