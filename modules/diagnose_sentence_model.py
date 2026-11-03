"""Audit v4 data, measure visual dependence, and memorize 32 training clips.

This experiment evaluates memorization on its own training clips, not generalization.
Run with the project interpreter: python -u -m modules.diagnose_sentence_model
"""
import csv
import json
import random
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from modules.sentence_model import VideoToTextModel, encode_bpe, greedy_decode, tokenize
from modules.train_sentence_translator import ROOT, SequenceDataset, collate_batch, read_rows, run_epoch


def main():
    torch.set_num_threads(4)
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = ROOT / 'output' / 'sentence_diagnostics_v4'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(ROOT / 'models/how2sign_video_to_text_hands_v4_demo.pt', map_location='cpu', weights_only=False)
    vocab, merges = checkpoint['vocabulary'], checkpoint['bpe_merges']
    pad = vocab['<pad>']
    tokens = sorted(vocab, key=vocab.get)
    train, validation = read_rows(ROOT / 'data/processed/how2sign_sequence_manifest_v4.csv')
    with (ROOT / 'data/processed/how2sign_manifest.csv').open(encoding='utf-8', newline='') as stream:
        originals = {(r['split'], r['clip_name']): r for r in csv.DictReader(stream)}
    report = {'device': str(device), 'seed': 42, 'audit': {}}

    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    for split, rows in [('train', train), ('validation', validation)]:
        stats = dict(clips=len(rows), mismatched_targets=0, truncated_targets=0,
                     unknown_tokens=0, text_tokens=0, frames_without_hands=0,
                     sampled_frames=0, errors=[])
        for row in rows:
            original = originals.get((split, row['clip_name']))
            stats['mismatched_targets'] += int(original is None or original['english_translation'] != row['english_translation'])
            ids = encode_bpe(row['english_translation'], vocab, merges, 100000)
            stats['truncated_targets'] += int(len(ids) > checkpoint['max_target_tokens'])
            stats['unknown_tokens'] += ids.count(vocab['<unk>'])
            stats['text_tokens'] += len(ids) - 2
            try:
                with np.load(ROOT / row['sequence_path']) as archive:
                    frames, landmarks, presence = archive['frames'], archive['hand_landmarks'], archive['hand_presence']
                    if frames.shape != (16, 112, 112, 3) or landmarks.shape != (16, 2, 21, 3) or presence.shape != (16, 2):
                        raise ValueError('Unexpected array shape')
                    if not np.isfinite(landmarks).all() or not np.isfinite(presence).all():
                        raise ValueError('Non-finite landmark values')
                    stats['frames_without_hands'] += int((presence.sum(axis=1) == 0).sum())
                    stats['sampled_frames'] += len(frames)
            except Exception as error:
                stats['errors'].append({'clip': row['clip_name'], 'error': str(error)})
        report['audit'][split] = stats
        save()
        print(split, stats, flush=True)
    report['audit']['overlapping_clip_names'] = len({r['clip_name'] for r in train} & {r['clip_name'] for r in validation})
    if any(s['errors'] or s['mismatched_targets'] for s in report['audit'].values() if isinstance(s, dict)):
        save()
        raise RuntimeError('Resolve data errors before continuing; see report.json')

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)
    model = VideoToTextModel(len(vocab), pad, use_hand_landmarks=True).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    # Fixed random sample and global cyclic permutation: every video changes.
    sampled = random.Random(42).sample(validation, 128)
    items = list(SequenceDataset(sampled, vocab, merges, checkpoint['max_target_tokens'], True))
    swapped = [(*items[(i + 1) % len(items)][:3], item[3]) for i, item in enumerate(items)]
    collate = partial(collate_batch, pad_id=pad)
    for name, data in [('correct_video', items), ('wrong_video', swapped)]:
        loss, accuracy = run_epoch(model, DataLoader(data, batch_size=8, collate_fn=collate), None, loss_fn, device)
        report[name] = {'loss': loss, 'teacher_forced_accuracy': accuracy, 'examples': len(data)}
        print(name, report[name], flush=True)
        save()

    # Use a fresh model and the existing tokenizer; exclude truncated/unknown
    # targets so this experiment can actually reproduce complete sentences.
    eligible = [r for r in train if len(encode_bpe(r['english_translation'], vocab, merges, 100000)) <= 80
                and vocab['<unk>'] not in encode_bpe(r['english_translation'], vocab, merges, 100000)]
    tiny_rows = random.Random(42).sample(eligible, 32)
    tiny = list(SequenceDataset(tiny_rows, vocab, merges, 80, True))
    loader = DataLoader(tiny, batch_size=8, shuffle=True, collate_fn=collate)
    model = VideoToTextModel(len(vocab), pad, use_hand_landmarks=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    report['memorization'] = {'clips': [r['clip_name'] for r in tiny_rows], 'history': [], 'limit_epochs': 150,
                              'learning_rate': 0.001, 'evaluation': 'same 32 training clips; not held out'}
    for epoch in range(1, 151):
        loss, accuracy = run_epoch(model, loader, optimizer, loss_fn, device, scaler)
        if epoch % 10 == 0:
            hypotheses = [greedy_decode(model, item[0].unsqueeze(0).to(device), tokens, 80,
                          item[1].unsqueeze(0).to(device), item[2].unsqueeze(0).to(device)) for item in tiny]
            matches = sum(tokenize(r['english_translation']) == tokenize(h) for r, h in zip(tiny_rows, hypotheses))
            entry = {'epoch': epoch, 'training_loss': loss, 'training_token_accuracy': accuracy, 'greedy_exact_matches': matches}
            report['memorization']['history'].append(entry)
            print('memorization', entry, flush=True)
            with (output / 'memorization_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['clip_name', 'reference', 'hypothesis'])
                writer.writeheader()
                writer.writerows({'clip_name': r['clip_name'], 'reference': r['english_translation'], 'hypothesis': h} for r, h in zip(tiny_rows, hypotheses))
            save()
            if matches == 32:
                break
    torch.save({'state_dict': model.state_dict(), 'vocabulary': vocab, 'bpe_merges': merges,
                'use_hand_landmarks': True, 'max_target_tokens': 80,
                'warning': '32-clip memorization diagnostic only; not a translation checkpoint'}, output / 'memorization.pt')
    report['complete'] = True
    save()
    print('Diagnostics complete:', output, flush=True)


if __name__ == '__main__':
    main()
