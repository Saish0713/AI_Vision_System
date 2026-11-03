"""Reproduce memorization and measure sensitivity to mismatched videos.

Fixed 32 training clips from the earlier diagnostic; no claims of generalization.
"""
import csv
import json
import random
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader
from sacrebleu.metrics import BLEU, CHRF

from modules.sentence_model import VideoToTextModel, greedy_decode, tokenize
from modules.train_sentence_translator import ROOT, SequenceDataset, collate_batch, read_rows, run_epoch

OUT = ROOT / 'output/visual_grounding'


def score(refs, hypotheses):
    return {'exact_matches': sum(tokenize(r) == tokenize(h) for r, h in zip(refs, hypotheses)),
            'examples': len(refs), 'bleu': BLEU().corpus_score(hypotheses, [refs]).score,
            'chrf': CHRF().corpus_score(hypotheses, [refs]).score}


def evaluate(model, items, rows, vocab, limit, device, name):
    tokens = sorted(vocab, key=vocab.get)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    collate = partial(collate_batch, pad_id=vocab['<pad>'])
    refs = [r['english_translation'] for r in rows]
    result, records, original_hyps = {}, [], None
    for offset in (0, 1, 7, 13):
        # All offsets are derangements: videos and landmarks move together;
        # reference text and its position remain fixed.
        data = [(*items[(i + offset) % len(items)][:3], item[3]) for i, item in enumerate(items)]
        loss, acc = run_epoch(model, DataLoader(data, batch_size=8, collate_fn=collate), None, criterion, device)
        hyps = [greedy_decode(model, x[0][None].to(device), tokens, limit,
                             x[1][None].to(device), x[2][None].to(device)) for x in data]
        if offset == 0:
            original_hyps = hyps
        result[str(offset)] = dict(loss=loss, teacher_forced_accuracy=acc, **score(refs, hyps),
                                  changed_predictions=sum(a != b for a, b in zip(original_hyps, hyps)))
        records.extend({'offset': offset, 'target_clip': r['clip_name'],
                        'input_clip': rows[(i + offset) % len(rows)]['clip_name'],
                        'reference': refs[i], 'hypothesis': hyps[i]} for i, r in enumerate(rows))
    with (OUT / f'{name}_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(name, result, flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    old = json.loads((ROOT / 'output/sentence_diagnostics_v4/report.json').read_text())
    checkpoint = torch.load(ROOT / 'models/how2sign_video_to_text_hands_v5_tokens160.pt', map_location='cpu', weights_only=False)
    vocab, merges = checkpoint['vocabulary'], checkpoint['bpe_merges']
    train, val = read_rows(ROOT / 'data/processed/how2sign_sequence_manifest_v4.csv')
    index = {r['clip_name']: r for r in train}
    rows = [index[name] for name in old['memorization']['clips']]
    items = list(SequenceDataset(rows, vocab, merges, 160, True))
    report = {'seed': 42, 'clip_ids': [r['clip_name'] for r in rows], 'stages': {},
              'gate': 'At least 29/32 exact, >=80% loss decrease, and each shuffle: >=26 predictions change and <=16 exact',
              'interpretation': 'Clip memorization and visual dependence, not sign semantics or unseen signer generalization.'}

    def save():
        (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    with (OUT / 'tiny_manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    model = VideoToTextModel(len(vocab), vocab['<pad>'], use_hand_landmarks=True).to(device)
    memorized = torch.load(ROOT / 'output/sentence_diagnostics_v4/memorization.pt', map_location='cpu', weights_only=False)
    if memorized['vocabulary'] != vocab or memorized['bpe_merges'] != merges:
        raise ValueError('Tokenizer changed: cannot compare checkpoints using the same encodings')
    model.load_state_dict(memorized['state_dict'])
    report['stages']['saved_memorization'] = evaluate(model, items, rows, vocab, 160, device, 'saved_memorization')
    save()
    sampled = random.Random(42).sample(val, 128)
    validation = list(SequenceDataset(sampled, vocab, merges, 160, True))
    model.load_state_dict(checkpoint['state_dict'])
    report['stages']['v5_validation'] = evaluate(model, validation, sampled, vocab, 160, device, 'v5_validation')
    save()
    del validation

    torch.manual_seed(42)
    model = VideoToTextModel(len(vocab), vocab['<pad>'], use_hand_landmarks=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    loader = DataLoader(items, batch_size=8, shuffle=True, collate_fn=partial(collate_batch, pad_id=vocab['<pad>']))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    initial, _ = run_epoch(model, loader, None, criterion, device)
    report['fresh'] = {'initial_loss': initial, 'history': [], 'passed': False}
    for epoch in range(1, 151):
        loss, acc = run_epoch(model, loader, optimizer, criterion, device, scaler)
        if epoch % 20 != 0:
            continue
        result = evaluate(model, items, rows, vocab, 160, device, 'fresh')
        passed = (result['0']['exact_matches'] >= 29 and result['0']['loss'] < initial * 0.2
                  and all(result[str(k)]['changed_predictions'] >= 26 and result[str(k)]['exact_matches'] <= 16 for k in (1, 7, 13)))
        report['fresh']['history'].append(dict(epoch=epoch, train_loss=loss, train_accuracy=acc, evaluation=result))
        report['fresh']['passed'] = passed
        save()
        if passed:
            break
    torch.save(dict(state_dict=model.state_dict(), vocabulary=vocab, bpe_merges=merges,
                    max_target_tokens=160, use_hand_landmarks=True, warning='32-clip grounding diagnostic'), OUT / 'fresh.pt')
    report['complete'] = True
    save()
    print('Grounding gate passed:', report['fresh']['passed'], flush=True)


if __name__ == '__main__':
    main()
