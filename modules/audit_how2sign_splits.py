"""CPU-only pose Stage 1 audit. Resume with --resume; stop safely with Ctrl+C.

Video durations use container metadata plus first-frame decode, not a full decode.
Filename suffixes are reported as unverified codes, never assumed to be signers.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import time
import unicodedata

import cv2
import numpy as np
import torch
from sacrebleu.metrics import BLEU, CHRF
from modules.sentence_model import encode_bpe, tokenize

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/how2sign'
MANIFEST = ROOT / 'data/processed/how2sign_manifest.csv'
SUBSET = ROOT / 'data/processed/how2sign_sequence_manifest_v4.csv'
TOKENIZER = ROOT / 'models/how2sign_video_to_text_hands_v4_demo.pt'
VIDEO_DIRS = {'train': RAW / 'archives/train_rgb_front_clips',
              'validation': RAW / 'archives/val_rgb_front_clips'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path, delimiter=','):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def normalize(text):
    return ' '.join(''.join(c for c in text.lower()
                           if not unicodedata.category(c).startswith('P')).split())


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Empty or non-finite distribution')
    return dict(count=len(a), min=float(a.min()), mean=float(a.mean()),
                **{f'p{p}': float(np.percentile(a, p)) for p in (1, 25, 50, 75, 90, 95, 99)},
                max=float(a.max()))


def probe(item):
    key, path, fingerprint = item
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Video cannot be opened')
        frames, fps, width, height = [cap.get(p) for p in (
            cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS,
            cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
        if not all(math.isfinite(v) and v > 0 for v in (frames, fps, width, height)):
            raise ValueError('Invalid video metadata')
        ok, first = cap.read()
        if not ok or first is None or first.shape[:2] != (int(height), int(width)):
            raise ValueError('First frame unreadable or shape inconsistent')
        return key, fingerprint, dict(frames=int(frames), fps=fps,
                                      seconds=frames / fps, width=int(width), height=int(height)), None
    except Exception as exc:
        return key, fingerprint, None, f'{type(exc).__name__}: {exc}'
    finally:
        cap.release()


def metrics(refs, hyps):
    result = {}
    for name, metric in [('bleu', BLEU()), ('chrf', CHRF())]:
        result[name] = metric.corpus_score(hyps, [refs]).score
        result[name + '_signature'] = str(metric.get_signature())
    result['unique_outputs'] = len(set(hyps))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/pose')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'audit.json').exists():
        raise FileExistsError('Completed audit exists; use a new --output directory to preserve it')
    if (out / 'pose_probe.sqlite').exists() and not args.resume:
        raise FileExistsError('Partial audit exists; use --resume')
    log = (out / 'pose_audit.log').open('a', encoding='utf-8', buffering=1)
    def say(message):
        print(message, flush=True)
        log.write(message + '\n')
    errors_file = (out / 'pose_audit_errors.csv').open('a', encoding='utf-8', newline='')
    errors = csv.writer(errors_file)
    if errors_file.tell() == 0:
        errors.writerow(['split', 'clip_name', 'reason'])
    error_count = 0
    def error(split, clip, reason):
        nonlocal error_count
        error_count += 1
        errors.writerow([split, clip, reason])
        errors_file.flush()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    rows = read_csv(MANIFEST)
    counts = Counter(r['split'] for r in rows)
    if counts != {'train': 31047, 'validation': 1739}:
        raise ValueError(f'Unexpected split counts: {counts}')
    keys = [(r['split'], r['clip_name']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate manifest clip keys')
    annotations, identity = {}, {}
    for split, filename in [('train', 'how2sign_train.tsv'), ('validation', 'how2sign_val.tsv')]:
        ann = read_csv(RAW / 'annotations' / filename, '\t')
        index = {r['SENTENCE_NAME'] + '.mp4': r for r in ann}
        if len(index) != len(ann):
            raise ValueError('Duplicate annotation sentence names')
        annotations[split] = index
        fields = list(ann[0])
        identity[split] = {'annotation_columns': fields,
                           'explicit_identity_columns': [f for f in fields if re.search('signer|speaker|person|subject|participant', f, re.I)],
                           'annotation_rows': len(ann)}
    checkpoint = torch.load(TOKENIZER, map_location='cpu', weights_only=False)
    vocab, merges = checkpoint['vocabulary'], checkpoint['bpe_merges']
    tokenizer_data = dict(vocabulary=vocab, bpe_merges=merges)
    tokenizer_hash = hashlib.sha256(json.dumps(tokenizer_data, sort_keys=True).encode()).hexdigest()
    say(f'Auditing {len(rows)} clips on CPU; tokenizer size {len(vocab)}; no training.')
    conn = sqlite3.connect(out / 'pose_probe.sqlite')
    conn.execute('CREATE TABLE IF NOT EXISTS probes (key TEXT PRIMARY KEY, fingerprint TEXT, result TEXT)')
    cached = {k: (f, json.loads(r)) for k, f, r in conn.execute('SELECT * FROM probes')}
    metadata, pending = {}, []
    for row in rows:
        split, clip = row['split'], row['clip_name']
        ann = annotations[split].get(clip)
        if not row['english_translation'].strip():
            error(split, clip, 'Empty target')
        if ann is None or ann['SENTENCE'].strip() != row['english_translation'] or ann['VIDEO_ID'] != row['video_id'] or ann['SENTENCE_ID'] != row['sentence_id']:
            error(split, clip, 'Annotation/manifest mismatch')
        elif any(abs(float(ann[a]) - float(row[b])) > 0.001 for a, b in [('START', 'source_start_seconds'), ('END', 'source_end_seconds')]):
            error(split, clip, 'Annotation timestamp mismatch')
        path = VIDEO_DIRS[split] / clip
        if not path.is_file():
            error(split, clip, 'Missing source file')
            continue
        stat = path.stat()
        fingerprint = f'{stat.st_size}:{stat.st_mtime_ns}'
        key = split + '/' + clip
        old = cached.get(key)
        if old and old[0] == fingerprint and all(math.isfinite(old[1][f]) and old[1][f] > 0 for f in ('frames', 'fps', 'seconds', 'width', 'height')):
            metadata[key] = old[1]
        else:
            pending.append((key, path, fingerprint))
    say(f'Reusing {len(metadata)} validated probes; {len(pending)} remaining. Ctrl+C stops; --resume continues.')
    started = time.monotonic()
    try:
        # Bounded batches keep cancellation responsive and avoid queuing the entire dataset.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(pending), 100):
                for key, fingerprint, result, failure in pool.map(probe, pending[start:start + 100]):
                    if failure:
                        error(*key.split('/', 1), failure)
                    else:
                        metadata[key] = result
                        conn.execute('INSERT OR REPLACE INTO probes VALUES (?, ?, ?)', (key, fingerprint, json.dumps(result)))
                conn.commit()
                done = min(start + 100, len(pending))
                elapsed = time.monotonic() - started
                say(f'Video checks {done}/{len(pending)}; elapsed {elapsed:.0f}s; ETA {elapsed / done * (len(pending) - done):.0f}s')
    finally:
        conn.commit()
        conn.close()
    if error_count:
        raise RuntimeError(f'{error_count} audit errors: see pose_audit_errors.csv; no success report written')
    disk = {}
    for split in VIDEO_DIRS:
        expected = {r['clip_name'] for r in rows if r['split'] == split}
        actual = {p.name for p in VIDEO_DIRS[split].glob('*.mp4')}
        disk[split] = dict(manifest=len(expected), on_disk=len(actual), missing=sorted(expected - actual), extra=sorted(actual - expected),
                           annotation_rows_not_in_manifest=sorted(set(annotations[split]) - expected))
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'validation']
    subset = read_csv(SUBSET)
    row_index = {(r['split'], r['clip_name']): r for r in rows}
    for r in subset:
        if row_index.get((r['split'], r['clip_name']), {}).get('english_translation') != r['english_translation']:
            raise ValueError('v4 subset misaligned with main manifest')
    val512 = {r['clip_name'] for r in subset if r['split'] == 'validation'}
    if len(val512) != 512:
        raise ValueError('Expected 512 unique v4 validation clips')
    overlap = {}
    for name, fn in [('exact', lambda s: s), ('normalized', normalize)]:
        seen = {fn(r['english_translation']) for r in train}
        overlap[name] = {}
        for group, selected in [('full_validation', val), ('v4_validation_512', [r for r in val if r['clip_name'] in val512])]:
            n = sum(fn(r['english_translation']) in seen for r in selected)
            overlap[name][group] = dict(count=n, total=len(selected), fraction=n / len(selected))
    stats, detailed = {}, []
    for split in VIDEO_DIRS:
        selected = [r for r in rows if r['split'] == split]
        for r in selected:
            m = metadata[split + '/' + r['clip_name']]
            ids = encode_bpe(r['english_translation'], vocab, merges, 2 * len(r['english_translation']) + 2)[1:-1]
            if not ids:
                raise ValueError('Empty encoded target')
            required = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
            record = dict(split=split, clip_name=r['clip_name'], **m,
                          words=len(re.findall(r"\b\w+(?:'\w+)*\b", r['english_translation'])),
                          bpe_tokens=len(ids), target_with_markers=len(ids) + 2,
                          unknown_tokens=ids.count(vocab['<unk>']), required_ctc_steps=required,
                          tokens_per_second=len(ids) / m['seconds'], tokens_per_frame=len(ids) / m['frames'])
            for stride in (1, 2, 4):
                record[f'ctc_impossible_stride_{stride}'] = required > math.ceil(m['frames'] / stride)
            detailed.append(record)
        items = [r for r in detailed if r['split'] == split]
        stats[split] = {field: distribution([r[field] for r in items]) for field in
                        ('frames', 'fps', 'seconds', 'words', 'bpe_tokens', 'target_with_markers', 'tokens_per_second', 'tokens_per_frame')}
        stats[split]['ctc_impossible'] = {str(s): sum(r[f'ctc_impossible_stride_{s}'] for r in items) for s in (1, 2, 4)}
        stats[split]['truncated_at_limit_including_markers'] = {str(n): sum(r['target_with_markers'] > n for r in items) for n in (80, 160)}
        stats[split]['unknown_tokens'] = sum(r['unknown_tokens'] for r in items)
        stats[split]['sentences_with_unknown'] = sum(r['unknown_tokens'] > 0 for r in items)
    with (out / 'pose_clip_audit.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed[0])); writer.writeheader(); writer.writerows(detailed)
    # Train-text-only fixed modal sentence and sampled bigram reference; no val tuning.
    frequencies = Counter(r['english_translation'] for r in train)
    modal = sorted(frequencies, key=lambda s: (-frequencies[s], s))[0]
    transitions = defaultdict(Counter)
    lengths = []
    for r in train:
        words = tokenize(r['english_translation'])
        lengths.append(len(words))
        for a, b in zip(['<start>'] + words, words + ['<end>']):
            transitions[a][b] += 1
    tables = {key: (sorted(c), [c[w] for w in sorted(c)]) for key, c in transitions.items()}
    max_length = int(math.ceil(np.percentile(lengths, 99)))
    predictions = []
    for r in val:
        seed = int(hashlib.sha256(f"{args.seed}:{r['clip_name']}".encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        previous, words = '<start>', []
        for _ in range(max_length):
            choices, weights = tables[previous]
            previous = rng.choices(choices, weights=weights, k=1)[0]
            if previous == '<end>':
                break
            words.append(previous)
        hypothesis = re.sub(r'\s+([.,!?;:])', r'\1', ' '.join(words))
        predictions.append(dict(clip_name=r['clip_name'], reference=r['english_translation'], modal=modal, bigram=hypothesis))
    baseline = {'method': 'Training-only modal sentence and seeded sampled word/punctuation bigram; no video or validation tuning.',
                'training_clips': len(train), 'bigram_max_tokens_train_p99': max_length, 'modal_sentence': modal}
    for name, selected in [('full_validation', predictions), ('v4_validation_512', [r for r in predictions if r['clip_name'] in val512])]:
        baseline[name] = {method: metrics([r['reference'] for r in selected], [r[method] for r in selected]) for method in ('modal', 'bigram')}
    with (out / 'pose_lm_predictions.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(predictions[0])); w.writeheader(); w.writerows(predictions)
    for split in VIDEO_DIRS:
        selected = [r for r in rows if r['split'] == split]
        codes = Counter()
        for r in selected:
            match = re.fullmatch(re.escape(r['sentence_id']) + r'-(\d+)-rgb_front\.mp4', r['clip_name'])
            codes[match.group(1) if match else 'UNPARSED'] += 1
        identity[split]['filename_suffix_code_counts_UNVERIFIED'] = dict(sorted(codes.items()))
        identity[split]['unique_source_video_ids'] = len({r['video_id'] for r in selected})
    identity['conclusion'] = 'No explicit signer/speaker column in local TSVs. Numeric filename suffix codes exist but no local identity mapping establishes their meaning. Signer counts and overlap UNKNOWN; do not equate VIDEO_ID or suffix codes with people.'
    identity['source_video_id_overlap_not_signer_overlap'] = sorted({r['video_id'] for r in train} & {r['video_id'] for r in val})
    identity['unverified_suffix_code_overlap'] = sorted(set(identity['train']['filename_suffix_code_counts_UNVERIFIED']) & set(identity['validation']['filename_suffix_code_counts_UNVERIFIED']))
    test_candidates = [str(p.relative_to(ROOT)) for p in (ROOT / 'data').rglob('*') if 'test' in p.name.lower()]
    source_paths = [Path(__file__), ROOT / 'modules/sentence_model.py', MANIFEST, SUBSET, TOKENIZER] + list((RAW / 'annotations').glob('*.tsv'))
    report = dict(stage=1, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        provenance=dict(seed=args.seed, config={'workers': args.workers, 'device': 'cpu', 'ctc_strides': [1, 2, 4], 'ctc_length_rule': 'ceil(frames/stride), requires same-padding convs; includes adjacent repeats; no duration cap applied'},
                        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                        hashes={str(p.relative_to(ROOT)): sha(p) for p in source_paths},
                        versions={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'opencv-python', 'sacrebleu', 'mediapipe')},
                        tokenizer_sha256=tokenizer_hash),
        identity=identity, manifest_disk=disk, sentence_overlap=overlap,
        frequent_sentences={split: {kind: Counter(fn(r['english_translation']) for r in rows if r['split'] == split).most_common(20)
                                   for kind, fn in [('exact', lambda s: s), ('normalized', normalize)]} for split in VIDEO_DIRS},
        distributions=stats, language_only_baselines=baseline,
        test_split={'candidates_under_data': test_candidates, 'conclusion': 'No test-named files/directories under data; official test split not found.' if not test_candidates else 'Candidates require inspection; see paths.'},
        limitations=['Video frame count/fps from OpenCV metadata; first frame decoded for every clip, not full-stream corruption detection.',
                     'Annotation text/IDs/timestamps checked, not semantic video/text alignment.',
                     'Signer independence unknown. Keypoints remove RGB backgrounds/clothing but can retain identity cues.',
                     'Text-only baselines are weak references, not an upper bound for ignoring video.'],
        future_success_criteria={'declared_before_pose_training': True, 'correct_vs_zero': 'at least +2 BLEU OR +4 chrF',
            'correct_vs_shuffled': 'at least 20% relative drop in BOTH generated BLEU and chrF for each of 3 shuffle seeds; shuffled loss >1% higher',
            'diversity': 'report unique sentences, lengths, first-five-word concentration and repeated ngrams; qualitative sanity check',
            'training_seeds': '2; roughly consistent, both must meet numeric criteria', 'if_failed': 'Do not scale; report evidence and propose controlled experiment'})
    text = '# Pose pipeline — Stage 1 audit\n\n'
    text += identity['conclusion'] + '\n\n'
    text += '| Split | Clips checked | Median seconds | p99 seconds | p99 frames | p99 target tokens incl. markers | CTC invalid stride 2 / 4 |\n|---|---:|---:|---:|---:|---:|---|\n'
    for split, s in stats.items():
        text += f"| {split} | {counts[split]} | {s['seconds']['p50']:.2f} | {s['seconds']['p99']:.2f} | {s['frames']['p99']:.0f} | {s['target_with_markers']['p99']:.0f} | {s['ctc_impossible']['2']} / {s['ctc_impossible']['4']} |\n"
    text += '\n## Sentence overlap\n\n' + json.dumps(overlap, indent=2) + '\n\n## Text-only reference\n\n'
    for group in ('full_validation', 'v4_validation_512'):
        for method, result in baseline[group].items():
            text += f"- {group}, {method}: BLEU {result['bleu']:.4f}; chrF {result['chrf']:.4f}.\n"
    text += '\n## Details and limits\n\n' + '\n'.join('- ' + s for s in report['limitations'])
    text += '\n- ' + report['test_split']['conclusion']
    text += '\n- Complete distributions, top 20 sentences, filename codes, provenance and metric signatures: `audit.json`.\n- Per-clip CTC flags, rates and lengths: `pose_clip_audit.csv`.\n- Text-only predictions: `pose_lm_predictions.csv`.\n'
    text += '\n## Reproduce / resume\n\n```powershell\n.\\.venv\\Scripts\\python.exe -m modules.audit_how2sign_splits --resume\n```\n\nCompleted reports are protected. For a new independent audit use `--output output/pose_audit_repeat`. Interrupted probes resume after size/mtime and metadata validation. No new dependencies installed.\n\n## Next stage\n\nAwait user confirmation before implementing/executing the 20-clip keypoint benchmark. No extraction or training started.\n'
    for filename in ('audit.json', 'report.json'):
        with (out / filename).open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, allow_nan=False)
    for filename in ('AUDIT.md', 'RESULTS.md'):
        with (out / filename).open('x', encoding='utf-8') as f:
            f.write(text)
    say('Stage 1 complete. Reports saved; stop before Stage 2 pending user confirmation.')


if __name__ == '__main__':
    main()
