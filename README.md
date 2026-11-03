# AI Vision System

A Python computer vision project for detecting hand landmarks and recognizing static American Sign Language (ASL) fingerspelling from a webcam or video.

The project also contains an experimental video-to-English sentence translation pipeline. That work is an early prototype and is not a usable translator.

## Features

Latest sentence experiments: a fresh 32-clip attention model memorized all targets and responds to shuffled inputs. A CTC model reached 13/32 exact after 200 epochs, below the predeclared scaling threshold. The larger comparison has not started. See [experiment results](output/visual_grounding/RESULTS.md) for the protocol, limits and saved predictions.

Sentence training now defaults to `--max-target-tokens 160`, including sentence markers, and reports truncation before training. This covers 99.85% of the v4 training targets. Existing checkpoints retain their saved limits. Commands explicitly specifying 80 override the new default.

- Detects up to two hands and displays their landmarks.
- Recognizes 24 static ASL letters: **A–I** and **K–Y**. **J** and **Z** are excluded because they require motion.
- Supports OpenCV KNN and PyTorch landmark classifiers.
- Can use a webcam or a recorded video as input.
- Includes scripts for collecting landmark samples, preparing data, and training models.

## Requirements

- Windows
- Python 3.11.9
- A webcam for live detection
- The project’s model files and data for running inference or training

Model weights and datasets are intentionally not included in this repository. You’ll need to provide those files locally before running the project.

## Setup

From the project directory, create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run hand tracking

The hand tracking script requires the MediaPipe Hand Landmarker model at `models\hand_landmarker.task`.

With a webcam and a local PyTorch classifier:

```powershell
python -m modules.hand_tracking --camera 0 --classifier models\asl_mlp_cuda.pt
```

With a recorded video:

```powershell
python -m modules.hand_tracking --video path\to\video.mp4 --classifier models\asl_mlp_cuda.pt
```

The classifier path is optional if you only want to view hand landmarks. Press **Q** to close the preview.

The repository does not include `hand_landmarker.task` or classifier weights. Add the required model files to the local `models/` directory before running inference.

## Training

Training scripts expect the relevant datasets to exist locally under `data/`. The dataset is excluded from this repository.

For example, the static-letter pipeline uses:

```powershell
python -m modules.prepare_signalphaset
python -m modules.train_sign_classifier
```

The PyTorch landmark classifier can be trained with:

```powershell
python -m modules.train_torch_sign_classifier
```

Check each script’s options and expected input files before running it.

## Experimental sentence translation

The repository includes scripts for preparing How2Sign video sequences and training a video-to-English baseline. This pipeline is experimental; its current model outputs are not reliable translations.

The How2Sign data and trained checkpoints are not included. The preparation and training scripts require local data files.

## Project structure

```text
modules/       Hand tracking, recognition, data preparation, and training code
requirements.txt
test_setup.py
```

Dataset files, trained model weights, generated outputs, and temporary files are kept out of source control.

## Limitations

- Static fingerspelling recognition does not handle motion-based letters **J** or **Z**.
- The static classifier does not recognize complete words or sentences.
- The sentence translation pipeline is a prototype and should not be treated as a working translator.
- Recognition quality depends on the model, input conditions, and training data supplied locally.


## 2026-11-03 — Pose direction and completed Stage 1 audit

This supersedes the earlier next step of tuning the 32-clip CTC model. We are moving to full-frame-rate body, hand and reduced-face keypoints. Existing RGB/hand data and experiment checkpoints are preserved.

- Audited 31,047 training and 1,739 validation videos: no missing/extra MP4s, no manifest/annotation mismatches, no metadata/first-frame decode errors. This is not full-video corruption or semantic-alignment verification.
- The source annotations contain 118 train and 2 validation rows outside the existing manifest, as previously recorded; their clips are not in the extracted folders.
- Signer counts/overlap remain unknown. No explicit signer/speaker columns exist; numeric filename codes lack a verified local identity mapping. Source-video ID overlap is zero, which does not prove signer independence.
- Validation sentences appearing in training: exact 43/1,739 (2.47%); lowercased/punctuation-stripped 51/1,739 (2.93%). Existing 512 subset: exact 9 (1.76%), normalized 10 (1.95%).
- Median duration: train 5.08 s, validation 5.00 s. Training p99: 25.42 s and about 682 frames; validation p99: 22.15 s and about 615 frames. FPS varies from about 23.976 to 60; do not assume 30 fps.
- With the saved 512-piece v4 tokenizer, CTC-invalid targets at stride 1/2/4: train 39/214/9,976; validation 0/20/650. Counts include adjacent repeats and assume ceil(frame_count/stride), with no duration cap. Stride 4 plus mandatory auxiliary CTC is not feasible for many clips. Do not silently omit these losses, truncate targets or repeat cached frames.
- Full-target lengths including BOS/EOS: p99 train 114, validation 103. A 160-token cap still truncates 49 train and 1 validation targets; 80 truncates 1,296 and 69. Report any chosen future cap explicitly. Unknown tokens: 46 across 41 training sentences and 6 across 6 validation sentences. These findings alone do not prove the vocabulary is the translation bottleneck.
- Training-text-only sampled bigram reference: full validation BLEU 0.1776 / chrF 13.5456; fixed 512 BLEU 0.2295 / chrF 13.9378. Modal sentence reference is also saved. These are weak references, not an upper bound on text-only performance.
- No official test split was found under local data. Nothing downloaded; no dependencies installed; no GPU training or keypoint extraction started.
- Four helper tests passed. Saved scores, per-clip CTC flags, source hashes and cache integrity were independently checked.
- Stage 6 thresholds are recorded before pose training in output/pose/report.json. Pose-only inputs still may retain signer identity cues.

Reports: `output/pose/AUDIT.md`, `output/pose/audit.json`, `output/pose/RESULTS.md`. Per-clip flags: `output/pose/pose_clip_audit.csv`. Text-only predictions: `output/pose/pose_lm_predictions.csv`. Verification: `output/pose/pose_verification.json`.

Reproduce with the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m modules.audit_how2sign_splits --resume
.\.venv\Scripts\python.exe -m modules.pose_audit_checks
```

`--resume` resumes an interrupted audit. A completed report is protected; use a new `--output` directory for a fresh audit. The CPU video probes persist every 100 clips and print progress/ETA; Ctrl+C stops safely. No new requirements are needed (existing OpenCV, NumPy, PyTorch and SacreBLEU).

Next: await user confirmation for Stage 2's 20-clip extraction benchmark, then stop again to review time, storage, detection quality and handedness before subset extraction. The larger pose training stages have not started. The CTC feasibility failures must be resolved explicitly before enabling the auxiliary loss.


## 2026-11-03 — Stage 2 pose benchmark completed; larger extraction paused

- User approved continuing to the 20-clip benchmark. MediaPipe 1.1.1 has no legacy Holistic API; implemented Tasks-based every-frame pose/hand/70-point face extraction with separate process-owned clip tracking, zero/mask missing data, validated resume and explicit errors. Existing RGB/CTC experiments are preserved.
- 20 clips / 2,613 frames completed in 93.22 seconds with 4 workers; 1.12 MiB cache total. Mean worker time 17.35 s/clip; effective wall time 4.66 s/clip.
- Presence: pose 100%, left hand 82.63%, right hand 86.11%, any hand 92.27%, face **11.98%**. Face coverage is too poor to commit to the full proposed face-feature extraction. Yellow pose-face points in overlays are not FaceLandmarker results.
- Raw handedness agreed with confident pose-wrist association 98.43%; no blanket mirroring/swap is applied. Five random overlays reviewed. All caches validated, including all 2,613 source frames re-decoded; resume reused all 20 clips without extraction.
- Frame-adjusted projections at the benchmark setup: 5,000/512 roughly 8.90 h and 0.376 GiB; all 32,786 roughly 52.80 h and 2.233 GiB. These estimates are tentative and face recovery would increase storage/time.
- Downloaded the missing official Google pose-lite (5,777,746 bytes) and face (3,758,596 bytes) model bundles. URLs/hashes: `output/pose/pose_model_downloads.json`. Existing hand model reused; no Python packages installed.
- Full report and exact commands: `output/pose/benchmark/RESULTS.md`. Visuals: `output/pose/debug/pose_debug_contact_sheet.png`. Extraction entry point: `.\.venv\Scripts\python.exe -m modules.extract_pose_keypoints --benchmark --limit 20 --workers 4 --seed 42 --resume`. Completed reports are protected; interrupted jobs resume, and a fresh report needs a new `--output` directory.
- Next proposed controlled experiment: pose-guided face-region crops on the same 20 clips, preserving current caches and checking coordinate remapping/visual quality. Await user approval. **No larger subset extraction or training has started.**
