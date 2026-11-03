# AI Vision System

A Python computer vision project for detecting hand landmarks and recognizing static American Sign Language (ASL) fingerspelling from a webcam or video.

The project also contains an experimental video-to-English sentence translation pipeline. That work is an early prototype and is not a usable translator.

## Features

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
