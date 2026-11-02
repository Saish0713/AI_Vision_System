"""Convert the local SignAlphaSet archive into ASL landmark features."""

import argparse
import csv
import random
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import mediapipe as mp
import numpy as np

from modules.sign_features import STATIC_ASL_LETTERS, normalize_hand_landmarks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FEATURE_NAMES = [
    f"{axis}{index}"
    for index in range(21)
    for axis in ("x", "y", "z")
]
CSV_FIELDS = ["split", "label", "hand", "source", *FEATURE_NAMES]
TRAIN_NAMES = {"train", "training"}
TEST_NAMES = {"test", "testing", "valid", "validation"}


def label_from_path(path: str) -> str | None:
    for part in reversed(PurePosixPath(path.replace("\\", "/")).parts[:-1]):
        label = part.strip().upper()
        if label in STATIC_ASL_LETTERS:
            return label
    return None


def split_from_path(path: str) -> str | None:
    for part in PurePosixPath(path.replace("\\", "/")).parts[:-1]:
        name = part.strip().lower()
        if name in TRAIN_NAMES:
            return "train"
        if name in TEST_NAMES:
            return "test"
    return None


def make_fallback_splits(items: list[tuple[str, str]]) -> dict[str, str]:
    """Create a deterministic, per-letter 80/20 split if the source has none."""
    by_label = {label: [] for label in STATIC_ASL_LETTERS}
    for path, label in items:
        by_label[label].append(path)

    split_for = {}
    rng = random.Random(42)
    for label_paths in by_label.values():
        rng.shuffle(label_paths)
        test_count = max(1, round(len(label_paths) * 0.2)) if label_paths else 0
        for index, path in enumerate(label_paths):
            split_for[path] = "test" if index < test_count else "train"
    return split_for


def prepare_dataset(archive_path: Path, output_path: Path, model_path: Path) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Hand Landmarker model not found: {model_path}")

    with zipfile.ZipFile(archive_path) as archive:
        image_names = [
            name
            for name in archive.namelist()
            if PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS
            and label_from_path(name)
        ]
        if not image_names:
            raise ValueError(f"No labeled ASL image folders found in {archive_path}")

        items = [(name, label_from_path(name)) for name in image_names]
        fallback_splits = make_fallback_splits(items)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )

        totals = {"train": 0, "test": 0}
        skipped = 0
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            with vision.HandLandmarker.create_from_options(options) as landmarker:
                for index, (name, label) in enumerate(items, start=1):
                    try:
                        encoded = np.frombuffer(archive.read(name), dtype=np.uint8)
                        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    except (OSError, RuntimeError, zipfile.BadZipFile):
                        skipped += 1
                        continue
                    if frame is None:
                        skipped += 1
                        continue

                    height, width = frame.shape[:2]
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )
                    result = landmarker.detect(image)
                    if not result.hand_landmarks:
                        skipped += 1
                        continue

                    try:
                        features = normalize_hand_landmarks(
                            result.hand_landmarks[0], width, height
                        )
                    except ValueError:
                        skipped += 1
                        continue

                    split = split_from_path(name) or fallback_splits[name]
                    hand = result.handedness[0][0].category_name
                    writer.writerow(
                        {
                            "split": split,
                            "label": label,
                            "hand": hand,
                            "source": name,
                            **dict(zip(FEATURE_NAMES, features)),
                        }
                    )
                    totals[split] += 1
                    if index % 1000 == 0:
                        print(f"Processed {index:,}/{len(items):,} images")

        print(
            f"Finished: {totals['train']:,} train, {totals['test']:,} test rows; "
            f"skipped {skipped:,} images.\nSaved to: {output_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract normalized hand landmarks from SignAlphaSet."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "SignAlphaSet.zip",
        help="SignAlphaSet ZIP archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "signalphaset_landmarks.csv",
        help="CSV output path (kept separate from existing data files)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "hand_landmarker.task",
        help="MediaPipe Hand Landmarker .task model path",
    )
    args = parser.parse_args()
    prepare_dataset(args.archive, args.output, args.model)


if __name__ == "__main__":
    main()
