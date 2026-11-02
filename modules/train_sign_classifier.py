"""Train and evaluate a lightweight OpenCV k-nearest-neighbors classifier."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from modules.sign_features import STATIC_ASL_LETTERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = [
    f"{axis}{index}"
    for index in range(21)
    for axis in ("x", "y", "z")
]


def read_split(csv_path: Path, wanted_split: str):
    features = []
    labels = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        expected = {"split", "label", *FEATURE_NAMES}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected landmark CSV format: {csv_path}")
        for row in reader:
            if row["split"] != wanted_split:
                continue
            label = row["label"].upper()
            if label not in STATIC_ASL_LETTERS:
                continue
            features.append([float(row[name]) for name in FEATURE_NAMES])
            labels.append(label)

    if not features:
        raise ValueError(f"No {wanted_split} samples found in {csv_path}")
    return np.asarray(features, dtype=np.float32), labels


def train(csv_path: Path, model_path: Path, labels_path: Path, neighbors: int) -> None:
    train_x, train_labels = read_split(csv_path, "train")
    test_x, test_labels = read_split(csv_path, "test")
    label_to_id = {label: index for index, label in enumerate(STATIC_ASL_LETTERS)}
    id_to_label = {str(index): label for label, index in label_to_id.items()}

    missing = sorted(set(STATIC_ASL_LETTERS) - set(train_labels))
    if missing:
        raise ValueError(f"Training data is missing letters: {', '.join(missing)}")

    train_y = np.asarray(
        [[label_to_id[label]] for label in train_labels], dtype=np.float32
    )
    model = cv2.ml.KNearest_create()
    model.setIsClassifier(True)
    model.setDefaultK(neighbors)
    if not model.train(train_x, cv2.ml.ROW_SAMPLE, train_y):
        raise RuntimeError("OpenCV could not train the classifier")

    _, predictions, neighbor_ids, _ = model.findNearest(test_x, k=neighbors)
    predicted_labels = [id_to_label[str(int(value))] for value in predictions[:, 0]]
    accuracy = sum(
        actual == predicted for actual, predicted in zip(test_labels, predicted_labels)
    ) / len(test_labels)
    per_letter = {}
    for letter in STATIC_ASL_LETTERS:
        indices = [i for i, label in enumerate(test_labels) if label == letter]
        if indices:
            per_letter[letter] = sum(
                test_labels[i] == predicted_labels[i] for i in indices
            ) / len(indices)

    # Store the model plus an explicit label map used by live inference.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    labels_path.write_text(
        json.dumps(id_to_label, indent=2), encoding="utf-8"
    )

    print(f"Training samples: {len(train_labels):,}")
    print(f"Evaluation samples: {len(test_labels):,}")
    print(f"Top-{neighbors} neighbor accuracy: {accuracy:.1%}")
    print("Per-letter accuracy:")
    print("  " + "  ".join(f"{letter}: {per_letter.get(letter, 0):.0%}" for letter in STATIC_ASL_LETTERS))
    print(f"Model saved to: {model_path}")
    print(f"Label map saved to: {labels_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an ASL static-letter classifier from landmark CSV data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "signalphaset_landmarks.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "asl_knn.xml",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "models" / "asl_label_map.json",
    )
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()
    if args.neighbors < 1:
        parser.error("--neighbors must be at least 1")
    train(args.data, args.model, args.labels, args.neighbors)


if __name__ == "__main__":
    main()
