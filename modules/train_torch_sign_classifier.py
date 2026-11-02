"""Train a PyTorch MLP for static ASL letters from normalized landmarks."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from modules.sign_features import FEATURE_NAMES, STATIC_ASL_LETTERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIDDEN_DIMS = (128, 64)


class LandmarkMLP(nn.Module):
    def __init__(self, input_size: int, class_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIMS[1], class_count),
        )

    def forward(self, features):
        return self.layers(features)


def read_split(
    csv_path: Path,
    wanted_split: str,
    label_to_id: dict[str, int],
    allow_empty: bool = False,
):
    features = []
    labels = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        expected = {"split", "label", *FEATURE_NAMES}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected landmark CSV format: {csv_path}")
        for row in reader:
            label = row["label"].upper()
            if row["split"] != wanted_split or label not in label_to_id:
                continue
            features.append([float(row[name]) for name in FEATURE_NAMES])
            labels.append(label_to_id[label])

    if not features and not allow_empty:
        raise ValueError(f"No {wanted_split} samples found in {csv_path}")
    return (
        torch.tensor(
            np.asarray(features, dtype=np.float32).reshape(-1, len(FEATURE_NAMES)),
            dtype=torch.float32,
        ),
        torch.tensor(labels, dtype=torch.long),
    )


def train(
    csv_path: Path,
    extra_data_paths: list[Path],
    model_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_to_id = {label: index for index, label in enumerate(STATIC_ASL_LETTERS)}
    train_x, train_y = read_split(csv_path, "train", label_to_id)
    test_x, test_y = read_split(csv_path, "test", label_to_id)
    personal_train = []
    personal_eval = []
    for extra_path in extra_data_paths:
        extra_train_x, extra_train_y = read_split(
            extra_path, "train", label_to_id, allow_empty=True
        )
        if len(extra_train_y):
            personal_train.append((extra_train_x, extra_train_y))
        extra_test = read_split(
            extra_path, "test", label_to_id, allow_empty=True
        )
        if len(extra_test[1]):
            personal_eval.append(extra_test)
    if personal_train:
        train_x = torch.cat([train_x, *(part[0] for part in personal_train)])
        train_y = torch.cat([train_y, *(part[1] for part in personal_train)])
    missing_ids = set(range(len(STATIC_ASL_LETTERS))) - set(train_y.tolist())
    if missing_ids:
        missing = [STATIC_ASL_LETTERS[index] for index in sorted(missing_ids)]
        raise ValueError(f"Training data is missing letters: {', '.join(missing)}")

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    model = LandmarkMLP(len(FEATURE_NAMES), len(STATIC_ASL_LETTERS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    personal_train_count = sum(len(part[1]) for part in personal_train)
    personal_eval_count = sum(len(part[1]) for part in personal_eval)
    print(
        f"Training samples: {len(train_y):,} (including {personal_train_count:,} personal); "
        f"public image-split evaluation samples: {len(test_y):,}"
    )
    if personal_eval_count:
        print(f"Personal held-out evaluation samples: {personal_eval_count:,}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        print(f"Epoch {epoch:02d}/{epochs}: loss={total_loss / len(train_y):.4f}")

    model.eval()
    with torch.inference_mode():
        logits = model(test_x.to(device))
        predictions = logits.argmax(dim=1).cpu()
    accuracy = (predictions == test_y).float().mean().item()
    print(f"Public image-split evaluation accuracy: {accuracy:.1%}")
    print("Per-letter accuracy:")
    for label, label_id in label_to_id.items():
        mask = test_y == label_id
        if mask.any():
            letter_accuracy = (predictions[mask] == test_y[mask]).float().mean().item()
            print(f"  {label}: {letter_accuracy:.1%}")

    personal_accuracy = None
    if personal_eval:
        personal_test_x = torch.cat([part[0] for part in personal_eval]).to(device)
        personal_test_y = torch.cat([part[1] for part in personal_eval])
        with torch.inference_mode():
            personal_predictions = model(personal_test_x).argmax(dim=1).cpu()
        personal_accuracy = (
            (personal_predictions == personal_test_y).float().mean().item()
        )
        print(f"Personal held-out evaluation accuracy: {personal_accuracy:.1%}")
        print("Personal per-letter accuracy:")
        for label, label_id in label_to_id.items():
            mask = personal_test_y == label_id
            if mask.any():
                score = (personal_predictions[mask] == personal_test_y[mask]).float().mean().item()
                print(f"  {label}: {score:.1%}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "landmark_mlp_v1",
            "input_size": len(FEATURE_NAMES),
            "hidden_dims": list(HIDDEN_DIMS),
            "labels": list(STATIC_ASL_LETTERS),
            "state_dict": model.cpu().state_dict(),
            "evaluation_accuracy": accuracy,
            "personal_evaluation_accuracy": personal_accuracy,
            "personal_training_samples": personal_train_count,
            "personal_evaluation_samples": personal_eval_count,
            "seed": seed,
        },
        model_path,
    )
    print(f"Model saved to: {model_path}")
    print("Note: this is an image-level split, not signer-independent evaluation.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a CUDA-enabled PyTorch ASL landmark classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "signalphaset_landmarks.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "asl_mlp_cuda.pt",
    )
    parser.add_argument(
        "--extra-data",
        type=Path,
        nargs="*",
        default=[],
        help="Optional compatible CSVs from the sample collector; train rows augment training, test rows are evaluated separately",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs and batch-size must be positive; learning-rate must be > 0")
    train(
        args.data,
        args.extra_data,
        args.model,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.seed,
    )


if __name__ == "__main__":
    main()
