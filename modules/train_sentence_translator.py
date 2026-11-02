"""Train an experimental How2Sign video-to-English baseline."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from modules.sentence_model import (
    PAD_TOKEN,
    VideoToTextModel,
    build_vocabulary,
    encode_text,
    greedy_decode,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "how2sign_sequence_manifest.csv"
DEFAULT_OUTPUT = ROOT / "models" / "how2sign_video_to_text_demo.pt"


class SequenceDataset(Dataset):
    def __init__(self, rows, vocabulary, max_tokens):
        self.rows = rows
        self.vocabulary = vocabulary
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = ROOT / row["sequence_path"]
        with np.load(path) as archive:
            frames = archive["frames"].copy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
        target = torch.tensor(
            encode_text(row["english_translation"], self.vocabulary, self.max_tokens),
            dtype=torch.long,
        )
        return frames, target


def collate_batch(items, pad_id):
    frames, targets = zip(*items)
    max_length = max(target.numel() for target in targets)
    padded = torch.full((len(targets), max_length), pad_id, dtype=torch.long)
    for index, target in enumerate(targets):
        padded[index, : target.numel()] = target
    return torch.stack(frames), padded


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Manifest must contain train and validation rows")
    return train, validation


def run_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    for frames, targets in loader:
        frames, targets = frames.to(device), targets.to(device)
        decoder_input, expected = targets[:, :-1], targets[:, 1:]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(frames, decoder_input)
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-target-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_target_tokens < 3:
        parser.error("epochs/batch-size must be positive and max-target-tokens >= 3")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_rows, val_rows = read_rows(args.manifest)
    vocabulary = build_vocabulary([row["english_translation"] for row in train_rows])
    id_to_token = [token for token, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
    pad_id = vocabulary[PAD_TOKEN]
    train_data = SequenceDataset(train_rows, vocabulary, args.max_target_tokens)
    val_data = SequenceDataset(val_rows, vocabulary, args.max_target_tokens)
    collate = lambda items: collate_batch(items, pad_id)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoToTextModel(len(vocabulary), pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print(f"Device: {device}; train: {len(train_rows)}; validation: {len(val_rows)}; vocab: {len(vocabulary)}")

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
                    "max_target_tokens": args.max_target_tokens,
                    "architecture": "FrameCNN-BiGRU-attention-GRU",
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
    print("Validation examples (reference => greedy output):")
    for index in range(min(5, len(val_rows))):
        frames, _ = val_data[index]
        prediction = greedy_decode(model, frames.unsqueeze(0).to(device), id_to_token, args.max_target_tokens)
        print(f"- {val_rows[index]['english_translation']} => {prediction}")


if __name__ == "__main__":
    main()
