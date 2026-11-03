"""Generate and save validation hypotheses from a trained checkpoint."""

import argparse
import csv
from pathlib import Path

import torch

from modules.sentence_model import VideoToTextModel, greedy_decode
from modules.train_sentence_translator import read_rows, SequenceDataset


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    args.manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    args.output = args.output if args.output.is_absolute() else ROOT / args.output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocabulary = checkpoint["vocabulary"]
    merges = checkpoint["bpe_merges"]
    use_hands = checkpoint.get("use_hand_landmarks", False)
    model = VideoToTextModel(
        len(vocabulary),
        vocabulary["<pad>"],
        use_hand_landmarks=use_hands,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _, validation_rows = read_rows(args.manifest)
    dataset = SequenceDataset(
        validation_rows,
        vocabulary,
        merges,
        checkpoint["max_target_tokens"],
        use_hand_landmarks=use_hands,
    )
    id_to_token = [token for token, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["reference", "hypothesis"])
        writer.writeheader()
        for index, row in enumerate(validation_rows):
            item = dataset[index]
            frames = item[0].unsqueeze(0).to(device)
            if use_hands:
                hands = item[1].unsqueeze(0).to(device)
                presence = item[2].unsqueeze(0).to(device)
            else:
                hands = presence = None
            hypothesis = greedy_decode(
                model,
                frames,
                id_to_token,
                checkpoint["max_target_tokens"],
                hands,
                presence,
            )
            writer.writerow(
                {"reference": row["english_translation"], "hypothesis": hypothesis}
            )
            if (index + 1) % 32 == 0 or index + 1 == len(validation_rows):
                print(f"Generated {index + 1}/{len(validation_rows)} validation predictions")
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
