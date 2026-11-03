"""Compute standard SacreBLEU and chrF scores from saved predictions."""

import argparse
import csv
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "data" / "processed" / "how2sign_video_to_text_hands_v3_demo_validation_predictions.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="?", default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()
    if not args.predictions.is_absolute():
        args.predictions = ROOT / args.predictions
    with args.predictions.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or not {"reference", "hypothesis"}.issubset(rows[0]):
        raise ValueError("Predictions CSV must contain reference and hypothesis columns")
    references = [row["reference"] for row in rows]
    hypotheses = [row["hypothesis"] for row in rows]
    for metric in (BLEU(), CHRF()):
        score = metric.corpus_score(hypotheses, [references])
        print(f"{metric.__class__.__name__}: {score.score:.2f}")
        print(f"Signature: {metric.get_signature()}")
    print(f"Scored {len(rows)} validation examples from {args.predictions}")


if __name__ == "__main__":
    main()
