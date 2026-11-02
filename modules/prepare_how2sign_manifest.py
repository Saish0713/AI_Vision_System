"""Build a clip-to-English manifest for the local How2Sign train/val archives."""

import argparse
import csv
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "how2sign"
REQUIRED_COLUMNS = {
    "VIDEO_ID",
    "SENTENCE_ID",
    "SENTENCE_NAME",
    "START",
    "END",
    "SENTENCE",
}
MANIFEST_FIELDS = [
    "split",
    "video_id",
    "sentence_id",
    "clip_name",
    "archive_path",
    "source_start_seconds",
    "source_end_seconds",
    "english_translation",
]


def read_annotations(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"How2Sign annotations not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as annotation_file:
        reader = csv.DictReader(annotation_file, delimiter="\t")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Unexpected How2Sign annotation columns in {path}; "
                f"missing: {', '.join(sorted(missing))}"
            )
        return list(reader)


def build_split(
    split: str,
    annotations_path: Path,
    archive_path: Path,
) -> tuple[list[dict[str, str]], int, list[str]]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"How2Sign clip archive not found: {archive_path}")
    annotations = read_annotations(annotations_path)
    with zipfile.ZipFile(archive_path) as archive:
        available = set(archive.namelist())

    records = []
    missing_clip = []
    for row in annotations:
        clip_name = f"{row['SENTENCE_NAME']}.mp4"
        archive_member = f"raw_videos/{clip_name}"
        if archive_member not in available:
            missing_clip.append(clip_name)
            continue
        try:
            start = float(row["START"])
            end = float(row["END"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid timestamps for {clip_name} in {annotations_path}"
            ) from error
        if start < 0 or end <= start:
            raise ValueError(
                f"Invalid timestamp range for {clip_name}: {start} to {end}"
            )
        translation = row["SENTENCE"].strip()
        if not translation:
            continue
        records.append(
            {
                "split": split,
                "video_id": row["VIDEO_ID"],
                "sentence_id": row["SENTENCE_ID"],
                "clip_name": clip_name,
                "archive_path": archive_member,
                "source_start_seconds": f"{start:.3f}",
                "source_end_seconds": f"{end:.3f}",
                "english_translation": translation,
            }
        )
    return records, len(annotations), missing_clip


def prepare_manifest(
    train_annotations: Path,
    val_annotations: Path,
    train_archive: Path,
    val_archive: Path,
    output_path: Path,
) -> None:
    train_records, train_rows, missing_train = build_split(
        "train", train_annotations, train_archive
    )
    val_records, val_rows, missing_val = build_split(
        "validation", val_annotations, val_archive
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(train_records)
        writer.writerows(val_records)

    print(f"Training pairs: {len(train_records):,}/{train_rows:,}")
    print(f"Validation pairs: {len(val_records):,}/{val_rows:,}")
    print(f"Saved manifest: {output_path}")
    if missing_train or missing_val:
        print(
            "Annotations without matching clips: "
            f"train={len(missing_train):,}, validation={len(missing_val):,}"
        )
        for split, missing in (("train", missing_train), ("validation", missing_val)):
            if missing:
                print(f"  {split} examples: {', '.join(missing[:5])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join How2Sign clip archives to English sentence annotations."
    )
    parser.add_argument(
        "--train-annotations",
        type=Path,
        default=DATA_ROOT / "annotations" / "how2sign_train.tsv",
    )
    parser.add_argument(
        "--val-annotations",
        type=Path,
        default=DATA_ROOT / "annotations" / "how2sign_val.tsv",
    )
    parser.add_argument(
        "--train-archive",
        type=Path,
        default=DATA_ROOT / "archives" / "train_rgb_front_clips.zip",
    )
    parser.add_argument(
        "--val-archive",
        type=Path,
        default=DATA_ROOT / "archives" / "val_rgb_front_clips.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "how2sign_manifest.csv",
    )
    args = parser.parse_args()
    prepare_manifest(
        args.train_annotations,
        args.val_annotations,
        args.train_archive,
        args.val_archive,
        args.output,
    )


if __name__ == "__main__":
    main()
