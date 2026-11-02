"""Sample ordered RGB frame sequences from a manageable How2Sign subset."""

import argparse
import csv
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "how2sign"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "how2sign_manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "how2sign_sequences"
SEQUENCE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "how2sign_sequence_manifest.csv"


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"How2Sign clip manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {
            "split", "clip_name", "archive_path", "english_translation"
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def sample_subset(
    rows: list[dict[str, str]],
    split: str,
    limit: int,
    seed: int,
) -> list[dict[str, str]]:
    candidates = [row for row in rows if row["split"] == split]
    if limit < 1:
        raise ValueError("Subset limits must be positive")
    rng = random.Random(seed + (0 if split == "train" else 1))
    rng.shuffle(candidates)
    return sorted(candidates[:limit], key=lambda row: row["clip_name"])


def decode_sequence(
    archive: zipfile.ZipFile,
    archive_member: str,
    temp_dir: Path,
    frame_count: int,
    frame_size: int,
) -> tuple[np.ndarray, float, int]:
    temp_video = temp_dir / Path(archive_member).name
    with archive.open(archive_member) as source, temp_video.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)

    capture = cv2.VideoCapture(str(temp_video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open clip {archive_member}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total_frames < 1:
            raise RuntimeError(f"Could not read frame count for {archive_member}")
        if fps <= 0:
            fps = 30.0

        target_indices = np.rint(
            np.linspace(0, total_frames - 1, frame_count)
        ).astype(np.int64)
        selected = []
        target_position = 0
        frame_index = 0
        while target_position < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            while (
                target_position < frame_count
                and target_indices[target_position] == frame_index
            ):
                height, width = frame.shape[:2]
                scale = frame_size / max(height, width)
                resized = cv2.resize(
                    frame,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                canvas = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
                y = (frame_size - resized.shape[0]) // 2
                x = (frame_size - resized.shape[1]) // 2
                canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                selected.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                target_position += 1
            frame_index += 1

        if target_position != frame_count:
            raise RuntimeError(
                f"Expected {frame_count} sampled frames, got {target_position} "
                f"from {archive_member}"
            )
        return np.stack(selected), fps, total_frames
    finally:
        capture.release()
        temp_video.unlink(missing_ok=True)


def process_split(
    rows: list[dict[str, str]],
    split: str,
    archive_path: Path,
    limit: int,
    frame_count: int,
    frame_size: int,
    seed: int,
    output_root: Path,
) -> list[dict[str, str]]:
    selected = sample_subset(rows, split, limit, seed)
    if not archive_path.is_file():
        raise FileNotFoundError(f"How2Sign archive not found: {archive_path}")
    split_output = output_root / split
    split_output.mkdir(parents=True, exist_ok=True)
    output_records = []

    with tempfile.TemporaryDirectory(prefix="how2sign_clip_") as temporary:
        temp_dir = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            archive_names = set(archive.namelist())
            for index, row in enumerate(selected, start=1):
                member = row["archive_path"]
                if member not in archive_names:
                    print(f"Skipping missing clip: {member}")
                    continue
                frames, fps, original_frames = decode_sequence(
                    archive,
                    member,
                    temp_dir,
                    frame_count,
                    frame_size,
                )
                sequence_name = Path(row["clip_name"]).stem + ".npz"
                sequence_path = split_output / sequence_name
                np.savez_compressed(
                    sequence_path,
                    frames=frames,
                    fps=np.float32(fps),
                    original_frame_count=np.int32(original_frames),
                )
                output_records.append(
                    {
                        "split": split,
                        "clip_name": row["clip_name"],
                        "sequence_path": str(sequence_path.relative_to(PROJECT_ROOT)),
                        "frame_count": str(frame_count),
                        "frame_size": str(frame_size),
                        "fps": f"{fps:.3f}",
                        "original_frame_count": str(original_frames),
                        "english_translation": row["english_translation"],
                    }
                )
                if index % 25 == 0 or index == len(selected):
                    print(f"{split}: processed {index}/{len(selected)} clips")
    return output_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ordered, fixed-length RGB sequences from How2Sign clips."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
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
    parser.add_argument("--max-train", type=int, default=256)
    parser.add_argument("--max-validation", type=int, default=64)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--sequence-manifest", type=Path, default=SEQUENCE_MANIFEST)
    args = parser.parse_args()
    if args.frames < 2 or args.size < 32:
        parser.error("--frames must be at least 2 and --size at least 32")
    if not args.output_root.is_absolute():
        args.output_root = PROJECT_ROOT / args.output_root
    if not args.sequence_manifest.is_absolute():
        args.sequence_manifest = PROJECT_ROOT / args.sequence_manifest

    rows = read_manifest(args.manifest)
    train_rows = process_split(
        rows,
        "train",
        args.train_archive,
        args.max_train,
        args.frames,
        args.size,
        args.seed,
        args.output_root,
    )
    val_rows = process_split(
        rows,
        "validation",
        args.val_archive,
        args.max_validation,
        args.frames,
        args.size,
        args.seed,
        args.output_root,
    )
    args.sequence_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.sequence_manifest.open("w", newline="", encoding="utf-8") as output_file:
        fields = [
            "split", "clip_name", "sequence_path", "frame_count", "frame_size",
            "fps", "original_frame_count", "english_translation",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(train_rows)
        writer.writerows(val_rows)
    print(
        f"Saved {len(train_rows)} train and {len(val_rows)} validation sequences "
        f"to {args.sequence_manifest}"
    )
    print("Sequences are RGB uint8 arrays; each retains its original frame order.")


if __name__ == "__main__":
    main()
