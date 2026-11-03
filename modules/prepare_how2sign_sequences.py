"""Sample ordered RGB frame sequences from a manageable How2Sign subset."""

import argparse
import csv
import random
import shutil
import tempfile
import zipfile
from contextlib import nullcontext
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "how2sign"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "how2sign_manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "how2sign_sequences"
SEQUENCE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "how2sign_sequence_manifest.csv"
HAND_LANDMARKER_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"


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


def decode_video_path(
    video_path: Path,
    frame_count: int,
    frame_size: int,
    landmarker=None,
) -> tuple[np.ndarray, float, int, np.ndarray | None, np.ndarray | None]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open clip {video_path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total_frames < 1:
            raise RuntimeError(f"Could not read frame count for {video_path}")
        if fps <= 0:
            fps = 30.0

        target_indices = np.rint(
            np.linspace(0, total_frames - 1, frame_count)
        ).astype(np.int64)
        selected = []
        hand_landmarks = (
            np.zeros((frame_count, 2, 21, 3), dtype=np.float32)
            if landmarker is not None else None
        )
        hand_presence = (
            np.zeros((frame_count, 2), dtype=np.float32)
            if landmarker is not None else None
        )
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
                if landmarker is not None:
                    rgb_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_original)
                    )
                    for points, handedness in zip(result.hand_landmarks, result.handedness):
                        label = handedness[0].category_name.lower() if handedness else ""
                        slot = 0 if label == "left" else 1
                        hand_landmarks[target_position, slot] = [
                            (point.x, point.y, point.z) for point in points
                        ]
                        hand_presence[target_position, slot] = 1.0
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
                f"from {video_path}"
            )
        return np.stack(selected), fps, total_frames, hand_landmarks, hand_presence
    finally:
        capture.release()


def decode_sequence(
    archive: zipfile.ZipFile,
    archive_member: str,
    temp_dir: Path,
    frame_count: int,
    frame_size: int,
    landmarker=None,
):
    temp_video = temp_dir / Path(archive_member).name
    with archive.open(archive_member) as source, temp_video.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    try:
        return decode_video_path(temp_video, frame_count, frame_size, landmarker)
    finally:
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
    hand_landmarker_model: Path | None,
    video_root: Path | None,
    resume: bool,
) -> list[dict[str, str]]:
    selected = sample_subset(rows, split, limit, seed)
    if video_root is not None and not video_root.is_dir():
        raise FileNotFoundError(f"Extracted video directory not found: {video_root}")
    if video_root is None and not archive_path.is_file():
        raise FileNotFoundError(f"How2Sign archive not found: {archive_path}")
    split_output = output_root / split
    split_output.mkdir(parents=True, exist_ok=True)
    output_records = []
    detected_frames = 0
    sampled_frames = 0

    if hand_landmarker_model is not None:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_landmarker_model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
        )
        landmarker_context = mp.tasks.vision.HandLandmarker.create_from_options(options)
    else:
        landmarker_context = nullcontext(None)

    archive_context = (
        nullcontext(None)
        if video_root is not None
        else zipfile.ZipFile(archive_path)
    )

    with tempfile.TemporaryDirectory(prefix="how2sign_clip_") as temporary:
        temp_dir = Path(temporary)
        with landmarker_context as landmarker, archive_context as archive:
            archive_names = set(archive.namelist()) if archive is not None else None
            for index, row in enumerate(selected, start=1):
                member = row["archive_path"]
                sequence_name = Path(row["clip_name"]).stem + ".npz"
                sequence_path = split_output / sequence_name
                if resume and sequence_path.is_file():
                    with np.load(sequence_path) as cached:
                        cache_usable = (
                            cached["frames"].shape
                            == (frame_count, frame_size, frame_size, 3)
                            and (
                                hand_landmarker_model is None
                                or "hand_landmarks" in cached.files
                            )
                        )
                        if cache_usable:
                            fps = float(cached["fps"])
                            original_frames = int(cached["original_frame_count"])
                        if cache_usable and "hand_presence" in cached.files:
                            cached_presence = cached["hand_presence"]
                            detected_frames += int((cached_presence.sum(axis=1) > 0).sum())
                            sampled_frames += frame_count
                    if not cache_usable:
                        print(f"Reprocessing incompatible cached sequence: {sequence_path.name}")
                    else:
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
                            print(f"{split}: resumed {index}/{len(selected)} clips")
                        continue

                if video_root is not None:
                    video_path = video_root / Path(member)
                    # Some extraction tools strip the archive's raw_videos/ folder.
                    if not video_path.is_file():
                        video_path = video_root / Path(member).name
                    if not video_path.is_file():
                        print(f"Skipping missing extracted clip: {video_path}")
                        continue
                    frames, fps, original_frames, hand_landmarks, hand_presence = decode_video_path(
                        video_path, frame_count, frame_size, landmarker
                    )
                else:
                    if member not in archive_names:
                        print(f"Skipping missing clip: {member}")
                        continue
                    frames, fps, original_frames, hand_landmarks, hand_presence = decode_sequence(
                        archive, member, temp_dir, frame_count, frame_size, landmarker
                    )
                sequence_arrays = {
                    "frames": frames,
                    "fps": np.float32(fps),
                    "original_frame_count": np.int32(original_frames),
                }
                if hand_landmarks is not None:
                    sequence_arrays["hand_landmarks"] = hand_landmarks
                    sequence_arrays["hand_presence"] = hand_presence
                    detected_frames += int((hand_presence.sum(axis=1) > 0).sum())
                    sampled_frames += frame_count
                np.savez_compressed(sequence_path, **sequence_arrays)
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
    if hand_landmarker_model is not None:
        print(
            f"{split}: hands detected in {detected_frames}/{sampled_frames} sampled frames "
            f"({detected_frames / max(sampled_frames, 1):.1%})"
        )
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
    parser.add_argument("--train-video-root", type=Path)
    parser.add_argument("--val-video-root", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing NPZ sequences in the output folder and continue preparing missing clips.",
    )
    parser.add_argument(
        "--with-hand-landmarks",
        action="store_true",
        help="Extract pretrained MediaPipe hand landmarks from each sampled source frame.",
    )
    parser.add_argument("--hand-landmarker-model", type=Path, default=HAND_LANDMARKER_MODEL)
    args = parser.parse_args()
    if args.frames < 2 or args.size < 32:
        parser.error("--frames must be at least 2 and --size at least 32")
    if not args.output_root.is_absolute():
        args.output_root = PROJECT_ROOT / args.output_root
    if not args.sequence_manifest.is_absolute():
        args.sequence_manifest = PROJECT_ROOT / args.sequence_manifest
    if args.train_video_root is not None and not args.train_video_root.is_absolute():
        args.train_video_root = PROJECT_ROOT / args.train_video_root
    if args.val_video_root is not None and not args.val_video_root.is_absolute():
        args.val_video_root = PROJECT_ROOT / args.val_video_root
    hand_model = args.hand_landmarker_model
    if not hand_model.is_absolute():
        hand_model = PROJECT_ROOT / hand_model
    if args.with_hand_landmarks and not hand_model.is_file():
        parser.error(f"Hand Landmarker model not found: {hand_model}")
    hand_model_arg = hand_model if args.with_hand_landmarks else None

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
        hand_model_arg,
        args.train_video_root,
        args.resume,
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
        hand_model_arg,
        args.val_video_root,
        args.resume,
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
    if args.with_hand_landmarks:
        print("Sequences also include MediaPipe hand landmarks and per-hand presence flags.")


if __name__ == "__main__":
    main()
