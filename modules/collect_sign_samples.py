"""Collect labeled MediaPipe landmark samples from a camera without saving frames."""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from modules.sign_features import FEATURE_NAMES, STATIC_ASL_LETTERS, normalize_hand_landmarks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = ["split", "label", "hand", "source", *FEATURE_NAMES]


def collect(
    camera_index: int,
    label: str | None,
    split: str,
    session: str,
    output_path: Path,
    sample_count: int,
    interval: float,
    model_path: Path,
) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Hand Landmarker model not found: {model_path}")
    if sample_count < 1 or interval <= 0:
        raise ValueError("sample-count and interval must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    saved_by_label = {letter: 0 for letter in STATIC_ASL_LETTERS}
    single_label = label is not None
    active_label = label
    last_sample_at = 0.0
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    try:
        with output_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                frame_index = 0
                while not single_label or saved_by_label[label] < sample_count:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError("Camera stopped before collection completed")

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = round(frame_index * 1000 / fps)
                    result = landmarker.detect_for_video(image, timestamp_ms)
                    frame_index += 1

                    if active_label is None:
                        status = "Choose an ASL letter with your keyboard"
                        help_text = "A-I or K-Y; hold the sign and vary it slightly; Q quits"
                    else:
                        status = (
                            f"{split.upper()} {active_label}: "
                            f"{saved_by_label[active_label]}/{sample_count}"
                        )
                        help_text = "Hold the sign; vary angle/distance slightly; Q quits"
                    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, help_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 2, cv2.LINE_AA)

                    now = time.monotonic()
                    if (
                        active_label is not None
                        and saved_by_label[active_label] < sample_count
                        and result.hand_landmarks
                        and now - last_sample_at >= interval
                    ):
                        height, width = frame.shape[:2]
                        try:
                            features = normalize_hand_landmarks(
                                result.hand_landmarks[0], width, height
                            )
                        except ValueError:
                            features = None
                        if features is not None:
                            hand = (
                                result.handedness[0][0].category_name
                                if result.handedness and result.handedness[0]
                                else "unknown"
                            )
                            writer.writerow({
                                "split": split,
                                "label": active_label,
                                "hand": hand,
                                "source": session,
                                **dict(zip(FEATURE_NAMES, features)),
                            })
                            csv_file.flush()
                            saved_by_label[active_label] += 1
                            last_sample_at = now

                    cv2.imshow("ASL Landmark Sample Collector", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if label is None and key != 255:
                        selected = chr(key).upper()
                        if selected in STATIC_ASL_LETTERS:
                            active_label = selected
                            last_sample_at = 0.0
    finally:
        capture.release()
        cv2.destroyAllWindows()

    saved_total = sum(saved_by_label.values())
    counts = ", ".join(
        f"{letter}={count}" for letter, count in saved_by_label.items() if count
    ) or "none"
    print(f"Saved {saved_total} {split} samples to {output_path} ({counts})")
    if single_label and saved_by_label[label] < sample_count:
        print("Collection ended early; rerun to collect more samples.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect one labeled ASL letter as hand landmarks; video is not saved."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (Camo is often 0)")
    parser.add_argument(
        "--label", choices=STATIC_ASL_LETTERS,
        help="Collect one letter only; omit to select letters using keyboard keys",
    )
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument(
        "--session",
        help="Unique capture-session name; defaults to split_letter_timestamp",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "personal_landmarks.csv",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--interval", type=float, default=0.15,
                        help="Seconds between saved samples")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "hand_landmarker.task",
    )
    args = parser.parse_args()
    session = args.session or f"{args.split}_{args.label}_{datetime.now():%Y%m%d_%H%M%S}"
    collect(
        args.camera,
        args.label,
        args.split,
        session,
        args.output,
        args.samples,
        args.interval,
        args.model,
    )


if __name__ == "__main__":
    main()
