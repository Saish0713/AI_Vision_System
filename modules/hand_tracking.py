"""Display MediaPipe hand landmarks and optional ASL predictions."""

import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp

from modules.sign_recognition import load_static_asl_classifier


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def draw_hand_landmarks(frame, landmarks, handedness):
    """Draw the 21 normalized landmarks and their connections on a frame."""
    height, width = frame.shape[:2]
    points = [
        (int(point.x * width), int(point.y * height))
        for point in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 220, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 100, 255), -1)

    if handedness:
        label = handedness[0].category_name
        wrist_x, wrist_y = points[0]
        cv2.putText(
            frame,
            label,
            (wrist_x + 10, wrist_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def process_input(
    source,
    model_path: Path,
    camera: bool = False,
    classifier_path: Path | None = None,
    labels_path: Path | None = None,
) -> None:
    if not camera and not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Hand Landmarker model not found: {model_path}\n"
            "Download the official hand_landmarker.task model and place it there."
        )
    classifier = None
    if classifier_path is not None:
        classifier = load_static_asl_classifier(classifier_path, labels_path)

    capture = cv2.VideoCapture(source if camera else str(source))
    if not capture.isOpened():
        input_name = f"camera {source}" if camera else f"video: {source}"
        raise RuntimeError(f"Could not open {input_name}")

    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            histories = {}
            fps = 30.0 if camera else capture.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0

            while True:
                success, frame = capture.read()
                if not success:
                    break

                timestamp_ms = round(frame_index * 1000 / fps)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(image, timestamp_ms)

                for landmarks, handedness in zip(
                    result.hand_landmarks, result.handedness
                ):
                    draw_hand_landmarks(frame, landmarks, handedness)
                    if classifier is not None:
                        height, width = frame.shape[:2]
                        prediction, prediction_confidence = classifier.predict(
                            landmarks, width, height
                        )
                        hand = handedness[0].category_name if handedness else "hand"
                        history = histories.setdefault(hand, deque(maxlen=7))
                        history.append((prediction, prediction_confidence))
                        label_votes = Counter(item[0] for item in history)
                        label, votes = label_votes.most_common(1)[0]
                        matching_confidence = sum(
                            confidence
                            for predicted, confidence in history
                            if predicted == label
                        ) / votes
                        confidence = (votes / len(history)) * matching_confidence
                        text = (
                            f"ASL: {label} ({confidence:.0%})"
                            if confidence >= 0.6
                            else "ASL: uncertain"
                        )
                        wrist_x = int(landmarks[0].x * width)
                        wrist_y = int(landmarks[0].y * height)
                        cv2.putText(
                            frame,
                            text,
                            (wrist_x + 10, wrist_y + 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                if not result.hand_landmarks:
                    histories.clear()

                title = "ASL Sign Recognition - press Q to quit"
                cv2.imshow(title, frame)
                frame_index += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show hand landmarks and optional static ASL predictions."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Path to a recorded video file")
    source.add_argument(
        "--camera",
        type=int,
        metavar="INDEX",
        help="Camera index (Camo often appears as 0 or 1)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task",
        help="Path to the MediaPipe Hand Landmarker .task model",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        help="Optional trained ASL classifier (.xml for KNN or .pt for PyTorch)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional label-map JSON file (defaults beside the classifier)",
    )
    args = parser.parse_args()
    if args.camera is not None:
        process_input(
            args.camera,
            args.model,
            camera=True,
            classifier_path=args.classifier,
            labels_path=args.labels,
        )
    else:
        process_input(
            args.video,
            args.model,
            classifier_path=args.classifier,
            labels_path=args.labels,
        )


if __name__ == "__main__":
    main()
