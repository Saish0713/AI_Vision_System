"""Shared landmark features for ASL static-letter data and recognition."""

import math


STATIC_ASL_LETTERS = tuple(letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter not in "JZ")
FEATURE_NAMES = [
    f"{axis}{index}"
    for index in range(21)
    for axis in ("x", "y", "z")
]
CSV_FIELDS = ["label", "hand", *FEATURE_NAMES]


def normalize_hand_landmarks(landmarks, image_width: int, image_height: int) -> list[float]:
    """Return 63 wrist-centered, palm-size-normalized landmark coordinates."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    scale_x = image_width
    scale_y = image_height

    wrist_x = wrist.x * scale_x
    wrist_y = wrist.y * scale_y
    palm_size = math.hypot(
        (middle_mcp.x - wrist.x) * scale_x,
        (middle_mcp.y - wrist.y) * scale_y,
    )
    if palm_size < 1e-6:
        raise ValueError("Hand landmarks have an invalid palm size")

    return [
        coordinate
        for point in landmarks
        for coordinate in (
            (point.x * scale_x - wrist_x) / palm_size,
            (point.y * scale_y - wrist_y) / palm_size,
            (point.z * scale_x - wrist.z * scale_x) / palm_size,
        )
    ]
