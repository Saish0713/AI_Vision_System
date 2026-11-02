"""Load and run the trained static ASL letter classifier."""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from modules.sign_features import normalize_hand_landmarks


class StaticASLClassifier:
    def __init__(self, model_path: Path, labels_path: Path, neighbors: int = 5):
        if not model_path.is_file():
            raise FileNotFoundError(f"ASL classifier not found: {model_path}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"ASL label map not found: {labels_path}")

        self.model = cv2.ml.KNearest_load(str(model_path))
        self.id_to_label = json.loads(labels_path.read_text(encoding="utf-8"))
        self.neighbors = neighbors

    def predict(self, landmarks, image_width: int, image_height: int):
        values = normalize_hand_landmarks(landmarks, image_width, image_height)
        sample = np.asarray([values], dtype=np.float32)
        _, prediction, neighbor_ids, _ = self.model.findNearest(
            sample, k=self.neighbors
        )
        predicted_id = str(int(prediction[0, 0]))
        neighbor_labels = [
            self.id_to_label[str(int(value))] for value in neighbor_ids[0]
        ]
        counts = Counter(neighbor_labels)
        label, votes = counts.most_common(1)[0]
        # Resolve exact vote ties using the classifier's own top prediction.
        if sum(count == votes for count in counts.values()) > 1:
            label = self.id_to_label[predicted_id]
        confidence = votes / self.neighbors
        return label, confidence


class TorchStaticASLClassifier:
    """Run the trained PyTorch landmark model on CUDA when available."""

    def __init__(self, model_path: Path):
        import torch
        from torch import nn

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "landmark_mlp_v1":
            raise ValueError(f"Unsupported PyTorch ASL model: {model_path}")

        hidden_dims = checkpoint["hidden_dims"]

        class LandmarkMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(checkpoint["input_size"], hidden_dims[0]),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dims[0], hidden_dims[1]),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dims[1], len(checkpoint["labels"])),
                )

            def forward(self, features):
                return self.layers(features)

        self.torch = torch
        self.labels = checkpoint["labels"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LandmarkMLP()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

    def predict(self, landmarks, image_width: int, image_height: int):
        values = normalize_hand_landmarks(landmarks, image_width, image_height)
        sample = self.torch.tensor([values], dtype=self.torch.float32, device=self.device)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(sample), dim=1)[0]
            confidence, predicted_id = probabilities.max(dim=0)
        return self.labels[predicted_id.item()], confidence.item()


def load_static_asl_classifier(model_path: Path, labels_path: Path | None = None):
    """Select the matching classifier loader from the model file extension."""
    if model_path.suffix.lower() == ".pt":
        return TorchStaticASLClassifier(model_path)
    return StaticASLClassifier(
        model_path,
        labels_path or model_path.with_name("asl_label_map.json"),
    )
