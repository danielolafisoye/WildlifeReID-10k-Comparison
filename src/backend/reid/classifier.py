"""Classifier adapters used by the Flask API."""

import json
from pathlib import Path

from PIL import Image

from .preprocessing import convnext_transform


class ModelUnavailableError(RuntimeError):
    """Raised when a requested model cannot be loaded."""


def _resolve_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ConvNeXtClassifier:
    name = "convnext"
    display_name = "ConvNeXt Base"

    def __init__(self, checkpoint: Path, label_map_path: Path, device: str = "auto"):
        import torch
        import torch.nn as nn
        from torchvision.models import convnext_base

        checkpoint = Path(checkpoint)
        label_map_path = Path(label_map_path)
        if not checkpoint.exists():
            raise ModelUnavailableError(f"ConvNeXt checkpoint not found: {checkpoint}")
        if not label_map_path.exists():
            raise ModelUnavailableError(f"ConvNeXt label map not found: {label_map_path}")

        label_data = json.loads(label_map_path.read_text(encoding="utf-8"))
        self.labels = list(label_data["index_to_identity"])
        if not self.labels or len(self.labels) != len(set(self.labels)):
            raise ModelUnavailableError("ConvNeXt label map is empty or contains duplicate identities.")

        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ModelUnavailableError(
                "ConvNeXt checkpoint could not be read. If Git LFS is in use, run 'git lfs pull'."
            ) from exc
        if "model_state_dict" in state:
            state = state["model_state_dict"]

        classifier_weight = state.get("classifier.2.weight")
        if classifier_weight is None:
            raise ModelUnavailableError("Checkpoint does not contain a ConvNeXt classifier head.")
        num_classes = int(classifier_weight.shape[0])
        if num_classes != len(self.labels):
            raise ModelUnavailableError(
                f"Checkpoint has {num_classes} outputs but label map has {len(self.labels)} identities."
            )

        model = convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        model.load_state_dict(state, strict=True)

        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.model = model.to(self.device).eval()
        self.transform = convnext_transform()
        self.num_classes = num_classes

    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(tensor)
            probabilities = self.torch.softmax(logits, dim=1)
            scores, indexes = probabilities.topk(min(top_k, self.num_classes), dim=1)

        ranked = [
            {
                "identity": self.labels[int(index)],
                "confidence": float(score),
            }
            for score, index in zip(scores[0].cpu(), indexes[0].cpu())
        ]
        return {
            "predicted_identity": ranked[0]["identity"],
            "confidence": ranked[0]["confidence"],
            "top_k": ranked,
        }
