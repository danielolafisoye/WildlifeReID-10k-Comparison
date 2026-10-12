"""Classifier adapters used by the Flask API."""

import importlib
import json
import sys
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


class MiewIDClassifier:
    """Adapt MiewID embedding retrieval to the API's ranked-result contract."""

    name = "miewid"
    display_name = "MiewID"

    def __init__(
        self,
        checkpoint: Path,
        gallery_path: Path,
        inference_dir: Path,
        unknown_threshold: float = 0.50,
        device: str = "auto",
    ):
        checkpoint = Path(checkpoint)
        gallery_path = Path(gallery_path)
        inference_dir = Path(inference_dir)

        for path, description in (
            (checkpoint, "checkpoint"),
            (gallery_path, "embedding gallery"),
            (inference_dir / "inference.py", "inference module"),
        ):
            if not path.exists():
                raise ModelUnavailableError(f"MiewID {description} not found: {path}")

        # MiewID remains self-contained in src/models/miewid. Import its
        # standalone inference module lazily so Flask can start without loading
        # the large model and gallery into memory.
        inference_dir_text = str(inference_dir.resolve())
        sys.path.insert(0, inference_dir_text)
        try:
            inference_module = importlib.import_module("inference")
            inference_class = inference_module.MiewIDInference
            self.inference = inference_class(
                weights_path=checkpoint,
                gallery_path=gallery_path,
                unknown_threshold=unknown_threshold,
                device=device,
            )
        except Exception as exc:
            raise ModelUnavailableError(
                "MiewID could not be loaded. Confirm the Git LFS files are present, "
                "then install src/backend/requirements.txt."
            ) from exc
        finally:
            if sys.path[0] == inference_dir_text:
                sys.path.pop(0)

    @staticmethod
    def _display_score(similarity: float) -> float:
        # Cosine similarity is normally positive for the strongest gallery
        # match. Clamp rare negative values so the shared percentage UI remains
        # within its expected 0..1 range.
        return min(1.0, max(0.0, float(similarity)))

    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        result = self.inference.predict(image, top_k=top_k)
        ranked = [
            {
                "identity": match["identity"],
                "confidence": self._display_score(match["similarity"]),
            }
            for match in result["matches"]
        ]
        return {
            # Keep the best identity even below threshold so the frontend can
            # show it as a best guess while marking the query as unknown.
            "predicted_identity": result["best_match_identity"],
            "confidence": ranked[0]["confidence"],
            "top_k": ranked,
        }
