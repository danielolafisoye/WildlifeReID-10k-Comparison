"""Feature extractors. MiewID and ConvNeXt Base are interchangeable behind one interface, so the
two candidates from the proposal are trained/evaluated separately and only the winner is served.

Every embedder returns a 1-D, L2-normalised float32 vector, so cosine similarity is a dot product.
"""
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from .preprocessing import to_chw_array


def l2_normalise(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Embedder(ABC):
    name: str = "base"

    @abstractmethod
    def embed(self, image: Image.Image) -> np.ndarray:
        """Return an L2-normalised embedding for one RGB image."""


class DummyEmbedder(Embedder):
    """Weight-free stand-in so the API and UI can be built and tested before training.

    It is NOT a re-identification model: it just compares coarse colour layout.
    """

    name = "dummy"

    def embed(self, image: Image.Image) -> np.ndarray:
        small = np.asarray(image.resize((8, 8), Image.BILINEAR), dtype=np.float32) / 255.0
        flat = small.reshape(-1)
        return l2_normalise(flat - flat.mean())


class _TorchEmbedder(Embedder):
    input_size = 224

    def __init__(self, device: str = "cpu", checkpoint: str = ""):
        import torch  # imported lazily so the dummy model needs no torch install

        self.torch = torch
        self.device = torch.device(device)
        self.model = self._build().to(self.device).eval()
        if checkpoint:
            state = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state, strict=False)

    @abstractmethod
    def _build(self):
        """Return a torch module that maps (B, 3, H, W) to (B, D) embeddings."""

    def embed(self, image: Image.Image) -> np.ndarray:
        arr = to_chw_array(image, self.input_size)
        batch = self.torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            out = self.model(batch)
        return l2_normalise(out.squeeze(0).cpu().numpy())


class ConvNeXtEmbedder(_TorchEmbedder):
    """ConvNeXt Base, ImageNet pretrained. The classifier is removed so the backbone outputs embeddings.

    After fine-tuning with an ArcFace head, save only the backbone weights and point
    REID_CHECKPOINT at them.
    """

    name = "convnext"
    input_size = 224

    def _build(self):
        import timm

        return timm.create_model("convnext_base", pretrained=True, num_classes=0)


class MiewIDEmbedder(_TorchEmbedder):
    """MiewID-msv3 (EfficientNetV2 backbone, Sub-center ArcFace).

    TODO: confirm the model id and expected input size against the official model card
    (reference [8] in the proposal) before relying on the numbers below.
    """

    name = "miewid"
    input_size = 440
    model_id = "conservationxlabs/miewid-msv3"

    def _build(self):
        from transformers import AutoModel

        return AutoModel.from_pretrained(self.model_id, trust_remote_code=True)


def load_embedder(name: str, device: str = "cpu", checkpoint: str = "") -> Embedder:
    name = name.lower()
    if name == "dummy":
        return DummyEmbedder()
    if name == "convnext":
        return ConvNeXtEmbedder(device, checkpoint)
    if name == "miewid":
        return MiewIDEmbedder(device, checkpoint)
    raise ValueError(f"Unknown model '{name}'. Use dummy, convnext or miewid.")
