"""Standalone MiewID embedding-gallery inference."""

import argparse
import io
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from model import MODEL_TAG, extract_embeddings, inference_transform, load_miewid


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = MODEL_DIR / "miewid_weights.pth"
DEFAULT_GALLERY = MODEL_DIR / "known_gallery.pth"
DEFAULT_UNKNOWN_THRESHOLD = 0.50


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_finetuned_model(weights_path=DEFAULT_WEIGHTS, device="auto"):
    device = choose_device(device)
    model = load_miewid(pretrained=True)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device


def load_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as opened:
            return opened.convert("RGB")
    if hasattr(image, "read"):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    raise TypeError("image must be a path, bytes, file-like object, or PIL image")


def build_gallery(
    model,
    gallery_loader,
    index_to_identity,
    output_path=DEFAULT_GALLERY,
    device="auto",
):
    """Embed every known training image and save the retrieval gallery."""
    device = choose_device(device) if isinstance(device, str) else device
    model.eval()
    all_embeddings = []
    all_identities = []

    with torch.inference_mode():
        for images, labels in tqdm(gallery_loader, desc="Building MiewID gallery"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                embeddings = extract_embeddings(model, images)
            all_embeddings.append(embeddings.cpu())
            all_identities.extend(index_to_identity[label] for label in labels.tolist())

    gallery = {
        "model_tag": MODEL_TAG,
        "embeddings": torch.cat(all_embeddings).float(),
        "identities": all_identities,
    }
    output_path = Path(output_path)
    torch.save(gallery, output_path)
    print(
        f"Saved {len(all_identities)} embeddings for "
        f"{len(set(all_identities))} identities to {output_path}",
        flush=True,
    )
    return gallery


def identity_matches(query_embedding, gallery, top_k=5):
    """Return the strongest cosine match for each known identity."""
    gallery_embeddings = F.normalize(gallery["embeddings"].float(), dim=1)
    query_embedding = F.normalize(query_embedding.float(), dim=1).cpu()
    similarities = (query_embedding @ gallery_embeddings.T).squeeze(0)

    best_by_identity = {}
    for identity, similarity in zip(gallery["identities"], similarities.tolist()):
        best_by_identity[identity] = max(
            similarity,
            best_by_identity.get(identity, float("-inf")),
        )

    ranked = sorted(best_by_identity.items(), key=lambda item: item[1], reverse=True)
    return [
        {"identity": identity, "similarity": float(similarity)}
        for identity, similarity in ranked[:top_k]
    ]


class MiewIDInference:
    def __init__(
        self,
        weights_path=DEFAULT_WEIGHTS,
        gallery_path=DEFAULT_GALLERY,
        unknown_threshold=DEFAULT_UNKNOWN_THRESHOLD,
        device="auto",
    ):
        self.model, self.device = load_finetuned_model(weights_path, device)
        self.transform = inference_transform()
        self.unknown_threshold = float(unknown_threshold)
        self.gallery = torch.load(gallery_path, map_location="cpu", weights_only=True)
        if not self.gallery.get("identities") or self.gallery["embeddings"].ndim != 2:
            raise ValueError("The MiewID gallery is empty or invalid.")

    def embed(self, image):
        tensor = self.transform(load_image(image)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):
                return extract_embeddings(self.model, tensor).cpu()

    def predict(self, image, top_k=5, unknown_threshold=None):
        threshold = self.unknown_threshold if unknown_threshold is None else float(unknown_threshold)
        matches = identity_matches(self.embed(image), self.gallery, top_k=top_k)
        best = matches[0]
        is_unknown = best["similarity"] < threshold
        return {
            "predicted_identity": "UNKNOWN" if is_unknown else best["identity"],
            "best_match_identity": best["identity"],
            "similarity": best["similarity"],
            "is_unknown": is_unknown,
            "unknown_threshold": threshold,
            "matches": matches,
        }


def main():
    parser = argparse.ArgumentParser(description="Run MiewID gallery inference.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--threshold", type=float, default=DEFAULT_UNKNOWN_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    predictor = MiewIDInference(
        weights_path=args.weights,
        gallery_path=args.gallery,
        unknown_threshold=args.threshold,
        device=args.device,
    )
    print(predictor.predict(args.image, top_k=args.top_k))


if __name__ == "__main__":
    main()
