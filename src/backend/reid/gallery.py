"""Precomputed gallery of known individuals and cosine-similarity search (step 4)."""
import json
from pathlib import Path

import numpy as np

EMBEDDINGS_FILE = "embeddings.npy"
META_FILE = "meta.json"


class Gallery:
    """Embeddings are stored L2-normalised, so a dot product is the cosine similarity.

    For a large gallery (the benchmark has >140k images) swap `search` for a FAISS index; the
    interface stays the same.
    """

    def __init__(self, embeddings: np.ndarray, identities: list[str], image_ids: list[str], model_name: str = ""):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.identities = list(identities)
        self.image_ids = list(image_ids)
        self.model_name = model_name
        if len(self.identities) != len(self.embeddings) or len(self.image_ids) != len(self.embeddings):
            raise ValueError("embeddings, identities and image_ids must have the same length")

    def __len__(self) -> int:
        return len(self.identities)

    @property
    def num_identities(self) -> int:
        return len(set(self.identities))

    @classmethod
    def empty(cls) -> "Gallery":
        return cls(np.zeros((0, 0), dtype=np.float32), [], [])

    @classmethod
    def load(cls, directory: Path) -> "Gallery":
        directory = Path(directory)
        emb_path, meta_path = directory / EMBEDDINGS_FILE, directory / META_FILE
        if not emb_path.exists() or not meta_path.exists():
            return cls.empty()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(np.load(emb_path), meta["identities"], meta["image_ids"], meta.get("model", ""))

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / EMBEDDINGS_FILE, self.embeddings)
        meta = {"identities": self.identities, "image_ids": self.image_ids, "model": self.model_name}
        (directory / META_FILE).write_text(json.dumps(meta), encoding="utf-8")

    def search(self, query: np.ndarray, k: int = 5) -> list[dict]:
        """Return the top-k identities, each with its best image score, highest first."""
        if len(self) == 0:
            return []
        scores = self.embeddings @ np.asarray(query, dtype=np.float32)
        best: dict[str, tuple[float, str]] = {}
        for idx in np.argsort(-scores):
            identity = self.identities[idx]
            if identity not in best:
                best[identity] = (float(scores[idx]), self.image_ids[idx])
                if len(best) >= k:
                    break
        return [{"identity": ident, "score": round(s, 4), "image_id": img} for ident, (s, img) in best.items()]
