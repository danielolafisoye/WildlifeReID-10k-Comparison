"""Runtime configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    # dummy = no weights needed, lets the whole stack run end to end while the real models train.
    # miewid | convnext = the two candidates from the proposal (see reid/model.py).
    MODEL_NAME = os.getenv("REID_MODEL", "dummy")
    DEVICE = os.getenv("REID_DEVICE", "cpu")
    CHECKPOINT = os.getenv("REID_CHECKPOINT", "")  # optional fine-tuned weights

    GALLERY_DIR = Path(os.getenv("REID_GALLERY_DIR", BASE_DIR / "data" / "gallery"))

    # Open-set score (max_sim or margin) and its threshold tau. Left empty, both come from the
    # exported gallery, where they were chosen on validation data for the model being served.
    SCORE = os.getenv("REID_SCORE", "")
    THRESHOLD = float(os.environ["REID_THRESHOLD"]) if os.getenv("REID_THRESHOLD") else None
    TOP_K = int(os.getenv("REID_TOP_K", "5"))

    MAX_UPLOAD_MB = int(os.getenv("REID_MAX_UPLOAD_MB", "10"))
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
