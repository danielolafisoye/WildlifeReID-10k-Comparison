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

    # Open-set threshold tau. The WildlifeReID-10k paper reports a stable region of about
    # 0.6-0.7 for MiewID; treat this only as a starting point and tune per model on a
    # validation split built from the training identities.
    THRESHOLD = float(os.getenv("REID_THRESHOLD", "0.65"))
    TOP_K = int(os.getenv("REID_TOP_K", "5"))

    MAX_UPLOAD_MB = int(os.getenv("REID_MAX_UPLOAD_MB", "10"))
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
