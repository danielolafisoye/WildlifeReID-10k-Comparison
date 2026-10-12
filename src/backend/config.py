"""Runtime configuration for the inference API."""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]


class Config:
    DEVICE = os.getenv("REID_DEVICE", "auto")
    DEFAULT_MODEL = os.getenv("REID_DEFAULT_MODEL", "convnext")

    CONVNEXT_CHECKPOINT = Path(
        os.getenv(
            "CONVNEXT_CHECKPOINT",
            REPO_ROOT / "src" / "models" / "convnext" / "convnext_weights.pth",
        )
    )
    CONVNEXT_LABEL_MAP = Path(
        os.getenv(
            "CONVNEXT_LABEL_MAP",
            REPO_ROOT / "src" / "models" / "convnext" / "label_map.json",
        )
    )

    MIEWID_DIR = Path(
        os.getenv(
            "MIEWID_DIR",
            REPO_ROOT / "src" / "models" / "miewid",
        )
    )
    MIEWID_CHECKPOINT = Path(
        os.getenv(
            "MIEWID_CHECKPOINT",
            MIEWID_DIR / "miewid_weights.pth",
        )
    )
    MIEWID_GALLERY = Path(
        os.getenv(
            "MIEWID_GALLERY",
            MIEWID_DIR / "known_gallery.pth",
        )
    )

    DATASET_ROOT = Path(
        os.getenv("WILDLIFE_DATASET_ROOT", REPO_ROOT / "data" / "wildlifereid-10k")
    )
    DATASET_ARCHIVE = Path(
        os.getenv("WILDLIFE_DATASET_ARCHIVE", REPO_ROOT / "data" / "archive.zip")
    )
    KAGGLE_DATASET_HANDLE = os.getenv(
        "WILDLIFE_KAGGLE_DATASET",
        "wildlifedatasets/wildlifereid-10k/versions/10",
    )
    KAGGLE_ON_DEMAND = os.getenv("WILDLIFE_KAGGLE_ON_DEMAND", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    THRESHOLD = float(os.getenv("REID_THRESHOLD", "0.50"))
    TOP_K = int(os.getenv("REID_TOP_K", "5"))
    UNKNOWN_TEST_IDENTITIES = int(os.getenv("REID_UNKNOWN_TEST_IDENTITIES", "20"))

    MAX_UPLOAD_MB = int(os.getenv("REID_MAX_UPLOAD_MB", "10"))
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:4173,http://127.0.0.1:4173",
        ).split(",")
        if origin.strip()
    ]
