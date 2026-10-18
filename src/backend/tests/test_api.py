import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "#9f7a52").save(buffer, format="PNG")
    return buffer.getvalue()


class TestConfig:
    DEVICE = "cpu"
    DEFAULT_MODEL = "convnext"
    CONVNEXT_CHECKPOINT = Path("missing.pth")
    CONVNEXT_LABEL_MAP = Path("missing.json")
    MIEWID_DIR = Path("missing-miewid")
    MIEWID_CHECKPOINT = Path("missing-miewid/miewid_weights.pth")
    MIEWID_GALLERY = Path("missing-miewid/known_gallery.pth")
    DATASET_ROOT = Path("missing-dataset")
    DATASET_ARCHIVE = Path("missing.zip")
    THRESHOLD = 0.50
    TOP_K = 5
    UNKNOWN_TEST_IDENTITIES = 20
    MAX_UPLOAD_MB = 10
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    CORS_ORIGINS = ["http://localhost:4173"]


class FakePredictor:
    def __init__(self, confidence=0.82):
        self.confidence = confidence

    def predict(self, image, top_k):
        ranked = [
            {"identity": f"animal_{index}", "confidence": max(0.01, self.confidence - index * 0.1)}
            for index in range(top_k)
        ]
        return {
            "predicted_identity": ranked[0]["identity"],
            "confidence": ranked[0]["confidence"],
            "top_k": ranked,
        }


class FakeCatalog:
    def list_images(self, page, page_size, query_type):
        return {
            "items": [
                {
                    "id": "42",
                    "query_type": "known",
                    "image_url": "/api/test-images/42/image",
                }
            ],
            "page": page,
            "page_size": page_size,
            "total": 1,
            "pages": 1,
        }

    def image_bytes(self, query_id):
        if query_id != "42":
            raise KeyError(query_id)
        return image_bytes(), "animal.png", "image/png"

    def identity_image_bytes(self, identity):
        if not identity.startswith("animal_"):
            raise KeyError(identity)
        return image_bytes(), "reference.png", "image/png"


def client(confidence=0.82, include_miewid=False):
    predictors = {"convnext": FakePredictor(confidence)}
    if include_miewid:
        predictors["miewid"] = FakePredictor(confidence)
    app = create_app(
        TestConfig,
        predictors=predictors,
        catalog=FakeCatalog(),
    )
    return app.test_client()


def test_health_and_model_availability():
    body = client().get("/api/health").get_json()
    assert body["status"] == "ok"
    assert body["dataset_available"] is True
    assert body["models"][0]["available"] is True
    assert body["models"][1]["available"] is False


def test_upload_prediction_returns_frontend_contract():
    response = client().post(
        "/api/predict",
        data={
            "model": "convnext",
            "image": (io.BytesIO(image_bytes()), "query.png"),
        },
        content_type="multipart/form-data",
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["predicted_identity"] == "animal_0"
    assert body["confidence"] == pytest.approx(0.82)
    assert body["is_unknown"] is False
    assert body["top_5"] == [f"animal_{index}" for index in range(5)]
    assert body["top_5_details"][0]["image_url"].endswith("identity=animal_0")


def test_miewid_uses_the_same_frontend_contract():
    api = client(include_miewid=True)
    health = api.get("/api/health").get_json()
    assert health["models"][1]["available"] is True

    response = api.post(
        "/api/predict",
        data={
            "model": "miewid",
            "image": (io.BytesIO(image_bytes()), "query.png"),
        },
        content_type="multipart/form-data",
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["model"] == "miewid"
    assert body["predicted_identity"] == "animal_0"
    assert body["top_5"][0] == "animal_0"


def test_low_confidence_is_unknown_but_keeps_best_guess():
    response = client(0.40).post(
        "/api/predict",
        data={
            "model": "convnext",
            "image": (io.BytesIO(image_bytes()), "query.png"),
        },
        content_type="multipart/form-data",
    )
    body = response.get_json()
    assert body["is_unknown"] is True
    assert body["predicted_identity"] == "animal_0"


def test_predicts_selected_test_image():
    response = client().post(
        "/api/predict",
        data={"model": "convnext", "test_image_id": "42"},
    )
    assert response.status_code == 200
    assert response.get_json()["source"] == "test"


def test_lists_and_serves_test_images():
    api = client()
    listing = api.get("/api/test-images").get_json()
    assert listing["items"][0]["id"] == "42"
    assert api.get("/api/test-images/42/image").mimetype == "image/png"
    assert api.get("/api/identity-image?identity=animal_0").mimetype == "image/png"


def test_rejects_invalid_inputs():
    api = client()
    assert api.post("/api/predict").status_code == 400
    assert (
        api.post(
            "/api/predict",
            data={"image": (io.BytesIO(b"not an image"), "query.png")},
            content_type="multipart/form-data",
        ).status_code
        == 400
    )
    assert (
        api.post(
            "/api/predict",
            data={"image": (io.BytesIO(image_bytes()), "query.gif")},
            content_type="multipart/form-data",
        ).status_code
        == 415
    )
