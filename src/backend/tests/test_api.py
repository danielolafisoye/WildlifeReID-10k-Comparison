import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from config import Config  # noqa: E402
from reid.gallery import Gallery  # noqa: E402
from reid.model import DummyEmbedder  # noqa: E402
from reid.openset import decide  # noqa: E402


def make_image(kind: str) -> bytes:
    """Two visually different synthetic images (horizontal vs vertical gradient)."""
    ramp = np.linspace(0, 255, 64, dtype=np.uint8)
    arr = np.tile(ramp, (64, 1)) if kind == "h" else np.tile(ramp[:, None], (1, 64))
    img = Image.fromarray(np.stack([arr, arr, arr], axis=-1))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client():
    embedder = DummyEmbedder()
    vecs, ids = [], []
    for ident, kind in [("animal_h", "h"), ("animal_v", "v")]:
        img = Image.open(io.BytesIO(make_image(kind))).convert("RGB")
        vecs.append(embedder.embed(img))
        ids.append(ident)
    gallery = Gallery(np.stack(vecs), ids, [f"{i}.png" for i in ids], model_name="dummy")
    return create_app(Config, embedder=embedder, gallery=gallery).test_client()


def post(client, data, name="q.png", **form):
    return client.post("/api/predict", data={"image": (io.BytesIO(data), name), **form}, content_type="multipart/form-data")


def test_health(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok" and body["gallery_identities"] == 2


def test_known_individual(client):
    body = post(client, make_image("h")).get_json()
    assert body["status"] == "known"
    assert body["prediction"]["identity"] == "animal_h"
    assert body["top_k"][0]["identity"] == "animal_h"


def test_unknown_when_threshold_not_met(client):
    body = post(client, make_image("h"), threshold="1.1").get_json()
    assert body["status"] == "unknown" and body["reason"] == "below_threshold"


def test_rejects_non_image(client):
    res = post(client, b"not an image", name="q.png")
    assert res.status_code == 400


def test_rejects_wrong_extension(client):
    res = post(client, make_image("h"), name="q.gif")
    assert res.status_code == 415


def test_missing_file(client):
    assert client.post("/api/predict").status_code == 400


def test_decide_empty_gallery():
    assert decide([], 0.65)["reason"] == "gallery_empty"
