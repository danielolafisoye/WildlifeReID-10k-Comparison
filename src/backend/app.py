"""Flask REST API for open-set animal re-identification.

    POST /api/predict   multipart form field "image" (optional "threshold") -> ranked matches or unknown
    GET  /api/health    liveness + model / gallery info
    GET  /api/config    settings the UI needs (threshold, top-k, accepted types)
"""
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import Config
from reid.gallery import Gallery
from reid.model import load_embedder
from reid.openset import FALLBACK_THRESHOLDS, SCORES, decide
from reid.preprocessing import InvalidImageError, load_image


def create_app(config=Config, embedder=None, gallery=None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    CORS(app, origins=config.CORS_ORIGINS)

    # Loaded once at start-up, not per request (gallery embeddings are precomputed).
    app.extensions["embedder"] = embedder or load_embedder(config.MODEL_NAME, config.DEVICE, config.CHECKPOINT)
    app.extensions["gallery"] = gallery if gallery is not None else Gallery.load(config.GALLERY_DIR)

    gallery_default = app.extensions["gallery"].default_score
    score_name = config.SCORE or (gallery_default if gallery_default in SCORES else "max_sim")
    if score_name not in SCORES:
        raise ValueError(f"Unknown REID_SCORE '{score_name}'. Use one of: {', '.join(SCORES)}.")
    default_threshold = config.THRESHOLD
    if default_threshold is None:
        default_threshold = app.extensions["gallery"].thresholds.get(score_name, FALLBACK_THRESHOLDS[score_name])
    default_threshold = round(float(default_threshold), 4)

    def error(message: str, status: int):
        return jsonify({"error": message}), status

    @app.get("/api/health")
    def health():
        gal = app.extensions["gallery"]
        return jsonify({
            "status": "ok",
            "model": app.extensions["embedder"].name,
            "gallery_images": len(gal),
            "gallery_identities": gal.num_identities,
        })

    @app.get("/api/config")
    def get_config():
        return jsonify({
            "threshold": default_threshold,
            "score": score_name,
            "threshold_max": 1.0 if score_name == "max_sim" else 0.5,
            "top_k": config.TOP_K,
            "max_upload_mb": config.MAX_UPLOAD_MB,
            "allowed_extensions": sorted(config.ALLOWED_EXTENSIONS),
        })

    @app.post("/api/predict")
    def predict():
        upload = request.files.get("image")
        if upload is None or upload.filename == "":
            return error("No image uploaded. Send a multipart form field named 'image'.", 400)
        ext = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
        if ext not in config.ALLOWED_EXTENSIONS:
            return error(f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}.", 415)

        try:
            threshold = float(request.form.get("threshold", default_threshold))
        except ValueError:
            return error("threshold must be a number.", 400)

        started = time.perf_counter()
        try:
            image = load_image(upload.read())  # 1. validate and load
        except InvalidImageError as exc:
            return error(str(exc), 400)

        query = app.extensions["embedder"].embed(image)  # 2-3. preprocess + embedding
        candidates = app.extensions["gallery"].search(query, max(config.TOP_K, 2))  # 4-5. similarity search + ranking
        decision = decide(candidates, threshold, score_name)  # 6. open-set threshold
        candidates = candidates[: config.TOP_K]

        return jsonify({  # 7. JSON result
            "status": decision["status"],
            "prediction": {"identity": decision["identity"], "score": decision["score"]},
            "reason": decision["reason"],
            "top_k": candidates,
            "score": score_name,
            "decision_score": decision["decision_score"],
            "threshold": threshold,
            "model": app.extensions["embedder"].name,
            "inference_ms": round((time.perf_counter() - started) * 1000, 1),
        })

    @app.errorhandler(413)
    def too_large(_):
        return error(f"Image is larger than {config.MAX_UPLOAD_MB} MB.", 413)

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
