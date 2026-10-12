"""Flask API for open-set wildlife identity prediction."""

import io
import time

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from config import Config
from reid.classifier import ConvNeXtClassifier, MiewIDClassifier, ModelUnavailableError
from reid.dataset import DatasetUnavailableError, TestImageCatalog
from reid.preprocessing import InvalidImageError, load_image


def create_app(config=Config, predictors=None, catalog=None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    CORS(app, origins=config.CORS_ORIGINS)

    app.extensions["predictors"] = dict(predictors or {})
    app.extensions["catalog_error"] = None
    if catalog is not None:
        app.extensions["test_catalog"] = catalog
    else:
        try:
            app.extensions["test_catalog"] = TestImageCatalog(
                config.DATASET_ROOT,
                config.DATASET_ARCHIVE,
                config.CONVNEXT_LABEL_MAP,
                config.UNKNOWN_TEST_IDENTITIES,
                kaggle_dataset_handle=getattr(config, "KAGGLE_DATASET_HANDLE", None),
                allow_remote=getattr(config, "KAGGLE_ON_DEMAND", False),
            )
        except (DatasetUnavailableError, OSError, ValueError) as exc:
            app.extensions["test_catalog"] = None
            app.extensions["catalog_error"] = str(exc)

    def error(message: str, status: int):
        return jsonify({"error": message}), status

    def model_options():
        convnext_available = (
            "convnext" in app.extensions["predictors"]
            or (
                config.CONVNEXT_CHECKPOINT.exists()
                and config.CONVNEXT_LABEL_MAP.exists()
                and config.CONVNEXT_CHECKPOINT.stat().st_size > 1024
            )
        )
        miewid_available = (
            "miewid" in app.extensions["predictors"]
            or (
                config.MIEWID_CHECKPOINT.exists()
                and config.MIEWID_GALLERY.exists()
                and (config.MIEWID_DIR / "inference.py").exists()
                and config.MIEWID_CHECKPOINT.stat().st_size > 1024
                and config.MIEWID_GALLERY.stat().st_size > 1024
            )
        )
        return [
            {
                "id": "convnext",
                "name": "ConvNeXt Base",
                "available": convnext_available,
                "reason": None if convnext_available else "Checkpoint or label map is unavailable.",
            },
            {
                "id": "miewid",
                "name": "MiewID",
                "available": miewid_available,
                "reason": None if miewid_available else "Checkpoint or embedding gallery is unavailable.",
            },
        ]

    def get_predictor(name: str):
        predictors = app.extensions["predictors"]
        if name in predictors:
            return predictors[name]
        if name == "convnext":
            predictor = ConvNeXtClassifier(
                config.CONVNEXT_CHECKPOINT,
                config.CONVNEXT_LABEL_MAP,
                config.DEVICE,
            )
        elif name == "miewid":
            predictor = MiewIDClassifier(
                config.MIEWID_CHECKPOINT,
                config.MIEWID_GALLERY,
                config.MIEWID_DIR,
                config.THRESHOLD,
                config.DEVICE,
            )
        else:
            raise ModelUnavailableError(f"Model '{name}' is not available.")
        predictors[name] = predictor
        return predictor

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "models": model_options(),
                "dataset_available": app.extensions["test_catalog"] is not None,
                "dataset_error": app.extensions["catalog_error"],
            }
        )

    @app.get("/api/config")
    def get_config():
        return jsonify(
            {
                "default_model": config.DEFAULT_MODEL,
                "models": model_options(),
                "threshold": config.THRESHOLD,
                "top_k": config.TOP_K,
                "max_upload_mb": config.MAX_UPLOAD_MB,
                "allowed_extensions": sorted(config.ALLOWED_EXTENSIONS),
            }
        )

    @app.get("/api/test-images")
    def test_images():
        catalog = app.extensions["test_catalog"]
        if catalog is None:
            return error(app.extensions["catalog_error"] or "Test dataset is unavailable.", 503)
        try:
            page = max(1, int(request.args.get("page", 1)))
            page_size = min(48, max(1, int(request.args.get("page_size", 18))))
        except ValueError:
            return error("page and page_size must be integers.", 400)
        query_type = request.args.get("type", "all").lower()
        if query_type not in {"all", "known", "unknown"}:
            return error("type must be all, known or unknown.", 400)
        return jsonify(catalog.list_images(page, page_size, query_type))

    @app.get("/api/test-images/<query_id>/image")
    def test_image(query_id):
        catalog = app.extensions["test_catalog"]
        if catalog is None:
            return error(app.extensions["catalog_error"] or "Test dataset is unavailable.", 503)
        try:
            data, filename, mime_type = catalog.image_bytes(query_id)
        except KeyError:
            return error("Test image not found.", 404)
        except DatasetUnavailableError as exc:
            return error(str(exc), 404)
        return send_file(io.BytesIO(data), mimetype=mime_type, download_name=filename)

    @app.post("/api/predict")
    def predict():
        model_name = request.form.get("model", config.DEFAULT_MODEL).lower()
        upload = request.files.get("image")
        test_image_id = request.form.get("test_image_id")
        if bool(upload and upload.filename) == bool(test_image_id):
            return error("Provide exactly one uploaded image or test_image_id.", 400)

        source = "upload"
        try:
            if test_image_id:
                catalog = app.extensions["test_catalog"]
                if catalog is None:
                    return error(app.extensions["catalog_error"] or "Test dataset is unavailable.", 503)
                image_bytes, _, _ = catalog.image_bytes(test_image_id)
                source = "test"
            else:
                extension = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
                if extension not in config.ALLOWED_EXTENSIONS:
                    allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
                    return error(f"Unsupported file type '.{extension}'. Allowed: {allowed}.", 415)
                image_bytes = upload.read()
            image = load_image(image_bytes)
            predictor = get_predictor(model_name)
        except InvalidImageError as exc:
            return error(str(exc), 400)
        except KeyError:
            return error("Test image not found.", 404)
        except DatasetUnavailableError as exc:
            return error(str(exc), 404)
        except ModelUnavailableError as exc:
            return error(str(exc), 503)

        started = time.perf_counter()
        prediction = predictor.predict(image, config.TOP_K)
        confidence = prediction["confidence"]
        return jsonify(
            {
                "predicted_identity": prediction["predicted_identity"],
                "confidence": round(confidence, 6),
                "is_unknown": confidence < config.THRESHOLD,
                "top_5": [candidate["identity"] for candidate in prediction["top_k"]],
                "top_5_details": [
                    {
                        "identity": candidate["identity"],
                        "confidence": round(candidate["confidence"], 6),
                    }
                    for candidate in prediction["top_k"]
                ],
                "model": model_name,
                "threshold": config.THRESHOLD,
                "source": source,
                "inference_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )

    @app.errorhandler(413)
    def too_large(_):
        return error(f"Image is larger than {config.MAX_UPLOAD_MB} MB.", 413)

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
