"""Evaluate the existing ConvNeXt and MiewID artifacts on identical test queries.

This script does not train or modify either model. Closed-set Top-1/Top-5 are
computed only for known identities. Open-set AUROC uses each model's native
ranking score and does not require the scores to share a common threshold.
"""

import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BACKEND_DIR = REPO_ROOT / "src" / "backend"
MIEWID_DIR = SCRIPT_DIR / "miewid"

sys.path.insert(0, str(BACKEND_DIR))

from config import Config  # noqa: E402
from reid.classifier import ConvNeXtClassifier  # noqa: E402
from reid.dataset import DatasetUnavailableError, TestImageCatalog  # noqa: E402
from reid.preprocessing import load_image  # noqa: E402


def load_miewid_inference_class():
    """Load MiewID's isolated inference module without changing its source."""
    module_name = "comparison_miewid_inference"
    spec = importlib.util.spec_from_file_location(module_name, MIEWID_DIR / "inference.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the MiewID inference module.")

    sys.path.insert(0, str(MIEWID_DIR))
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.MiewIDInference
    finally:
        sys.path.pop(0)


def read_identities(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    identities = data.get("index_to_identity")
    if not identities:
        raise RuntimeError(f"Identity map is empty or invalid: {path}")
    return identities


def verify_shared_identities():
    convnext_identities = read_identities(Config.CONVNEXT_LABEL_MAP)
    miewid_identities = read_identities(MIEWID_DIR / "label_map.json")
    if convnext_identities != miewid_identities:
        raise RuntimeError(
            "ConvNeXt and MiewID do not have the same ordered identity list. "
            "A direct comparison would be invalid."
        )
    return convnext_identities


def resolve_dataset_root():
    """Prefer the configured dataset, then the standard Kaggle cache."""
    if (Config.DATASET_ROOT / "metadata.csv").is_file():
        return Config.DATASET_ROOT
    kaggle_cache = (
        Path.home()
        / ".cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/10"
    )
    if (kaggle_cache / "metadata.csv").is_file():
        return kaggle_cache
    return Config.DATASET_ROOT


def select_queries(frame, limit_per_group):
    """Use the full shared test set, or a balanced subset for a smoke test."""
    if not limit_per_group:
        return frame.copy()
    known = frame[frame["query_type"].eq("known")].head(limit_per_group)
    unknown = frame[frame["query_type"].eq("unknown")].head(limit_per_group)
    return frame.loc[known.index.union(unknown.index)].copy()


def load_queries(catalog, frame, workers):
    """Load each query once so both models receive identical image bytes."""
    os.environ["DISABLE_COLAB_CACHE"] = "1"

    def load_query(row):
        try:
            image_bytes, _, _ = catalog.image_bytes(row["query_id"])
        except DatasetUnavailableError:
            if not Config.KAGGLE_ON_DEMAND:
                raise
            try:
                import kagglehub
            except ImportError as exc:
                raise RuntimeError(
                    "kagglehub is required to fetch missing test images."
                ) from exc
            downloaded = Path(
                kagglehub.dataset_download(
                    Config.KAGGLE_DATASET_HANDLE,
                    path=str(row["path"]),
                )
            )
            if not downloaded.is_file():
                raise RuntimeError(f"Kaggle did not return the test image: {row['path']}")
            image_bytes = downloaded.read_bytes()
        return {
            "query_id": row["query_id"],
            "identity": row["identity"],
            "query_type": row["query_type"],
            "image": load_image(image_bytes),
        }

    rows = [row for _, row in frame.iterrows()]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(load_query, rows),
                total=len(rows),
                desc="Loading shared queries",
            )
        )


def evaluate_model(name, predictor, queries):
    known_total = 0
    top1_correct = 0
    top5_correct = 0
    unknown_labels = []
    unknown_scores = []
    predictions = []
    inference_times = []

    for query in tqdm(queries, desc=f"Evaluating {name}"):
        started = time.perf_counter()
        result = predictor(query["image"])
        inference_ms = (time.perf_counter() - started) * 1000
        inference_times.append(inference_ms)

        ranked_identities = result["identities"]
        best_score = float(result["score"])
        is_unknown = query["query_type"] == "unknown"

        unknown_labels.append(int(is_unknown))
        unknown_scores.append(1.0 - best_score)

        if not is_unknown:
            known_total += 1
            top1_correct += int(ranked_identities[0] == query["identity"])
            top5_correct += int(query["identity"] in ranked_identities[:5])

        predictions.append(
            {
                "model": name,
                "query_id": query["query_id"],
                "query_type": query["query_type"],
                "true_identity": query["identity"],
                "predicted_identity": ranked_identities[0],
                "best_score": best_score,
                "top_5": "|".join(ranked_identities[:5]),
                "inference_ms": inference_ms,
            }
        )

    if not known_total:
        raise RuntimeError("The selected test queries contain no known identities.")
    if len(set(unknown_labels)) != 2:
        raise RuntimeError("Open-set AUROC requires both known and unknown test queries.")

    metrics = {
        "top_1": top1_correct / known_total,
        "top_5": top5_correct / known_total,
        "open_set_auroc": float(roc_auc_score(unknown_labels, unknown_scores)),
        "known_queries": known_total,
        "unknown_queries": sum(unknown_labels),
        "mean_inference_ms": sum(inference_times) / len(inference_times),
    }
    return metrics, predictions


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Compare existing ConvNeXt and MiewID weights without retraining."
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument(
        "--limit-per-group",
        type=int,
        default=0,
        help="smoke test only: evaluate at most N known and N unknown queries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "comparison_results.json",
        help="aggregated JSON results path",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=SCRIPT_DIR / "comparison_predictions.csv",
        help="per-query CSV results path",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="parallel workers for cached/on-demand test image loading",
    )
    args = parser.parse_args()

    if args.limit_per_group < 0:
        parser.error("--limit-per-group cannot be negative")
    if args.download_workers < 1:
        parser.error("--download-workers must be at least 1")

    identities = verify_shared_identities()
    dataset_root = resolve_dataset_root()
    catalog = TestImageCatalog(
        dataset_root=dataset_root,
        archive_path=Config.DATASET_ARCHIVE,
        label_map_path=Config.CONVNEXT_LABEL_MAP,
        unknown_identity_count=Config.UNKNOWN_TEST_IDENTITIES,
        kaggle_dataset_handle=Config.KAGGLE_DATASET_HANDLE,
        allow_remote=Config.KAGGLE_ON_DEMAND,
    )
    query_frame = select_queries(catalog.frame, args.limit_per_group)
    queries = load_queries(catalog, query_frame, args.download_workers)

    print(
        f"Verified the same {len(identities)} ordered identities. "
        f"Evaluating {len(queries)} identical test queries.",
        flush=True,
    )

    results = {}
    all_predictions = []

    convnext = ConvNeXtClassifier(
        Config.CONVNEXT_CHECKPOINT,
        Config.CONVNEXT_LABEL_MAP,
        args.device,
    )

    def predict_convnext(image):
        prediction = convnext.predict(image, 5)
        return {
            "identities": [item["identity"] for item in prediction["top_k"]],
            "score": prediction["confidence"],
        }

    metrics, predictions = evaluate_model("convnext", predict_convnext, queries)
    results["convnext"] = {
        "method": "classification_softmax",
        "unknown_score": "1 - maximum softmax probability",
        **metrics,
    }
    all_predictions.extend(predictions)
    del predict_convnext
    del convnext
    release_memory()

    MiewIDInference = load_miewid_inference_class()
    miewid = MiewIDInference(
        weights_path=Config.MIEWID_CHECKPOINT,
        gallery_path=Config.MIEWID_GALLERY,
        device=args.device,
    )

    def predict_miewid(image):
        prediction = miewid.predict(image, top_k=5)
        return {
            "identities": [item["identity"] for item in prediction["matches"]],
            "score": prediction["similarity"],
        }

    metrics, predictions = evaluate_model("miewid", predict_miewid, queries)
    results["miewid"] = {
        "method": "embedding_cosine_retrieval",
        "unknown_score": "1 - maximum cosine similarity",
        **metrics,
    }
    all_predictions.extend(predictions)
    del predict_miewid
    del miewid
    release_memory()

    payload = {
        "protocol": {
            "dataset": "WildlifeReID-10k version 10",
            "known_identities": len(identities),
            "same_ordered_identities": True,
            "same_test_queries": True,
            "query_count": len(queries),
            "limited_smoke_test": bool(args.limit_per_group),
            "top_k": 5,
            "open_set_metric": "threshold-independent AUROC",
        },
        "models": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_predictions[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(all_predictions)

    print(json.dumps(payload, indent=2))
    print(f"Saved metrics to {args.output}")
    print(f"Saved per-query predictions to {args.predictions}")


if __name__ == "__main__":
    main()
