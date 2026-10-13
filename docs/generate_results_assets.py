"""Generate report figures from the reproducible comparison outputs."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "src" / "models" / "comparison_results.json"
PREDICTIONS_PATH = REPO_ROOT / "src" / "models" / "comparison_predictions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

COLORS = {
    "convnext": "#86A89A",
    "miewid": "#2E7D4F",
    "grid": "#D9DED8",
    "text": "#1F2A24",
}


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def finish_figure(fig, output_name):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def metrics_chart(results):
    metric_keys = ["top_1", "top_5", "open_set_auroc"]
    labels = ["Top-1", "Top-5", "Open-set AUROC"]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for offset, model in ((-width / 2, "convnext"), (width / 2, "miewid")):
        values = [results["models"][model][key] * 100 for key in metric_keys]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label="ConvNeXt" if model == "convnext" else "MiewID",
            color=COLORS[model],
        )
        ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4)

    ax.set_title("Identification and open-set performance", loc="left", weight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, -0.2))
    finish_figure(fig, "metrics_comparison.png")


def inference_chart(results):
    models = ["convnext", "miewid"]
    labels = ["ConvNeXt", "MiewID"]
    values = [results["models"][model]["mean_inference_ms"] for model in models]

    fig, ax = plt.subplots(figsize=(8, 5.6))
    bars = ax.bar(labels, values, width=0.55, color=[COLORS[model] for model in models])
    ax.bar_label(bars, labels=[f"{value:.1f} ms" for value in values], padding=5)
    ax.set_title("Mean inference time per image", loc="left", weight="bold")
    ax.set_ylabel("Milliseconds (lower is better)")
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, "inference_time.png")


def roc_chart(predictions, results):
    fig, ax = plt.subplots(figsize=(8, 6.2))
    for model in ("convnext", "miewid"):
        rows = [row for row in predictions if row["model"] == model]
        labels = np.array([1 if row["query_type"] == "unknown" else 0 for row in rows])
        unknown_scores = np.array([1.0 - float(row["best_score"]) for row in rows])
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, unknown_scores)
        auc = results["models"][model]["open_set_auroc"]
        display_name = "ConvNeXt" if model == "convnext" else "MiewID"
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            color=COLORS[model],
            linewidth=2.5,
            label=f"{display_name} (AUROC {auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="#9AA39D", linewidth=1.2, linestyle="--", label="Chance")
    ax.set_title("Unknown-animal detection", loc="left", weight="bold")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    finish_figure(fig, "open_set_roc.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    with PREDICTIONS_PATH.open(newline="", encoding="utf-8") as file:
        predictions = list(csv.DictReader(file))

    metrics_chart(results)
    inference_chart(results)
    roc_chart(predictions, results)
    print(f"Saved report figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
