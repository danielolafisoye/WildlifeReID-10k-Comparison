import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def balanced_weights(identities: np.ndarray, datasets: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"identity": identities, "dataset": datasets})
    per_identity = frame.groupby("identity")["identity"].transform("size")
    ids_per_dataset = frame.groupby("dataset")["identity"].transform("nunique")
    return (1.0 / (frame["dataset"].nunique() * ids_per_dataset * per_identity)).to_numpy()


def closed_set_metrics(top_ids: np.ndarray, average_precision: np.ndarray, true_ids: np.ndarray) -> dict:
    hits = top_ids == true_ids[:, None]
    return {
        "top1": float(hits[:, 0].mean()),
        "top5": float(hits[:, :5].any(axis=1).mean()),
        "mAP": float(np.nanmean(average_precision)),
    }


class OpenSetEvaluator:
    def __init__(self, correct: np.ndarray, is_unknown: np.ndarray, identities: np.ndarray, datasets: np.ndarray):
        self.correct = correct & ~is_unknown
        self.is_unknown = is_unknown
        self.weights = np.zeros(len(correct))
        for mask in (is_unknown, ~is_unknown):
            if mask.any():
                self.weights[mask] = balanced_weights(identities[mask], datasets[mask])

    def at_threshold(self, score: np.ndarray, tau: float) -> dict:
        accepted = score >= tau
        baks = float(self.weights[self.correct & accepted].sum())
        baus = float(self.weights[self.is_unknown & ~accepted].sum())
        return {"threshold": float(tau), "BAKS": baks, "BAUS": baus, "normalised_accuracy": float(np.sqrt(baks * baus))}

    def best_threshold(self, score: np.ndarray) -> float:
        order = np.argsort(-score, kind="stable")
        s = score[order]
        baks = np.cumsum(self.weights[order] * self.correct[order])
        baus = self.weights[self.is_unknown].sum() - np.cumsum(self.weights[order] * self.is_unknown[order])
        valid = np.r_[s[1:] != s[:-1], True]
        best = int(np.argmax(np.where(valid, np.sqrt(baks * np.clip(baus, 0, None)), -1.0)))
        return float(s[best])

    def threshold_free(self, score: np.ndarray) -> dict:
        if self.is_unknown.all() or not self.is_unknown.any():
            return {"AUROC": float("nan"), "TNR@95TPR": float("nan")}
        tau = np.quantile(score[~self.is_unknown], 0.05)
        return {
            "AUROC": float(roc_auc_score(~self.is_unknown, score)),
            "TNR@95TPR": float((score[self.is_unknown] < tau).mean()),
        }
