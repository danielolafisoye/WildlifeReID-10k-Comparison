"""Open-set decision rule (step 6).

    y_hat = identity of the best match   if score(q) >= tau
          = unknown                       otherwise

Two scores are supported:

    max_sim  similarity to the best match (the rule in the proposal)
    margin   how far the best identity is ahead of the second-best one. A known animal matches
             one identity clearly; an unknown one tends to match several about equally.
"""

SCORES = ("max_sim", "margin")
FALLBACK_THRESHOLDS = {"max_sim": 0.65, "margin": 0.1}


def decision_score(candidates: list[dict], score: str) -> float:
    best = candidates[0]["score"]
    if score == "margin" and len(candidates) > 1:
        return round(best - candidates[1]["score"], 4)
    return best


def decide(candidates: list[dict], threshold: float, score: str = "max_sim") -> dict:
    """candidates: Gallery.search output, best first. Returns the accepted/rejected top match."""
    if not candidates:
        return {"status": "unknown", "identity": None, "score": None, "decision_score": None, "reason": "gallery_empty"}
    best = candidates[0]
    value = decision_score(candidates, score)
    if value >= threshold:
        return {"status": "known", "identity": best["identity"], "score": best["score"], "decision_score": value, "reason": None}
    return {"status": "unknown", "identity": None, "score": best["score"], "decision_score": value, "reason": "below_threshold"}
