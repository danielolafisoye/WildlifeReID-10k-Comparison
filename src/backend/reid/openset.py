"""Open-set decision rule (step 6).

    y_hat = identity of the best match   if max_i S(q, g_i) >= tau
          = unknown                       otherwise
"""


def decide(candidates: list[dict], threshold: float) -> dict:
    """candidates: Gallery.search output, best first. Returns the accepted/rejected top match."""
    if not candidates:
        return {"status": "unknown", "identity": None, "score": None, "reason": "gallery_empty"}
    best = candidates[0]
    if best["score"] >= threshold:
        return {"status": "known", "identity": best["identity"], "score": best["score"], "reason": None}
    return {"status": "unknown", "identity": None, "score": best["score"], "reason": "below_threshold"}
