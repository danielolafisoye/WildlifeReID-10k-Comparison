from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

SCORES = ("max_sim", "knn", "prototype", "margin")


@dataclass
class SearchResult:
    top_ids: np.ndarray
    top_sims: np.ndarray
    scores: dict[str, np.ndarray]
    average_precision: np.ndarray


def search(
    query: np.ndarray,
    gallery: np.ndarray,
    gallery_ids: np.ndarray,
    query_ids: np.ndarray | None = None,
    top_k: int = 5,
    knn_k: int = 5,
    chunk: int = 256,
    device: str = "cpu",
) -> SearchResult:
    g = torch.from_numpy(gallery).to(device)
    g_ids = torch.from_numpy(gallery_ids).to(device)
    n_ids = int(g_ids.max()) + 1
    top_k, knn_k = min(top_k, n_ids), min(knn_k, len(g))

    prototypes = torch.zeros(n_ids, g.shape[1], device=device).index_add_(0, g_ids, g)
    prototypes = F.normalize(prototypes, dim=1)

    out = {name: [] for name in ("ids", "sims", "ap", *SCORES)}
    for start in range(0, len(query), chunk):
        q = torch.from_numpy(query[start : start + chunk]).to(device)
        sims = q @ g.T

        per_id = torch.full((len(q), n_ids), -2.0, device=device)
        per_id.scatter_reduce_(1, g_ids.expand(len(q), -1), sims, reduce="amax")
        id_sims, ids = per_id.topk(top_k, dim=1)

        out["ids"].append(ids)
        out["sims"].append(id_sims)
        out["max_sim"].append(id_sims[:, 0])
        out["knn"].append(sims.topk(knn_k, dim=1).values.mean(dim=1))
        out["prototype"].append((q @ prototypes.T).max(dim=1).values)
        out["margin"].append(id_sims[:, 0] - id_sims[:, 1] if top_k > 1 else id_sims[:, 0])

        if query_ids is not None:
            labels = torch.from_numpy(query_ids[start : start + chunk]).to(device)
            relevant = (g_ids[sims.argsort(dim=1, descending=True)] == labels[:, None]).float()
            precision = relevant.cumsum(dim=1) / torch.arange(1, len(g) + 1, device=device)
            n_relevant = relevant.sum(dim=1)
            ap = (precision * relevant).sum(dim=1) / n_relevant.clamp(min=1)
            out["ap"].append(torch.where(n_relevant > 0, ap, torch.nan))

    cat = {k: torch.cat(v).cpu().numpy() for k, v in out.items() if v}
    return SearchResult(
        top_ids=cat["ids"],
        top_sims=cat["sims"],
        scores={name: cat[name] for name in SCORES},
        average_precision=cat.get("ap", np.full(len(query), np.nan, dtype=np.float32)),
    )
