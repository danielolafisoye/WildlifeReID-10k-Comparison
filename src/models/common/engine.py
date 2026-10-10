import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .arcface import ArcFaceHead
from .data import FIT, VAL_KNOWN, VAL_UNKNOWN, ReIDDataset, build_transform
from .metrics import OpenSetEvaluator, closed_set_metrics
from .retrieval import SearchResult, search

DEFAULT_WORKERS = min(4, os.cpu_count() or 1)


def pick_device(name: str = "auto") -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


@torch.no_grad()
def embed_images(
    backbone: nn.Module, df: pd.DataFrame, root: Path, size: int, batch_size: int, device: str, workers: int, desc: str
) -> tuple[np.ndarray, float]:
    loader = DataLoader(ReIDDataset(df, root, build_transform(size, train=False)), batch_size=batch_size, num_workers=workers, pin_memory=device == "cuda")
    backbone.eval()
    chunks, seconds = [], 0.0
    for images, _ in tqdm(loader, desc=desc):
        images = images.to(device)
        started = time.perf_counter()
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            out = backbone(images)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - started
        chunks.append(F.normalize(out.float(), dim=1).cpu().numpy())
    return np.concatenate(chunks), 1000 * seconds / max(1, len(df))


def embed_images_resumable(
    backbone: nn.Module, df: pd.DataFrame, root: Path, size: int, batch_size: int, device: str, workers: int,
    cache_dir: Path, tag: str, chunk: int = 10000,
) -> tuple[np.ndarray, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    info_file = cache_dir / "embeddings.json"
    info = {"tag": tag, "images": len(df), "chunk": chunk}
    if not info_file.exists() or json.loads(info_file.read_text()) != info:
        for stale in cache_dir.glob("embeddings_*.npz"):
            stale.unlink()
        info_file.write_text(json.dumps(info))

    parts, total_ms = [], 0.0
    starts = range(0, len(df), chunk)
    for i, start in enumerate(starts):
        part_file = cache_dir / f"embeddings_{i:03d}.npz"
        rows = df.iloc[start : start + chunk]
        if part_file.exists():
            part = np.load(part_file)
            emb, ms = part["emb"], float(part["ms"])
        else:
            emb, ms = embed_images(backbone, rows, root, size, batch_size, device, workers, f"embedding {i + 1}/{len(starts)}")
            tmp = cache_dir / f"partial_{i:03d}.npz"
            np.savez(tmp, emb=emb, ms=ms)
            tmp.replace(part_file)
        parts.append(emb)
        total_ms += ms * len(rows)
    return np.concatenate(parts), total_ms / max(1, len(df))


def evaluate_queries(
    q_emb: np.ndarray, q_df: pd.DataFrame, g_emb: np.ndarray, g_df: pd.DataFrame, device: str = "cpu"
) -> tuple[SearchResult, OpenSetEvaluator, dict]:
    g_codes, g_names = pd.factorize(g_df["identity"])
    q_codes = pd.Index(g_names).get_indexer(q_df["identity"])
    is_unknown = q_codes < 0

    result = search(q_emb, g_emb, g_codes.astype(np.int64), q_codes.astype(np.int64), device=device)
    evaluator = OpenSetEvaluator(
        result.top_ids[:, 0] == q_codes, is_unknown, q_df["identity"].to_numpy(), q_df["dataset"].to_numpy()
    )
    closed = closed_set_metrics(result.top_ids[~is_unknown], result.average_precision[~is_unknown], q_codes[~is_unknown])
    return result, evaluator, closed


@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    input_size: int
    epochs: int = 6
    batch_size: int = 32
    eval_batch_size: int = 64
    lr_backbone: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    head_epochs: int = 5
    warmup_frac: float = 0.05
    arcface_scale: float = 30.0
    arcface_margin: float = 0.3
    val_identities: int = 1500
    workers: int = DEFAULT_WORKERS
    seed: int = 0
    device: str = "auto"


def _validation_frames(df: pd.DataFrame, n_identities: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    known = np.sort(df.loc[df["role"] == VAL_KNOWN, "identity"].unique())
    unknown = np.sort(df.loc[df["role"] == VAL_UNKNOWN, "identity"].unique())
    keep_known = rng.choice(known, size=min(n_identities, len(known)), replace=False)
    n_unknown = max(1, round(len(unknown) * len(keep_known) / max(1, len(known))))
    keep_unknown = rng.choice(unknown, size=min(n_unknown, len(unknown)), replace=False)

    gallery = df[(df["role"] == FIT) & df["identity"].isin(keep_known)]
    queries = df[
        ((df["role"] == VAL_KNOWN) & df["identity"].isin(keep_known))
        | ((df["role"] == VAL_UNKNOWN) & df["identity"].isin(keep_unknown))
    ]
    return gallery, queries


def _train_head_on_cached(head: ArcFaceHead, embeddings: np.ndarray, labels: np.ndarray, epochs: int, lr: float, device: str):
    x, y = torch.from_numpy(embeddings).to(device), torch.from_numpy(labels).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr)
    for epoch in range(epochs):
        total = 0.0
        for idx in torch.randperm(len(x), device=device).split(512):
            loss = F.cross_entropy(head(x[idx], y[idx]), y[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        print(f"stage 1 (head only) epoch {epoch + 1}/{epochs}: loss {total / len(x):.4f}")


def fit(backbone: nn.Module, unfrozen: list[nn.Parameter], df: pd.DataFrame, cfg: TrainConfig, resume: bool = False) -> dict:
    device = pick_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    fit_df = df[df["role"] == FIT]
    labels, classes = pd.factorize(fit_df["identity"])
    val_gallery, val_queries = _validation_frames(df, cfg.val_identities, cfg.seed)
    print(f"train: {len(fit_df)} images, {len(classes)} identities | "
          f"val: {len(val_queries)} queries vs {len(val_gallery)} gallery images | device: {device}")

    def validate() -> dict:
        g, _ = embed_images(backbone, val_gallery, cfg.data_root, cfg.input_size, cfg.eval_batch_size, device, cfg.workers, "val gallery")
        q, _ = embed_images(backbone, val_queries, cfg.data_root, cfg.input_size, cfg.eval_batch_size, device, cfg.workers, "val queries")
        result, evaluator, closed = evaluate_queries(q, val_queries, g, val_gallery, device)
        score = result.scores["max_sim"]
        return {**closed, **evaluator.at_threshold(score, evaluator.best_threshold(score)), **evaluator.threshold_free(score)}

    def save_best(metrics: dict, epoch: int) -> None:
        torch.save(backbone.state_dict(), out_dir / "best_backbone.pt")
        (out_dir / "best_val.json").write_text(json.dumps({"epoch": epoch, **metrics}, indent=2), encoding="utf-8")

    embedding_dim = backbone(torch.zeros(1, 3, cfg.input_size, cfg.input_size, device=device)).shape[1]
    head = ArcFaceHead(embedding_dim, len(classes), cfg.arcface_scale, cfg.arcface_margin).to(device)

    for p in unfrozen:
        p.requires_grad_(True)
    optimiser = torch.optim.AdamW(
        [{"params": unfrozen, "lr": cfg.lr_backbone}, {"params": head.parameters(), "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )
    loader = DataLoader(
        ReIDDataset(fit_df, cfg.data_root, build_transform(cfg.input_size, train=True), labels),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, drop_last=True,
    )
    total_steps = cfg.epochs * len(loader)
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))),
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    last_path = out_dir / "last.pt"
    history, start_epoch, best = [], 0, -1.0

    def save_last(epoch: int) -> None:
        torch.save(
            {"backbone": backbone.state_dict(), "head": head.state_dict(), "optimiser": optimiser.state_dict(),
             "scheduler": scheduler.state_dict(), "history": history, "epoch": epoch, "best": best},
            last_path,
        )
        (out_dir / "history.json").write_text(json.dumps({"config": asdict(cfg), "history": history}, indent=2), encoding="utf-8")

    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        backbone.load_state_dict(state["backbone"])
        head.load_state_dict(state["head"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        history, start_epoch, best = state["history"], state["epoch"], state["best"]
        print(f"resumed after epoch {start_epoch}")
    else:
        metrics = validate()
        best = metrics["normalised_accuracy"]
        history.append({"epoch": 0, **metrics})
        save_best(metrics, 0)
        print(f"epoch 0 (pretrained): {json.dumps(metrics)}")

        fit_emb, _ = embed_images(backbone, fit_df, cfg.data_root, cfg.input_size, cfg.eval_batch_size, device, cfg.workers, "stage 1 embeddings")
        prototypes = torch.zeros(len(classes), embedding_dim).index_add_(0, torch.from_numpy(labels), torch.from_numpy(fit_emb))
        head.init_from_prototypes(prototypes.to(device))
        _train_head_on_cached(head, fit_emb, labels, cfg.head_epochs, cfg.lr_head, device)
        save_last(0)

    for epoch in range(start_epoch, cfg.epochs):
        head.train()
        running, seen = 0.0, 0
        progress = tqdm(loader, desc=f"stage 2 epoch {epoch + 1}/{cfg.epochs}")
        for images, targets in progress:
            images, targets = images.to(device), targets.to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                embeddings = backbone(images)
            loss = F.cross_entropy(head(embeddings, targets), targets)
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_([*unfrozen, *head.parameters()], 5.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            running, seen = running + loss.item() * len(images), seen + len(images)
            progress.set_postfix(loss=f"{running / seen:.4f}")

        metrics = validate()
        history.append({"epoch": epoch + 1, "train_loss": running / max(1, seen), **metrics})
        print(f"epoch {epoch + 1}: {json.dumps(history[-1])}")
        if metrics["normalised_accuracy"] > best:
            best = metrics["normalised_accuracy"]
            save_best(metrics, epoch + 1)
        save_last(epoch + 1)

    return {"history": history, "best_val_normalised_accuracy": best}
