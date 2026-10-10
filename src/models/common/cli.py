import argparse
import json
from pathlib import Path

import numpy as np

from .data import FIT, TEST, VAL_KNOWN, VAL_UNKNOWN, limit_identities, load_metadata, make_validation_split
from .engine import DEFAULT_WORKERS, TrainConfig, embed_images_resumable, evaluate_queries, fit, pick_device
from .retrieval import SCORES


def train_main(model) -> None:
    parser = argparse.ArgumentParser(description=f"Fine-tune {model.NAME} on WildlifeReID-10k with an ArcFace head.")
    parser.add_argument("--data-root", required=True, type=Path, help="folder holding metadata.csv and images/")
    parser.add_argument("--out", required=True, type=Path, help="where checkpoints and history.json are written")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs, help="stage 2 epochs")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr-backbone", type=float, default=TrainConfig.lr_backbone)
    parser.add_argument("--lr-head", type=float, default=TrainConfig.lr_head)
    parser.add_argument("--unfreeze-stages", type=int, default=1, help="how many of the last backbone stages to train")
    parser.add_argument("--val-identities", type=int, default=TrainConfig.val_identities)
    parser.add_argument("--workers", type=int, default=TrainConfig.workers)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    parser.add_argument("--limit-identities", type=int, default=0, help="smoke test: keep only this many identities")
    args = parser.parse_args()

    df = load_metadata(args.data_root)
    if args.limit_identities:
        df = limit_identities(df, args.limit_identities, args.seed)
    df = make_validation_split(df, seed=args.seed)

    cfg = TrainConfig(
        data_root=str(args.data_root), out_dir=str(args.out), input_size=model.INPUT_SIZE, epochs=args.epochs,
        batch_size=args.batch_size, lr_backbone=args.lr_backbone, lr_head=args.lr_head,
        val_identities=args.val_identities, workers=args.workers, seed=args.seed, device=args.device,
    )
    backbone = model.build_backbone()
    result = fit(backbone, model.unfreeze_parameters(backbone, args.unfreeze_stages), df, cfg, resume=args.resume)
    print(f"best validation normalised accuracy: {result['best_val_normalised_accuracy']:.4f}")
    print(f"best backbone saved to {args.out / 'best_backbone.pt'}")



def evaluate_main(model) -> None:
    parser = argparse.ArgumentParser(description=f"Evaluate {model.NAME} on the official WildlifeReID-10k test split.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="results.json and the resumable embedding cache go here")
    parser.add_argument("--checkpoint", default="", help="fine-tuned backbone; omit for the pretrained baseline")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0, help="must match the seed used for training")
    parser.add_argument("--limit-identities", type=int, default=0, help="smoke test: keep only this many identities")
    parser.add_argument("--export-gallery", type=Path, help="also write the training-split gallery for the API")
    args = parser.parse_args()

    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    df = load_metadata(args.data_root)
    if args.limit_identities:
        df = limit_identities(df, args.limit_identities, args.seed)
    df = make_validation_split(df, seed=args.seed).reset_index(drop=True)

    backbone = model.build_backbone(args.checkpoint).to(device)
    emb, ms_per_image = embed_images_resumable(
        backbone, df, args.data_root, model.INPUT_SIZE, args.batch_size, device, args.workers, args.out, args.checkpoint or "pretrained"
    )

    def rows(*roles: str) -> np.ndarray:
        return df["role"].isin(roles).to_numpy()

    val_q, val_g = rows(VAL_KNOWN, VAL_UNKNOWN), rows(FIT)
    val_result, val_eval, val_closed = evaluate_queries(emb[val_q], df[val_q], emb[val_g], df[val_g], device)
    thresholds = {name: val_eval.best_threshold(val_result.scores[name]) for name in SCORES}

    test_q, test_g = rows(TEST), ~rows(TEST)
    result, evaluator, closed = evaluate_queries(emb[test_q], df[test_q], emb[test_g], df[test_g], device)

    open_set = {}
    for name in SCORES:
        score = result.scores[name]
        open_set[name] = {
            **evaluator.at_threshold(score, thresholds[name]),
            **evaluator.threshold_free(score),
            "validation": val_eval.at_threshold(val_result.scores[name], thresholds[name]),
        }

    report = {
        "model": model.MODEL_ID,
        "checkpoint": args.checkpoint or "pretrained",
        "counts": {
            "gallery_images": int(test_g.sum()),
            "queries_known": int((~evaluator.is_unknown).sum()),
            "queries_unknown": int(evaluator.is_unknown.sum()),
        },
        "closed_set": {**closed, "BAKS": evaluator.at_threshold(result.scores["max_sim"], -np.inf)["BAKS"]},
        "validation_closed_set": val_closed,
        "open_set": open_set,
        "efficiency": {
            "device": device,
            "batch_size": args.batch_size,
            "ms_per_image": ms_per_image,
            "parameters_millions": sum(p.numel() for p in backbone.parameters()) / 1e6,
            "embedding_dim": int(emb.shape[1]),
        },
    }
    (args.out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    c = report["closed_set"]
    print(f"\nclosed set (known test queries): Top-1 {c['top1']:.4f}  Top-5 {c['top5']:.4f}  mAP {c['mAP']:.4f}  BAKS {c['BAKS']:.4f}")
    print(f"{'open-set score':<16}{'tau':>8}{'BAKS':>8}{'BAUS':>8}{'norm.acc':>10}{'AUROC':>8}{'TNR@95':>8}")
    for name, m in open_set.items():
        print(f"{name:<16}{m['threshold']:>8.3f}{m['BAKS']:>8.4f}{m['BAUS']:>8.4f}{m['normalised_accuracy']:>10.4f}{m['AUROC']:>8.4f}{m['TNR@95TPR']:>8.4f}")
    print(f"model time: {ms_per_image:.1f} ms/image on {device}. Full report: {args.out / 'results.json'}")

    if args.export_gallery:
        gallery_df = df[test_g]
        args.export_gallery.mkdir(parents=True, exist_ok=True)
        np.save(args.export_gallery / "embeddings.npy", emb[test_g])
        meta = {"identities": gallery_df["identity"].tolist(), "image_ids": gallery_df["path"].tolist(), "model": model.NAME}
        (args.export_gallery / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"gallery for the API written to {args.export_gallery}")

