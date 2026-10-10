import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data import limit_identities, load_metadata, make_validation_split  # noqa: E402
from common.engine import TrainConfig, fit  # noqa: E402
from backbone import INPUT_SIZE, build_backbone, unfreeze_parameters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MiewID-msv3 on WildlifeReID-10k with an ArcFace head.")
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
        data_root=str(args.data_root), out_dir=str(args.out), input_size=INPUT_SIZE, epochs=args.epochs,
        batch_size=args.batch_size, lr_backbone=args.lr_backbone, lr_head=args.lr_head,
        val_identities=args.val_identities, workers=args.workers, seed=args.seed, device=args.device,
    )
    backbone = build_backbone()
    result = fit(backbone, unfreeze_parameters(backbone, args.unfreeze_stages), df, cfg, resume=args.resume)
    print(f"best validation normalised accuracy: {result['best_val_normalised_accuracy']:.4f}")
    print(f"best backbone saved to {args.out / 'best_backbone.pt'}")


if __name__ == "__main__":
    main()
