"""Fine-tune official MiewID-msv3 on the shared 100-identity subset."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from inference import build_gallery
from model import (
    EMBEDDING_DIM,
    MODEL_TAG,
    MiewIdScheduler,
    SubCenterArcFaceLoss,
    extract_embeddings,
    load_miewid,
)
from prepare_data import (
    IDENTITY_RECORD_PATH,
    gallery_loader,
    identities,
    label_map,
    train_df,
    train_loader,
)


MODEL_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = MODEL_DIR / "miewid_weights.pth"
CHECKPOINT_PATH = MODEL_DIR / "miewid_checkpoint.pth"
GALLERY_PATH = MODEL_DIR / "known_gallery.pth"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dynamic_margins():
    """Official frequency-adaptive ArcFace margin calculation."""
    counts = train_df["identity"].value_counts()
    class_counts = np.asarray([counts[identity] for identity in identities], dtype=np.float32)
    values = np.sqrt(1.0 / np.sqrt(class_counts))
    if np.isclose(values.max(), values.min()):
        return np.full(len(identities), 0.5, dtype=np.float32)
    return 0.2 + (values - values.min()) / (values.max() - values.min()) * 0.3


def save_checkpoint(model, arcface, optimizer, scheduler, stage, epoch):
    torch.save(
        {
            "model_tag": MODEL_TAG,
            "model_state_dict": model.state_dict(),
            "arcface_state_dict": arcface.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "stage": stage,
            "epoch": epoch,
            "index_to_identity": identities,
        },
        CHECKPOINT_PATH,
    )


def train_stage(model, arcface, device, epochs, stage, train_backbone):
    for parameter in model.parameters():
        parameter.requires_grad = train_backbone

    parameters = list(arcface.parameters())
    if train_backbone:
        parameters = list(model.parameters()) + parameters

    optimizer = torch.optim.Adam(parameters, lr=5e-6)
    scheduler = MiewIdScheduler(
        optimizer,
        lr_start=5e-6,
        lr_max=1e-5,
        lr_min=1e-6,
        lr_ramp_epochs=min(5, max(1, epochs)),
        lr_decay=0.8,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(epochs):
        model.train()
        arcface.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"{stage} {epoch + 1}/{epochs}")

        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                embeddings = extract_embeddings(model, images, normalize=False)
                loss = arcface(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        average_loss = total_loss / len(train_loader)
        print(f"{stage} epoch {epoch + 1}/{epochs} loss: {average_loss:.4f}", flush=True)
        save_checkpoint(model, arcface, optimizer, scheduler, stage, epoch)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiewID on WildlifeReID-10k.")
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    print("Training on:", device, flush=True)
    print(f"Known identities: {len(label_map)}", flush=True)

    model = load_miewid(pretrained=True).to(device)
    embedding_dim = int(getattr(model, "final_in_features", EMBEDDING_DIM))
    arcface = SubCenterArcFaceLoss(
        embedding_dim=embedding_dim,
        num_classes=len(label_map),
        margins=dynamic_margins(),
        scale=30.0,
        k=3,
    ).to(device)

    train_stage(
        model,
        arcface,
        device,
        epochs=args.head_epochs,
        stage="Frozen-backbone ArcFace",
        train_backbone=False,
    )
    train_stage(
        model,
        arcface,
        device,
        epochs=args.finetune_epochs,
        stage="Full MiewID fine-tune",
        train_backbone=True,
    )

    identity_record = json.loads(IDENTITY_RECORD_PATH.read_text(encoding="utf-8"))
    torch.save(
        {
            "model_tag": MODEL_TAG,
            "model_state_dict": model.state_dict(),
            "index_to_identity": identities,
            "identity_record": identity_record,
        },
        WEIGHTS_PATH,
    )
    LABEL_MAP_PATH.write_text(
        json.dumps(
            {
                **identity_record,
                "identity_to_index": label_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved trained weights to {WEIGHTS_PATH}", flush=True)

    build_gallery(
        model,
        gallery_loader,
        identities,
        output_path=GALLERY_PATH,
        device=device,
    )


if __name__ == "__main__":
    main()
