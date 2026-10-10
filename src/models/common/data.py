from pathlib import Path

import numpy as np
import pandas as pd

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FIT, VAL_KNOWN, VAL_UNKNOWN, TEST = "fit", "val_known", "val_unknown", "test"


def load_metadata(root: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(root) / "metadata.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def make_validation_split(
    df: pd.DataFrame,
    seed: int = 0,
    unknown_frac: float = 0.05,
    holdout_frac: float = 0.2,
    min_images: int = 4,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["role"] = np.where(df["split"] == "test", TEST, FIT)
    train = df[df["split"] == "train"]

    unknown_ids = []
    for _, ids in train.groupby("dataset")["identity"].unique().items():
        n = max(1, round(unknown_frac * len(ids)))
        unknown_ids.extend(rng.choice(np.sort(ids), size=n, replace=False))
    df.loc[(df["split"] == "train") & df["identity"].isin(unknown_ids), "role"] = VAL_UNKNOWN

    remaining = df[df["role"] == FIT].sort_values(["identity", "date", "cluster_id", "path"], na_position="first")
    size = remaining.groupby("identity")["path"].transform("size")
    from_end = remaining.groupby("identity").cumcount(ascending=False)
    n_holdout = np.maximum(1, np.round(holdout_frac * size)).astype(int)
    held = remaining.index[((size >= min_images) & (from_end < n_holdout)).to_numpy()]
    df.loc[held, "role"] = VAL_KNOWN
    return df


def limit_identities(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    ids = np.sort(df["identity"].unique())
    keep = np.random.default_rng(seed).choice(ids, size=min(n, len(ids)), replace=False)
    return df[df["identity"].isin(keep)].reset_index(drop=True)


def build_transform(size: int, train: bool):
    from torchvision import transforms as T

    steps = [T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)]
    if train:
        steps += [
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


class ReIDDataset:
    def __init__(self, df: pd.DataFrame, root: Path, transform, labels: np.ndarray | None = None):
        self.paths = [str(Path(root) / p) for p in df["path"]]
        self.labels = np.full(len(df), -1, dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        with Image.open(self.paths[idx]) as img:
            image = self.transform(img.convert("RGB"))
        return image, int(self.labels[idx])
