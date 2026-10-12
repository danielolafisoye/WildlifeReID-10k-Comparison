"""WildlifeReID-10k subset shared by MiewID training, testing, and gallery build."""

import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from model import IMAGE_SIZE, MEAN, STD, inference_transform


MODEL_DIR = Path(__file__).resolve().parent
KAGGLE_DATASET_HANDLE = "wildlifedatasets/wildlifereid-10k/versions/10"
DATA_ROOT = Path(
    os.getenv(
        "WILDLIFE_DATASET_ROOT",
        Path.home()
        / ".cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/10",
    )
)
METADATA_PATH = Path(os.getenv("WILDLIFE_METADATA_PATH", DATA_ROOT / "metadata.csv"))
IDENTITY_RECORD_PATH = MODEL_DIR / "selected_identities.json"

NUM_KNOWN_IDENTITIES = 100
NUM_UNKNOWN_IDENTITIES = 20
RANDOM_STATE = 42
BATCH_SIZE = int(os.getenv("MIEWID_BATCH_SIZE", "8"))
NUM_WORKERS = int(os.getenv("MIEWID_NUM_WORKERS", "0"))


def download_dataset_file(relative_path):
    """Download one WildlifeReID-10k file without fetching the full dataset."""
    # Colab's dataset-mount resolver can return intermittent 403 responses for
    # individual files. The normal Kaggle HTTP resolver supports public files,
    # writes them to ~/.cache/kagglehub, and is stable for sequential prefetch.
    os.environ["DISABLE_COLAB_CACHE"] = "1"
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download missing WildlifeReID files. "
            "Run: pip install -r requirements.txt"
        ) from exc

    try:
        return Path(
            kagglehub.dataset_download(
                KAGGLE_DATASET_HANDLE,
                path=str(relative_path),
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download '{relative_path}' from Kaggle. "
            "Check the Colab internet connection and Kaggle access."
        ) from exc


def resolve_metadata_path():
    if METADATA_PATH.is_file():
        return METADATA_PATH
    print("metadata.csv is missing; downloading only that file from Kaggle...", flush=True)
    return download_dataset_file("metadata.csv")


def resolve_image_path(relative_path):
    local_path = DATA_ROOT / relative_path
    if local_path.is_file():
        return local_path
    return download_dataset_file(relative_path)


def prefetch_images(dataframe, description):
    """Cache selected files sequentially before DataLoader iteration."""
    paths = dataframe["path"].drop_duplicates().tolist()
    for relative_path in tqdm(paths, desc=description):
        resolve_image_path(relative_path)


df = pd.read_csv(resolve_metadata_path(), low_memory=False)
all_train_df = df[df["split"] == "train"]
all_test_df = df[df["split"] == "test"]

# This is intentionally identical to ConvNeXt's identity selection.
known_identities = all_train_df["identity"].drop_duplicates().sample(
    NUM_KNOWN_IDENTITIES,
    random_state=RANDOM_STATE,
)

train_df = all_train_df[all_train_df["identity"].isin(known_identities)].copy()

unknown_identities = all_test_df[
    ~all_test_df["identity"].isin(all_train_df["identity"])
]["identity"].drop_duplicates().sample(
    NUM_UNKNOWN_IDENTITIES,
    random_state=RANDOM_STATE,
)

test_df = all_test_df[
    all_test_df["identity"].isin(known_identities)
    | all_test_df["identity"].isin(unknown_identities)
].copy()

# Preserve metadata order, exactly as ConvNeXt does after filtering.
identities = train_df["identity"].unique().tolist()
label_map = {identity: index for index, identity in enumerate(identities)}


def _verify_identity_record():
    record = json.loads(IDENTITY_RECORD_PATH.read_text(encoding="utf-8"))
    if record["selection"] != {
        "num_identities": NUM_KNOWN_IDENTITIES,
        "random_state": RANDOM_STATE,
    }:
        raise RuntimeError("selected_identities.json has different selection settings.")
    if record["index_to_identity"] != identities:
        raise RuntimeError(
            "The selected identities do not match the permanent MiewID identity record. "
            "Check that WildlifeReID-10k metadata version 10 is being used."
        )


_verify_identity_record()


train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomApply(
            [
                transforms.RandomChoice(
                    [
                        transforms.RandomAdjustSharpness(1.5),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.RandomAutocontrast(),
                    ]
                )
            ],
            p=0.5,
        ),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.25, 0.25),
            scale=(0.8, 1.2),
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


class WildlifeDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = resolve_image_path(row["path"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        image = self.transform(image)
        label = label_map.get(row["identity"], -1)
        return image, label


train_dataset = WildlifeDataset(train_df, train_transform)
gallery_dataset = WildlifeDataset(train_df, inference_transform())
test_dataset = WildlifeDataset(test_df, inference_transform())

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

gallery_loader = DataLoader(
    gallery_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
