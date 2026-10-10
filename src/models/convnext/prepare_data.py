import pandas as pd
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, InterpolationMode, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader

to_tensor = ToTensor()

DATA_ROOT = "/root/.cache/kagglehub/datasets/wildlifedatasets/wildlifereid-10k/versions/10"
df = pd.read_csv(f"{DATA_ROOT}/metadata.csv", low_memory=False)

train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

# Pick 100 known animals from training
known_identities = train_df["identity"].drop_duplicates().sample(
    5000,
    random_state=42
)

# Training = only these 100 animals
train_df = train_df[
    train_df["identity"].isin(known_identities)
]

# Find animals that were NEVER in training
unknown_identities = test_df[
    ~test_df["identity"].isin(df[df["split"] == "train"]["identity"])
]["identity"].drop_duplicates().sample(
    20,
    random_state=42
)

# Test contains:
# known animals + completely unseen animals
test_df = test_df[
    test_df["identity"].isin(known_identities) |
    test_df["identity"].isin(unknown_identities)
]

identities = train_df["identity"].unique()

label_map = {
    identity: index
    for index, identity in enumerate(identities)
}


class WildlifeDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
     return len(self.df)

    def __getitem__(self, index):
        # Get one row from metadata
        row = self.df.iloc[index]

        # Get actual image path
        image_path = f"{DATA_ROOT}/{row['path']}"

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Resize, crop, tensor, normalize
        image = self.transform(image)
        
        # Correct animal identity
        identity = row["identity"]

        if identity in label_map:
            label = label_map[identity]
        else:
            label = -1 

        return image, label

    def transform(self, image):
       resize = Resize(
            232,
            interpolation=InterpolationMode.BILINEAR
        )

       resized_image = resize(image)

       crop = CenterCrop(224)

       cropped_image = crop(resized_image) 

       image_tensor = to_tensor(cropped_image)

       normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

       return normalize(image_tensor)

# Create test dataset object
train_dataset = WildlifeDataset(
    train_df,
)

test_dataset = WildlifeDataset(
    test_df,
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
