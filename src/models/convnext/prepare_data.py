import pandas as pd
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, InterpolationMode, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader

to_tensor = ToTensor()

df = pd.read_csv("../../data/metadata.csv")

train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

DATA_ROOT = "../../data"  

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
        label = label_map[row["identity"]]

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
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)
