import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from tqdm.auto import tqdm

from prepare_data import train_loader, label_map

num_epochs = 15
scaler = torch.amp.GradScaler("cuda")

# Choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device, flush=True)

# Number of animal identities
num_classes = len(label_map)

# Load pretrained ConvNeXt
model = convnext_base(
    weights=ConvNeXt_Base_Weights.DEFAULT
)

# Replace ImageNet classifier
model.classifier[2] = nn.Linear(
    model.classifier[2].in_features,
    num_classes
)

# Move model to GPU
model = model.to(device)

# Loss
loss_function = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


for epoch in range(num_epochs):

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs}"
    )

    total_loss = 0

    for images, labels in progress:

        # Move images and labels to GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # AMP
        with torch.amp.autocast("cuda"):
            predictions = model(images)
            loss = loss_function(predictions, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{num_epochs} Loss: {average_loss:.4f}",
        flush=True
    )

torch.save(model.state_dict(), "convnext_weights.pth")