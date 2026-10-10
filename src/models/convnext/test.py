import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import convnext_base

from prepare_data import test_loader, label_map


num_classes = len(label_map)

# Recreate model
model = convnext_base(weights=None)

model.classifier[2] = nn.Linear(
    model.classifier[2].in_features,
    num_classes
)

# Load trained weights
model.load_state_dict(
    torch.load("convnext_weights.pth")
)

model.eval()


correct_top1 = 0
correct_top5 = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        predictions = model(images)

        # Top 1
        top1 = predictions.argmax(dim=1)

        correct_top1 += (top1 == labels).sum().item()


        # Top 5
        top5 = predictions.topk(5, dim=1).indices

        correct_top5 += (
            top5 == labels.unsqueeze(1)
        ).any(dim=1).sum().item()

        total += labels.size(0)


top1_accuracy = correct_top1 / total
top5_accuracy = correct_top5 / total

print(f"Top 1 Accuracy: {top1_accuracy * 100:.2f}%")
print(f"Top 5 Accuracy: {top5_accuracy * 100:.2f}%")