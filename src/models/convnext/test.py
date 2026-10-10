import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import convnext_base
import torch.nn.functional as F

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


threshold = 0.5  # example only

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        # Convert outputs to probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Highest confidence + predicted identity
        confidence, predictions = probabilities.max(dim=1)

        # Anything below threshold becomes UNKNOWN
        predictions[confidence < threshold] = -1

        print("Predictions:", predictions)
        print("Actual labels:", labels)
        print("Confidence:", confidence)