import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base

from prepare_data import test_loader, label_map


# Number of identities the model learned during training
num_classes = len(label_map)


# Recreate ConvNeXt
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


# Example threshold
threshold = 0.5


correct = 0
total = 0

known_correct = 0
known_total = 0

unknown_correct = 0
unknown_total = 0


with torch.no_grad():

    for images, labels in test_loader:

        # Model predicts
        outputs = model(images)

        # Convert outputs into probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Get highest probability
        confidence, predictions = probabilities.max(dim=1)

        # Low confidence = UNKNOWN
        predictions[confidence < threshold] = -1


        # Overall accuracy
        correct += (predictions == labels).sum().item()
        total += labels.size(0)


        # Known animals
        known_mask = labels != -1

        known_correct += (
            predictions[known_mask] == labels[known_mask]
        ).sum().item()

        known_total += known_mask.sum().item()


        # Unknown animals
        unknown_mask = labels == -1

        unknown_correct += (
            predictions[unknown_mask] == -1
        ).sum().item()

        unknown_total += unknown_mask.sum().item()


# Results
overall_accuracy = correct / total

print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

if known_total > 0:
    known_accuracy = known_correct / known_total
    print(f"Known Animal Accuracy: {known_accuracy * 100:.2f}%")

if unknown_total > 0:
    unknown_accuracy = unknown_correct / unknown_total
    print(f"Unknown Detection Accuracy: {unknown_accuracy * 100:.2f}%")