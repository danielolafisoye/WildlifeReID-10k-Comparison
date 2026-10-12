"""Official MiewID-msv3 backbone and ArcFace fine-tuning components.

The backbone and preprocessing contract follow conservationxlabs/miewid-msv3.
The sub-center ArcFace head follows WildMeOrg/wbia-plugin-miew-id.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


MODEL_TAG = "conservationxlabs/miewid-msv3"
IMAGE_SIZE = 440
EMBEDDING_DIM = 2152
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_miewid(pretrained=True):
    """Load the official wildlife-pretrained MiewID-msv3 embedding model."""
    if not pretrained:
        raise ValueError(
            "MiewID must start from the official pretrained checkpoint; "
            "training from scratch is intentionally unsupported."
        )

    from transformers import AutoConfig, AutoModel

    try:
        model = AutoModel.from_pretrained(MODEL_TAG, trust_remote_code=True)
    except AttributeError as exc:
        if "all_tied_weights_keys" not in str(exc):
            raise

        # MiewID's custom class predates the Transformers 5 rename from
        # _tied_weights_keys to all_tied_weights_keys. Patch the downloaded
        # class and retry without changing the official checkpoint.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(MODEL_TAG, trust_remote_code=True)
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModel"],
            MODEL_TAG,
        )
        if not hasattr(model_class, "all_tied_weights_keys"):
            model_class.all_tied_weights_keys = property(lambda self: {})
        model = AutoModel.from_pretrained(MODEL_TAG, trust_remote_code=True)
    if not hasattr(model, "final_in_features"):
        model.final_in_features = EMBEDDING_DIM
    return model


def embedding_output(output):
    """Normalize the possible Hugging Face custom-model output containers."""
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise RuntimeError("MiewID must return a [batch, embedding_dim] tensor.")
    return output


def extract_embeddings(model, images, normalize=True):
    embeddings = embedding_output(model(images))
    return F.normalize(embeddings, dim=1) if normalize else embeddings


def inference_transform():
    """Official MiewID-msv3 440x440 preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


class ArcMarginProductSubcenter(nn.Module):
    """Sub-center cosine classifier used by the official MiewID trainer."""

    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.out_features = out_features
        self.k = k
        self.weight = nn.Parameter(torch.empty(out_features * k, in_features))
        nn.init.uniform_(
            self.weight,
            -1.0 / math.sqrt(in_features),
            1.0 / math.sqrt(in_features),
        )

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine = cosine.view(-1, self.out_features, self.k)
        return cosine.max(dim=2).values


class AdaptiveArcFaceMargin(nn.Module):
    def __init__(self, margins, out_features, scale=30.0):
        super().__init__()
        self.out_features = out_features
        self.scale = scale
        self.register_buffer("margins", torch.as_tensor(margins, dtype=torch.float32))

    def forward(self, cosine, labels):
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        margins = self.margins[labels]
        cos_m = torch.cos(margins).unsqueeze(1)
        sin_m = torch.sin(margins).unsqueeze(1)
        threshold = torch.cos(math.pi - margins).unsqueeze(1)
        correction = (torch.sin(math.pi - margins) * margins).unsqueeze(1)
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > threshold, phi, cosine - correction)
        one_hot = F.one_hot(labels, self.out_features).float()
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.scale


class SubCenterArcFaceLoss(nn.Module):
    """Dynamic-margin sub-center ArcFace objective from official fine-tuning."""

    def __init__(self, embedding_dim, num_classes, margins, scale=30.0, k=3):
        super().__init__()
        self.classifier = ArcMarginProductSubcenter(
            embedding_dim,
            num_classes,
            k=k,
        )
        self.margin = AdaptiveArcFaceMargin(margins, num_classes, scale=scale)

    def forward(self, embeddings, labels):
        cosine = self.classifier(embeddings.float())
        logits = self.margin(cosine, labels)
        return F.cross_entropy(logits, labels)


class MiewIdScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Warm-up and exponential decay schedule used by official MiewID."""

    def __init__(
        self,
        optimizer,
        lr_start=5e-6,
        lr_max=1e-5,
        lr_min=1e-6,
        lr_ramp_epochs=5,
        lr_sustain_epochs=0,
        lr_decay=0.8,
    ):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_epochs = lr_ramp_epochs
        self.lr_sustain_epochs = lr_sustain_epochs
        self.lr_decay = lr_decay
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.lr_ramp_epochs:
            lr = self.lr_start + (
                (self.lr_max - self.lr_start) * epoch / max(1, self.lr_ramp_epochs)
            )
        elif epoch < self.lr_ramp_epochs + self.lr_sustain_epochs:
            lr = self.lr_max
        else:
            decay_epoch = epoch - self.lr_ramp_epochs - self.lr_sustain_epochs
            lr = (self.lr_max - self.lr_min) * self.lr_decay**decay_epoch + self.lr_min
        return [lr for _ in self.optimizer.param_groups]
