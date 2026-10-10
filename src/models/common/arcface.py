import math

import torch
import torch.nn.functional as F
from torch import nn


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def init_from_prototypes(self, prototypes: torch.Tensor) -> None:
        with torch.no_grad():
            self.weight.copy_(F.normalize(prototypes, dim=1))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings.float()), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target = F.one_hot(labels, cosine.size(1)).bool()
        with_margin = torch.where(
            theta + self.margin < math.pi, torch.cos(theta + self.margin), cosine - self.margin * math.sin(self.margin)
        )
        return self.scale * torch.where(target, with_margin, cosine)
