import torch
import torchvision
from torch import nn as nn

from torchgeo.models.fcn import FCN as _FCN


class FCN(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=3,
        num_filters=64,
    ):
        super().__init__()
        self.backbone = _FCN(in_channels=in_channels, num_filters=num_filters, classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
