import torch
import torchvision
from torch import nn as nn
from collections import OrderedDict


class FCN_RESNET101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.segmentation.fcn_resnet101(
            torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT
        )
        in_channels = self.backbone.classifier[-1].in_channels
        self.backbone.classifier[-1] = nn.Conv2d(in_channels, num_classes, 1, 1)

        in_channels = self.backbone.aux_classifier[-1].in_channels
        self.backbone.aux_classifier[-1] = nn.Conv2d(in_channels, num_classes, 1, 1)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        return self.backbone(x)
