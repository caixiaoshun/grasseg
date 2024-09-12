import torch
from torchgeo.models.farseg import FarSeg as FarSegBase
from typing import Literal


class Farseg(torch.nn.Module):
    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "resnet50", "resnet101"] = "resnet50",
        num_classes=6,
        backbone_pretrained=True,
    ):
        super().__init__()
        self.backbone = FarSegBase(
            backbone=backbone,
            classes=num_classes,
            backbone_pretrained=backbone_pretrained,
        )

    def forward(self, x):
        return self.backbone(x)
