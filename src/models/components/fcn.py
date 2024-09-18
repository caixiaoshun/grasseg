import torch
import torchvision
import torch.nn as nn
from typing import Literal


class FCN(nn.Module):
    def __init__(
        self, num_classes=6, weights: Literal["resnet50", "resnet101"] = "resnet50"
    ):
        super().__init__()
        self.num_classes = num_classes
        if weights == "resnet50":
            self.backbone = torchvision.models.segmentation.fcn_resnet50(
                pretrained=True,
                weights=torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT,
            )
        elif weights == "resnet101":
            self.backbone = torchvision.models.segmentation.fcn_resnet101(
                pretrained=True,
                weights=torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT,
            )
        else:
            raise ValueError("Invalid weights")
        in_channels = self.backbone.classifier[-1].in_channels
        self.backbone.classifier[-1] = nn.Conv2d(
            in_channels, num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)["out"]
        return x


if __name__ == "__main__":
    model = FCN(weights="resnet101")
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(out.shape)
