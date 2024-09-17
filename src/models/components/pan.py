import torch
from torch import nn as nn
import segmentation_models_pytorch as smp
from typing import Optional, Union,List


class PAN(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 32,
        in_channels: int = 3,
        num_classes: int = 6,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_output_stride=encoder_output_stride,
            decoder_channels=decoder_channels,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            upsampling=upsampling,
            aux_params=aux_params,
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    x = torch.rand((2, 3, 256, 256))
    model = PAN()
    out = model(x)
    print(out.shape)  # torch.Size([2, 6, 256, 256])
