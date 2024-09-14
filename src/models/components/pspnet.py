import torch
from torch import nn as nn
import segmentation_models_pytorch as smp
from typing import Optional, Union


class PSPNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_depth: int = 3,
        psp_out_channels: int = 512,
        psp_use_batchnorm: bool = True,
        psp_dropout: float = 0.2,
        in_channels: int = 3,
        num_classes: int = 6,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 8,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            psp_out_channels=psp_out_channels,
            psp_use_batchnorm=psp_use_batchnorm,
            psp_dropout=psp_dropout,
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
    model = PSPNet()
    out = model(x)
    print(out.shape)  # torch.Size([2, 6, 256, 256])
