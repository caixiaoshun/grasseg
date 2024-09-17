import torch
from torch import nn as nn
import segmentation_models_pytorch as smp
from typing import Optional, Union,List


class Linknet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        in_channels: int = 3,
        num_classes: int = 6,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone = smp.Linknet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            aux_params=aux_params,
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    x = torch.rand((2, 3, 256, 256))
    model = Linknet()
    out = model(x)
    print(out.shape)  # torch.Size([2, 6, 256, 256])
