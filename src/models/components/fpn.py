import torch
from typing import Optional
from torch import nn as nn
import segmentation_models_pytorch as smp


class FPN(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        num_classes: int = 6,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone = smp.FPN(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_pyramid_channels=decoder_pyramid_channels,
            decoder_segmentation_channels=decoder_segmentation_channels,
            decoder_merge_policy=decoder_merge_policy,
            decoder_dropout=decoder_dropout,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            upsampling=upsampling,
            aux_params=aux_params,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
if __name__ == "__main__":
    x = torch.rand((2, 3, 256, 256))
    model = FCN()
    out = model(x)
    print(out.shape)  # torch.Size([2, 6, 256, 256])
