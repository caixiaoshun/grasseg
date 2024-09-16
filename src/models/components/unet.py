import torch
from typing import Optional,List,Union
from torch import nn as nn
import segmentation_models_pytorch as smp


class Unet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        num_classes: int = 6,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            aux_params=aux_params,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
if __name__ == "__main__":
    x = torch.rand((2, 3, 256, 256))
    model = Unet()
    out = model(x)
    print(out.shape)  # torch.Size([2, 6, 256, 256])
