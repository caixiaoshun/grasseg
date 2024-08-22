import torch
from torch import nn as nn
from torch.nn import functional as F


class CrossEntropyWithAux(nn.Module):
    def __init__(self,scale:float=0.4):
        super().__init__()
        self.scale = scale

    def forward(
        self, out: torch.Tensor, aux: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        out_loss = F.cross_entropy(out, target)
        aux_loss = F.cross_entropy(aux, target)
        loss = out_loss + self.scale * aux_loss
        return loss
