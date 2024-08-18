# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 下午4:19
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : unetmobv2.py
# @Software: PyCharm
import segmentation_models_pytorch as smp
import torch
from torch import nn as nn


class UNetMobV2(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone = smp.Unet(
            encoder_name='mobilenet_v2',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == '__main__':
    fake_image = torch.rand(1, 3, 224, 224)
    model = UNetMobV2(num_classes=2)
    output = model(fake_image)
    print(output.size())