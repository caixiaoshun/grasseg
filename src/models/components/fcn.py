import torch
import torchvision
from torch import nn as nn
from collections import OrderedDict
from typing import Literal, Union


class FCNVGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # conv1
        # 输入图像为3通道，输出64个特征图，卷积核大小为（3，3），步长为1，padding为100（避免图片不兼容，其实也可以为1的）
        # 卷积输出公式：output=(input+2*padding-kernel_size)/stride+1
        #  512=(512+2*1-3)/1+1
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn1_1 = nn.BatchNorm2d(num_features=64)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn1_2 = nn.BatchNorm2d(num_features=64)
        self.relu1_2 = nn.ReLU(inplace=True)

        # 最大池化层进行下采样
        # 采样输出公式：output=(input+2*padding-kernel_size)/stride+1
        # 256=(512+2*0-2)/2+1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2
        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.bn2_1 = nn.BatchNorm2d(num_features=128)
        self.relu2_1 = nn.ReLU(inplace=True)

        self.conv2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.bn2_2 = nn.BatchNorm2d(num_features=128)
        self.relu2_2 = nn.ReLU(inplace=True)

        # 最大池化层进行下采样
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3
        self.conv3_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn3_1 = nn.BatchNorm2d(num_features=256)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn3_2 = nn.BatchNorm2d(num_features=256)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.conv3_3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn3_3 = nn.BatchNorm2d(num_features=256)
        self.relu3_3 = nn.ReLU(inplace=True)

        # 最大池化层进行下采样
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4
        self.conv4_1 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.bn4_1 = nn.BatchNorm2d(num_features=512)
        self.relu4_1 = nn.ReLU(inplace=True)

        self.conv4_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.bn4_2 = nn.BatchNorm2d(num_features=512)
        self.relu4_2 = nn.ReLU(inplace=True)

        self.conv4_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.bn4_3 = nn.BatchNorm2d(num_features=512)
        self.relu4_3 = nn.ReLU(inplace=True)

        # 最大池化层进行下采样
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv5
        # 输入图像为3通道，输出64个特征图，卷积核大小为（3，3），步长为1，padding为100（避免图片不兼容）
        self.conv5_1 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.bn5_1 = nn.BatchNorm2d(num_features=512)
        self.relu5_1 = nn.ReLU(inplace=True)

        self.conv5_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.bn5_2 = nn.BatchNorm2d(num_features=512)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.conv5_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.bn5_3 = nn.BatchNorm2d(num_features=512)
        self.relu5_3 = nn.ReLU(inplace=True)

        # 最大池化层进行下采样
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # cnov6
        self.conv6 = nn.Conv2d(
            in_channels=512, out_channels=4096, kernel_size=7, stride=1, padding=1
        )
        self.bn6 = nn.BatchNorm2d(num_features=4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(p=0.5)

        # cnov7
        self.conv7 = nn.Conv2d(
            in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=1
        )
        self.bn7 = nn.BatchNorm2d(num_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=0.5)

        # cnov8，本项目有20个类别，一个背景，一共21类
        self.conv8 = nn.Conv2d(
            in_channels=4096,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=1,
        )

        # 上采样2倍（16，16，21）————>（32，32，21）
        self.up_conv8_2 = nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=2,
            stride=2,
            bias=False,
        )

        # 反卷积ConvTranspose2d操作输出宽高公式
        # output=((input-1)*stride)+outputpadding-(2*padding)+kernelsize
        # 34=(16-1)*2+0-(2*0)+4

        # 第4层maxpool值做卷积运算
        self.pool4_conv = nn.Conv2d(
            in_channels=512, out_channels=num_classes, kernel_size=1, stride=1
        )

        # 利用反卷积上采样2倍
        self.up_pool4_2 = nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=2,
            stride=2,
            bias=False,
        )

        # 第3层maxpool值做卷积运算
        self.pool3_conv = nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1, stride=1
        )

        # 利用反卷积上采样8倍
        self.up_pool3_8 = nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=8,
            stride=8,
            bias=False,
        )

    def forward(self, x):
        """正向传播"""

        # 记录初始图片的大小（32，21，512，512）
        h = x

        # conv1
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.maxpool1(x)

        # conv2
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.maxpool2(x)

        # conv3
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.relu3_3(self.bn3_3(self.conv3_3(x)))
        x = self.maxpool3(x)
        pool3 = x

        # conv4
        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = self.relu4_2(self.bn4_2(self.conv4_2(x)))
        x = self.relu4_3(self.bn4_3(self.conv4_3(x)))
        x = self.maxpool4(x)
        pool4 = x

        # conv5
        x = self.relu5_1(self.bn5_1(self.conv5_1(x)))
        x = self.relu5_2(self.bn5_2(self.conv5_2(x)))
        x = self.relu5_3(self.bn5_3(self.conv5_3(x)))
        x = self.maxpool5(x)

        # conv6
        #         print(self.conv6(x).shape)
        #         print(self.bn6(self.conv6(x)).shape)
        #         print(self.relu6(self.bn6(self.conv6(x))).shape)
        #         print(self.drop6(self.relu6(self.bn6(self.conv6(x)))).shape)
        x = self.drop6(self.relu6(self.bn6(self.conv6(x))))

        # conv7
        x = self.drop7(self.relu7(self.bn7(self.conv7(x))))

        # conv8
        x = self.up_conv8_2(self.conv8(x))
        up_conv8 = x

        # 计算第4层的值
        x2 = self.pool4_conv(pool4)
        # 相加融合
        x2 = up_conv8 + x2
        # 反卷积上采样8倍
        x2 = self.up_pool4_2(x2)
        up_pool4 = x2

        # 计算第3层的值
        x3 = self.pool3_conv(pool3)
        x3 = up_pool4 + x3

        # 反卷积上采样8倍
        x3 = self.up_pool3_8(x3)
        return x3


class FCN(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone: Literal["resnet101", "resnet50", "vgg16"] = "resnet101",
        pretrain=True,
    ):
        super().__init__()
        self.backbone = None
        if backbone == "resnet101":
            if pretrain:
                self.backbone = torchvision.models.segmentation.fcn_resnet101(
                    torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT
                )
                in_channels = self.backbone.classifier[-1].in_channels
                self.backbone.classifier[-1] = nn.Conv2d(in_channels, num_classes, 1, 1)
            else:
                self.backbone = torchvision.models.segmentation.fcn_resnet101(
                    weights=None, num_classes=num_classes, aux_loss=True
                )
        elif backbone == "resnet50":

            if pretrain:
                self.backbone = torchvision.models.segmentation.fcn_resnet50(
                    torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
                )
                in_channels = self.backbone.classifier[-1].in_channels
                self.backbone.classifier[-1] = nn.Conv2d(in_channels, num_classes, 1, 1)
            else:
                self.backbone = torchvision.models.segmentation.fcn_resnet50(
                    weights=None, num_classes=num_classes, aux_loss=True
                )
        elif backbone == "vgg16":
            self.backbone = FCNVGG16(num_classes=num_classes)

        else:
            raise ValueError(
                f"backbone should in ['resnet101', 'resnet50','vgg16'],but actually is {backbone}"
            )

    def forward(self, x: torch.Tensor) -> Union[OrderedDict, torch.Tensor]:
        return self.backbone(x)
