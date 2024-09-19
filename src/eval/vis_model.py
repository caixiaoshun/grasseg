import argparse
from collections import OrderedDict
from glob import glob
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch
from torch import nn as nn
import torchvision
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from rich.progress import track
from src.utils.draw import pasteImages, give_colors_to_mask
from src.data.components.grass import Grass
from src.models.components.farseg import Farseg
from src.models.components.fcn import FCN
from src.models.components.linknet import Linknet
from src.models.components.pspnet import PSPNet
from src.models.components.unet_plus_plus import UnetPlusPlus
from src.models.components.pan import PAN
from src.models.components.unet import Unet
from src.models.components.deeplabv3plus import DeepLabV3Plus
from src.models.components.manet import MAnet
from src.models.components.fpn import FPN
from src.models.components.deeplabv3 import DeepLabV3


class VisModel:
    def __init__(self, device: str):
        self.device = device
        self.models = OrderedDict(
            {
                "linknet-timm-resnest101e": Linknet(encoder_name="timm-resnest101e").to(
                    self.device
                ),
                "fpn-timm-regnetx_320": FPN(encoder_name="timm-regnetx_320").to(
                    self.device
                ),
                "manet-se_resnext101_32x4d": MAnet(
                    encoder_name="se_resnext101_32x4d"
                ).to(self.device),
                "deeplabv3plus-timm-efficientnet-l2": DeepLabV3Plus(
                    encoder_name="timm-efficientnet-l2",encoder_weights="noisy-student-475"
                ).to(self.device),
                "farseg_resnet50": Farseg(backbone="resnet50").to(self.device),
                "unet-timm-efficientnet-l2": Unet(
                    encoder_name="timm-efficientnet-l2",encoder_weights="noisy-student-475"
                ).to(self.device),
                "pan-se_resnext101_32x4d": PAN(encoder_name="se_resnext101_32x4d").to(
                    self.device
                ),
                "unet_plus_plus-se_resnext101_32x4d": UnetPlusPlus(
                    encoder_name="se_resnext101_32x4d"
                ).to(self.device),
                "fcn-resnet50": FCN(weights="resnet50",num_classes=6).to(self.device),
                "pspnet-timm-efficientnet-l2": PSPNet(
                    encoder_name="timm-efficientnet-l2",encoder_weights="noisy-student-475"
                ).to(self.device),
                "deeplabv3-resnet152": DeepLabV3(encoder_name="resnet152").to(
                    self.device
                ),
            }
        )
        self.load_weights()
        self.dataloader = self.get_dataloader()

    def __load_state_dict(self, filename: str):
        ckpt = torch.load(filename, map_location=self.device)
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            state_dict[k[4:]] = v
        return state_dict

    def load_weights(self):
        for name, model in self.models.items():
            filename = glob(f"logs/grasseg/{name}/*/checkpoints/*epoch*.ckpt")[0]
            model.load_state_dict(self.__load_state_dict(filename))
            model.eval()
        print("weights loaded")

    def get_dataloader(self):
        img_transform = albu.Compose([albu.ToFloat(), ToTensorV2()])
        dataset = Grass(root="data/grass", phase="val", img_transform=img_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        print("dataloader loaded")
        return dataloader

    @torch.no_grad()
    def inference(self, model: nn.Module, img: torch.Tensor) -> np.ndarray:
        logits = model(img)
        preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        return preds

    def get_image(self, img: torch.Tensor) -> np.ndarray:
        """由数据范围为[0,1]tensor[1,c,h,w]数组转为转换为数据范围为[0,255] [h,w,c]的np.ndarray

        Args:
            img (torch.Tensor): [0,1]tensor数组

        Returns:
            np.ndarray: [0,255]的np.ndarray
        """
        img = (img * 255).type(torch.uint8).squeeze().permute(1, 2, 0)
        return img.detach().cpu().numpy()

    def add_title(self, image: np.ndarray, add_height: int = 50) -> None:
        """给图片添加标题

        Args:
            image (np.ndarray)
        """
        image = Image.fromarray(image)

        width, height = image.size

        new_height = height + add_height
        new_image = Image.new("RGB", (width, new_height), (255, 255, 255))

        new_image.paste(image, (0, add_height))
        draw = ImageDraw.Draw(new_image)
        font = ImageFont.truetype("resource/Times New Roman.ttf", size=30)

        titles = ["input", "label"] + list(self.models.keys())
        titles = [title.split("-")[0] for title in titles]
        num_cols = len(titles)
        col_width = width // num_cols

        for i, title in enumerate(titles):
            left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
            text_width = right - left
            text_height = bottom - top

            x = (i * col_width) + (col_width - text_width) // 2
            y = (add_height - text_height) // 2

            draw.text((x, y), title, font=font, fill=(0, 0, 0))
        new_image.save("images/model_eval.png")

    def vis(self):
        show_images = None
        for data in track(self.dataloader, total=len(self.dataloader)):

            paste_image_array = []

            img: torch.Tensor = data["image"].to(self.device)
            mask: np.ndarray = data["mask"].squeeze().cpu().numpy()
            image: np.ndarray = self.get_image(img)

            color_true = give_colors_to_mask(image, mask)

            paste_image_array.append(image)
            paste_image_array.append(color_true)
            for name, model in self.models.items():

                preds: np.ndarray = self.inference(model, img)
                color_pred = give_colors_to_mask(image, preds)
                paste_image_array.append(color_pred)

            paste_image = pasteImages(paste_image_array)
            paste_image = np.transpose(paste_image, (2, 0, 1))[np.newaxis]
            show_images = (
                paste_image
                if show_images is None
                else np.concatenate((show_images, paste_image), axis=0)
            )

        show_image_tensor = torch.from_numpy(show_images)
        grid_images = (
            torchvision.utils.make_grid(show_image_tensor, nrow=1, padding=2)
            .permute(1, 2, 0)
            .numpy()
        )
        self.add_title(grid_images)


def parse_args() -> str:
    parse = argparse.ArgumentParser()
    parse.add_argument("--device", type=str, default="cuda:0")

    args = parse.parse_args()
    return args.device


if __name__ == "__main__":
    # example usage: python src/eval/vis_model.py --device cuda:0
    device = parse_args()
    vis = VisModel(device=device)
    vis.vis()
