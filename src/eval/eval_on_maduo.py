import torch
from PIL import Image
from glob import glob
from typing import Tuple, List, Dict
from rich.progress import track
from src.data.components.grass import Grass
from torch import nn as nn
from src.models.components.linknet import Linknet
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import os
import numpy as np
from natsort import natsorted
import argparse
from src.utils.draw import give_colors_to_mask


def load_weight(filename: str, device: str) -> Dict:
    ckpt = torch.load(filename, map_location=device)
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        state_dict[k[4:]] = v
    return state_dict


def load_module(device) -> nn.Module:
    model = Linknet(encoder_name="timm-resnest101e").to(device)
    ckpt_path = glob(
        "logs/grasseg/linknet-timm-resnest101e/*/checkpoints/*epoch*.ckpt"
    )[0]
    model.load_state_dict(load_weight(ckpt_path, device))
    model.eval()
    return model


def load_data(input_dir:str) -> List[str]:
    filenames = glob(os.path.join(input_dir, "*.png"))
    filenames = natsorted(filenames)
    return filenames


def read_image(filename: str) -> Tuple[np.ndarray, torch.Tensor]:
    image = np.array(Image.open(filename))
    transform = albu.Compose([albu.ToFloat(), ToTensorV2()])
    img_transform:torch.Tensor = transform(image=image)["image"]
    image_tensor = img_transform.unsqueeze(0)
    return image, image_tensor


@torch.no_grad()
def inference(model: nn.Module, img: torch.Tensor) -> np.ndarray:
    logits = model(img)
    preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype("uint8")
    return preds


def main(input_dir:str,device: str,color_mask_dir: str,mask_path:str):
    os.makedirs(color_mask_dir, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    model = load_module(device)
    filenames = load_data(input_dir)
    for filename in track(filenames, total=len(filenames)):
        image, x = read_image(filename)
        if np.min(image) == np.max(image):
            color_mask = np.ones_like(image, dtype=np.uint8)
            color_mask = color_mask * 255
            mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            x = x.to(device)
            mask = inference(model, x)
            color_mask = give_colors_to_mask(image,mask)
        
        base_name = os.path.basename(filename)
        output_path = os.path.join(color_mask_dir, base_name)
        Image.fromarray(color_mask).save(output_path)
        Image.fromarray(mask).save(os.path.join(mask_path, base_name))
    print("Done")

def get_args()->Tuple[str,str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,required=True,help="input directory")
    parser.add_argument("--device", type=str, default="cpu",help="device to use")
    parser.add_argument("--color_mask_dir", type=str, default="data/output",help="output directory",required=True)
    parser.add_argument("--mask_path", type=str, help="mask path",required=True)
    args = parser.parse_args()
    return args.input_dir,args.device, args.color_mask_dir,args.mask_path

if __name__ == "__main__":
    input_dir,device, color_mask_dir,mask_path = get_args()
    main(input_dir,device,color_mask_dir,mask_path)