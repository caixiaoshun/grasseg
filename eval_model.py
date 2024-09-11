import torchvision
import torch
from torch.utils.data import DataLoader
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from src.models.components.fcn import FCN
from src.data.components.grass import Grass
from typing import List
from rich.progress import track
import numpy as np
from PIL import Image


def pasteImages(images: List[np.ndarray]):
    """
    将两张图片粘贴在一起
    """
    widths = [img.shape[1] for img in images]
    heights = [img.shape[0] for img in images]
    width = sum(widths)
    height = max(heights)
    paste_img = Image.new("RGB", (width, height))
    offset_x = 0
    for img in images:
        pil_img = Image.fromarray(img)
        paste_img.paste(pil_img, (offset_x, 0))
        offset_x += img.shape[1]

    paste_img = np.array(paste_img)
    return paste_img


def give_colors_to_mask(image: np.ndarray, mask: np.ndarray, num_classes=4):
    """
    将mask转换为彩色
    image: np.ndarray, (H, W, 3)
    mask: np.ndarray, (H, W)
    """
    image_tensor = torch.tensor(image).permute(2, 0, 1)
    masks = [mask == v for v in range(num_classes)]
    mask = np.stack(masks, axis=0).astype("bool")
    mask_tensor = torch.tensor(mask)
    colors = [
        (255, 255, 255),  # 白色    荒地
        (173, 255, 173),  # 浅绿    低覆盖度
        (60, 179, 113),  # 中绿   中低覆盖度
        (0, 128, 0),  # 深绿绿     中覆盖度
    ]

    mask_colors = (
        torchvision.utils.draw_segmentation_masks(
            image_tensor, mask_tensor, colors=colors, alpha=1.0
        )
        .permute(1, 2, 0)
        .numpy()
        .astype(np.uint8)
    )
    return mask_colors


class Eval:
    def __init__(self):
        self.model = FCN(num_classes=4)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.__load_weights()
        self.model.eval()
        self.dataloader = self.__load_data()

    def __load_weights(self):
        ckpt = torch.load(
            "logs/train/runs/2024-09-06_17-00-00/checkpoints/epoch_035.ckpt",
            map_location=self.device,
        )
        state_dict = {}
        for k, value in ckpt["state_dict"].items():
            k = k.replace("net.", "")
            state_dict[k] = value
        self.model.load_state_dict(state_dict)

    def __load_data(self):
        img_transform = albu.Compose([albu.ToFloat(), ToTensorV2()])
        train_pipeline = val_pipeline = test_pipeline = {
            "all_transform": None,
            "img_transform": img_transform,
            "ann_transform": None,
        }
        dataset = Grass(root="data/grass", phase="val",all_transform=None, img_transform=img_transform, ann_transform=None)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataloader

    @torch.no_grad()
    def infer(self, img):
        logits = self.model(img)["out"]
        pred = torch.argmax(logits, dim=1)
        return pred

    def eval(self):
        show_images = None
        for data in track(self.dataloader,total=len(self.dataloader)):
            img:torch.Tensor = data["image"].to(self.device)
            mask:torch.Tensor = data["mask"]
            pred:torch.Tensor = self.infer(img)

            image_np = img.permute(0,2,3,1).squeeze().cpu().numpy()
            image_np = image_np * 255
            image_np = image_np.astype(np.uint8)
            mask_np = mask.squeeze().cpu().numpy()
            pred_np = pred.squeeze().cpu().numpy()
            color_label = give_colors_to_mask(image_np, mask_np)
            color_pred = give_colors_to_mask(image_np, pred_np)
            paste_image = pasteImages([image_np, color_label, color_pred])
            paste_image = np.transpose(paste_image,(2,0,1))[np.newaxis]
            
            if show_images is None:
                show_images = paste_image
            else:
                show_images = np.concatenate([show_images, paste_image], axis=0)
        show_images_tensor = torch.tensor(show_images)
        grid_images = torchvision.utils.make_grid(show_images_tensor, nrow=5, padding=2).permute(1,2,0).numpy()
        im = Image.fromarray(grid_images)
        im.save("results.png",dpi=(300,300))


if __name__ == "__main__":
    Eval().eval()
