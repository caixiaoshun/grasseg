import numpy as np
from PIL import Image
from typing import List
import torch
import torchvision


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


def give_colors_to_mask(image: np.ndarray, mask: np.ndarray, num_classes=6)->np.ndarray:
    """将mask转换为彩色

    Args:
        image (np.ndarray): np.ndarray, (H, W, 3) dtype: uint8
        mask (np.ndarray): np.ndarray, (H, W)    dtype: uint8
        num_classes (int, optional): numclasses of mask. Defaults to 6.

    Returns:
        _type_: np.ndarray
    """
    image_tensor = torch.tensor(image).permute(2, 0, 1)
    masks = [mask == v for v in range(num_classes)]
    mask = np.stack(masks, axis=0).astype("bool")
    mask_tensor = torch.tensor(mask)
    colors = [
        (255, 255, 255),  # 白色    荒地
        (173, 255, 47),  # 浅绿    低覆盖度
        (144, 238, 144),  # 中绿   中低覆盖度
        (60, 179, 113),  # 深绿绿     中覆盖度
        (34, 139, 34),
        (0, 100, 0),
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