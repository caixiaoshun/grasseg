# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 下午19:42
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : grass.py
# @Software: PyCharm

import os
from glob import glob
from typing import Literal
import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Grass(Dataset):
    METAINFO = dict(
        classes=("背景", "低覆盖度", "中低覆盖度","中覆盖度", "中高覆盖度", "高覆盖度"),
        palette=(
            (255, 255, 255),  #  荒地
            (173, 255, 47),  #  低覆盖度
            (144, 238, 144),  # 中低覆盖度
            (60, 179, 113),  #  中覆盖度
            (34, 139, 34),   #  中高覆盖度
            (0, 100, 0),     #  高覆盖度
        ),
        img_size=(256, 256),  # H, W
        ann_size=(256, 256),  # H, W
    )

    def __init__(
        self,
        root: str,
        phase: Literal["train", "val"] = "train",
        all_transform: albumentations.Compose = None,
        img_transform: albumentations.Compose = None,
        ann_transform: albumentations.Compose = None,
    ):
        self.root = root
        self.phase = phase
        self.all_transform = all_transform
        self.img_transform = img_transform
        self.ann_transform = ann_transform
        self.image_paths, self.mask_paths = self.__load_data()

    def __load_data(self):
        image_paths = glob(os.path.join(self.root, self.phase, "img", "*.tif"))
        masks = [filename.replace("img", "ann") for filename in image_paths]
        return image_paths, masks

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = np.array(image)
        mask = np.array(mask)

        if self.all_transform is not None:
            transformed = self.all_transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        if self.img_transform is not None:
            image = self.img_transform(image=image)["image"]
        if self.ann_transform is not None:
            mask = self.ann_transform(image=mask)["image"]

        return {
            "image": image,
            "mask": np.int64(mask),
            "filename": image_path.split(os.path.sep)[-1],
        }


if __name__ == "__main__":
    dataset = Grass(
        root="data/grass", all_transform=None, img_transform=None, ann_transform=None
    )
    print(len(dataset))
    data = dataset[0]
    print(data["image"].shape, data["mask"].shape, data["filename"])
