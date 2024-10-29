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
        classes=("backgroud", "low", "middle-low","middle", "middle-high", "high"),
        palette=(
            (255, 255, 255),  #  荒地
            (185,101,71),  #  低覆盖度
            (248,202,155),  # 中低覆盖度
            (211,232,158),  #  中覆盖度
            (138,191,104),   #  中高覆盖度
            (92,144,77),     #  高覆盖度
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
        image_paths = glob(os.path.join(self.root, "img_dir", self.phase,"*.tif"))
        masks = [filename.replace("img_dir", "ann_dir") for filename in image_paths]
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
