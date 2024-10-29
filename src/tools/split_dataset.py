import os
from glob import glob
from typing import List, Literal
import shutil
import json
import numpy as np
from rich.progress import track
import cv2
from sklearn.model_selection import train_test_split
import argparse


def get_mask_by_json(filename: str) -> np.ndarray:
    """
    将json文件转换为mask
    """
    json_file = json.load(open(filename))
    img_height = json_file["imageHeight"]
    img_width = json_file["imageWidth"]
    mask = np.zeros((img_height, img_width), dtype="int8")
    for shape in json_file["shapes"]:
        label = int(shape["label"])
        points = np.array(shape["points"]).astype(np.int32)
        cv2.fillPoly(mask, [points], label)
    return mask


def json_to_image(json_path, image_path):
    """
    将json文件中的标注信息转换为图片,并保存至指定路径
    """
    mask = get_mask_by_json(json_path)
    cv2.imwrite(image_path, mask)


def create_dataset(
    image_paths: List[str],
    ann_paths: List[str],
    phase: Literal["train", "val"],
    output_dir: str,
):
    """
    划分数据集，将标注信息转换为图片，并保存至指定路径
    """
    for image_path, ann_path in track(
        zip(image_paths, ann_paths),
        description=f"{phase} dataset",
        total=len(image_paths),
    ):
        ann_save_path = os.path.join(
            output_dir,
            "ann_dir",
            phase,
            os.path.basename(ann_path).replace(".json", ".tif"),
        )

        # 将image复制到指定路径
        new_image_path = os.path.join(
            output_dir, "img_dir", phase, os.path.basename(image_path)
        )
        shutil.copy(image_path, new_image_path)

        # 将ann保存到指定路径
        json_to_image(ann_path, ann_save_path)


def split_dataset(
    root_path: str,
    output_path: str,
    split_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42,
) -> None:
    """
    Split a dataset into train, test, and validation sets.

    Args:
        root_path (str): Path to the dataset. The dataset should be organized as follows:
            dataset_path/
                image1.tif
                image2.tif
                ...
                imageN.tif
                label1.tif
                label2.tif
                ...
                labelN.tif
        output_path (str): Path to the output directory where the split dataset will be saved.
        split_ratio (float, optional): Ratio of the dataset to be used for training. Defaults to 0.8.
        seed (int, optional): Seed for the random number generator. Defaults to 42.
    """
    image_paths = glob(os.path.join(root_path, "*.tif"))
    ann_paths = [filename.replace("tif", "json") for filename in image_paths]
    assert len(image_paths) == len(
        ann_paths
    ), "Number of images and annotations do not match"
    print(f"images: {len(image_paths)}, annotations: {len(ann_paths)}")

    image_train, image_test, ann_train, ann_test = train_test_split(
        image_paths,
        ann_paths,
        train_size=split_ratio,
        random_state=seed,
        shuffle=shuffle,
    )
    print(f"train: {len(image_train)}, test: {len(image_test)}")

    os.makedirs(os.path.join(output_path, "img_dir", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "img_dir", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "ann_dir", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "ann_dir", "val"), exist_ok=True)

    create_dataset(image_train, ann_train, "train", output_path)
    create_dataset(image_test, ann_test, "val", output_path)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--root", type=str, default="data/raw_data")
    args.add_argument("--output", type=str, default="data/grass")
    args.add_argument("--split_ratio", type=float, default=0.8)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--shuffle", type=bool, default=True)
    args = args.parse_args()

    root: str = args.root
    output_path: str = args.output
    split_ratio: float = args.split_ratio
    seed: int = args.seed
    shuffle: bool = args.shuffle

    split_dataset(
        root_path=root,
        output_path=output_path,
        split_ratio=split_ratio,
        shuffle=shuffle,
        seed=seed,
    )

    print("数据集划分完成")


if __name__ == "__main__":

    # 使用示例 : python src/tools/split_dataset.py --root data/raw_data --output data/grass --split_ratio 0.8 --seed 42 --shuffle True
    main()
