from PIL import Image, ImageFile
import os
from natsort import natsorted
from rich.progress import track
import argparse
from glob import glob
from typing import List, Tuple
import numpy as np
import json
from src.data.components.grass import Grass
from src.utils.invert_crop import invert_crop_images

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def get_args() -> Tuple[str, str, str, str]:
    parser = argparse.ArgumentParser(description="Invert and crop images")
    parser.add_argument(
        "--image_folder",
        type=str,
        default="data/output",
        help="Path to the input folder containing the images",
    )
    parser.add_argument(
        "--color_mask_folder",
        type=str,
        default="data/color_mask",
        help="Path to the color mask folder containing the masks",
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        default="data/mask",
        help="Path to the mask folder containing the masks",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Path to the output folder where the inverted and cropped images will be saved",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="res.png",
        help="Filename of the image to be inverted and cropped",
    )
    parser.add_argument(
        "--statistic_path",
        type=str,
        default="res.png",
        help="statistic path of the image to be saved",
    )
    args = parser.parse_args()
    return (
        args.image_folder,
        args.color_mask_folder,
        args.mask_folder,
        args.output_folder,
        args.filename,
        args.statistic_path,
    )


def statistic_pixel(mask: Image.Image, classes=len(Grass.METAINFO["palette"])):
    mask = np.array(mask)
    statistic = {}
    for i in range(classes):
        statistic[i] = int(np.sum(mask == i))
    total = sum(list(statistic.values()))

    vegetation_coverage = 0
    ratio = [0,0.05,0.15,0.3,0.6,0.9]

    for i in range(classes):
        statistic[i] = statistic[i] / total
        statistic[i] = statistic[i] * 25253
    return statistic


def main():
    (
        image_folder,
        color_mask_folder,
        mask_folder,
        output_folder,
        filename,
        statistic_path,
    ) = get_args()

    image = invert_crop_images(image_folder, work_name="invert images")
    color_mask = invert_crop_images(color_mask_folder, work_name="invert color masks")
    mask = invert_crop_images(mask_folder, work_name="invert masks")
    assert image.size == color_mask.size, "image and mask size not match"
    res = Image.new("RGB", (image.size[0] * 2, image.size[1]))
    res.paste(image, (0, 0))
    res.paste(color_mask, (image.size[0], 0))
    res.save(os.path.join(output_folder, filename))

    statistic = statistic_pixel(mask)
    with open(statistic_path, "w") as f:
        f.write(json.dumps(statistic, indent=4))


if __name__ == "__main__":
    main()
