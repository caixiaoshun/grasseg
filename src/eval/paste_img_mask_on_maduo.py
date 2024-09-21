from PIL import Image, ImageFile
import os
from natsort import natsorted
from rich.progress import track
import argparse
from glob import glob
from typing import List, Tuple
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
        "--mask_folder",
        type=str,
        default="data/mask",
        help="Path to the input folder containing the masks",
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
    args = parser.parse_args()
    return args.image_folder, args.mask_folder, args.output_folder, args.filename


def main():
    image_folder, mask_folder, output_folder, filename = get_args()
    image = invert_crop_images(image_folder)
    mask = invert_crop_images(mask_folder)
    assert image.size == mask.size, "image and mask size not match"
    res = Image.new("RGB", (image.size[0] * 2, image.size[1]))
    res.paste(image, (0, 0))
    res.paste(mask, (image.size[0], 0))
    res.save(os.path.join(output_folder, filename))


if __name__ == "__main__":
    main()
