from glob import glob
from rich.progress import track
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from PIL import Image
import torch
import torchvision
import argparse
from src.utils.draw import give_colors_to_mask,pasteImages,draw_text_in_image

def get_args()->Tuple[str,str]:
    parser = argparse.ArgumentParser(description='Dataset Visualizer')
    parser.add_argument('--root', type=str, default='data/grass', help='root directory of dataset')
    parser.add_argument('--phase', type=str, default="val", help='phase of dataset')

    args = parser.parse_args()
    root = args.root
    phase = args.phase
    assert os.path.exists(root), f"root directory {root} does not exist"
    assert phase in ["train","val"], f"phase {phase} is not valid"
    return root, phase

def main():
    root, phase = get_args()
    image_paths = glob(os.path.join(root, phase,"img","*.tif"))
    ann_paths = [image_path.replace("img","ann") for image_path in image_paths]
    show_images = None
    for image_path, ann_path in track(zip(image_paths, ann_paths), total=len(image_paths)):
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(ann_path))
        color_mask = give_colors_to_mask(image, mask, num_classes=6)
        image_mask = pasteImages([image, color_mask])

        image_mask = draw_text_in_image(image_mask, f"{image_path.split('/')[-1]}", (10, 10), color="red")
        image_mask = np.transpose(image_mask, (2,0,1))[np.newaxis]
        if show_images is None:
            show_images = image_mask
        else:
            show_images = np.concatenate([show_images, image_mask], axis=0)

    show_images_tensor = torch.tensor(show_images)
    grid_images = torchvision.utils.make_grid(show_images_tensor, nrow=10, padding=2).permute(1,2,0).numpy()

    im = Image.fromarray(grid_images)

    im.save(f"images/dataset_vis/grass_{phase}.png",dpi=(300,300))

    print(f"Saved {len(image_paths)} images to images/dataset_vis/grass_{phase}.png")

if __name__ == '__main__':
    # 使用示例: python src/tools/dataset_visiual.py --root data/grass --phase train
    main()