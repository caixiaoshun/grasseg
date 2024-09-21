from PIL import Image
import os
import argparse

def crop_image(image_path, output_path, patch_size:int=256):
    """_summary_: Crops an image into patches of a given size

    Args:
        image_path (_type_): _path to the image to be cropped
        output_path (_type_): _path to the output directory
        patch_size (_type_): _size of the patches to be cropped
    """
    image = Image.open(image_path)
    width, height = image.size
    n_rows = height // patch_size
    n_cols = width // patch_size

    for i in range(n_rows):
        for j in range(n_cols):
            patch = image.crop(
                (
                    patch_size * i,
                    patch_size * j,
                    patch_size * (i + 1),
                    patch_size * (j + 1)
                )
            )
            save_name = f"patch_{i}_{j}.png"
            file_path = os.path.join(output_path, save_name)
            patch.save(file_path,dpi=(300,300))
    print(f"Image cropped into {n_rows * n_cols} patches")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop an image into patches")
    parser.add_argument("image_path", type=str, help="Path to the image to be cropped")
    parser.add_argument("output_path", type=str, help="Path to the output directory")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches to be cropped")
    args = parser.parse_args()
    crop_image(args.image_path, args.output_path, args.patch_size)