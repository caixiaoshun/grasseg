import seaborn as sns
import numpy as np
import argparse
from PIL import Image
from src.utils.invert_crop import invert_crop_images
from src.utils.draw import give_colors_to_mask
from src.utils.const import MULTI_CHANGE_COLORS

class MultiClassChangeDetection:
    """
    Class for evaluating multi-class change detection results.
    """

    def __init__(self,cur_path,prev_path,image_path,save_path):
        """
        :param num_classes: Number of classes.
        """
        self.cur_path = cur_path
        self.prev_path = prev_path
        self.image_path = image_path
        self.save_path = save_path
        print("save path:",self.save_path)
        self.prev_mask = np.array(invert_crop_images(self.prev_path, work_name="prev mask"))[:,:,0]
        self.cur_mask = np.array(invert_crop_images(self.cur_path, work_name="cur mask"))[:,:,0]
        self.image = np.array(invert_crop_images(self.image_path, work_name="image"))


    def evaluate(self):
        """
        Evaluate multi-class change detection results.
        :param cur: Current mask.
        :param prev: Previous mask.
        :return: Evaluation results.
        """
        change_matrix =self.cur_mask - self.prev_mask

        change_matrix = change_matrix + 5

        change_matrix = change_matrix.astype(np.uint8)
        
        color_mask = give_colors_to_mask(self.image,change_matrix,num_classes=len(MULTI_CHANGE_COLORS),colors=MULTI_CHANGE_COLORS)
        color_mask_image = Image.fromarray(color_mask)
        color_mask_image.save(f"{self.save_path}",dpi=(300,300))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cur_path', type=str, default=None, help='Current mask')
    parser.add_argument('--prev_path', type=str, default=None, help='Previous mask')
    parser.add_argument('--image_path', type=str, default=None, help='image path')
    parser.add_argument('--save_path', type=str, default=None, help='save path')
    args = parser.parse_args()
    return args.cur_path,args.prev_path,args.image_path,args.save_path

if __name__ == '__main__':
    cur_path,prev_path,image_path,save_path = get_args()
    MultiClassChangeDetection(cur_path,prev_path,image_path,save_path).evaluate()
