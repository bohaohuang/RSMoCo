"""

"""


# Built-in
import os

# Libs
import torch
import numpy as np
import toolman as tm
from torch.utils import data

# Own modules


def get_file_paths(parent_path, file_list):
    """
    Parse the paths into absolute paths
    :param parent_path: the parent paths of all the data files
    :param file_list: the list of files
    :return:
    """
    img_list = []
    lbl_list = []
    for fl in file_list:
        img_filename, lbl_filename = [os.path.join(parent_path, a) for a in fl.strip().split(' ')]
        img_list.append(img_filename)
        lbl_list.append(lbl_filename)
    return img_list, lbl_list


class RSDataLoader(data.Dataset):
    def __init__(self, parent_path, file_list, transforms=None):
        """
        A data reader for the remote sensing dataset
        The dataset storage structure should be like
        /parent_path
            /patches
                img0.png
                img1.png
            file_list.txt
        Normally the downloaded remote sensing dataset needs to be preprocessed
        :param parent_path: path to a preprocessed remote sensing dataset
        :param file_list: a text file where each row contains rgb and gt files separated by space
        :param transforms: albumentation transforms
        """
        try:
            file_list = tm.misc_utils.load_file(file_list)
            self.img_list, self.lbl_list = get_file_paths(parent_path, file_list)
        except OSError:
            file_list = eval(file_list)
            parent_path = eval(parent_path)
            self.img_list, self.lbl_list = [], []
            for fl, pp in zip(file_list, parent_path):
                img_list, lbl_list = get_file_paths(pp, tm.misc_utils.load_file(fl))
                self.img_list.extend(img_list)
                self.lbl_list.extend(lbl_list)
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        rgb = tm.misc_utils.load_file(self.img_list[index])
        rgb1 = self.transforms(image=rgb)['image']
        rgb2 = self.transforms(image=rgb)['image']

        rot_rand = np.random.randint(0, 3)
        if rot_rand == 0:
            rgb_rot = rgb1.transpose(1, 2).flip(1)
        elif rot_rand == 1:
            rgb_rot = rgb1.flip(1).flip(2)
        else:
            rgb_rot = rgb1.transpose(1, 2).flip(2)

        img = torch.cat([rgb1, rgb2, rgb_rot], dim=0)
        return img, rot_rand
