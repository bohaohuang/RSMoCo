"""
Make a dataset for evaluation purpose of the feature extractor
"""


# Built-in
import os

# Libs
import numpy as np

# PyTorch
import torch
from torch import nn
from torch.utils import data

# Own modules
from mrs_utils import misc_utils

# Settings
misc_utils.set_random_seed(0)
NUM_PER_CITY = 200


def get_data(data_dir=r'/hdd/moco/duke/ps224_pd0_ol0'):
    def get_city_name(line):
        return line.split('_')[0][:-1]

    def update_dict_list(d, city_name, file_name):
        if city_name not in d:
            d[city_name] = [file_name]
        else:
            d[city_name].append(file_name)

    file_list = os.path.join(data_dir, 'file_list_train.txt')
    patch_names = misc_utils.load_file(file_list)
    data_list = dict()
    for f_line in patch_names:
        f_name = os.path.join(data_dir, 'patches', f_line.split(' ')[0])
        update_dict_list(data_list, get_city_name(f_line), f_name)

    for k, v in data_list.items():
        data_list[k] = np.random.permutation(data_list[k])[:NUM_PER_CITY]

    eval_data_x, eval_data_y = [], []
    for cnt, (_, v) in enumerate(data_list.items()):
        assert len(v) == NUM_PER_CITY
        for f_name in v:
            eval_data_x.append(f_name)
            eval_data_y.append(cnt)

    return eval_data_x, eval_data_y, list(data_list.keys())


if __name__ == '__main__':
    select_data = get_data()
    for line in select_data:
        print(line)
