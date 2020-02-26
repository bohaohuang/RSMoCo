"""
Load the pretrained RSMoCo model, extract the features and save them
"""


# Built-in
import os

# Libs
import umap
import sklearn.utils
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2

# PyTorch
import torch

# Own modules
import models
from mrs_utils import misc_utils
from eval import prepare_dataset
from network import network_utils

# Settings
GPU = 0
MODEL_DIR = r'/hdd6/Models/mrs/rsmoco_w/ecresnet50_ds8ds_lr1e-02_ep500_bs768_ds100_200_300_400_dr0p5_crxent/epoch-500.pth.tar'


def get_model(model_dir):
    model = models.InsResNet50()
    network_utils.load(model, model_dir, relax_load=True)
    return model


def get_features(model_dir, eval_data, save_dir, force_run=False):
    # ftr file
    ftr_file = os.path.join(save_dir, '{}.npy'.format(model_dir.split('/')[-1]))

    # run
    if not os.path.exists(ftr_file) or force_run:
        misc_utils.make_dir_if_not_exist(save_dir)
        # prepare model and data
        device, parallel = misc_utils.set_gpu(GPU)
        model = get_model(model_dir).to(device)

        ftr_list = []

        tsfms = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        for patch_file in tqdm(eval_data):
            rgb = misc_utils.load_file(patch_file)
            for tsfm in tsfms:
                tsfm_image = tsfm(image=rgb)
                rgb = tsfm_image['image']
            ftr = model(torch.unsqueeze(rgb, 0).to(device))
            ftr_list.append(ftr.detach().cpu().numpy()[0, :])

        ftrs = np.stack(ftr_list, 0)
        misc_utils.save_file(ftr_file, ftrs)
        return ftrs
    else:
        return misc_utils.load_file(ftr_file)


def eval_ftr(ftr, lbl):
    ftr, lbl = sklearn.utils.shuffle(ftr, lbl)
    clf = svm.SVC()
    return cross_val_score(clf, ftr, lbl, cv=5)


def umap_visualize(ftr, lbl, lbl_names):
    acc = eval_ftr(ftr, lbl)

    project_ftr = umap.UMAP(n_neighbors=50).fit_transform(ftr)
    for cnt, class_id in enumerate(sorted(np.unique(lbl))):
        plt.scatter(project_ftr[lbl == class_id, 0], project_ftr[lbl == class_id, 1], label=lbl_names[cnt])
    plt.legend()
    plt.title('Acc={:.2f}%'.format(acc.mean() * 100))
    plt.tight_layout()
    plt.show()


def main():
    data_x, data_y, city_names = prepare_dataset.get_data()
    ftrs = get_features(MODEL_DIR, data_x, r'/media/ei-edl01/user/bh163/tasks/RSMoCo', force_run=True)
    umap_visualize(ftrs, data_y, city_names)


if __name__ == '__main__':
    main()
