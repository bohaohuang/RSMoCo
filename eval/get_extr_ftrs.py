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
import path_utils
from mrs_utils import misc_utils
from eval import prepare_dataset
from network import network_utils

# Settings
GPU = 0
task_dir, img_dir = path_utils.get_task_img_folder()
MODEL_DIR = r'/hdd6/Models/mrs/rsmoco_w/ecresnet50_ds8ds_equi_lr1e-03_ep200_bs200_ds100_150_dr0p1_crxent/epoch-200.pth.tar'


def get_model(model_dir):
    model = network_utils.DataParallelPassThrough(models.InsResNet50())
    network_utils.load(model, model_dir)
    print('Model loaded!')
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
            if isinstance(ftr, tuple):
                ftr = ftr[0]
            ftr_list.append(ftr.detach().cpu().numpy()[0, :])

        ftrs = np.stack(ftr_list, 0)
        misc_utils.save_file(ftr_file, ftrs)
        return ftrs
    else:
        return misc_utils.load_file(ftr_file)


def eval_ftr(ftr, lbl):
    ftr, lbl = sklearn.utils.shuffle(ftr, lbl)
    clf = svm.SVC(kernel='linear')
    return cross_val_score(clf, ftr, lbl, cv=5)


def umap_visualize(ftr, lbl, p, lbl_names):
    acc_1 = eval_ftr(ftr, lbl)
    acc_2 = eval_ftr(ftr, (np.array(p) > 100).astype(int))

    project_ftr = umap.UMAP(n_neighbors=50).fit_transform(ftr)

    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)
    for cnt, class_id in enumerate(sorted(np.unique(lbl))):
        plt.scatter(project_ftr[lbl == class_id, 0], project_ftr[lbl == class_id, 1], label=lbl_names[cnt])
    plt.legend()
    plt.title('City: Acc={:.2f}%'.format(acc_1.mean() * 100))

    plt.subplot(122, sharex=ax1, sharey=ax1)
    plt.scatter(project_ftr[:, 0], project_ftr[:, 1], c=p)
    plt.colorbar()
    plt.title('Building: Acc={:.2f}%'.format(acc_2.mean() * 100))

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'rsmoco.png'))
    plt.show()


def main():
    data_x, data_y, data_p, city_names = prepare_dataset.get_data()
    ftrs = get_features(MODEL_DIR, data_x, r'/media/ei-edl01/user/bh163/tasks/RSMoCo', force_run=True)
    umap_visualize(ftrs, data_y, data_p, city_names)


if __name__ == '__main__':
    main()
