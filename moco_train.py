"""
Momentum Contrastive Learning
This file comes from: https://github.com/HobbitLong/CMC/blob/master/train_moco_ins.py
"""


# Built-in
import os
import sys
import json
import timeit
import argparse

# Libs
import albumentations as A
from tensorboardX import SummaryWriter
from albumentations.pytorch import ToTensorV2

# Pytorch
import torch
import torch.backends.cudnn as cudnn
from torch import optim, nn
from torch.utils.data import DataLoader

# Own modules
import models
import moco_utils
from network import network_utils
from mrs_utils import misc_utils

# Settings
CONFIG_FILE = 'config.json'

# TODO
"""
1. args.softmax as parameter
2. resume training
"""


def unique_model_name(cfg):
    """
    Make a unique model name based on the config file arguments
    :param cfg: config dictionary
    :return: unique model string
    """
    criterion_str = network_utils.make_criterion_str(cfg)
    decay_str = '_'.join(str(ds) for ds in eval(cfg['optimizer']['decay_step']))
    dr_str = str(cfg['optimizer']['decay_rate']).replace('.', 'p')
    return 'ec{}_ds{}_lr{:.0e}_ep{}_bs{}_ds{}_dr{}_cr{}'.format(
        cfg['encoder_name'], cfg['dataset']['ds_name'], cfg['optimizer']['learn_rate'], cfg['trainer']['epochs'],
        cfg['dataset']['batch_size'],
        decay_str, dr_str, criterion_str)


def read_config():
    parser = argparse.ArgumentParser()
    args, extras = parser.parse_known_args(sys.argv[1:])
    cfg_dict = misc_utils.parse_args(extras)
    if 'config' not in cfg_dict:
        cfg_dict['config'] = CONFIG_FILE
    flags = json.load(open(cfg_dict['config']))
    flags = misc_utils.update_flags(flags, cfg_dict)
    flags['save_dir'] = os.path.join(flags['trainer']['save_root'], unique_model_name(flags))
    return flags


def main(args, device):
    # prepare log directory
    log_dir = os.path.join(args['save_dir'], 'log')
    writer = SummaryWriter(log_dir=log_dir)

    # prepare the dataset
    mean = eval(args['dataset']['mean'])
    std = eval(args['dataset']['std'])
    crop_size = eval(args['dataset']['crop_size'])
    # define transforms
    tsfm_train = A.Compose([
        A.RandomResizedCrop(*crop_size),
        # A.Flip(),
        # A.ShiftScaleRotate(),
        # A.RandomRotate90(),
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
        A.RGBShift(),
        A.HueSaturationValue(),
        A.ImageCompression(),
        A.ToGray(),
        A.GaussNoise(),
        A.GaussianBlur(),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    train_loader = DataLoader(moco_utils.RSDataLoader(
        args['dataset']['data_dir'], args['dataset']['train_file'], transforms=tsfm_train),
        batch_size=args['dataset']['batch_size'], shuffle=True, num_workers=args['dataset']['num_workers'],
        drop_last=True)
    print('Training model on the {} dataset'.format(args['dataset']['ds_name']))
    n_data = len(train_loader)

    # create model
    model, model_ema = models.create_model(args['encoder_name'])
    moco_utils.moment_update(model, model_ema, 0)
    model = model.to(device)
    model_ema = model_ema.to(device)
    model = nn.DataParallel(model)
    model_ema = nn.DataParallel(model_ema)

    # set the momentum memory and criterion
    contrast = moco_utils.MemoryMoCo(128, n_data, args['trainer']['nce_k'], args['trainer']['nce_t'], True).to(device)
    criterion = moco_utils.NCESoftmaxLoss().to(device)
    rot_criterion = nn.MSELoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args['optimizer']['learn_rate'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=eval(args['optimizer']['decay_step']),
                                               gamma=args['optimizer']['decay_rate'])
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_data, eta_min=0, last_epoch=-1)
    cudnn.benchmark = True

    if eval(args['trainer']['finetune_dir']):
        print('Finetuning model from {}'.format(args['trainer']['finetune_dir']))
        network_utils.load(model, args['trainer']['finetune_dir'], relax_load=False)

    # train the model
    for epoch in range(0, args['trainer']['epochs']):
        start_time = timeit.default_timer()
        loss, rot, prob = moco_utils.train_moco(train_loader, model, model_ema, contrast, criterion, rot_criterion,
                                                optimizer, args, epoch, writer, scheduler)
        # moco_utils.adjust_learning_rate(epoch, args['optimizer']['learn_rate'], eval(args['optimizer']['decay_step']),
        #                                 args['optimizer']['decay_rate'], optimizer)
        scheduler.step()
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_data, 0)
        writer.add_scalar('loss_epoch', loss, epoch)
        writer.add_scalar('prob_epoch', prob, epoch)
        writer.add_scalar('rot_epoch', rot, epoch)
        end_time = timeit.default_timer()
        print('Epoch: {}/{} duration: {:.2f}s loss:{:.3f} rot:{:.3f} prob:{:.3f}'.format(
            epoch, args['trainer']['epochs'], float(str(end_time-start_time)), loss, rot, prob))

        # save the model
        if epoch % args['trainer']['save_epoch'] == 0 and epoch != 0:
            save_name = os.path.join(args['save_dir'], 'epoch-{}.pth.tar'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_name)
            print('Saved model at {}'.format(save_name))

        torch.cuda.empty_cache()

    # save model one last time
    save_name = os.path.join(args['save_dir'], 'epoch-{}.pth.tar'.format(args['trainer']['epochs']))
    torch.save({
        'epoch': args['trainer']['epochs'],
        'state_dict': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'contrast': contrast.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_name)
    writer.close()


if __name__ == '__main__':
    # settings
    cfg = read_config()
    # set gpu to use
    device, parallel = misc_utils.set_gpu(cfg['gpu'])

    main(cfg, device)
