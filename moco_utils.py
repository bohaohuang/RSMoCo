"""

"""


# Built-in
import os
import math

# Libs
import numpy as np
from tqdm import tqdm

# Pytorch
import torch
from torch import nn
from torch.utils import data

# Own modules
from mrs_utils import misc_utils


def moment_update(model, model_ema, m):
    """
    Momentum update in the paper
    model_ema = m * model_ema + (1-m) model
    """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1-m, p1.detach().data)


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out


class NCECriterion(nn.Module):
    def __init__(self, n_data, eps=1e-7):
        super(NCECriterion, self).__init__()
        self.n_data = n_data
        self.eps = eps

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + self.eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + self.eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


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
            file_list = misc_utils.load_file(file_list)
            self.img_list, self.lbl_list = get_file_paths(parent_path, file_list)
        except OSError:
            file_list = eval(file_list)
            parent_path = eval(parent_path)
            self.img_list, self.lbl_list = [], []
            for fl, pp in zip(file_list, parent_path):
                img_list, lbl_list = get_file_paths(pp, misc_utils.load_file(fl))
                self.img_list.extend(img_list)
                self.lbl_list.extend(lbl_list)
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        rgb = misc_utils.load_file(self.img_list[index])
        # rgb90, rgb180, rgb270 = np.rot90(rgb, 1), np.rot90(rgb, 2), np.rot90(rgb, 3)
        rgb1 = self.transforms(image=rgb)['image']
        rgb2 = self.transforms(image=rgb)['image']
        img = torch.cat([rgb1, rgb2, rgb1.transpose(1, 2).flip(1), rgb1.flip(1).flip(2), rgb1.transpose(1, 2).flip(2)],
                        dim=0)
        return img


def adjust_learning_rate(epoch, learn_rate, decay_step, decay_rate, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.1 every steep step"""
    steps = np.sum(epoch > np.asarray(decay_step))
    if steps > 0:
        new_lr = learn_rate * (decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_moco(train_loader, model, model_ema, contrast, criterion, rot_criterion, optimizer, opt,
               epoch, writer, scheduler):
    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != 1:
            m.train()

    def get_shuffle_ids(bsz):
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    loss_meter = AverageMeter()
    rot_meter = AverageMeter()
    prob_meter = AverageMeter()

    model.train()
    model_ema.eval()
    model_ema.apply(set_bn_train)

    for idx, inputs in enumerate(tqdm(train_loader)):
        writer.add_scalar('lr', scheduler.get_lr(), epoch*len(train_loader)+idx)
        bsz = inputs.size(0)
        inputs = inputs.float()
        inputs = inputs.cuda('cuda', non_blocking=True)

        # forward
        # ids for ShuffleBN
        x1, x2, x90, x180, x270 = torch.split(inputs, [3, 3, 3, 3, 3], dim=1)
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)

        # rotate
        feat_q, map_q = model(x1)
        _, map_q90 = model(x90)
        _, map_q180 = model(x180)
        _, map_q270 = model(x270)
        map_q90 = map_q90.transpose(2, 3).flip(3)
        map_q180 = map_q180.flip(2).flip(3)
        map_q270 = map_q270.transpose(2, 3).flip(2)

        with torch.no_grad():
            x2 = x2[shuffle_ids]
            feat_k, map_k = model_ema(x2)
            feat_k = feat_k[reverse_ids]
        out = contrast(feat_q, feat_k)

        loss = criterion(out)
        rot_loss = rot_criterion(map_q, torch.mean(torch.stack([map_q90, map_q180, map_q270]), dim=0))
        prob = out[:, 0].mean()

        # backprop
        optimizer.zero_grad()
        (loss+rot_loss).backward()
        optimizer.step()

        loss_meter.update(loss.item(), bsz)
        rot_meter.update(rot_loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        moment_update(model, model_ema, opt['trainer']['alpha'])

        torch.cuda.synchronize()

    return loss_meter.avg, rot_meter.avg, prob_meter.avg


if __name__ == '__main__':
    pass
