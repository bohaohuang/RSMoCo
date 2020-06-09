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

# Own modules
from moco_utils import metric_utils


class DataParallelPassThrough(torch.nn.DataParallel):
    """
    Access model attributes after DataParallel wrapper
    this code comes from: https://github.com/pytorch/pytorch/issues/16885#issuecomment-551779897
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


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


def adjust_learning_rate(epoch, learn_rate, decay_step, decay_rate, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.1 every steep step"""
    steps = np.sum(epoch > np.asarray(decay_step))
    if steps > 0:
        new_lr = learn_rate * (decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


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

    loss_meter = metric_utils.AverageMeter()
    rot_meter = metric_utils.AverageMeter()
    prob_meter = metric_utils.AverageMeter()

    model.train()
    model_ema.eval()
    model_ema.apply(set_bn_train)

    for idx, (inputs, rot_ind) in enumerate(tqdm(train_loader)):
        writer.add_scalar('lr', scheduler.get_lr(), epoch*len(train_loader)+idx)
        bsz = inputs.size(0)
        inputs = inputs.float()
        inputs = inputs.cuda('cuda', non_blocking=True)

        # forward
        # ids for ShuffleBN
        x1, x2, x_rot = torch.split(inputs, [3, 3, 3], dim=1)
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)

        '''from data import data_utils
        from mrs_utils import vis_utils
        x_rot = metric_utils.rotate_back(x_rot, rot_ind)
        temp = [a[:, :, :3] for a in data_utils.change_channel_order(x1.detach().cpu().numpy())]
        temp.extend([a[:, :, :3] for a in data_utils.change_channel_order(x_rot.detach().cpu().numpy())])
        vis_utils.compare_figures(temp, (2, 4), (8, 4))
        exit(0)'''

        # rotate
        feat_q, map_q = model(x1)

        '''from data import data_utils
        from mrs_utils import vis_utils
        print(map_q.shape)
        temp = [a[:, :, :3] for a in data_utils.change_channel_order(map_q.detach().cpu().numpy())]
        temp.extend([a[:, :, :3] for a in data_utils.change_channel_order(map_rot.detach().cpu().numpy())])
        vis_utils.compare_figures(temp, (4, 4), (8, 8))
        exit(0)'''

        with torch.no_grad():
            x2 = x2[shuffle_ids]
            feat_k, map_k = model_ema(x2)
            feat_k = feat_k[reverse_ids]
        out = contrast(feat_q, feat_k)

        loss = criterion(out)
        prob = out[:, 0].mean()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # rot loss
        _, map_rot = model.forward(x_rot)
        rot_loss = rot_criterion(map_q.detach(), map_rot, rot_ind)
        rot_loss.backward()

        # update meter
        loss_meter.update(loss.item(), bsz)
        rot_meter.update(rot_loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        moment_update(model, model_ema, opt['trainer']['alpha'])

        torch.cuda.synchronize()

    return loss_meter.avg, rot_meter.avg, prob_meter.avg


def load_optim(optim, state_dict, device):
    """
    Load the optimizer and then individually transfer the optimizer parts, this part comes from
    https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/4
    :param optim: the optimizer
    :param state_dict: state dictionary
    :param device:  device to place the models
    :return:
    """
    optim.load_state_dict(state_dict)
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def load_epoch(save_dir, resume_epoch, model, optm, device, model_key='state_dict'):
    """
    Load model from a snapshot, this function can be used to resume training
    :param save_dir: directory that saved the model
    :param resume_epoch: the epoch number to continue training
    :param model: the model created by classes defined in network/
    :param optm: a torch optimizer
    :return:
    """
    checkpoint = torch.load(
        os.path.join(save_dir, 'epoch-' + str(resume_epoch) + '.pth.tar'),
        map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'epoch-' + str(resume_epoch) + '.pth.tar')))
    model.load_state_dict(checkpoint[model_key])
    load_optim(optm, checkpoint['opt_dict'], device)
