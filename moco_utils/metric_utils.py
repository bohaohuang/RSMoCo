"""

"""


# Built-in

# Libs
import torch
from torch import nn

# Own modules


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


def rotate_back(x_rot, rand_ind):
    """
    map_rot[rot_ind == 0, :] = map_rot[rot_ind == 0, :].transpose(2, 3).flip(3)
    map_rot[rot_ind == 1, :] = map_rot[rot_ind == 1, :].flip(2).flip(3)
    map_rot[rot_ind == 2, :] = map_rot[rot_ind == 2, :].transpose(2, 3).flip(2)
    """
    def rotate_helper(x, ri):
        if ri == 0:
            return x.transpose(1, 2).flip(2)
        elif ri == 1:
            return x.flip(1).flip(2)
        elif ri == 2:
            return x.transpose(1, 2).flip(1)
        else:
            raise NotImplementedError
    for cnt, r in enumerate(rand_ind):
        x_rot[cnt, :, :, :] = rotate_helper(x_rot[cnt, :, :, :], r)
    return x_rot


class RotCriterion(nn.Module):
    def __init__(self):
        super(RotCriterion, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, x_rot, rand_int):
        return self.criterion(x, rotate_back(x_rot, rand_int))


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
