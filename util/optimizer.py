import copy
from torch.autograd import Variable
from torch.optim import Optimizer
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import torch

class TwinOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(TwinOptimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = copy.deepcopy(local_weight_updated)
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)
        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = copy.deepcopy(local_weight_updated)
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        # return  p.data
        return group['params']


def filter_noisy_data(input: Tensor, target: Tensor):
    loss = F.cross_entropy(input, target, reduction='none')  # crossentropy loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)  # number of batch
    loss_v = np.zeros(num_batch)  # selected tag
    loss_ = -torch.log(F.softmax(input, dim=1) + 1e-8)
    # sel metric
    loss_sel = loss - torch.mean(loss_, 1)  # CRLOSS - alpha
    loss_div_numpy = loss_sel.data.cpu().numpy()
    for i in range(len(loss_numpy)):
        if loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)

    return Variable(torch.from_numpy(loss_v)).bool()

def f_beta(round):
    # beta1 = np.linspace(0.0, 0.0, num=2)
    # beta2 = np.linspace(0.0, 2, num=6)
    # beta3 = np.linspace(2, 2, num=100-8)
    #
    # beta = np.concatenate((beta1, beta2, beta3), axis=0)
    beta1 = np.linspace(0.0, 0.0, num=10)
    beta2 = np.linspace(0.0, 2, num=30)
    beta3 = np.linspace(2, 2, num=60)

    beta = np.concatenate((beta1, beta2, beta3), axis=0)
    return beta[round]

# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, round, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']= alpha_plan[round] / (1 + f_beta(round))


alpha_plan = [0.1] * 50 + [0.01] * 50