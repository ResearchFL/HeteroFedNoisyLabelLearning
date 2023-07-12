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


# def f_beta(epoch, args):
#     max_beta = 2.0
#     beta1 = np.linspace(0.0, 0.0, num=1)  # 这么搞是用来让刚开始学习的时候学习率要稳定一些
#     # beta2 = np.linspace(0.0, max_beta/2, num=args.local_ep * 9)
#     beta2 = np.linspace(0, max_beta, num=args.local_ep * args.begin_sel)
#     beta3 = np.linspace(max_beta, max_beta, num=args.rounds2 * args.local_ep)
#     beta = np.concatenate((beta1, beta2, beta3), axis=0)
#     return beta[epoch]
def f_beta(epoch, args):
    max_beta = 2.0
    beta1 = np.linspace(0.0, 0.0, num=args.local_ep * 2)
    beta2 = np.linspace(0.0, max_beta, num=args.local_ep * args.begin_sel)
    beta3 = np.linspace(max_beta, max_beta, num=args.rounds2 * args.local_ep)
    beta = np.concatenate((beta1, beta2, beta3), axis=0)
    return beta[epoch]

# Adjust learning rate and for SGD Optimizer
# def adjust_learning_rate(optimizer, round, args):
#     alpha_plan = [args.plr] * 3 + [args.plr * 0.1] * 50
#     if type(optimizer).__name__ == "TwinOptimizer":
#         for param_group in optimizer.param_groups:
#             param_group['lr']= alpha_plan[round] / (1 + f_beta(round, args))
#     else:
#         for param_group in optimizer.param_groups:
#             param_group['lr']= alpha_plan[round] / (1 + f_beta(round, args))

# def adjust_learning_rate(epoch, args, optimizer=None):
#     alpha_plan = [[args.plr] * int(args.local_ep)
#                   + [args.plr * 0.1] * int(args.local_ep * args.rounds2),
#                   [args.lr] * int(args.local_ep)
#                   + [args.lr * 0.1] * int(args.local_ep * args.rounds2)]
#     if optimizer is None:
#         lr = alpha_plan[1][epoch] / (1 + f_beta(epoch, args))
#         return lr
#     else:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = alpha_plan[0][epoch] / (1 + f_beta(epoch, args))
def adjust_learning_rate(epoch, args, optimizer=None):
    # 需要后期再确认一次是否是args.begin_sel，之前是10
    alpha_plan = [[args.plr] * args.local_ep * args.begin_sel + [args.plr * 0.1] * args.local_ep * args.rounds2,
                  [args.lr] * args.local_ep * args.begin_sel + [args.lr * 0.1] * args.local_ep * args.rounds2]
    if optimizer is None:
        lr = alpha_plan[1][epoch] / (1 + f_beta(epoch, args))
        return lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = alpha_plan[0][epoch] / (1 + f_beta(epoch, args))