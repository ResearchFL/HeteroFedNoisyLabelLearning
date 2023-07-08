from typing import Optional
from torch import Tensor
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import numpy as np
class CORESLoss(CrossEntropyLoss):
    r"""
    Examples::
        >>> # Example of target with class indices
        >>> loss = CORESLoss()
        >>> beta = 0
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss.forward(input, target, beta)
        >>> output.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> output = loss.forward(input, target, beta)
        >>> output.backward()
    """
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduce = reduce

    def forward(self, input: Tensor, target: Tensor, beta, noise_prior=None) -> Tensor:
        # beta = f_beta(epoch)
        # if epoch == 1:
            # print(f'current beta is {beta}')
        loss = F.cross_entropy(input, target, reduce=self.reduce) # crossentropy loss
        loss_ = -torch.log(F.softmax(input) + 1e-8)
        if noise_prior is None:
            loss = loss - beta * torch.mean(loss_, 1)  # CORESLoss
        else:
            loss = loss - beta * torch.sum(torch.mul(noise_prior, loss_), 1)
        loss_ = loss
        return loss_

def filter_noisy_data(input: Tensor, target: Tensor):
    loss = F.cross_entropy(input, target, reduce=False)  # crossentropy loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)  # number of batch
    loss_v = np.zeros(num_batch)  # selected tag
    loss_ = -torch.log(F.softmax(input) + 1e-8)
    # sel metric
    loss_sel = loss - torch.mean(loss_, 1)  # CRLOSS - alpha
    loss_div_numpy = loss_sel.data.cpu().numpy()
    for i in range(len(loss_numpy)):
        if loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)

    return Variable(torch.from_numpy(loss_v)).cuda()
def f_beta(round):
    beta1 = np.linspace(0.0, 0.0, num=2)
    beta2 = np.linspace(0.0, 2, num=6)
    beta3 = np.linspace(2, 2, num=100-8)

    beta = np.concatenate((beta1, beta2, beta3), axis=0)
    # max_beta = 0.1
    # beta1 = np.linspace(0.0, 0.0, num=1)
    # beta2 = np.linspace(0.0, max_beta, num=50)
    # beta3 = np.linspace(max_beta, max_beta, num=5000)
    #
    # beta = np.concatenate((beta1, beta2, beta3), axis=0)
    return beta[round]


class FedTwinCRLoss(CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduce = reduce

    def forward(self, input_p, input_g, target, rounds, noise_prior=None):
        coresloss = CORESLoss(reduce=self.reduce)
        Beta = f_beta(rounds)
        if rounds <= 1:  # 如果在前30epoch集中式，对应联邦应该是30/local_epoch
            loss_p_update = coresloss(input_p, target, Beta, noise_prior)
            loss_g_update = coresloss(input_g, target, Beta, noise_prior)
        else:
            ind_p_update = filter_noisy_data(input_p, target)
            ind_g_update = filter_noisy_data(input_g, target)

            loss_p_update = coresloss(input_p[ind_g_update], target[ind_g_update], Beta, noise_prior)
            loss_g_update = coresloss(input_g[ind_p_update], target[ind_p_update], Beta, noise_prior)
        loss_batch_p = loss_p_update.data.cpu().numpy() # number of batch loss1
        loss_batch_g = loss_g_update.data.cpu().numpy()  # number of batch loss1
        if len(loss_batch_p) == 0.0:
            loss_p = torch.mean(loss_p_update) / 100000000
        else:
            loss_p = torch.sum(loss_p_update) / len(loss_batch_p)
        if len(loss_batch_g) == 0.0:
            loss_g = torch.mean(loss_g_update) / 100000000
        else:
            loss_g = torch.sum(loss_g_update) / len(loss_batch_g)
        return loss_p, loss_g, len(loss_batch_p), len(loss_batch_g)
