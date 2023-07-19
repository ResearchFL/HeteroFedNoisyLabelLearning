# python version 3.7.1
# -*- coding: utf-8 -*-
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import random
import torch
from util.options import args_parser
from baselines.FedCorr import FedCorr
from baselines.RFL import RFL
from baselines.FedTwin import FedTwin
from baselines.MR import MR
from baselines.FedAvg import FedAVG
from baselines.FedProx import FedProx
np.set_printoptions(threshold=np.inf)
"""
Major framework of noise FL
"""


def run(args):
    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu',
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if args.dataset == "mnist":
       args.K = 1
    else:
       args.K = 5
    for x in vars(args).items():
        print(x)
    # run Algorithm
    eval(args.algorithm)(args)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # parse args
    args = args_parser()
    # float for some parameters
    args.lr = float(args.lr)
    args.plr = float(args.plr)
    args.frac2 = float(args.frac2)
    args.begin_sel = int(args.begin_sel)
    run(args)