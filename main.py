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
    # path
    # args.save_dir += f'{args.algorithm}/'
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    # args.txtname = '%s_%s_NL_%.1f_LB_%.1f_Rnd_%d_ep_%d_Frac_%.2f_LR_%.3f_Seed_%d' % (
    #     args.dataset, args.model, args.level_n_system, args.level_n_lowerb,
    #     args.rounds2, args.local_ep, args.frac2, args.lr, args.seed)
    # if args.iid:
    #     args.txtname += "_IID"
    # else:
    #     args.txtname += "_nonIID_p_%.1f_dirich_%.1f" % (args.non_iid_prob_class, args.alpha_dirichlet)
    # print args
    for x in vars(args).items():
        print(x)
    # run Algorithm
    eval(args.algorithm)(args)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # parse args
    args = args_parser()
    print(args)
    # float for some parameters
    args.lr = float(args.lr)
    args.plr = float(args.plr)
    args.frac2 = float(args.frac2)
    run(args)