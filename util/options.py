# python version 3.7.1
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-algo', "--algorithm", type=str, default="FedTwin")
    parser.add_argument('-gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    #######################################################################
    ################### federated common arguments#########################
    parser.add_argument('--seed', type=int, default=13, help="random seed, default: 1")
    parser.add_argument('--save_dir', type=str, default='./record/', help="name of save directory")
    parser.add_argument('--rounds2', type=int, default=200, help="rounds of training in usual training stage")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs")
    parser.add_argument('--frac2', type=float, default=0.1,
                        help="fration of selected clients in fine-tuning and usual training stage")
    parser.add_argument('--num_users', type=int, default=100, help="number of uses: K")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate") #0.01
    parser.add_argument('--model', type=str, default='lenet', help="model name")
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--pretrained', action='store_true', help="whether to use pre-trained model")
    parser.add_argument('--iid', action='store_true', help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=10)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--alpha', type=float, default=1, help="0.1,1,5")
    ## noise arguments
    parser.add_argument('--LID_k', type=int, default=20, help="lid")
    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")



    ###### FedCorr ####################
    parser.add_argument('--iteration1', type=int, default=5, help="enumerate iteration in preprocessing stage")
    parser.add_argument('--rounds1', type=int, default=200, help="rounds of training in fine_tuning stage")
    parser.add_argument('--frac1', type=float, default=0.01, help="fration of selected clients in preprocessing stage")

    parser.add_argument('--beta', type=float, default=0, help="coefficient for local proximal，0 for fedavg, 1 for fedprox, 5 for noise fl")
    ## correction
    parser.add_argument('--relabel_ratio', type=float, default=0.5, help="proportion of relabeled samples among selected noisy samples")
    parser.add_argument('--confidence_thres', type=float, default=0.5, help="threshold of model's confidence on each sample")
    parser.add_argument('--clean_set_thres', type=float, default=0.1, help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")
    ## ablation study
    parser.add_argument('--fine_tuning', action='store_false', help='whether to include fine-tuning stage')
    parser.add_argument('--correction', action='store_false', help='whether to correct noisy labels')




    ###### RFL arguments###################
    parser.add_argument('--T_pl', type=int, help='T_pl: When to start using global guided pseudo labeling', default=100)
    parser.add_argument('--lambda_cen', type=float, help='lambda_cen', default=1.0)
    parser.add_argument('--lambda_e', type=float, help='lambda_e', default=0.8)


    # uncheck parameter
    parser.add_argument('--num_shards', type=int, default=200, help="number of shards")
    # noise arguments
    parser.add_argument('--noise_type', type=str, default='symmetric', choices=['symmetric', 'pairflip', 'clean'],
                        help='noise type of each clients')
    parser.add_argument('--noise_rate', type=float, default=0.2, help="noise rate of each clients")
    parser.add_argument('--num_gradual', type=int, default=10, help='T_k')
    parser.add_argument('--forget_rate', type=float, default=0.2, help="forget rate")
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay size")
    parser.add_argument('--schedule', nargs='+', default=[], help='decrease learning rate at these epochs.')
    # parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="sgd weight decay")
    parser.add_argument('--feature_dim', type=int, help='feature dimension', default=128)



    ###### FedTwinCORES####################
    parser.add_argument('--plr', help="--personal_learning_rate", type=str, default=0.09)
    # same with FedCorr
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument('--gamma', type=float, default=1, help="personalized aggregation")
    return parser.parse_args()

    # # training arguments
    # # parser.add_argument('--epochs', type=int, default=1000, help="rounds of training")
    # # parser.add_argument('--bs', type=int, default=128, help="test batch size")
    # # parser.add_argument('--lr', type=float, default=0.25, help="learning rate")
    # parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay size")
    # parser.add_argument('--schedule', nargs='+', default=[], help='decrease learning rate at these epochs.')
    # # parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    # parser.add_argument('--weight_decay', type=float, default=0.0001, help="sgd weight decay")
    # parser.add_argument('--feature_dim', type=int, help='feature dimension', default=128)
    #
    # # FL arguments
    # # parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    # # parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    # # parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    # # parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    #
    # # dataset arguments
    # # parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    # # parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    # # parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    # parser.add_argument('--num_shards', type=int, default=200, help="number of shards")
    #
    # # noise arguments
    # parser.add_argument('--noise_type', type=str, default='symmetric', choices=['symmetric', 'pairflip', 'clean'],
    #                     help='noise type of each clients')
    # parser.add_argument('--noise_rate', type=float, default=0.2, help="noise rate of each clients")
    # parser.add_argument('--num_gradual', type=int, default=10, help='T_k')
    # parser.add_argument('--forget_rate', type=float, default=0.2, help="forget rate")
    #
    # # # "Robust Federated Learning with Noisy Labels" arguments
    # # parser.add_argument('--T_pl', type=int, help='T_pl: When to start using global guided pseudo labeling', default=100)
    # # parser.add_argument('--lambda_cen', type=float, help='lambda_cen', default=1.0)
    # # parser.add_argument('--lambda_e', type=float, help='lambda_e', default=0.8)
    #
    # # Experiment arguments
    # # parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    # # # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    # # parser.add_argument('--save_dir', type=str, default=None, help="name of save directory")