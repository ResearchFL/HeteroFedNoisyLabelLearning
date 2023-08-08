from model.build_model import build_model
import copy
from util.aggregation import personalized_aggregation
import numpy as np
from util.load_data import load_data_with_noisy_label
# from data.dataloader_getter_utils import *
from util.local_training import FedTwinLocalUpdate, globaltest, personalizedtest
import time

from util.get_loss_of_clean_noisy_sample import get_clean_noisy_sample_loss
import torch.nn as nn
from util.loss import CORESLoss
from util.optimizer import f_beta
from metrics import cal_fscore
import torch

def FedTwin(args):
    # load dataset
    dataset_train, dataset_test, dict_users, y_train, gamma_s, noisy_sample_idx = load_data_with_noisy_label(args)
    start = time.time()
    targets = dataset_train.targets
    # begin training
    netglob = build_model(args)
    # parameter
    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1 / args.num_users for i in range(args.num_users)]
    for rnd in range(args.rounds2):
        if rnd <= args.begin_sel:
            print("\rRounds {:d} early training:"
                  .format(rnd), end='\n', flush=True)
        else:
            print("\rRounds {:d} filter noisy data:"
                  .format(rnd), end='\n', flush=True)
        w_locals, p_models, loss_locals, n_bar = [], [], [], []
        if args.without_regularization_term:
            p_models = [netglob.state_dict() for _ in range(args.num_users)]
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        noise_index_output_all = {}
        for idx in idxs_users:  # training over the subset
            local = FedTwinLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], client_idx=idx)
            if args.without_regularization_term:
                net_p = copy.deepcopy(netglob)
                net_p.load_state_dict(p_models[idx])
                p_model, w_local, loss_local, n_bar_k = local.update_weights(
                    net_p=net_p.to(args.device),
                    net_glob=copy.deepcopy(netglob).to(
                        args.device), rounds=rnd)
            else:
                p_model, w_local, loss_local, n_bar_k, noise_index_output = local.update_weights(net_p=copy.deepcopy(netglob).to(args.device),
                                                                         net_glob=copy.deepcopy(netglob).to(
                                                                             args.device), rounds=rnd)
            if rnd == args.begin_sel + 1:
                noise_index_output_all.update(noise_index_output)
                print("noise_index_output_all:{}".format(len(noise_index_output_all)))
                for index, output in noise_index_output_all.items():
                    #print("index:{}".format(index))
                    #print("outout_before:{}".format(output))
                    output = torch.softmax(output, dim=0)
                    #print("outout_after:{}".format(output))
                    if torch.max(output) > 0.2:
                        targets[index] = torch.argmax(output)


            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            if args.without_regularization_term:
                p_models[idx] = p_model
            loss_locals.append(loss_local)
            n_bar.append(n_bar_k)
            # print('\n')
        dataset_train.targets = targets
        loss_round = sum(loss_locals) / len(loss_locals)
        w_glob_fl = personalized_aggregation(netglob.state_dict(), w_locals, n_bar, args.gamma)
        netglob.load_state_dict(w_glob_fl)

        acc_s2 = globaltest(netglob.to(args.device), dataset_test, args)
        show_info_loss = "Round %d train loss  %.4f" % (rnd, loss_round)
        show_info_test_acc = "Round %d global test acc  %.4f \n" % (rnd, acc_s2)
        print(show_info_loss)
        print(show_info_test_acc)

        f_scores = []
        all_idxs_users = [i for i in range(args.num_users)]
        if rnd == args.rounds2 - 1:
            for idx in all_idxs_users:
                f_scores.append(
                    cal_fscore(args, netglob.to(args.device), dataset_train, y_train, dict_users[idx]))
            show_info_Fscore = "Round %d Fscore \n" % (rnd)
            print(show_info_Fscore)
            print(str(f_scores))

    show_time_info = f"time : {time.time() - start}"
    print(show_time_info)
    torch.save(netglob.state_dict(), f'{args.save_dir}_{time.time()}_fedtwin.pt')
