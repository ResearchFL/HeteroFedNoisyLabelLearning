from model.build_model import build_model
from util.local_training import FedCorrLocalUpdate, globaltest
from util.aggregation import FedAvg
import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.mixture import GaussianMixture
import torch.nn as nn
import copy
from util.util import lid_term, get_output
from util.load_data import load_data_with_noisy_label
import time
from metrics import cal_fscore
from math import ceil

def FedCorr(args):
    # if args.mixup:
    #     args.txtname += "_Mix_%.1f" % (args.alpha)
    # if args.fine_tuning:
    #     args.txtname += "_FT"
    # if args.correction:
    #     args.txtname += "_CORR"
    # f_save = open(args.save_dir + args.txtname + '_acc.txt', 'a')
    ##############################
    #  Load Dataset
    ##############################
    dataset_train, dataset_test, dict_users, y_train, gamma_s, _ = load_data_with_noisy_label(args)

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # build model
    start = time.time()
    ##############################
    # Build model
    ##############################
    netglob = build_model(args)
    print(netglob)
    net_local = build_model(args)

    # client_p_index = np.where(gamma_s == 0)[0]
    # client_n_index = np.where(gamma_s > 0)[0]
    criterion = nn.CrossEntropyLoss(reduction='none')
    LID_accumulative_client = np.zeros(args.num_users)

    for iteration in range(args.iteration1):
        LID_whole = np.zeros(len(y_train))
        loss_whole = np.zeros(len(y_train))
        LID_client = np.zeros(args.num_users)
        loss_accumulative_whole = np.zeros(len(y_train))

        # ---------Broadcast global model----------------------
        if iteration == 0:
            mu_list = np.zeros(args.num_users)
        else:
            mu_list = estimated_noisy_level

        prob = [1 / args.num_users] * args.num_users
        prob_normalized = prob / np.sum(prob)
        for _ in range(int(1 / args.frac1)):
            idxs_users = np.random.choice(range(args.num_users), ceil(args.num_users * args.frac1), p=prob_normalized)
            w_locals = []
            for idx in idxs_users:
                prob[idx] = 0
                if sum(prob) > 0:
                    prob = [prob[i] / sum(prob) for i in range(len(prob))]

                net_local.load_state_dict(netglob.state_dict())
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)

                # proximal term operation
                mu_i = mu_list[idx]
                local = FedCorrLocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)
                w, loss = local.update_weights(net=copy.deepcopy(net_local).to(args.device),
                                               w_g=netglob.to(args.device), epoch=args.local_ep, mu=mu_i)

                net_local.load_state_dict(copy.deepcopy(w))
                w_locals.append(copy.deepcopy(w))
                acc_t = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
                print("iteration %d, client %d, acc: %.4f \n" % (iteration, idx, acc_t))
                #f_save.flush()

                local_output, loss = get_output(loader, net_local.to(args.device), args, False, criterion)
                LID_local = list(lid_term(local_output, local_output))
                LID_whole[sample_idx] = LID_local
                loss_whole[sample_idx] = loss
                LID_client[idx] = np.mean(LID_local)

            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob = FedAvg(w_locals, dict_len)

            netglob.load_state_dict(copy.deepcopy(w_glob))

        LID_accumulative_client = LID_accumulative_client + np.array(LID_client)
        loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)

        # Apply Gaussian Mixture Model to LID
        gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=args.seed).fit(
            np.array(LID_accumulative_client).reshape(-1, 1))
        labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(LID_accumulative_client).reshape(-1, 1))
        clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]

        noisy_set = np.where(labels_LID_accumulative != clean_label)[0]
        clean_set = np.where(labels_LID_accumulative == clean_label)[0]

        estimated_noisy_level = np.zeros(args.num_users)

        for client_id in noisy_set:
            sample_idx = np.array(list(dict_users[client_id]))

            loss = np.array(loss_accumulative_whole[sample_idx])
            gmm_loss = GaussianMixture(n_components=2, random_state=args.seed).fit(np.array(loss).reshape(-1, 1))
            labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
            gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

            pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
            estimated_noisy_level[client_id] = len(pred_n) / len(sample_idx)
            y_train_noisy_new = np.array(dataset_train.targets)

        if args.correction:
            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                loss = np.array(loss_accumulative_whole[sample_idx])
                local_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
                relabel_idx = list(
                    set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(relabel_idx))

                y_train_noisy_new = np.array(dataset_train.targets)
                y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]
                dataset_train.targets = y_train_noisy_new

    # reset the beta,
    args.beta = 0

    # ---------------------------- second stage training -------------------------------
    if args.fine_tuning:
        selected_clean_idx = np.where(estimated_noisy_level <= args.clean_set_thres)[0]

        prob = np.zeros(args.num_users)  # np.zeros(100)
        prob[selected_clean_idx] = 1 / len(selected_clean_idx)
        m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
        m = min(m, len(selected_clean_idx))
        netglob = copy.deepcopy(netglob)
        # add fl training
        for rnd in range(args.rounds1):
            w_locals, loss_locals = [], []
            idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
            for idx in idxs_users:  # training over the subset
                local = FedCorrLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device),
                                                           w_g=netglob.to(args.device), epoch=args.local_ep, mu=0)
                w_locals.append(copy.deepcopy(w_local))  # store every updated model
                loss_locals.append(copy.deepcopy(loss_local))

            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob_fl = FedAvg(w_locals, dict_len)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))

            acc_s2 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
            print("fine tuning stage round %d, test acc  %.4f \n" % (rnd, acc_s2))
            #f_save.flush()

        if args.correction:
            relabel_idx_whole = []
            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                glob_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                y_predicted = np.argmax(glob_output, axis=1)
                relabel_idx = np.where(np.max(glob_output, axis=1) > args.confidence_thres)[0]
                y_train_noisy_new = np.array(dataset_train.targets)
                y_train_noisy_new[sample_idx[relabel_idx]] = y_predicted[relabel_idx]
                dataset_train.targets = y_train_noisy_new

    # ---------------------------- third stage training -------------------------------
    # third stage hyper-parameter initialization
    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1 / args.num_users for i in range(args.num_users)]

    for rnd in range(args.rounds2):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:  # training over the subset
            local = FedCorrLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device),
                                                       w_g=netglob.to(args.device), epoch=args.local_ep, mu=0)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))

        # w_glob_fl = FedAvg(w_locals)  # global averaging
        # if args.iid:
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        acc_s2 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        show_info_test_acc = "Round %d global test acc  %.4f" % (rnd, acc_s2)
        print(show_info_test_acc)
        # f_save.write(show_info_test_acc)
        # f_save.flush()

        f_scores = []
        all_idxs_users = [i for i in range(args.num_users)]
        if rnd == args.rounds2 - 1:
            for idx in all_idxs_users:
                f_scores.append(cal_fscore(args, netglob.to(args.device), dataset_train, y_train, dict_users[idx]))
        show_info_Fscore = "Round %d Fscore \n" % (rnd)
        print(show_info_Fscore)
        print(str(f_scores))
        # file_name = "./fscore_" + args.algorithm + ".txt"
        # f = open(file_name, "w")
        # f.writelines(f_scores.to)
        # f.close()

    show_time_info = f"time : {time.time() - start}"
    print(show_time_info)
    # f_save.write(show_time_info)
    # f_save.flush()