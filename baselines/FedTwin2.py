import numpy as np
import torch
from util.load_data import load_data_with_noisy_label
from util.options import args_parser
from model.build_model import build_model
from torch.utils.data import Subset, DataLoader, SequentialSampler
from util.local_training import FedAVGLocalUpdate, globaltest
from util.aggregation import FedAvg
import copy

import random
# =========================================================stage2=========================================================
# 对噪声标签做修正
def FedTwin2(args):


    ind_noise = np.load('../ind_noise.npy', allow_pickle=True).item()

    noise_list = []
    for _, value in ind_noise.items():
        for index, if_noise in value.items():
            index = index.numpy()
            noise_list.extend(index[if_noise])

    dataset_train, dataset_test, dict_users, y_train, gamma_s, noisy_sample_idx = load_data_with_noisy_label(args)
    samplerC = SequentialSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=128, shuffle=False, sampler=samplerC)
    #noise_dataset = Subset(dataset_train, noise_list)
    #noise_loader = dataloader(noise_dataset, batch_size=128)

    model = build_model(args)
    model.load_state_dict(torch.load("../model_fedTwin.pth"))
    model.eval()

    new_labels = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(args.device)
            outputs, _ = model(images)
            outputs = outputs.numpy()
            out = np.argmax(outputs, axis=1)
            # print("out:{}".format(out))
            new_labels.extend(out)

    old_label = dataset_train.targets
    new_labels = np.array(new_labels)
    old_label[noise_list] = new_labels[noise_list]

    dataset_train.targets = new_labels


    model.train()

    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1/args.num_users for i in range(args.num_users)]

    for rnd in range(args.rounds2):
        w_locals, loss_locals = [], []

        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)

        for idx in idxs_users:  # training over the subset
            local = FedAVGLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(model).to(args.device))
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))

        loss_round = sum(loss_locals)/len(loss_locals)

        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = FedAvg(w_locals, dict_len)

        model.load_state_dict(copy.deepcopy(w_glob_fl)) # 全局模型

        acc_s2 = globaltest(model, dataset_test, args)

        show_info_loss = "Round %d train loss  %.4f" % (rnd, loss_round)
        show_info_test_acc = "Round %d global test acc  %.4f" % (rnd, acc_s2)

        print(show_info_test_acc)



