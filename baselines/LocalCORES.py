from util.load_data import load_data_with_noisy_label
import time
from model.build_model import build_model
import numpy as np
from util.local_training import LocalCORESUpdate, globaltest
import copy
from util.aggregation import FedAvg


def LocalCORES(args):
    dataset_train, dataset_test, dict_users, y_train, gamma_s, _ = load_data_with_noisy_label(args)
    # federated learning stage
    # print("FedAVG：")
    start = time.time()
    # 获取模型
    model = build_model(args)
    pnets = [copy.deepcopy(model.state_dict()) for _ in range(args.num_users)]
    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1 / args.num_users for i in range(args.num_users)]

    for rnd in range(args.rounds2):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)

        for idx in idxs_users:  # training over the subset
            local = LocalCORESUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(model).to(args.device), pnet_dict=pnets[idx], rounds=rnd)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))

        loss_round = sum(loss_locals)/len(loss_locals)

        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = FedAvg(w_locals, dict_len)

        model.load_state_dict(copy.deepcopy(w_glob_fl)) # 全局模型

        acc_s2 = globaltest(model, dataset_test, args)

        show_info_loss = "Round %d train loss  %.4f" % (rnd, loss_round)
        show_info_test_acc = "Round %d global test acc  %.4f" % (rnd, acc_s2)

        # print(show_info_loss)
        print(show_info_test_acc)
    show_time_info = f"time : {time.time() - start}"
    print(show_time_info)