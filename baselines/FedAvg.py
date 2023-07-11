from util.load_data import load_data_with_noisy_label
import time
from model.build_model import build_model
import numpy as np
from util.local_training import LocalUpdate, globaltest
import copy
from util.aggregation import FedAvg

def FedAVG(args):
    dataset_train, dataset_test, dict_users, y_train, gamma_s = load_data_with_noisy_label(args)

    # 开始联邦学习阶段
    print("FedAVG：")
    start = time.time()
    # 获取模型
    model = build_model(args)
    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1/args.num_users for i in range(args.num_users)]
    for rnd in range(args.rounds2):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:  # training over the subset
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(model).to(args.device), seed=args.seed,
                                                        w_g=model.to(args.device), epoch=args.local_ep, mu=0)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))
        loss_round = sum(loss_locals)/len(loss_locals)
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = FedAvg(w_locals, dict_len)
        model.load_state_dict(copy.deepcopy(w_glob_fl))
        acc_s2 = globaltest(copy.deepcopy(model).to(args.device), dataset_test, args)
        show_info_loss = "Round %d train loss  %.4f\n" % (rnd, loss_round)
        show_info_test_acc = "global test acc  %.4f \n\n" % (acc_s2)
        print(show_info_loss)
        print(show_info_test_acc)
    show_time_info = f"time : {time.time() - start}"
    print(show_time_info)