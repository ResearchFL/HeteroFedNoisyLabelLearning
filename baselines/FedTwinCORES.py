from model.build_model import build_model
import copy
from util.aggregation import personalized_aggregation
import numpy as np
from util.load_data import load_data_with_noisy_label
from data.dataloader_getter_utils import *
from util.local_training import FedTwinCORESLocalUpdate, globaltest, personalizedtest
from fed_utils.others import *
from fed_utils.loss import *



def FedDual(args, f_acc):
    # load dataset
    dataset_train, dataset_test, dict_users, y_train, gamma_s = load_data_with_noisy_label(args)

    # parameter
    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1 / args.num_users for i in range(args.num_users)]

    # begin training
    netglob = build_model(args)
    for rnd in range(args.rounds2):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:  # training over the subset
            local = FedTwinCORESLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, p_model, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                       w_g=netglob.to(args.device), epoch=args.local_ep, mu=0)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))

        # w_glob_fl = FedAvg(w_locals)  # global averaging
        # if args.iid:
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = personalized_aggregation(netglob, w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        acc_s1 = personalizedtest(copy.deepcopy(w_locals).to(args.device), dataset_test, args)
        acc_s2 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        f_acc.write("third stage round %d, personalized test acc  %.4f \n" % (rnd, acc_s1))
        f_acc.write("third stage round %d, global test acc  %.4f \n" % (rnd, acc_s2))
        f_acc.flush()
