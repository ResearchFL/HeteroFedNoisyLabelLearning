from model.build_model import build_model
import copy
from util.aggregation import personalized_aggregation
import numpy as np
from util.load_data import load_data_with_noisy_label
# from data.dataloader_getter_utils import *
from util.local_training import FedTwinLocalUpdate, globaltest, personalizedtest
# from fed_utils.others import *
# from fed_utils.loss import *



def FedTwin(args):
    f_acc = open(args.txtpath + '_acc.txt', 'w')
    # load dataset
    dataset_train, dataset_test, dict_users, y_train, gamma_s = load_data_with_noisy_label(args)

    # begin training
    netglob = build_model(args)
    # parameter
    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1 / args.num_users for i in range(args.num_users)]
    for rnd in range(args.rounds2):
        if rnd <=1:
            print("\rRounds {:d} early training:"
              .format(rnd), end='\n', flush=True)
        else:
            print("\rRounds {:d} filter noisy data:"
                  .format(rnd), end='\n', flush=True)
        w_locals, p_models, loss_locals, n_bar = [], [], [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:  # training over the subset
            local = FedTwinLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], client_idx=idx)
            p_model, w_local, loss_local, n_bar_k = local.update_weights(net_p=copy.deepcopy(netglob).to(args.device),
                                                                         net_glob=copy.deepcopy(netglob).to(args.device), rounds=rnd, epoch=args.local_ep)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            p_models.append(p_model)
            loss_locals.append(loss_local)
            n_bar.append(n_bar_k)
            print('\n')

        # dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = personalized_aggregation(netglob.state_dict(), w_locals, n_bar, args.gamma)
        netglob.load_state_dict(w_glob_fl)
        # acc_s1 = personalizedtest(args, p_models, dataset_test)
        acc_s2 = globaltest(netglob.to(args.device), dataset_test, args)
        # f_acc.write("third stage round %d, personalized test acc  %.4f \n" % (rnd, acc_s1))
        show_info = "\n Round %d global test acc  %.4f \n" % (rnd, acc_s2)
        print(show_info)
        f_acc.write(show_info)
        f_acc.flush()
