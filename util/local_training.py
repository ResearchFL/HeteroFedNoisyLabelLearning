# python version 3.7.1
# -*- coding: utf-8 -*-
import copy

from util.loss import CORESLoss
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from util.loss import FedTwinCRLoss
import numpy as np
from util.optimizer import TwinOptimizer, adjust_learning_rate
from util.optimizer import FedProxOptimizer, f_beta

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]
        return image, label, self.idxs[item]


class FedCorrLocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = CrossEntropyLoss()  # loss function -- cross entropy
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, w_g, epoch, mu=1, lr=None):
        net_glob = w_g

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels, _) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                if self.args.mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(images, labels, self.args.alpha)
                    net.zero_grad()
                    log_probs, _ = net(inputs)
                    loss = mixup_criterion(self.loss_func, log_probs, targets_a, targets_b, lam)
                else:
                    labels = labels.long()
                    net.zero_grad()
                    log_probs, _ = net(images)
                    loss = self.loss_func(log_probs, labels)

                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class FedTwinLocalUpdate:
    def __init__(self, args, dataset, idxs, client_idx):
        self.args = args
        self.loss_func = FedTwinCRLoss()  # loss function -- cross entropy
        self.cores_loss_fun = CORESLoss(reduction='none')
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))
        self.client_idx = client_idx

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net_p, net_glob, rounds, args):
        net_p.train()
        net_glob.train()
        # net_global_param = copy.deepcopy(list(net_glob.parameters()))
        # train and update
        optimizer_theta = TwinOptimizer(net_p.parameters(), lr=self.args.plr, lamda=self.args.lamda)
        optimizer_w = torch.optim.SGD(net_glob.parameters(), lr=self.args.lr)
        epoch_loss = []
        n_bar_k = []
        for iter in range(args.local_ep):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            b_bar_p = []
            # lr = args.lr
            adjust_learning_rate(rounds * args.local_ep + iter, args, optimizer_theta)
            adjust_learning_rate(rounds * args.local_ep + iter, args, optimizer_w)
            plr = adjust_learning_rate(rounds * args.local_ep + iter, args, 'plr')
            lr = adjust_learning_rate(rounds * args.local_ep + iter, args, 'lr')

            for batch_idx, (images, labels, _) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # K = 30 # K is number of personalized steps

                labels = labels.long()
                log_probs_p, _ = net_p(images)
                log_probs_g, _ = net_glob(images)
                # log_probs = net(images)
                loss_p, loss_g, len_loss_g, len_loss_g, ind_g = self.loss_func(log_probs_p, log_probs_g, labels, rounds, iter, args)
                for i in range(self.args.K):
                    net_p.zero_grad()
                    if i == 0:
                        loss_p.backward()
                        self.persionalized_model_bar, _ = optimizer_theta.step(list(net_glob.parameters()))
                    else:
                        log_probs_p, _ = net_p(images)
                        Beta = f_beta(rounds * args.local_ep + iter, args)
                        loss_p = self.cores_loss_fun(log_probs_p, labels, Beta)
                        loss_p = torch.sum(loss_p[ind_g]) / len(loss_p[ind_g])
                        loss_p.backward()
                        self.persionalized_model_bar, _ = optimizer_theta.step(list(net_glob.parameters()))

                # batch_loss.append(loss.item())
                # update local weight after finding aproximate theta
                for new_param, localweight in zip(self.persionalized_model_bar, net_glob.parameters()):
                    localweight.data = localweight.data - self.args.lamda * lr * (
                            localweight.data - new_param.data)

                net_glob.zero_grad()
                loss_g.backward()
                optimizer_w.step()
                batch_loss.append(loss_g.item())
                b_bar_p.append(len_loss_g)
            n_bar_k.append(sum(b_bar_p))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # print("\rRounds {:d} Client {:d} Epoch {:d}: train loss {:.4f}"
            #       .format(rounds, self.client_idx, iter, sum(epoch_loss) / len(epoch_loss)), end='\n', flush=True)
            # if any(math.isnan(loss) for loss in epoch_loss):
            #     print("debug epoch_loss")
        n_bar_k = sum(n_bar_k) / len(n_bar_k)
        return net_p, net_glob.state_dict(), sum(epoch_loss) / len(epoch_loss), n_bar_k


class RFLLocalUpdate:
    def __init__(self, args, dataset=None, user_idx=None, idxs=None):
        self.args = args
        self.dataset = dataset
        self.user_idx = user_idx
        self.idxs = idxs

        self.pseudo_labels = torch.zeros(len(self.dataset), dtype=torch.long, device=self.args.device)
        self.sim = torch.nn.CosineSimilarity(dim=1)
        self.loss_func = CrossEntropyLoss(reduction="none")
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))
        # self.ldr_train = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_train_tmp = DataLoader(DatasetSplit(dataset, idxs), batch_size=1, shuffle=True)

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def RFLloss(self, logit, labels, feature, f_k, mask, small_loss_idxs, new_labels):
        mse = torch.nn.MSELoss(reduction='none')
        ce = torch.nn.CrossEntropyLoss()
        sm = torch.nn.Softmax(dim=1)
        lsm = torch.nn.LogSoftmax(dim=1)

        L_c = ce(logit[small_loss_idxs], new_labels)
        L_cen = torch.sum(
            mask[small_loss_idxs] * torch.sum(mse(feature[small_loss_idxs], f_k[labels[small_loss_idxs]]), 1))
        L_e = -torch.mean(torch.sum(sm(logit[small_loss_idxs]) * lsm(logit[small_loss_idxs]), dim=1))

        lambda_e = self.args.lambda_e
        lambda_cen = self.args.lambda_cen
        if self.args.g_epoch < self.args.T_pl:
            lambda_cen = (self.args.lambda_cen * self.args.g_epoch) / self.args.T_pl

        return L_c + (lambda_cen * L_cen) + (lambda_e * L_e)

    def get_small_loss_samples(self, y_pred, y_true, forget_rate, args):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).to(args.device)
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        return ind_update

    def train(self, net, f_G, client_num):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        epoch_loss = []

        net.eval()
        f_k = torch.zeros(self.args.num_classes, net.fc1.in_features, device=self.args.device)
        n_labels = torch.zeros(self.args.num_classes, 1, device=self.args.device)

        # obtain global-guided pseudo labels y_hat by y_hat_k = C_G(F_G(x_k))
        # initialization of global centroids
        # obtain naive average feature
        with torch.no_grad():
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train_tmp):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                self.pseudo_labels[idxs] = torch.argmax(logit)
                if self.args.g_epoch == 0:
                    f_k[labels] += feature
                    n_labels[labels] += 1

        if self.args.g_epoch == 0:
            for i in range(len(n_labels)):
                if n_labels[i] == 0:
                    n_labels[i] = 1
            f_k = torch.div(f_k, n_labels)
        else:
            f_k = f_G

        net.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            correct_num = 0
            total = 0
            for batch_idx, batch in enumerate(self.ldr_train):
                net.zero_grad()
                images, labels, idx = batch
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                feature = feature.detach()
                f_k = f_k.to(self.args.device)

                small_loss_idxs = self.get_small_loss_samples(logit, labels, self.args.forget_rate, self.args)

                y_k_tilde = torch.zeros(self.args.local_bs, device=self.args.device)
                mask = torch.zeros(self.args.local_bs, device=self.args.device)
                for i in small_loss_idxs:
                    y_k_tilde[i] = torch.argmax(self.sim(f_k, torch.reshape(feature[i], (1, net.fc1.in_features))))
                    if y_k_tilde[i] == labels[i]:
                        mask[i] = 1

                # When to use pseudo-labels
                if self.args.g_epoch < self.args.T_pl:
                    for i in small_loss_idxs:
                        self.pseudo_labels[idx[i]] = labels[i]

                # For loss calculating
                new_labels = mask[small_loss_idxs] * labels[small_loss_idxs] + (1 - mask[small_loss_idxs]) * \
                             self.pseudo_labels[idx.to(self.args.device)[small_loss_idxs.to(self.args.device)]]
                new_labels = new_labels.type(torch.LongTensor).to(self.args.device)

                loss = self.RFLloss(logit, labels, feature, f_k, mask, small_loss_idxs, new_labels)

                # weight update by minimizing loss: L_total = L_c + lambda_cen * L_cen + lambda_e * L_e
                loss.backward()
                optimizer.step()

                # obtain loss based average features f_k,j_hat from small loss dataset
                f_kj_hat = torch.zeros(self.args.num_classes, net.fc1.in_features, device=self.args.device)
                n = torch.zeros(self.args.num_classes, 1, device=self.args.device)
                for i in small_loss_idxs:
                    f_kj_hat[labels[i]] += feature[i]
                    n[labels[i]] += 1
                for i in range(len(n)):
                    if n[i] == 0:
                        n[i] = 1
                f_kj_hat = torch.div(f_kj_hat, n)

                # update local centroid f_k
                one = torch.ones(self.args.num_classes, 1, device=self.args.device)
                f_k = (one - self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_k + (
                        self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_kj_hat

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), f_k


class FedAVGLocalUpdate:
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = CrossEntropyLoss()  # loss function -- cross entropy
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for images, labels, _ in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                outputs, _ = net(images)
                loss = self.loss_func(outputs, labels)
                # print("outputs={}, labels={}".format(outputs, labels))
                # print("loss={}".format(loss))
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                # print("batch_loss={}".format(batch_loss))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # print("epoch_loss={}".format(epoch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class FedProxLocalUpdate:
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = CrossEntropyLoss()  # loss function -- cross entropy
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net):
        old_net = copy.deepcopy(net)
        net.train()
        optimizer = FedProxOptimizer(net.parameters(), lr=self.args.lr, lamda=self.args.mu)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for images, labels, _ in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                outputs, _ = net(images)
                loss = self.loss_func(outputs, labels)
                # print("outputs={}, labels={}".format(outputs, labels))
                # print("loss={}".format(loss))
                loss.backward()
                optimizer.step(list(old_net.parameters()))
                batch_loss.append(loss.item())
                # print("batch_loss={}".format(batch_loss))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # print("epoch_loss={}".format(epoch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def get_local_update_objects(args, dataset_train, dict_users=None, net_glob=None):
    local_update_objects = []
    for idx in range(args.num_users):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
        )
        local_update_objects.append(RFLLocalUpdate(**local_update_args))

    return local_update_objects


def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs, _ = net(images)
            # outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = (correct / total) * 100
    return acc


def personalizedtest(args, p_models, dataset_test):
    pass
