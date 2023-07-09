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
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
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
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                if self.args.mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(images, labels, self.args.alpha)
                    net.zero_grad()
                    log_probs = net(inputs)
                    loss = mixup_criterion(self.loss_func, log_probs, targets_a, targets_b, lam)
                else:
                    labels = labels.long()
                    net.zero_grad()
                    log_probs = net(images)
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

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class FedTwinLocalUpdate:
    def __init__(self, args, dataset, idxs, client_idx):
        self.args = args
        self.loss_func = FedTwinCRLoss(reduction='none')  # loss function -- cross entropy
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
        optimizer_w = torch.optim.SGD(net_glob.parameters(), lr=self.args.plr)
        adjust_learning_rate(rounds, args, optimizer_theta)
        adjust_learning_rate(rounds, args, optimizer_w)
        # lr=args.lr
        lr = adjust_learning_rate(rounds, args)

        epoch_loss = []
        n_bar_k = []
        for iter in range(args.local_ep):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            b_bar_p = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # K = 30 # K is number of personalized steps

                labels = labels.long()
                log_probs_p = net_p(images)
                log_probs_g = net_glob(images)
                # log_probs = net(images)
                loss_p, loss_g, len_loss_p, len_loss_g = self.loss_func(log_probs_p, log_probs_g, labels, rounds, args)
                for i in range(self.args.K):
                    net_p.zero_grad()
                    if i == (self.args.K - 1):
                        loss_p.backward()
                    else:
                        loss_p.backward(retain_graph=True)
                    self.persionalized_model_bar, _ = optimizer_theta.step(list(net_glob.parameters()))

                # batch_loss.append(loss.item())
                # update local weight after finding aproximate theta
                for new_param, localweight in zip(self.persionalized_model_bar, net_glob.parameters()):
                    localweight.data = localweight.data - self.args.lamda * lr * (
                                localweight.data - new_param.data)

                for param, new_param in zip(net_glob.parameters(), net_glob.parameters()):
                    param.data = new_param.data.clone()

                net_glob.zero_grad()
                loss_g.backward()
                optimizer_w.step()
                batch_loss.append(loss_g.item())
                b_bar_p.append(len_loss_g)
            n_bar_k.append(sum(b_bar_p))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            print("\rRounds {:d} Client {:d} Epoch {:d}: train loss {:.4f}"
                  .format(rounds, self.client_idx, iter, sum(epoch_loss) / len(epoch_loss)), end='\n', flush=True)
            # if any(math.isnan(loss) for loss in epoch_loss):
            #     print("debug epoch_loss")
        n_bar_k = sum(n_bar_k)/len(n_bar_k)
        return net_p, net_glob.state_dict(), sum(epoch_loss) / len(epoch_loss), n_bar_k


class LocalUpdateRFL:
    def __init__(self, args, dataset=None, user_idx=None, idxs=None):
        self.args = args
        self.dataset = dataset
        self.user_idx = user_idx
        self.idxs = idxs

        self.pseudo_labels = torch.zeros(len(self.dataset), dtype=torch.long, device=self.args.device)
        self.sim = torch.nn.CosineSimilarity(dim=1)
        self.loss_func = CrossEntropyLoss()
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))
        # self.ldr_train = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        # self.ldr_train_tmp = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=1, shuffle=True)
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

    def get_small_loss_samples(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        return ind_update

    def train(self, net, f_G, client_num):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        epoch_loss = []

        net.eval()
        f_k = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
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

                small_loss_idxs = self.get_small_loss_samples(logit, labels, self.args.forget_rate)

                y_k_tilde = torch.zeros(self.args.local_bs, device=self.args.device)
                mask = torch.zeros(self.args.local_bs, device=self.args.device)
                for i in small_loss_idxs:
                    y_k_tilde[i] = torch.argmax(self.sim(f_k, torch.reshape(feature[i], (1, self.args.feature_dim))))
                    if y_k_tilde[i] == labels[i]:
                        mask[i] = 1

                # When to use pseudo-labels
                if self.args.g_epoch < self.args.T_pl:
                    for i in small_loss_idxs:
                        self.pseudo_labels[idx[i]] = labels[i]

                # For loss calculating
                new_labels = mask[small_loss_idxs] * labels[small_loss_idxs] + (1 - mask[small_loss_idxs]) * \
                             self.pseudo_labels[idx[small_loss_idxs]]
                new_labels = new_labels.type(torch.LongTensor).to(self.args.device)

                loss = self.RFLloss(logit, labels, feature, f_k, mask, small_loss_idxs, new_labels)

                # weight update by minimizing loss: L_total = L_c + lambda_cen * L_cen + lambda_e * L_e
                loss.backward()
                optimizer.step()

                # obtain loss based average features f_k,j_hat from small loss dataset
                f_kj_hat = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
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

def get_local_update_objects(args, dataset_train, dict_users=None, net_glob=None):
    local_update_objects = []
    for idx in range(args.num_users):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
        )
        local_update_objects.append(LocalUpdateRFL(**local_update_args))

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
            outputs = net(images)
            # outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = (correct / total)*100
    return acc


def personalizedtest(args, p_models, dataset_test):
    pass
#     num_samples = []
#     tot_correct = []
#     for p_model in p_models:
#         self.model.eval()
#         test_acc = 0
#         self.update_parameters(self.persionalized_model_bar)
#         for x, y in self.testloaderfull:
#             x, y = x.to(self.device), y.to(self.device)
#             output = self.model(x)
#             test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
#             # @loss += self.loss(output, y)
#             # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
#             # print(self.id + ", Test Loss:", loss)
#         self.update_parameters(self.local_model)
#         test_acc, y.shape[0]
#
#
#
#
#
#
#         ct, ns = c.test_persionalized_model()
#         tot_correct.append(ct * 1.0)
#         num_samples.append(ns)
#     ids = [c.id for c in self.users]
#
#     return ids, num_samples, tot_correct
#
#
#     stats = test_persionalized_model()
#         stats_train = train_error_and_loss_persionalized_model()
#         glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
#         train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
#         # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
#         train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
#         rs_glob_acc_per.append(glob_acc)
#         rs_train_acc_per.append(train_acc)
#         rs_train_loss_per.append(train_loss)
#         #print("stats_train[1]",stats_train[3][0])
#         print("Average Personal Accurancy: ", glob_acc)
#         print("Average Personal Trainning Accurancy: ", train_acc)
#         print("Average Personal Trainning Loss: ",train_loss)


# class CoteachingCRLoss(CrossEntropyLoss):
#     __constants__ = ['weight', 'ignore_index', 'reduction']
#
#     def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
#         super(CoteachingCRLoss, self).__init__(weight, reduction)
#         self.ignore_index = ignore_index
#
#     @staticmethod
#     def is_clean_judge(input1, input2, target):
#         loss1 = F.cross_entropy(input1, target, reduction='none')
#         loss2 = F.cross_entropy(input2, target, reduction='none')
#         loss1_ = -torch.log(F.softmax(input1, dim=1) + 1e-8)
#         loss2_ = -torch.log(F.softmax(input2, dim=1) + 1e-8)
#
#         loss1 = loss1 - torch.mean(loss1_, 1)
#         loss2 = loss2 - torch.mean(loss2_, 1)
#
#         thre = -0.0
#
#         ind_1 = (loss1.data.cpu().numpy() <= thre)  # global model Sieve
#         ind_2 = (loss2.data.cpu().numpy() <= thre)  # private model Sieve
#         return [ind_1, ind_2]
#
#     @staticmethod
#     def cal_prf(is_clean_real, is_clean_judge):
#         is_clean_judge_1 = is_clean_judge[0]
#         is_clean_judge_2 = is_clean_judge[1]
#         pure_ratio_1 = 100 * np.sum(is_clean_real * is_clean_judge_1) / (np.sum(is_clean_judge_1) + 1e-8)
#         pure_ratio_2 = 100 * np.sum(is_clean_real * is_clean_judge_2) / (np.sum(is_clean_judge_2) + 1e-8)
#
#         Recall_1 = 100 * np.sum(is_clean_real * is_clean_judge_1) / (np.sum(is_clean_real) + 1e-8)
#         Recall_2 = 100 * np.sum(is_clean_real * is_clean_judge_2) / (np.sum(is_clean_real) + 1e-8)
#
#         F_score1 = 2 * pure_ratio_1 * Recall_1 / (pure_ratio_1 + Recall_1 + 1e-8)
#         F_score2 = 2 * pure_ratio_2 * Recall_2 / (pure_ratio_2 + Recall_2 + 1e-8)
#         # clean_num = np.sum(ind_2)
#         return {'pure_ratio': [pure_ratio_1, pure_ratio_2],
#                 'recall': [Recall_1, Recall_2],
#                 'F_score': [F_score1, F_score2]}
#
#     def forward(self, input1, input2, target, round, noise_prior, epoch, begin_sel):
#         # input1: global_model output
#         # input2: private_model output
#         beta = self.f_beta((round - 1) * 5 + epoch)
#         [is_clean_judge_1, is_clean_judge_2] = FedTwinCRLoss.is_clean_judge(input1, input2, target)
#
#         if begin_sel:
#             if np.sum(is_clean_judge_2) == 0:
#                 loss_1_update = F.cross_entropy(input1, target, reduction='none')
#                 loss1 = torch.sum(loss_1_update) / len(is_clean_judge_2) * 1e-8
#             else:
#                 is_clean_judge_2 = torch.tensor(is_clean_judge_2)
#                 loss_1_update = F.cross_entropy(input1[is_clean_judge_2], target[is_clean_judge_2], reduction='none') - beta * torch.sum(
#                     torch.mul(noise_prior, -torch.log(F.softmax(input1[is_clean_judge_2], dim=1) + 1e-8)), 1)
#                 loss1 = torch.sum(loss_1_update) / np.sum(is_clean_judge_2.data.cpu().numpy())
#             if np.sum(is_clean_judge_1) == 0:
#                 loss_2_update = F.cross_entropy(input2, target, reduction='none')
#                 loss2 = torch.sum(loss_2_update) / len(is_clean_judge_1) * 1e-8
#                 # loss2 = None
#             else:
#                 ind_1 = torch.tensor(is_clean_judge_1)
#                 loss_2_update = F.cross_entropy(input2[ind_1], target[ind_1], reduction='none') - beta * torch.sum(
#                     torch.mul(noise_prior, -torch.log(F.softmax(input2[ind_1], dim=1) + 1e-8)), 1)
#                 loss2 = torch.sum(loss_2_update) / np.sum(ind_1.data.cpu().numpy())
#         else:
#             loss_1_update = F.cross_entropy(input1, target, reduction='none') - beta * torch.sum(
#                 torch.mul(noise_prior, -torch.log(F.softmax(input1, dim=1) + 1e-8)), 1)
#             loss_2_update = F.cross_entropy(input2, target, reduction='none') - beta * torch.sum(
#                 torch.mul(noise_prior, -torch.log(F.softmax(input2, dim=1) + 1e-8)), 1)
#             loss1 = torch.sum(loss_1_update) / len(is_clean_judge_1)
#             loss2 = torch.sum(loss_2_update) / len(is_clean_judge_2)
#
#         return [loss1, loss2]
#
#     def f_beta(self, batch):
#         max_beta = 0.1
#         beta1 = np.linspace(0.0, 0.0, num=1)
#         beta2 = np.linspace(0.0, max_beta, num=50)
#         beta3 = np.linspace(max_beta, max_beta, num=5000)
#
#         beta = np.concatenate((beta1, beta2, beta3), axis=0)
#         return beta[batch]

