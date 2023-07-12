from util.dataMR import split_data, my_split, get_n_sample_to_keep, MyDataset, wash_data
from model.build_model import build_model, eminist_build_model
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch import nn
from util.sampling import iid_sampling, non_iid_dirichlet_sampling
import torch
import numpy as np
from util.util import add_noise
from util.local_training import LocalUpdate, globaltest
import copy
from util.aggregation import FedAvg
import time
import math

from torchvision import transforms
from torchvision.datasets import EMNIST
from util.load_data import load_data_with_noisy_label


def MR2(args):

    # =======================================================获取Fedminist数据集并拆分出benchmark_dataset_train/test=======================================================
    data_path = '../data/mnist'

    trans_mnist_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trans_mnist_val = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    EMNIST_dataset_train = EMNIST(
        root=data_path,
        download=True,
        train=True,
        transform=trans_mnist_train,
        split="byclass"
    )
    EMNIST_dataset_test = EMNIST(
        root=data_path,
        download=True,
        train=False,
        transform=trans_mnist_train,
        split="byclass"
    )
    benchmark_ratio = 0.03  # benchmark dataset比例
    fliter_ratio = 0.97  # 待过滤 dataset比例
    benchmark_dataset_train, _ = my_split(args=args, ratio1=benchmark_ratio, ratio2=fliter_ratio,
                                          dataset=EMNIST_dataset_train);
    benchmark_dataset_test, _ = my_split(args=args, ratio1=benchmark_ratio, ratio2=fliter_ratio,
                                         dataset=EMNIST_dataset_test);
    benchmark_dataset = torch.utils.data.ConcatDataset([benchmark_dataset_train, benchmark_dataset_test])

    # 将benchmark_dataset划分成train和test
    train_ratio = 1 / 1.3  # benchmark dataset比例
    test_ratio = 1 - 1 / 1.3  # 待过滤 dataset比例
    train_size = int(train_ratio * len(benchmark_dataset))
    test_size = int(len(benchmark_dataset) - train_size)
    benchmark_dataset_train, benchmark_dataset_test = my_split(args, train_ratio, test_ratio, benchmark_dataset)
    print("将benchmark_dataset划分成train和test了")

    # =======================================================获取待训练数据集并加噪=======================================================
    dataset_train, dataset_test, dict_users, y_train, gamma_s = load_data_with_noisy_label(args)

    # ======================================================benchmark model训练===============================================================

    print("即将使用benchmark_dataset_train开始训练benchmark model")
    # 然后benchmark dataset train用于训练benchmark model
    # 获取模型
    print("使用设备为")
    print(args.device)
    benchmark_model = eminist_build_model(args)

    # dataloader
    benchmark_train_dataloader = DataLoader(benchmark_dataset_train, batch_size=16)
    benchmark_test_dataloader = DataLoader(benchmark_dataset_test, batch_size=16)
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(args.device)
    # 学习率、优化器
    learning_rate = args.lr
    optimizer = torch.optim.SGD(benchmark_model.parameters(), lr=learning_rate)
    # 训练参数
    total_train_step = 0
    total_test_step = 0
    total_accuracy = 0
    epoch = 500
    # 定义一个列表存储benchmark_test_dataset的loss
    list_loss_benchmark = []
    list_loss_fliter = {}
    # 开始训练
    max_val = math.inf
    counter_convergence_reached = 0
    i = 0
    while counter_convergence_reached < 200:
        print("----------benchmark模型第{}轮训练开始----------".format(i + 1))
        for data in benchmark_train_dataloader:
            imgs, targets = data
            imgs = imgs.to(args.device)
            targets = targets.to(args.device)
            outputs, _ = benchmark_model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print("benchmark模型训练次数：{}，loss：{}".format(total_train_step, loss.item()))

        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in benchmark_test_dataloader:
                imgs, targets = data
                imgs = imgs.to(args.device)
                targets = targets.to(args.device)
                outputs, _ = benchmark_model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy += accuracy
        print("benchmark模型整体测试集上的loss：{}".format(total_test_loss))
        print("benchmark模型整体测试集上的正确率：{}".format(total_accuracy / test_size))
        total_test_step += 1

        if total_test_loss < max_val:
            max_val = total_test_loss
            counter_convergence_reached = 0
        else:
            if i > 10:
                counter_convergence_reached += 1

        i = i + 1
        if i > 10000 or counter_convergence_reached == 200:
            with torch.no_grad():
                for data in benchmark_dataset_test:
                    img, target = data
                    img = img.expand(1, -1, -1, -1).to(args.device)
                    target = torch.from_numpy(np.array([target])).to(args.device)
                    output, _ = benchmark_model(img)
                    loss = loss_fn(output, target).cpu().numpy()
                    list_loss_benchmark.append(loss)

            count1 = 0

            with torch.no_grad():
                for data in dataset_train:
                    img, target = data
                    img = img.expand(1, -1, -1, -1).to(args.device)
                    target = torch.from_numpy(np.array([target])).to(args.device)
                    output, _ = benchmark_model(img)
                    loss = loss_fn(output, target).cpu().numpy()
                    list_loss_fliter[count1] = loss
                    count1 += 1
            break

    # ======================================================噪声过滤===============================================================
    # 将fliter训练集集loss排序并保存
    sorted_d = sorted(list_loss_fliter.items(), key=lambda kv: (kv[1], kv[0]))
    # 使用benchmark dataset test和其他所有数据用于计算阈值 用于筛选noise dataset
    n_sample_keep, indices_to_keep = get_n_sample_to_keep(
        sorted_d, list_loss_benchmark)
    # 更新dict_users（indices_to_keep）可以指定了可以用的索引
    dict_users = wash_data(dict_users, indices_to_keep)

    # ======================================================联邦训练===============================================================

    # 开始联邦学习阶段
    print("下面开始联邦训练阶段：")
    start = time.time()
    # 获取模型
    model = build_model(args)
    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1 / args.num_users for i in range(args.num_users)]
    for rnd in range(args.rounds2):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:  # training over the subset
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(model).to(args.device),
                                                       w_g=model.to(args.device), epoch=args.local_ep, mu=0)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))
        loss_round = sum(loss_locals) / len(loss_locals)
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
