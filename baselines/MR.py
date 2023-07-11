
from util.dataMR import split_data, my_split, get_n_sample_to_keep, MyDataset, wash_data
from model.build_model import build_model
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

def MR(args):
# ======================================================数据划分，加噪===============================================================
    # 首先从原始数据集中拆分出benchmark dataset 剩余其他数据
    benchmark_dataset, fliter_dataset_train, fliter_dataset_test = split_data(args);
    print("benchmark数据集拆分出来了")
    print("fliter_dataset_train的形状是：{}".format(len(fliter_dataset_train)))
    print(len(fliter_dataset_train[0]))
    # 按照客户端划分
    print("正在按照客户端划分fliter_dataset_train")
    n_train = len(fliter_dataset_train)
    # y_train = np.array(fliter_dataset_train.targets)
    print(type(fliter_dataset_train[0][1]))
    y_train = np.array([label for _, label in fliter_dataset_train])
    if args.iid:
        dict_users = iid_sampling(len(n_train), args.num_users, args.seed)
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)
    print("正在加噪声")
    # fliter_dataset_train数据进行加噪得到noise dataset
    y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users)
    x_train = [img for img, _ in fliter_dataset_train]
    fliter_dataset_train = MyDataset(x_train, y_train_noisy)
    for i in range(100):
        data = fliter_dataset_train.__getitem__(i)
        img, label = data
        if label != y_train_noisy[i]:
            print("true")
    print("加噪声完毕")

    # 将benchmark_dataset划分成train和test
    train_ratio = 1 / 1.3  # benchmark dataset比例
    test_ratio = 1 - 1 / 1.3    # 待过滤 dataset比例
    train_size = int(train_ratio * len(benchmark_dataset))
    test_size = int(len(benchmark_dataset) - train_size)
    benchmark_dataset_train, benchmark_dataset_test = my_split(args, train_ratio, test_ratio, benchmark_dataset)
    print("将benchmark_dataset划分成train和test了")

# ======================================================benchmark model训练===============================================================

    print("即将使用benchmark_dataset_train开始训练benchmark model")
    # 然后benchmark dataset train用于训练benchmark model
    # 获取模型
    print("使用设备为")
    print(args.device)
    benchmark_model = build_model(args)

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
        print("----------benchmark模型第{}轮训练开始----------".format(i+1))
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
                    img = torch.from_numpy(np.array([img])).to(args.device)
                    target = torch.from_numpy(np.array([target])).to(args.device)
                    output, _ = benchmark_model(img)
                    loss = loss_fn(output, target).cpu().numpy()
                    list_loss_benchmark.append(loss)

            count1 = 0 

            with torch.no_grad():
                for data in fliter_dataset_train:
                    img, target = data
                    img = torch.from_numpy(np.array([img])).to(args.device)
                    target = torch.from_numpy(np.array([target])).to(args.device)
                    output, _ = benchmark_model(img)
                    loss = loss_fn(output, target).cpu().numpy()
                    list_loss_fliter[count1] = loss
                    count1 += 1
            break

# ======================================================噪声过滤===============================================================
    # 将fliter训练集集loss排序并保存
    sorted_d = sorted(list_loss_fliter.items(), key = lambda kv:(kv[1], kv[0]))
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
    prob = [1/args.num_users for i in range(args.num_users)]
    for rnd in range(args.rounds2):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:  # training over the subset
            local = LocalUpdate(args=args, dataset=fliter_dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(model).to(args.device), seed=args.seed,
                                                        w_g=model.to(args.device), epoch=args.local_ep, mu=0)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))
        loss_round = sum(loss_locals)/len(loss_locals)
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = FedAvg(w_locals, dict_len)
        model.load_state_dict(copy.deepcopy(w_glob_fl))
        acc_s2 = globaltest(copy.deepcopy(model).to(args.device), fliter_dataset_test, args)
        show_info_loss = "Round %d train loss  %.4f\n" % (rnd, loss_round)
        show_info_test_acc = "global test acc  %.4f \n\n" % (acc_s2)
        print(show_info_loss)
        print(show_info_test_acc)
    show_time_info = f"time : {time.time() - start}"
    print(show_time_info)


