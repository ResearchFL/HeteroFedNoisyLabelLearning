import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist


def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    # 只是缩放到指定区间而已
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        # y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, args.num_classes - 1, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    # By comparing the labels of the dataset before and after adding noise
    # we can determine which samples have been affected by the noise
    noisy_samples_idx = np.where(y_train != y_train_noisy)[0]
    return y_train_noisy, gamma_s, real_noise_level, noisy_samples_idx



def add_non_iid_noise(args, y_train, dict_users, global_noise_ratio, alpha):
    np.random.seed(args.seed)
    direchlet1 = np.random.dirichlet([alpha] * args.num_users) * global_noise_ratio
    direchlet2 = np.random.dirichlet([alpha] * args.num_users) * (1 - global_noise_ratio)
    gamma_c = direchlet1 / (direchlet1 + direchlet2)
    gamma_s = np.ones(args.num_users) # 全是噪声client

    client_bias_list = np.random.randint(0, args.num_classes - 1, args.num_users)# 每个客户端固定偏移向某种类
    global_bias = np.random.randint(0, args.num_classes - 1)

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        # 加噪方法1：sample level
        if args.add_noise_method == 'sample_level':
            y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, args.num_classes - 1, len(noisy_idx))
        # 加噪方法2：client level
        if args.add_noise_method == 'client_level':
            y_train_noisy[sample_idx[noisy_idx]] = client_bias_list[i]
        # 加噪方法3：global level
        if args.add_noise_method == 'global_level':
            y_train_noisy[sample_idx[noisy_idx]] = global_bias
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    print("noise level varience: ", np.var(real_noise_level))
    # By comparing the labels of the dataset before and after adding noise
    # we can determine which samples have been affected by the noise
    noisy_samples_idx = np.where(y_train != y_train_noisy)[0]
    return y_train_noisy, gamma_s, real_noise_level, noisy_samples_idx


def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if latent == False:
                outputs, _ = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs, _ = net(images, True)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids

#
# if rnd == 1 or rnd == 20 or rnd == 50 or rnd == (args.rounds2 - 1):
#     # Record the loss for clean and noisy samples separately
#     clean_loss_s, noisy_loss_s = get_clean_noisy_sample_loss(
#         model=netglob.to(args.device),
#         dataset=dataset_train,
#         noisy_sample_idx=noisy_sample_idx,
#         round=rnd,
#         device=args.device,
#         beta=Beta
#     )
#     print(f"{loss_fn.__class__.__name__}:")
#     print("clean_loss:")
#     print(clean_loss_s)
#     print("noisy_loss:")
#     print(noisy_loss_s)