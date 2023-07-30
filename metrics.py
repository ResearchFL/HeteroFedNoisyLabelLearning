import torch
from util.optimizer import filter_noisy_data
from torch.utils.data import DataLoader, TensorDataset, Subset
from util.local_training import DatasetSplit


def cal_fscore(args, loss_fn, beta, net, dataset_train, y_train, idxs):
    net.eval()
    # 判断dataset_train.target与y_train对应下标是否相同
    client_y_train = y_train[list(idxs)]
    subset = DatasetSplit(dataset_train, idxs)
    dataloader = DataLoader(subset, batch_size=128, shuffle=False)
    filter_list = []
    target_list = []
    with torch.no_grad():
        for idx, (images, labels, _) in enumerate(dataloader):
            images, labels = images.to(args.device), labels.to(args.device)

            labels = labels.long()
            log_probs_g, _ = net(images)

            ind_g_update = filter_noisy_data(log_probs_g, labels, loss_fn, beta)
            filter_list.extend(ind_g_update)
            target_list.extend(labels)
    match_list = [1 if target == client_y_train[idx] else 0 for idx, target in enumerate(target_list)]
    temp_list = [1 if filter_list[i] and match_list[i] else 0 for i in range(len(match_list))]
    # 计算precision
    filter_num = sum(filter_list)
    pre = sum(temp_list) / filter_num
    # 计算recall
    match_num = sum(match_list)
    rec = sum(temp_list) / match_num
    # 计算f1-score
    f1_score = 2 * pre * rec / (pre + rec)
    return f1_score.numpy().item()

