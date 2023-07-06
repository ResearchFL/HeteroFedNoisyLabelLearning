# python version 3.7.1
# -*- coding: utf-8 -*-

import copy
import torch


def FedAvg(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():        
        w_avg[k] = w_avg[k] * dict_len[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg

def personalized_aggregation(w, netglob, dict_len, beta):
    w_agg = copy.deepcopy(w[0])
    for k in w_agg.keys():
        w_agg[k] = w_agg[k] * dict_len[0]
        for i in range(1, len(w)):
            w_agg[k] += w[i][k] * dict_len[i]
        w_agg[k] = beta * torch.div(w_agg[k], sum(dict_len)) + (1 - beta) * netglob[k]
    return w_agg
