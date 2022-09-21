import torch
from load_data import read_data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import faiss

def cal_loss(x, x_pos, x_neg):
    x, x_pos, x_neg = F.normalize(x, dim=1), F.normalize(x_pos, dim=1), F.normalize(x_neg, dim=1)
    tau = 0.1
    sim_pos = torch.exp((x*x_pos).sum(1)/tau)
    sim_neg = torch.exp((x*x_neg).sum(1)/tau)
    sim_all = torch.exp(torch.matmul(x, x.T)/tau).sum(1)
    loss = ((-sim_pos+sim_neg)/sim_all).mean()
    return loss
def cluser(x, k):
    x = x.detach().cpu().numpy()
    kmeans = faiss.Kmeans(d=x.shape[1], k=k, gpu=False)
    kmeans.train(x)
    cluster_cents = kmeans.centroids

    _, I = kmeans.index.search(x, 1)

    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(cluster_cents)
    centroids = F.normalize(centroids, p=2, dim=1)
    node2cluster = torch.LongTensor(I).squeeze()
    return node2cluster
def cal_label(cor_list, all_list):
    num = len(all_list)
    labels = torch.zeros(num, num)
    pos = 0
    for item in cor_list:
        for i in range(len(item)):
            for j in range(i, len(item)):
                labels[pos+i][pos+j] = 1
                labels[pos + j][pos + i] = 1
        pos += len(item)
    return labels