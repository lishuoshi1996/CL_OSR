import torch
import torch.nn as nn


def cos_dis(x, y):
    loss = 1 - torch.nn.CosineSimilarity()(x.view(x.shape[0], -1), y.view(y.shape[0], -1))
    return loss


