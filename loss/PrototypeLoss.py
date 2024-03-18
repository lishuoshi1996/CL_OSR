import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Dist import Dist
import numpy as np
from torch.autograd import Variable

def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H

def dot_dis(x,y):
    dis = x.mm(y.t())
    return dis


class PrototypeLoss(nn.Module):
    def __init__(self, **options):
        super(PrototypeLoss, self).__init__()
        self.temp = options['temp']
        self.W = nn.Parameter(torch.Tensor(rvs(options['feat_dim'])[:options['num_classes']]), requires_grad=True)

    def forward(self, x,  y, labels=None , train_model = True):

        prototype = self.W

        dist_dot = dot_dis(x, prototype)

        logits = dist_dot


        if train_model:
            if labels is None: return logits, 0
            loss = F.cross_entropy(logits / self.temp, labels)

            return logits, loss
        else:
            logits = F.softmax(logits, dim=1)
            loss = (logits * torch.log(logits)).sum(1).mean().exp()
            return logits, loss
