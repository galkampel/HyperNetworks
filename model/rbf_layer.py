from modulefinder import Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def RBFLayer(nn.Module):
    """
    Radial basis functions Layer.
    e(d) = exp(- gamma * ||d - mu_k||^2)
    default settings:
    gamma = 10
    0 <= mu_k <= 30 for k=1~300
    """
    def __init__(self, low=0, high=30, gap=0.1, coef=0.1, dim=1):
        super(RBFLayer, self).__init__()
        self.low = low
        self.high = high
        self.gap = gap
        self.dim = dim
        self.coef = coef
        self.n_centers = int(np.ceil((high - low) / gap))
        centers = np.linspace(low, high, self._n_centers)
        self.centers = th.tensor(centers, dtype=th.float, requires_grad=False)
        self.centers = nn.Parameter(self.centers, requires_grad=False)
        self._fan_out = self._dim * self._n_centers
        self._gap = centers[1] - centers[0]

