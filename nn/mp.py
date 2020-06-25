import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def shifted_softplus(x):
    x = F.softplus(x) - np.log(2)
    return x


class Message(nn.Module):
    def __init__(self, C=64, T=4):
        super(Message, self).__init__()
        self.W1s = [nn.Linear(C, C) for _ in range(T)]
        self.W2s = [nn.Linear(C, C) for _ in range(T)]
        self.W3s = [nn.Linear(C, C) for _ in range(T)]

    def get_filter(self, e_vw, t):
        flter = self.W2s[t](e_vw)
        flter = shifted_softplus(flter)
        flter = self.W3s[t](flter)
        flter  = shifted_softplus(flter)
        return flter

    def forward(self, h_w, e_vw, t):
        nbr = self.W1s[t](h_w)
        flter = self.get_filter(e_vw, t)
        m_t = nbr * flter
        return m_t


class StateTransition(nn.Module):
    def __init__(self, C=64, T=4):
        super(StateTransition, self).__init__()
        self.W4s = [nn.Linear(C, C) for _ in range(T)]
        self.W5s = [nn.Linear(C, C) for _ in range(T)]

    def forward(self, h_v, m_v, t):  # h_v and m_v^{t+1} as inputs
        x = self.W4s[t](m_v)
        x = shifted_softplus(x)
        x = self.W5s[t](x)
        x = h_v + x
        return x


class Readout(nn.Module):
    def __init__(self, C=64):
        super(Readout, self).__init__()
        self.W6 = nn.Linear(C, C // 2)
        self.W7 = nn.Linear(C // 2, 1)

    def forward(self, h):
        x = self.W6(h.T)
        x = shifted_softplus(x)
        x = self.W7(x)
        return x  # add torch.mean(x).item()


class Edge(nn.Module):
    def __init__(self, C):
        super(Readout, self).__init__()
        self.W6 = nn.Linear(3 * C, 2 * C)
        self.W7 = nn.Linear(2 * C, C)

    def forward(self, h):
        x = self.W6(h.T)
        x = shifted_softplus(x)
        x = self.W7(x)
        return x  # add torch.mean(x).item()
