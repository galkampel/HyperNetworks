# import os
# import os.path as osp
# from math import pi as PI
# import warnings
# import itertools
# from pstats import Stats
# from sre_parse import State
# from sys import _hash_info
from typing import re

import ase
import torch
import torch.nn.functional as F
import torch.nn as nn
# import numpy as np

from torch_scatter import scatter
from torch_geometric.nn import radius_graph, MessagePassing
# from torch_geometric.data.makedirs import makedirs
# from torch_geometric.data import download_url, extract_zip


try:
    import schnetpack as spk
except ImportError:
    spk = None

qm9_target_dict = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


class NMPEdge(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        state_hyper (bool, optionl): If true use hyperNetwork for the
            state-transition phase (https://arxiv.org/pdf/2002.00240.pdf)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(self, embed_size=256, hidden_channels=128, num_filters=128, num_interactions=4,
                 num_gaussians=150, cutoff=15.0, readout='add', state_hyper=False,
                 dipole=False, mean=None, std=None, atomref=None):
        super(NMPEdge, self).__init__()

        assert readout in ['add', 'sum', 'mean']

        self.hidden_channels = hidden_channels
        self.embed_size = embed_size
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None
        self.state_hyper = state_hyper

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = nn.Embedding(embed_size, hidden_channels) # 256 instead of 100?
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.edge_updates = nn.ModuleList()
        edge_net0 = EdgeUpdate(num_gaussians + 2 * hidden_channels, hidden_channels)
        self.edge_updates.append(edge_net0)
        for _ in range(num_interactions - 1):
            edge_net = EdgeUpdate(3 * hidden_channels, hidden_channels)
            self.edge_updates.append(edge_net)
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, hidden_channels, self.state_hyper)
            self.interactions.append(block)

        ### readout function ###
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.fc2 = nn.Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = nn.Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for i, interaction in enumerate(self.interactions):
            interaction.reset_parameters()
            self.edge_updates[i].reset_parameters()
        # torch.nn.init.xavier_uniform_(self.lin1.weight)
        # self.state_transition.reset_parameters()
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        # torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        e = edge_attr
        # h0 = h.clone()
        for i, interaction in enumerate(self.interactions):

            e = self.edge_updates[i](h, edge_index, e)
            h = h + interaction(h, edge_index, e)

        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c[batch])

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, state_hyper=False):
        super(InteractionBlock, self).__init__()
        self.filter_mlp = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
            ShiftedSoftplus()
        )
        self.cf_conv = CFConv(hidden_channels, num_filters, self.filter_mlp)
        if state_hyper:
            self.state_transition = ... # use HyperNetwork
        else:
            self.state_transition = StateMLP(num_filters)
        self.reset_parameters()

    def reset_parameters(self):
        self.cf_conv.reset_parameters()
        self.state_transition.reset_parameters()
        torch.nn.init.kaiming_uniform_(self.filter_mlp[0].weight)
        self.filter_mlp[0].bias.data.fill_(0)
        torch.nn.init.kaiming_uniform_(self.filter_mlp[2].weight)
        self.filter_mlp[2].bias.data.fill_(0)
        # torch.nn.init.kaiming_uniform_(self.lin.weight)
        # self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        x = self.cf_conv(x, edge_index, edge_attr)
        x = self.state_transition(x)
        return x


class EdgeUpdate(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(EdgeUpdate, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_filters * 2)
        self.fc2 = nn.Linear(num_filters * 2, num_filters)
        self.act = ShiftedSoftplus()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        x_j = x[edge_index[0, :], :]
        x_i = x[edge_index[1, :], :]
        edge_attr = torch.cat([x_i, x_j, edge_attr], dim=1)
        edge_attr = self.fc1(edge_attr)
        edge_attr = self.act(edge_attr)
        edge_attr = self.fc2(edge_attr)
        return edge_attr


class StateHyper(nn.Module):
    def __init__(self, num_filters):
        super(StateHyper, self).__init__()
        self.hyperparams_dim = num_filters // 2
        self.num_filters = num_filters
        self.c = nn.Parameter(torch.rand(1), requires_grad=True)
        self.fc1 = nn.Linear(self.num_filters, self.hyperparams_dim, bias=False)
        self.fc2 = nn.Linear(self.hyperparams_dim, self.hyperparams_dim, bias=False)
        self.fc3 = nn.Linear(self.hyperparams_dim, self.hyperparams_dim, bias=False)
        self.fc4 = nn.Linear(self.hyperparams_dim, 2 * self.num_filters, bias=False)
        self.act = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)
        torch.nn.init.kaiming_uniform_(self.fc4.weight)
        torch.nn.init.uniform_(self.c, 0, 1)

    def forward(self, h_0, h_t, msg):  # TODO: check that weight is clipped (between 0 and 1)
        h = self.c * h_0 + (1 - self.c) * h_t
        ######## f ########
        f1_out = self.fc1(h)
        f2_in = self.act(f1_out)
        f2_out = self.fc2(f2_in)
        f3_in = self.act(f2_out)
        f3_out = self.fc3(f3_in)
        f4_in = self.act(f3_out)
        f4_out = self.fc4(f4_in)
        ######## f ########
        ####### g #######
        g1_weights = f4_out[:, :self.num_filters]
        g2_weights = f4_out[:, self.num_filters:]
        g1_out = torch.nn.functional.linear(msg, g1_weights)
        g2_in = self.act(g1_out)
        g2_out = torch.nn.functional.linear(g2_in, g2_weights)
        ####### g #######
        return g2_out


class StateMLP(nn.Module):
    def __init__(self, num_filters):
        super(StateMLP, self).__init__()
        self.fc1 = nn.Linear(num_filters, num_filters, bias=False)
        self.fc2 = nn.Linear(num_filters, num_filters, bias=False)
        self.act = ShiftedSoftplus()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, num_filters, filter_nn):
        super(CFConv, self).__init__(aggr='add')
        self.fc1 = nn.Linear(in_channels, num_filters, bias=False)
        # self.lin2 = nn.Linear(num_filters, out_channels)
        self.filter_nn = filter_nn
        # self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.fc1.weight)

    def forward(self, x, edge_index, edge_attr):
        # x = self.fc1(x)
        W = self.filter_nn(edge_attr)
        x_msg = self.propagate(edge_index, x=x, W=W)
        return x_msg

    def message(self, x_j, W):
        x_j = self.fc1(x_j)
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        stop = stop - (stop - start) / num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
