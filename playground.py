
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from model.gatgcn import GATGCN
from model.gat import HyperGATConv
from model.gcn import HyperGCNConv
import argparse
import json

#TODO: add scheduler?, update every x iteration (c_hyper and or c_out).


def update_lr(optimizer, decay_factor):
    optimizer.param_groups[0]["lr"] *= decay_factor


def train(data):
    model.train()
    optimizer.zero_grad()
    pred = model(data)[data.train_mask]
    target = data.y[data.train_mask]
    loss = F.nll_loss(pred, target)
    loss.backward()
    optimizer.step()


def test(data):
    accs = []
    with torch.no_grad():
        model.eval()
        logits = model(data)
        # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        # for _, mask in data('train_mask', 'val_mask'):
        for _, mask in data('train_mask'):
            pred = logits[mask].max(1)[1]  #[0] is the values and [1] is the relevant class
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs

names = {"Cora", "CiteSeer", "PubMed"}
name = "PubMed"
cuda = 3
# name = "Cora"
path = os.path.join(os.getcwd(), "dataset", name)
os.makedirs(path, exist_ok=True)
# path = os.path.join('/home/galkampel/tmp', name)
dataset = Planetoid(path, name, split='public', transform=T.NormalizeFeatures())
data_obj = dataset[0]
# print((data_obj.x != 0).sum(1))
# print(data_obj.x.shape[1])
# exit()
# print(data_obj.x.nonzero().size(0)/ data_obj.x.numel(), data_obj.x.numel())
# exit()
device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

#  pubmed: 8 heads_out and weight decay = 0.001
# cora: 1 heads_out and weight decay = 5e-4

# GART params


# GAT paramsters: heads 1, heads 2, p_att, bias_gat, (normalize?), use_hypernetworks ("global")
# GCN parameters: bias_gcn, normalize, use_hypernetworks ("global")

# GATGCN parameters: c_dict = {"out": [False, 0.5], "hyper": [False, 0.5] # requires_grad, init val}
# models name, dropout, use_hypernetworks, f_n_hidden, f_hidden_size

# optimizer parameters: lr, weight decay

# def get_arguments(arg_list=None):
#     parser = argparse.ArgumentParser(description="Model parameters")
#     ###### dataset parameters ######
#     parser.add_argument("--dataset", type=str, default="QM9", choices=["QM9"])
#     parser.add_argument("--save_test", type=bool, default=True, choices=[True, False])
#     ###### run configuration ######
#     parser.add_argument("--c_dict", type=json.loads, default=dict())
#     parser.add_argument("--model_filename", type=str, default=None)  # load (partially trained) model
#     parser.add_argument("--max_iters", type=int, default=int(1e7))
#     parser.add_argument("--target", type=int, default=7, choices=range(12))  # checked on {7, 10}
#     parser.add_argument("--optimizer", type=str, default="Adam")
#     parser.add_argument("--learning_rate", type=float, default=1e-4)
#     parser.add_argument("--decay_every", type=int, default=100000)
#     parser.add_argument("--eval_every", type=int, default=50000)
#     parser.add_argument("--no_improvement_thres", type=int, default=1000000)  # if there is no improvement within no_improvement_thres update steps - stop
#     parser.add_argument("--lr_decay_factor", type=float, default=0.96)
#     ###### model parameters ######
#     parser.add_argument("--model_name", type=str, default="NMPEdge")
#     parser.add_argument("--cutoff", type=float, default=15.0)
#     parser.add_argument("--num_passes", type=int, default=4)
#     parser.add_argument("--num_gaussians", type=int, default=150)
#     parser.add_argument("--embed_size", type=int, default=256)
#     # parser.add_argument("--hidden_channels", type=int, default=256)
#     parser.add_argument("--num_filters", type=int, default=256)
#     parser.add_argument("--gpu_device", type=int, default=3, choices=[0, 1, 2, 3])
#     parser.add_argument("--readout", type=str, default="add", choices=["add", "mean"])
#     parser.add_argument("--hypernet_update", type=bool, default=True, choices=[True, False])
#     parser.add_argument("--f_hidden_channels", type=int, default=64)
#     parser.add_argument("--g_hidden_channels", type=int, default=128)
#     return parser.parse_args(arg_list)


# def set_model_layers(data, dataset_name, model_name):
#     if model_name == 'gcn' and dataset_name == 'PubMed':
#         heads_in = heads_out = 8
#         model_layers = nn.ModuleList([
#             HyperGATConv(data.num_features, out_channels),
#             HyperGATConv()
#         ])


p_att = 0.6  # default 0.6, 0.4 is good
use_hypernetworks = True
bias_gat = True  # use_hypernetworks = False, bias = True
bias_gcn = True

num_features = dataset.num_features
num_classes = dataset.num_classes
# note- second layer has no bias
model1_layers = nn.ModuleList([
    HyperGATConv(num_features, out_channels=8, heads=8, dropout=p_att, bias=bias_gat),
    HyperGATConv(8 * 8, out_channels=num_classes, heads=8, concat=False, dropout=p_att, bias=False,
                 use_hypernetworks=use_hypernetworks)
])

# model1_layers = nn.ModuleList([
#     HyperGCNConv(dataset.num_features, out_channels=64, cached=True, bias=bias_gcn, normalize=True),
#     HyperGCNConv(64, dataset.num_classes, cached=True, bias=False, normalize=True,
#                  use_hypernetworks=use_hypernetworks)
# ])

# model2_layers = nn.ModuleList([
#     HyperGATConv(dataset.num_features, out_channels=8, heads=8, dropout=p_att, bias=bias_gat),
#     HyperGATConv(8 * 8, out_channels=dataset.num_classes, heads=8, concat=False, dropout=p_att, bias=False,
#                      use_hypernetworks=use_hypernetworks)
# ])


model2_layers = nn.ModuleList([
    HyperGCNConv(num_features, out_channels=64, cached=True, bias=bias_gcn, normalize=True),
    HyperGCNConv(64, num_classes, cached=True, bias=False, normalize=True,
                 use_hypernetworks=use_hypernetworks)
])

# model1_layers.parameters()



c_dict = {
    "out":
        {
        "requires_grad": True,
        "init_val": 0.5,
        "update_every": 10  # 1
        },
    "hyper":
        {
        "requires_grad": True,
        "init_val": 0.5,
        "update_every": 10   # 1
        }
}

models_name = ['gat', 'gcn']
# models_name = ['gcn', 'gcn']
# models_name = ['gat', 'gat']
p_input = 0.6  # default 0.6
p_hyper = 0.1
f_n_hidden = 4  # best 4?
f_hidden_size = 128  # best 128
model = GATGCN(model1_layers, model2_layers, models_name=models_name, c_dict=c_dict, p=p_input, p_hyper=0.1,
               use_hypernetworks=use_hypernetworks, f_n_hidden=f_n_hidden, f_hidden_size=f_hidden_size).to(device)

## hypernetwork hyper parameters
# model.set_gatconv_hidden_device(device)  # heads_hidden=8
data_obj = data_obj.to(device)


# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)  # default: lr=0.005, weight_decay=5e-4
optimizer = torch.optim.Adam([
    dict(params=model1_layers.parameters(), lr=0.005, weight_decay=5e-4),
    dict(params=model2_layers[0].parameters(), lr=0.01),
    dict(params=model2_layers[1].parameters(), lr=0.01, weight_decay=0),
    dict(params=model.hyper_params.parameters(), lr=1e-4, weight_decay=5e-5),  # lr=4e-5, weight_decay=1e-5
    dict(params=model.c_hyper, lr=0.005, weight_decay=0),
    dict(params=model.c_out, lr=0.005, weight_decay=0)
    ], lr=0.005, weight_decay=5e-4) if use_hypernetworks else \
    torch.optim.Adam([
        dict(params=model1_layers.parameters()),
        dict(params=model2_layers[0].parameters(), lr=0.01),
        dict(params=model2_layers[1].parameters(), lr=0.01, weight_decay=0),
        dict(params=model.c_out, weight_decay=0)
    ], lr=0.005, weight_decay=5e-4)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # more epochs, lower lr
epochs = 1000
decay_every = epochs
decay_factor = 0.96  # default 0.96

for epoch in range(1, epochs + 1):
    # print(f'c_out = {model.c_out.item()}\tc_hyper = {model.c_hyper.item()}')
    model.clip_cs(device)
    model.set_cs_grad(epoch, requires_grad=True)
    train(data_obj)
    model.set_cs_grad(epoch, requires_grad=False)
    # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}'
    log = 'Epoch: {:03d}, Train: {:.4f}'
    print(log.format(epoch, *test(data_obj)))
    # print(f'c_out = {model.c_out.item()}\tc_hyper = {model.c_hyper.item()}')

    if epoch % decay_every == 0:
        update_lr(optimizer, decay_factor)
