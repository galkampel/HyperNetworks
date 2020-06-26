
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
from model.nmp_edge import NMPEdge
import argparse

SEED = 0
optimizers = {'Adam': nn.Adam, 'SGD': nn.SGD}

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network"
    )
    parser.add_argument("--dataset", type=str, default="QM9", choices=["QM9"])
    parser.add_argument("--path_root", type=str, default="'/home/galkampel/tmp/QM9'")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--cutoff", type=int, default=15)
    parser.add_argument("--num_passes", type=int, default=4)
    parser.add_argument("--num_gaussians", type=int, default=150)
    parser.add_argument("--node_embedding_size", type=int, default=256)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_filters", type=int, default=256)
    parser.add_argument("--gpu_device", type=int, default=3, choices=[0, 1, 2, 3, None])
    parser.add_argument("--readout", type=str, default="add")
    parser.add_argument("--max_iters", type=int, default=int(1e7))
    parser.add_argument("--target", type=int, default=0, choices=range(12))
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--decay_evey", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=50000)
    parser.add_argument("--no_improvement_thres", type=int, default=1000000)
    parser.add_argument("--lr_decay_factor", type=float, default=0.96)
    parser.add_argument("--hypernet_update", type=bool, default=False)

    return parser.parse_args(arg_list)


class trainer:
    def __init__(self, **kwargs): # optimizer parameters, optimizer_name, target, n_iters,
        # update_every, save
        self.model_name = kwargs.get('model_name", "nmp_edge')
        opt_dict = {'Adam': torch.optim.Adam}
        self.optimizer

    def fit(self):
        ...

    def evaluate(self):
        ...

    def save_model(self):
        ...

    def get_pretrained_model(self):
        2
# DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


class model_run:
    def __init__(self, configs):

        # hyper_update, embed_size, hidden_channels, num_filters, num_interactions,
        # num_gaussians, cutoff, readout, hyper_update, device, dipole, mean, std, atomref


def get_dataset(root): return QM9(root=root)


def train_test_split(dataset, train_size=120000, val_size=10000, seed=0):
    torch.manual_seed(seed)
    N = len(dataset)
    train_val_set, test_set = torch.utils.data.random_split(dataset, [train_size, N - train_size])
    train_set, val_set = torch.utils.data.random_split(train_val_set, [train_size - val_size, val_size])
    return train_set, val_set, test_set


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
for batch in train_loader:
    batch =batch.to(device)
# val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
data0 = dataset[0]

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = NMPEdge().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()

for batch in train_loader:
    batch = batch.to(device)
    optimizer.zero_grad()
    pred = model(batch.z, batch.pos, batch.batch)
    mae = (pred.view(-1) - batch.y[:, 1]).abs()
    loss.backward()
    optimizer.step()

    # batch = batch.to(DEVICE)
    x = 3
    y = 2


def main(args):
    torch.manual_seed(SEED)
    dataset = get_dataset(args.root)
    DataLoader = get_dataloader_class("qm9")
    graph_obj_list = DataLoader(cutoff_type="const", cutoff_radius=100).load()
    print('finished downloading Qm9DataLoader_const-100.00.pkz')
    exit()
    # print(f'graph_obj_list:\n{graph_obj_list}')
    data_handler = datahandler.EdgeSelectDataHandler(
        graph_obj_list, ["U"], 0)

    target_mean, target_std = data_handler.get_normalization(per_atom=True)
    x = 4


if __name__ == "__main__":
    args  =get_arguments()
    main(args)