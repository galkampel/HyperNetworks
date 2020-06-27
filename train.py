from csv import excel

from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
from model.nmp_edge import NMPEdge
import os
import numpy as np
import argparse

SEED = 0
optimizers = {'Adam': nn.Adam, 'SGD': nn.SGD}


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Model parameters")
    ###### run configuration ######
    parser.add_argument("--dataset", type=str, default="QM9", choices=["QM9"])
    parser.add_argument("--path_root", type=str, default="'/home/galkampel/tmp/QM9'")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_filename", type=str, default=None)
    parser.add_argument("--max_iters", type=int, default=int(1e7))
    parser.add_argument("--target", type=int, default=0, choices=range(12))
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--decay_evey", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=50000)
    parser.add_argument("--no_improvement_thres", type=int, default=1000000)
    parser.add_argument("--lr_decay_factor", type=float, default=0.96)
    ###### model parameters ######
    parser.add_argument("--cutoff", type=float, default=15.0)
    parser.add_argument("--num_passes", type=int, default=4)
    parser.add_argument("--num_gaussians", type=int, default=150)
    parser.add_argument("--node_embedding_size", type=int, default=256)
    # parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_filters", type=int, default=256)
    parser.add_argument("--gpu_device", type=int, default=3, choices=[0, 1, 2, 3, None])
    parser.add_argument("--readout", type=str, default="add")
    parser.add_argument("--hypernet_update", type=bool, default=False)

    return parser.parse_args(arg_list)


class trainer:
    def __init__(self, model_run, args): # optimizer parameters, optimizer_name, target, n_iters,
        # update_every, save
        self.model = model_run.get_model()
        self.model_name = model_run.get_model_name()
        self.device = model_run.get_device()
        self.optimizer = optimizers[args.optimizer](self.model.parameters(), lr=args.learning_rate)
        self.decay_factor = args.lr_decay_factor
        self.decay_every = args.decay_every
        self.eval_every = args.eval_every
        self.target = args.target
        self.max_iters = args.max_iters
        self.checkpoint_folder = os.path.join(os.getcwd(), 'chekpoint')
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        self.start_iter = 0
        self.model_filename = args.model_filename
        self.is_best_model = False
        if self.model_filename and os.path.exists(os.path.join(self.checkpoint_folder, f'{self.model_filename}.pth')):
            self.load_model()

    def update_lr(self):
        self.optimizer.param_groups[0]["lr"] *= args.lr_decay_factor

    def fit(self, train_loader, val_loader): #TODO: finish fit
        ...

    def predict(self, data_loader):
        maes = []
        with torch.no_grad():
            self.model.eval()
            for data_batch in data_loader:
                data_batch = data_batch.to(self.device)
                pred = self.model(data_batch.z, data_batch.pos, data_batch.batch)
                maes.append((pred.view(-1) - data_batch.y[:, self.target]).abs().cpu().numpy())
        mae = np.concatenate(maes).mean()
        return mae

    def save_model(self, iteration, is_best_model=False):
        model_saved_name = f'{self.model_name}_target={self.target}'
        full_path = os.path.join(self.checkpoint_folder, f'{model_saved_name}.pth')
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'iteration': iteration,
                    'is_best_model': is_best_model}, full_path)

    def load_model(self):
        path_model = os.path.join(self.checkpoint_folder, f'{self.model_filename}.pth')
        checkpoint = torch.load(path_model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_iter = checkpoint['iteration']
        self.is_best_model = checkpoint['is_best_model']


class ModelRun:
    def __init__(self, args):
        num_gaussians = args.num_gaussians
        hidden_channels = args.node_embedding_size
        num_interactions = args.num_passes
        cutoff = args.cutoff
        num_filters = args.num_filters
        readout = args.readout
        hypernet_update = args.hypernet_update
        gpu_device = args.gpu_device
        self.device = torch.device(f'cuda:{gpu_device}' if torch.cuda.is_available() else 'cpu')
        self.model = NMPEdge(num_gaussians=num_gaussians, cutoff=cutoff, num_interactions=num_interactions,
                             hidden_channels=hidden_channels, num_filters=num_filters, readout=readout,
                             hypernet_update=hypernet_update, device=self.device).to(self.device)  # no num_embeddings

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device

    def get_model_name(self):
        name = 'NMPEdge'
        if self.hypernet_update:
            name +=  '_hypernet'
        return name


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
    args = get_arguments()
    main(args)