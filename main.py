
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch
from model.nmp_edge import NMPEdge
import argparse


# DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
def get_dataset(root): return QM9(root=root)
dataset = QM9(root='/home/galkampel/tmp/QM9')  # transform=T.Distance(norm=False)

def train_test_split(train_size=12,seed=0,x=4 ):
    torch.manual_seed(seed)
    train_val_set, test_set = torch.utils.data.random_split(dataset, [120000, 9433])
    train_set, val_set = torch.utils.data.random_split(train_val_set, [110000, 10000])
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', default='/home/galkampel/tmp/QM9')  # download QM9
    parser.add_argument('-weights_file_path', default='input/weights_params.json')
    parser.add_argument('-LAMAS_weights_path', default='Data/weights/rel_weights.csv')
    parser.add_argument('-unemployment_path', default='Data/Unemployment rate (smoothend).xlsx')
    parser.add_argument('-path_dict_names', default='Data/NamesDict.xlsx')
    arguments = parser.parse_args()
    main(arguments)
    main()