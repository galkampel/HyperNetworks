
import os
import torch

# from torch_geometric.nn import SchNet
from model.gan import GATConv
from utils.preprocess import TransductivePreprocessing
from torch_geometric.data import DataLoader
from torch_geometric.datasets import CitationFull

names = {"Cora", "Cora_ML", "CiteSeer", "PubMed"}
name = "PubMed"
path = os.path.join('/home/galkampel/tmp', name)
dataset = CitationFull(path, name)
t_preprocess = TransductivePreprocessing(nodes_per_class=20, n_validation=500, n_test=1000)
train_idx, val_idx, test_idx = t_preprocess.train_val_test_idx_split(dataset.data.y)
train_set, validation_set, test_set = dataset[train_idx], dataset[val_idx], dataset[test_idx]
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)  # 1?
val_loader = DataLoader(validation_set, batch_size=1)  # 1?
test_loader = DataLoader(test_set, batch_size=1)  # 1?
for data_batch in train_loader:
    x = 5
    # data_batch = data_batch.to(device)
# device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
exit()

# for target in range(12):
#     model, datasets = SchNet.from_qm9_pretrained(path, dataset, target)
#     train_dataset, val_dataset, test_dataset = datasets

    # model = model.to(device)
    # loader = DataLoader(test_dataset, batch_size=256)

    # maes = []
    # for data in loader:
    #     data = data.to(device)
        # with torch.no_grad():
            # pred = model(data.z, data.pos, data.batch)
        # mae = (pred.view(-1) - data.y[:, target]).abs()
        # maes.append(mae)

    # mae = torch.cat(maes, dim=0)

    # Report meV instead of eV.
    # mae = 1000 * mae if target in [2, 3, 4, 6, 7, 8, 9, 10] else mae

    # print(f'Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f}')

