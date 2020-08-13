
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv


names = {"Cora", "CiteSeer", "PubMed"}
name = "Cora"
path = os.path.join('/home/galkampel/tmp', name)
dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
data = dataset[0]


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=0.6)

    def forward(self):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model, data = GAT().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    accs = []
    with torch.no_grad():
        model.eval()
        logits = model()
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs


for epoch in range(1, 301):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))

# import os
# import torch
# from torch.utils.data import Subset
# # from torch_geometric.nn import SchNet
# from model.gan import GATConv
# from utils.preprocess import TransductivePreprocessing
# from torch_geometric.data import DataLoader, Data, Dataset
# import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid  # QM9, CitationFull
#
# names = {"Cora", "CiteSeer", "PubMed"}
# name = "PubMed"
# path = os.path.join('/home/galkampel/tmp', name)
# train_batch_size = 1
# test_batch_size = 1
# dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
# data = dataset[0]
# x = 3
# # data = dataset[0]
# # path2 = os.path.join('/home/galkampel/tmp', "QM9")
# # dataset2 = QM9(root=path2)
# t_preprocess = TransductivePreprocessing(nodes_per_class=20, n_validation=500, n_test=1000)
# train_idx, validation_idx, test_idx = t_preprocess.train_val_test_idx_split(dataset.data.y)
# # train_idx = torch.from_numpy(train_idx).long()
# # validation_idx, test_idx = torch.from_numpy(validation_idx).long(), torch.from_numpy(test_idx).long()
# # train_idx, val_idx, test_idx = t_preprocess.train_val_test_idx_split(dataset.data.y)
#
# # data_train = Data(x=data.x[train_mask], y=data.y[train_mask], edge_index=data.edge_index)
# training_set = t_preprocess.create_dataset(data, train_idx, data_size=1)
# train_loader = DataLoader(training_set, batch_size=16, shuffle=True)
#
# # validation_set = t_preprocess.create_dataset(data, validation_idx, data_size=test_batch_size)
# # train_loader = DataLoader(validation_set, batch_size=16, shuffle=True)
#
# # training_set = t_preprocess.create_dataset(data, train_idx, data_size=train_batch_size)
# # train_loader = DataLoader(training_set, batch_size=train_batch_size, shuffle=True)
#
# count = 1
# for data_batch in train_loader:
#     print(f'count = {count}')
#     count += 1
# x = 5
