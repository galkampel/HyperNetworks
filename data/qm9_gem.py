from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch
# from torch_geometric.nn import SchNet
from model.nmp_edge import NMPEdge

# DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
dataset = QM9(root='/home/galkampel/tmp/QM9') # , transform=T.Distance(norm=False)
train_val_set, test_set = torch.utils.data.random_split(dataset, [120000, 9433])
train_set, val_set = torch.utils.data.random_split(train_val_set, [110000, 10000])


# dataset2 = QM9(root='/home/galkampel/tmp/QM9')

# model, (train_set, validation_set, test_set) = SchNet.from_qm9_pretrained(root='/home/galkampel/tmp/QM9', dataset='QM9', target=2)

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
model = NMPEdge().to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in train_loader:
    batch = batch.to(device)
    optimizer.zero_grad()
    pred = model(batch.z, batch.pos, batch.batch)
    mae = (pred.view(-1) - batch.y[:, 0]).abs()
    optimizer.step()
# val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
