import sys
import random
import numpy as np
import torch
import torch.nn as nn


class Network(torch.nn.Module):
    def __init__(self, args=None):
        super(Network, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, op, x):
        xy = torch.cat((op, x), dim=-1)
        return self.linear(xy)


class HyperNetwork1(torch.nn.Module):
    def __init__(self, args=None):
        super(HyperNetwork1, self).__init__()
        self.get_w = nn.Linear(1, 1)
        self.get_b = nn.Linear(1, 1)

    def forward(self, op, x):
        return self.get_w(op) * x + self.get_b(op)


class HyperNetwork2(torch.nn.Module):
    def __init__(self, args=None):
        super(HyperNetwork2, self).__init__()
        self.get_w = nn.Linear(1, 1)
        self.get_b = nn.Linear(1, 1)

    def forward(self, op, x):
        weight = self.get_w(op)
        bias = self.get_b(op)
        return torch.nn.functional.linear(x, weight, bias)


class HyperNetwork3(torch.nn.Module):
    def __init__(self, args=None):
        super(HyperNetwork3, self).__init__()
        self.get_w1 = nn.Linear(1, 3)
        self.get_b1 = nn.Linear(1, 3)

        self.get_w2 = nn.Linear(3, 1)
        self.get_b2 = nn.Linear(3, 1)

    def forward(self, op, x):
        ######### f ##########
        weight1 = self.get_w1(op)
        bias1 = self.get_b1(op)

        # weight2 = self.get_w2(op)
        weight2 = self.get_w2(weight1)
        bias2 = self.get_b2(bias1)

        ######### g ##########
        x = torch.nn.functional.linear(x, weight1, bias1)
        x = nn.functional.relu(x)
        x = torch.nn.functional.linear(x, weight2, bias2)
        return x


# if sys.argv[1] == "Network":
#     net = Network()
# elif sys.argv[1] == "HyperNetwork1":
#     net = HyperNetwork1()
# elif sys.argv[1] == "HyperNetwork2":
#     net = HyperNetwork2()
# elif sys.argv[1] == "HyperNetwork3":
#     net = HyperNetwork3()
# else:
#     assert False

net = HyperNetwork3()
learn_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)

batch_size = 5

while True:
    optimizer.zero_grad()

    x_batch = []
    op_batch = []
    ground_truth_batch = []
    for i in range(5):
        x = random.random() * 200 - 100
        op = random.randint(0, 1)
        if op == 0:
            ground_truth = x
        elif op == 1:
            ground_truth = x * 7

        op = torch.FloatTensor([op])
        x = torch.FloatTensor([x])
        x_batch.append(x)
        op_batch.append(op)
        ground_truth_batch.append(op)

    op = torch.stack(op_batch)
    x = torch.stack(x_batch)
    ground_truth = torch.stack(ground_truth_batch)

    prediction = net(op, x)
    loss = ((ground_truth - prediction) ** 2).mean()
    print(loss.item())

    loss.backward()
    optimizer.step()

