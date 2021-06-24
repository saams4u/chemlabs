import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, SGConv, AGNNConv, ARMAConv, global_add_pool

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    row = []
    for i in range(adj.shape[0]):
        sum = adj[i].sum()
        row.append(sum)
    rowsum = np.array(row)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    a = d_mat_inv_sqrt.dot(adj)
    return a

def preprocess_adj(adj, norm=True, sparse=False):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = adj + np.eye(len(adj))
    if norm:
        adj = normalize_adj(adj)
    return adj

class GAT(torch.nn.Module):
    """
    Graph Attention Networks
    <https://arxiv.org/abs/1710.10903>
    """
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(75, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(
            8 * 8, 128, heads=1, concat=True, dropout=0.6)

        self.gather_layer = nn.Linear(128, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = F.dropout(x, p=0.6, training=self.training)
        x2 = F.elu(self.conv1(x1, edge_index))
        x3 = F.dropout(x2, p=0.6, training=self.training)
        x4 = self.conv2(x3, edge_index)

        y_molecules = global_add_pool(x4, batch)
        z_molecules = self.gather_layer(y_molecules)
        return z_molecules

    def __call__(self, data, std, mean, train=True):

        target = torch.unsqueeze(data.y, 1)
        out = self.forward(data)
        if train:
            loss = F.mse_loss(out, target)
            return loss
        else:
            loss = F.mse_loss(out, target)
            out = out.to('cpu').data.numpy()
            target = target.to('cpu').data.numpy()
            z, t = std * out + mean, std * target + mean
            return loss, z, t

class SGC(torch.nn.Module):
    """
    Simplifying Graph Convolutional Networks"
    <https://arxiv.org/abs/1902.07153>
    """
    def __init__(self):
        super(SGC, self).__init__()
        self.conv1 = SGConv(
            75, 128, K=2, cached=False)
        self.gather_layer = nn.Linear(128, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.gather_layer.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)

        y_molecules = global_add_pool(x1, batch)
        z_molecules = self.gather_layer(y_molecules)
        return z_molecules

    def __call__(self, data, std, mean, train=True):

        target = torch.unsqueeze(data.y, 1)
        out = self.forward(data)
        if train:
            loss = F.mse_loss(out, target)
            return loss
        else:
            loss = F.mse_loss(out, target)
            out = out.to('cpu').data.numpy()
            target = target.to('cpu').data.numpy()
            z, t = std * out + mean, std * target + mean
            return loss, z, t

class AGNN(torch.nn.Module):
    """
    Attention-based Graph Neural Network for Semi-Supervised Learning
    <https://arxiv.org/abs/1803.03735>
    """
    def __init__(self):
        super(AGNN, self).__init__()
        self.lin1 = torch.nn.Linear(75, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, 64)

        self.gather_layer = nn.Linear(64, 1)

    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)

        y_molecules = global_add_pool(x, batch)
        z_molecules = self.gather_layer(y_molecules)
        return z_molecules

    def __call__(self, data, std, mean, train=True):

        target = torch.unsqueeze(data.y, 1)
        out = self.forward(data)
        if train:
            loss = F.mse_loss(out, target)
            return loss
        else:
            loss = F.mse_loss(out, target)
            out = out.to('cpu').data.numpy()
            target = target.to('cpu').data.numpy()
            z, t = std * out + mean, std * target + mean
            return loss, z, t

class ARMA(torch.nn.Module):
    """
    Graph Neural Networks with Convolutional ARMA Filters
    <https://arxiv.org/abs/1901.01343>
    """
    def __init__(self):
        super(ARMA, self).__init__()

        self.conv1 = ARMAConv(
            75,
            16,
            num_stacks=3,
            num_layers=2,
            shared_weights=True,
            dropout=0.25)

        self.conv2 = ARMAConv(
            16,
            64,
            num_stacks=3,
            num_layers=2,
            shared_weights=True,
            dropout=0.25,
            act=None)

        self.gather_layer = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        y_molecules = global_add_pool(x, batch)
        z_molecules = self.gather_layer(y_molecules)
        return z_molecules

    def __call__(self, data, std, mean, train=True):

        target = torch.unsqueeze(data.y, 1)
        out = self.forward(data)
        if train:
            loss = F.mse_loss(out, target)
            return loss
        else:
            loss = F.mse_loss(out, target)
            out = out.to('cpu').data.numpy()
            target = target.to('cpu').data.numpy()
            z, t = std * out + mean, std * target + mean
            return loss, z, t