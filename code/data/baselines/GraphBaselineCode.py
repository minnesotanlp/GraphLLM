import torch
vers = torch.__version__
print("Torch vers: ", vers)

# PyG installation
# !pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu118.html
# !pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu118.html
# !pip install -q git+https://github.com/rusty1s/pytorch_geometric.git
# !pip install tensorboardX
# !pip install tensorboard
# !pip install ogb

import torch_geometric
import numpy as np
import pandas as pd
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric
from torch.nn import Parameter
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import urllib.request
import tarfile
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool as gmp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from ogb.nodeproppred import PygNodePropPredDataset

log_dir = '/content/graph_runs'

# Load Cora dataset
dataset = Planetoid(root="../data", name='Cora')
# dataset = PygNodePropPredDataset(root="../", name='ogbn-arxiv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = DataLoader(dataset, batch_size=1, shuffle=True)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

    def forward(self, x, edge_index, batch):
        # x: Node feature matrix
        # edge_index: Graph connectivity matrix
        #x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        h = gmp(x, batch)
        return h, F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                        lr=0.005,
                                        weight_decay=5e-4)

    def forward(self, x, edge_index, batch):
        x = self.sage1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        # Pooling
        h = gmp(x, batch)
        h = h.mean(dim = 0, keepdims = True) # this does not have a sampler marker to the batch, so we have to keep batch size as 1
        return h, F.log_softmax(x, dim=1)

'''
GAT- uses Attention stratgey
compute the hidden representations of each node in the Graph by attending
over its neighbors using a self-attention strategy
'''
from torch_geometric.nn import global_max_pool as gmp

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

    def forward(self,x, edge_index, batch):

        # Dropout before the GAT layer is used to avoid overfitting
        x = F.dropout(x, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        h = gmp(x, batch)
        return h,F.log_softmax(x, dim=1)

# model = GCN().to(device)
# model = GraphSAGE(dataset.num_features, 16, dataset.num_classes).to(device)
model = GAT().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
loss_fnc = torch.nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    total_loss = 0
    for train_data in loader:
      train_data = train_data.to(device)
      optimizer.zero_grad()
      _, out = model(train_data.x, train_data.edge_index, train_data.batch)
    #   oh = F.one_hot(train_data.y, dataset.num_classes).to(out)
    #   loss = -(out * oh).sum(dim=1).mean()
      loss = loss_fnc(out[train_data.train_mask], data.y[train_data.train_mask])
      loss.backward()
      optimizer.step()
      total_loss += float(loss) * train_data.num_graphs
    avg_train_loss = total_loss / train_data.train_mask.sum()
    return avg_train_loss

@torch.no_grad()
def test(epoch):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    mean_conf = 0
    threshold = 0.5
    for test_data in loader:
      test_data = test_data.to(device)
      _, out = model(test_data.x, test_data.edge_index, test_data.batch)
    #   oh = F.one_hot(test_data.y, dataset.num_classes).to(out)
    #   loss = -(out * oh).sum(dim=1).mean()
      loss = loss_fnc(out[test_data.test_mask], data.y[test_data.test_mask])
      total_loss += float(loss) * test_data.num_graphs
      preds = torch.softmax(out[test_data.test_mask], dim=1)
      conf, classes = torch.max(preds, 1)
      mean_conf = torch.mean(conf)
      threshold_confidence = (conf < threshold).sum()
      all_preds.append(preds)
      all_labels.append(test_data.y[test_data.test_mask].float())

    # Calculate Metrics
    accuracy = metrics(all_preds, all_labels)
    return total_loss / test_data.test_mask.sum(), accuracy, mean_conf, threshold_confidence

def metrics(preds, gts):
  preds = torch.cat(preds, dim=0).max(dim=1)[1].cpu().detach().numpy()
  gts = torch.cat(gts, dim=0).cpu().detach().numpy()
  acc = accuracy_score(preds, gts)
  # f1 = f1_score(preds, gts)
  # precision = precision_score(preds, gts)
  # recall = recall_score(preds, gts)
  return acc

# Train the model
writer = SummaryWriter(log_dir+"/gs") # change the log dir for every graph model
num_epochs = 200
for epoch in range(num_epochs):
    loss = train(epoch)
    val_loss, val_acc, val_conf, val_threshold = test(epoch)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {val_acc:.4f}, Confidence: {val_conf:.4f}, Threshold: {val_threshold}')