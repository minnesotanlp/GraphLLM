from utils import characterize_graph, load_dataset
import networkx as nx
import torch_geometric
from torch_geometric.utils import to_networkx
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid

data_dir = './data'
dataset_name = 'products'
dataset = load_dataset(data_dir, dataset_name)
data = dataset[0]
# Convert the PyG graph to NetworkX graph
nx_graph = to_networkx(data, to_undirected=True)
#G = nx.erdos_renyi_graph(100, 0.15)  # Replace with your graph or dataset
characterize_graph(nx_graph,dataset_name)