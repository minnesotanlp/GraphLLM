from utils import characterize_graph
import networkx as nx
G = nx.erdos_renyi_graph(100, 0.15)  # Replace with your graph or dataset
characterize_graph(G)