import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch_geometric
from torch_geometric.utils import to_networkx
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid

import json

def load_dataset(data_dir, dataset_name):
    if dataset_name == 'cora':
        dataset = Planetoid(root=data_dir, name='Cora')
    elif dataset_name == 'arxiv':
        dataset = PygNodePropPredDataset(root=data_dir, name='ogbn-arxiv')        
    elif dataset_name == 'products':
        dataset = PygNodePropPredDataset(root=data_dir, name='ogbn-products')
    elif dataset_name == 'pubmed':
        dataset = Planetoid(root=data_dir, name='Pubmed')
    elif dataset_name == 'citeseer':
        dataset = Planetoid(root=data_dir, name='Citeseer')
    else :
        print("ERROR : Dataset name not present in our list!")
    return dataset

def draw_graph(edge_example):
    node_example = np.unique(edge_example.flatten())
    plt.figure(figsize=(10, 6))
    G = nx.Graph()
    G.add_nodes_from(node_example)
    G.add_edges_from(list(zip(edge_example[0], edge_example[1])))
    nx.draw_networkx(G, with_labels=False)
    plt.show()

def characterize_graph(G):
    #Calculate the number of nodes and edges in the graph
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    graph_type = "Directed" if G.is_directed() else "Undirected"
    # Calculate the density of the graph
    density = nx.density(G)
    # Degree Distribution
    degree_sequence = [d for n, d in G.degree()]
    degree_counts = np.bincount(degree_sequence)
    # Calculate the average degree of nodes in the graph
    average_degree = np.mean([val for (node, val) in G.degree()])
    # Connected Components
    num_connected_components = nx.number_connected_components(G)
    # Centrality Measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)

    # Clustering Coefficient
    clustering_coefficient = nx.average_clustering(G)
    # Graph Diameter
    try:
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        diameter = "Infinite (Graph is not connected)"
    # Print Basic Graph Properties
    print("Number of Nodes:", num_nodes)
    print("Number of Edges:", num_edges)
    print("Graph Type:", graph_type)

    # Print Connectivity Metrics
    print("Density:", density)
    print("Degree Distribution:", degree_counts)
    print("Average Degree:", average_degree)
    print("Number of Connected Components:", num_connected_components)

    # Print Centrality Measures
    print("Degree Centrality:", degree_centrality)
    print("Betweenness Centrality:", betweenness_centrality)
    print("Closeness Centrality:", closeness_centrality)
    print("Eigenvector Centrality:", eigenvector_centrality)

    # Print Clustering Coefficient and Graph Diameter
    print("Clustering Coefficient:", clustering_coefficient)
    print("Graph Diameter:", diameter)

    # Plot Degree Distribution Histogram
    plt.hist(degree_sequence, bins=range(max(degree_sequence)+1), alpha=0.75)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()


def plot_label_distribution(data):
    # Count the occurrences of each y label
    label_counts = torch.bincount(data.y.view(-1))
    # Create a bar chart with rotated x-axis labels
    plt.bar(range(len(label_counts)), label_counts.numpy())
    plt.xlabel('Y Label')
    plt.ylabel('Frequency')
    plt.title('Distribution of Y Labels')
    plt.xticks(range(len(label_counts)), rotation=90)  # Rotate x-axis labels
    plt.tight_layout()  # Adjust layout for better readability
    plt.show()

def print_dataset_stats(data): # modify
    print(f'Number of nodes: {data.num_nodes}')
    # Number of nodes: 2708
    print(f'Number of edges: {data.num_edges}')
    # Number of edges: 10556
    print(f'Number of features: {data.num_node_features}')
    # Number of edge features: 0
    # Count the number of distinct y labels
    num_distinct_labels = torch.unique(data.y).size(0)
    print("Total number of distinct classes:", num_distinct_labels)
    # Number of classes: 7
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')  # False
    print(f'Has self-loops: {data.has_self_loops()}')  # False
    print(f'Is undirected: {data.is_undirected()}')  # True

def get_labels_for_dataset(dataset_name):
    label_filename = "./label_data/" + dataset_name + "_labels.json"
    with open(label_filename, 'r') as label_file:
        data = json.load(label_file)
    labels = dict()
    for item in data:
        labels[item['class']] = item['label']
    return labels