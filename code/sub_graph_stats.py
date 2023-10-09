import os
import json
import time
import statistics
import csv
import openai
import random
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from networkx.algorithms import community

from utils import draw_graph,plot_label_distribution, load_dataset,print_dataset_stats

# this function returns all possible no of neighbors in a 2 hop subgraph, given total no of nodes in a subgraph
def calculate_all_k_values(total_nodes):
    k1_values = []
    k2_values = []

    # Iterate through possible values of k1 (1-hop neighbors)
    for k1_candidate in range(total_nodes - 1):
        # Calculate the maximum possible k2 (2-hop neighbors) that doesn't exceed total_nodes
        max_k2_candidate = min(total_nodes - k1_candidate - 1, k1_candidate)

        # Calculate the total number of nodes in the ego graph for this k1_candidate
        total_nodes_candidate = 1 + k1_candidate + max_k2_candidate

        # Check if the total nodes match the desired value (20)
        if total_nodes_candidate == total_nodes:
            k1_values.append(k1_candidate)
            k2_values.append(max_k2_candidate)

    return k1_values, k2_values


# Function to sample a 2-hop ego graph centered at a node
def sample_k_hop_ego_graph(graph, center_node, no_of_hops):
    try:
        ego_graph = nx.ego_graph(graph, center_node, radius=no_of_hops, undirected=True)
        return ego_graph
    except nx.NetworkXError:
        return None

def calc_average_nodes_edges_khop_subgraphs(nx_graph, no_of_hops):
    # Initialize variables to store the total number of edges and ego graph count
    
    ego_graph_count = 0
    edges_per_ego_graph = []
    nodes_per_ego_graph = []

    # Iterate through all nodes in the graph
    for node in nx_graph.nodes():
        # Sample a 2-hop ego graph centered at the current node
        ego_graph = sample_k_hop_ego_graph(nx_graph, node, no_of_hops)
        
        if ego_graph is not None:
            # Calculate the number of edges in the sampled ego graph
            num_edges = ego_graph.number_of_edges()
            num_nodes = ego_graph.number_of_nodes()
            # Update total edges and ego graph count
            edges_per_ego_graph.append(num_edges)
            nodes_per_ego_graph.append(num_nodes)
            ego_graph_count += 1

    # Calculate the average number of edges
    average_edges = statistics.mean(edges_per_ego_graph)
    std_edges = statistics.stdev(edges_per_ego_graph)
    average_nodes = statistics.mean(nodes_per_ego_graph)
    std_nodes = statistics.stdev(nodes_per_ego_graph)
    print(f"Average number of edges in 2-hop subgraphs: {average_edges:.2f} with a std dev of {std_edges:.2f}")
    print(f"Average number of nodes in 2-hop subgraphs: {average_nodes:.2f} with a std dev of {std_nodes:.2f}")


def calc_average_nodes_edges_khop_subgraphs_dist(nx_graph, no_of_hops):
    # Initialize variables
    ego_graph_count = 0
    edges_per_ego_graph = []
    nodes_per_ego_graph = []

    # Histograms to store the distribution of nodes and edges
    node_histogram = {}
    edge_histogram = {}

    # Iterate through all nodes in the graph
    for node in nx_graph.nodes():
        # Sample a k-hop ego graph centered at the current node
        ego_graph = sample_k_hop_ego_graph(nx_graph, node, no_of_hops)
        
        if ego_graph is not None:
            # Calculate the number of edges and nodes in the sampled ego graph
            num_edges = ego_graph.number_of_edges()
            num_nodes = ego_graph.number_of_nodes()
            
            # Update lists and counters
            edges_per_ego_graph.append(num_edges)
            nodes_per_ego_graph.append(num_nodes)
            ego_graph_count += 1

            # Update node histogram
            node_histogram[num_nodes] = node_histogram.get(num_nodes, 0) + 1

            # Update edge histogram
            edge_histogram[num_edges] = edge_histogram.get(num_edges, 0) + 1

    # Calculate and print averages and standard deviations
    average_edges = statistics.mean(edges_per_ego_graph)
    std_edges = statistics.stdev(edges_per_ego_graph)
    average_nodes = statistics.mean(nodes_per_ego_graph)
    std_nodes = statistics.stdev(nodes_per_ego_graph)

    print(f"Average number of edges in {no_of_hops}-hop subgraphs: {average_edges:.2f} with a std dev of {std_edges:.2f}")
    print(f"Average number of nodes in {no_of_hops}-hop subgraphs: {average_nodes:.2f} with a std dev of {std_nodes:.2f}")

    # Plot histogram of node distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(node_histogram.keys(), node_histogram.values(), color='b')
    plt.xlabel('Number of Nodes in Ego Graph')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Nodes in {no_of_hops}-Hop Ego Graphs')

    # Plot histogram of edge distribution
    plt.subplot(1, 2, 2)
    plt.bar(edge_histogram.keys(), edge_histogram.values(), color='r')
    plt.xlabel('Number of Edges in Ego Graph')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Edges in {no_of_hops}-Hop Ego Graphs')
    plt.tight_layout()
    plt.show()

    return {
        'average_edges': average_edges,
        'std_edges': std_edges,
        'average_nodes': average_nodes,
        'std_nodes': std_nodes,
        'node_histogram': node_histogram,
        'edge_histogram': edge_histogram
    }

data_dir = './data'
dataset_name = 'cora'
#-- params -- 
no_of_hops = 2
random.seed(10)
no_of_ego_graphs = 10
max_nodes_in_egograph = 10
max_edges_in_egograph = 100
#-------------

dataset = load_dataset(data_dir, dataset_name)
data = dataset[0]
# Convert the PyG graph to NetworkX graph
nx_graph = to_networkx(data, to_undirected=True)

print("Dataset name : ", dataset_name)
dict_ret = calc_average_nodes_edges_khop_subgraphs_dist(nx_graph, no_of_hops)



"""
# Calculate all possible k1 and k2 values for a maximum of 20 nodes in the ego graph
total_nodes = 20
k1_values, k2_values = calculate_all_k_values(total_nodes)

print("Possible combinations of k1 and k2:")
for k1, k2 in zip(k1_values, k2_values):
    print(f"k1 (1-hop neighbors): {k1}, k2 (2-hop neighbors): {k2}")
"""




