
import os
import csv
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
import openai
import random

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def draw_graph(edge_example):
    node_example = np.unique(edge_example.flatten())
    plt.figure(figsize=(10, 6))
    G = nx.Graph()
    G.add_nodes_from(node_example)
    G.add_edges_from(list(zip(edge_example[0], edge_example[1])))
    nx.draw_networkx(G, with_labels=False)
    plt.show()

def plot_label_distribution(data):
    counter = collections.Counter(data.y.numpy())
    counter = dict(counter)
    print(counter)
    count = [x[1] for x in sorted(counter.items())]
    plt.figure(figsize=(10, 6))
    plt.bar(range(7), count)
    plt.xlabel("class", size=20)
    plt.show()

def print_dataset_stats(dataset):
    data = dataset[0]
    print(f'Number of nodes: {data.num_nodes}')
    # Number of nodes: 2708
    print(f'Number of edges: {data.num_edges}')
    # Number of edges: 10556
    print(f'Number of features: {data.num_node_features}')
    # Number of edge features: 0
    print(f'Number of classes: {dataset.num_classes}')
    # Number of classes: 7
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')  # False
    print(f'Has self-loops: {data.has_self_loops()}')  # False
    print(f'Is undirected: {data.is_undirected()}')  # True

def load_dataset(data_dir, dataset_name):
    dataset = Planetoid(root=data_dir, name=dataset_name)
    return dataset


def generate_graphlist(num_nodes_to_sample,no_of_hops,data):
    # List to store sampled graphs
    graph_list = []
    #choose some random nodes to sample
    sampled_indices = torch.randint(0, data.num_nodes, (num_nodes_to_sample,))
    # Convert the PyG graph to NetworkX graph
    nx_graph = to_networkx(data, to_undirected=True)

    # Convert sampled indices to integers, list of center nodes
    nx_ids = [int(node_id.item()) for node_id in sampled_indices]

    # Sample first 10 nodes and create a graph centered around each node -- modify this to be random!
    for center_node in nx_ids:
        sampled_subgraph = nx.ego_graph(nx_graph, center_node, radius=no_of_hops, undirected=True)
        graph_list.append(sampled_subgraph)
    return graph_list

def generate_edgelist(graph):
    # Print out the labels associated with a graph
    edge_list = list(graph.edges())
    return edge_list

def generate_textual_edgelist(edge_list):           
        # Convert the edge list information into text
        edgelist_converted = ''
        for edge in edge_list:
            source, target = edge
            edgelist_converted += f'Node {source} is connected to Node {target}. '
        return edgelist_converted

def generate_node_label_dict(graph, node_with_question_mark): 
    node_label_dict= {} # node: label
    for node in graph.nodes():
        if node == node_with_question_mark:
            ground_truth = data.y[node].item()
            label = "?"
        else:
            label = data.y[node].item()  # Extract node label
        node_label_dict[node]=label
    return ground_truth, node_label_dict
        
def generate_text_for_prompt(i, graph, edge_text_flag):
    text = ""
    ground_truth = ""
    text+= f"Graph {i+1}"+"\n"
    edge_list = generate_edgelist(graph)
    if edge_text_flag:
        edge_list_converted = generate_textual_edgelist(edge_list)
        text+="Edgelist: "+str(edge_list_converted)+"\n"
    else:
        text+="Edgelist: "+str(edge_list)+"\n"
    # Randomly choose a node to have a "?" label
    node_with_question_mark = random.choice(list(graph.nodes()))
    ground_truth, node_label_dict = generate_node_label_dict(graph, node_with_question_mark)

    text+=f"Node to Label Mapping : "+"\n"
    for node in node_label_dict:
        text+=f"Node {node}: Label {node_label_dict[node]}| "
    return text, node_with_question_mark, ground_truth


def parse_response(response, delimiter):
    try:
        start_index = response.index(delimiter) + len(delimiter)
        value = response[start_index:].strip()
        return value
    except ValueError:
        return None


def compute_accuracy(csv_filename):
    total_count = 0
    correct_count = 0

    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            total_count += 1
            ground_truth = row['GroundTruth']
            parsed_value = row['Parsed Value']
            if ground_truth == parsed_value:
                correct_count += 1

    if total_count == 0:
        return 0
    else:
        accuracy = correct_count / total_count
        return accuracy

if __name__== '__main__':  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    dataset = load_dataset(data_dir,'Cora')
    #print_dataset_stats(dataset)
    data = dataset[0]
    #lets print the label distribution
    #plot_label_distribution(data)

    #lets print a subgraph
    #edge_index = data.edge_index.numpy()
    #print(edge_index.shape)
    #edge_example = edge_index[:, np.where(edge_index[0]==30)[0]]
    #draw_graph(edge_example)

    #sample nodes and create the prompt for gpt 3.5-turbo
    random.seed(10)
    openai.api_key = os.environ["OPENAI_API_UMNKEY"]
    filename = "./results/cora_100nodes_1hop_edgelist.csv"
    no_of_hops = 1
    use_edge_text = False
    no_of_sampled_nodes = 100
    graph_list = generate_graphlist(no_of_sampled_nodes,no_of_hops,data)
    with open(filename,'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['GroundTruth', 'Parsed Value', 'Prompt', 'Response'])
        for i, graph in enumerate(graph_list):
            text, node_with_question_mark, ground_truth = generate_text_for_prompt(i, graph,use_edge_text)
            prompt = f"""
            Task : Node Label Prediction (Predict the label of the node marked with a ?, in the format "Label of Node = " : <predicted label>) given the edge connectivity and label information in the text enclosed in triple backticks.
            ```{text}```
            """
            response = get_completion(prompt)
            print(text)

            delimiter_options = ['=', ':']  # You can add more delimiters if needed
            parsed_value = None

            for delimiter in delimiter_options:
                parsed_value = parse_response(response, delimiter)
                if parsed_value is not None:
                    csv_writer.writerow([ground_truth, parsed_value, f'"{prompt}"', f'"{response}"'])
                    break

            print("RESPONSE --> ", response)
            print("Node with ?: ", node_with_question_mark, "Label: ",ground_truth)
            print("="*30)
    accuracy = compute_accuracy(filename)
    print(f"Accuracy: {accuracy:.2%}")

    
    