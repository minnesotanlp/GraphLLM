
import os
import time
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
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_networkx

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

def print_dataset_stats(dataset):
    data = dataset[0]
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


def generate_graphlist(num_nodes_to_sample,no_of_hops,data):
    # stores labels of each sub graph --> center node : {node: label}, ..
    y_labels_dict= {}
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
        y_labels_dict[center_node] = {}  # Initialize dictionary for this center node
        for node in sampled_subgraph.nodes():
            y_labels_dict[center_node][node] = data.y[node].item()  # Store y label
        graph_list.append(sampled_subgraph)
    return y_labels_dict, nx_ids, graph_list

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

def generate_node_label_dict(graph, node_with_question_mark, center_node, y_labels_dict): 
    ground_truth =  y_labels_dict[center_node][node_with_question_mark]
    node_label_dict= {} # node: label
    for node in graph.nodes():
        if node == node_with_question_mark:
            label = "?"
        else:
            label = y_labels_dict[center_node][node]  # Extract node label
        node_label_dict[node]=label
    return ground_truth, node_label_dict
        
def generate_text_for_prompt(i, nx_ids, graph, y_labels_dict, edge_text_flag):
    text = ""
    ground_truth = ""
    text+= f"Ego Graph {i+1} Attributes"+"\n"
    center_node = nx_ids[i]
    edge_list = generate_edgelist(graph)

    if edge_text_flag:
        edge_list_converted = generate_textual_edgelist(edge_list)
        text+="Edgelist: "+str(edge_list_converted)+"\n"
    else:
        text+="Edgelist: "+str(edge_list)+"\n"

    # Randomly choose a node to have a "?" label
    node_with_question_mark = random.choice(list(graph.nodes()))

    ground_truth, node_label_dict = generate_node_label_dict(graph, node_with_question_mark, center_node, y_labels_dict)

    text+=f"Node to Label Mapping : "+"\n"
    for node in node_label_dict:
        text+=f"Node {node}: Label {node_label_dict[node]}| "
    return text, node_with_question_mark, ground_truth


def parse_response(response, delimiter):
    try:
        start_index = response.index(delimiter) + len(delimiter)
        value = response[start_index:].strip()
        if '?' in value :
            return '?'
        else:
            return value
    except ValueError:
        return None


def compute_accuracy(csv_filename):
    total_count = 0
    correct_count = 0
    fail_count = 0
    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            total_count += 1
            ground_truth = row['GroundTruth']
            parsed_value = row['Parsed Value']
            if ground_truth == parsed_value:
                correct_count += 1
            if parsed_value == '?':
                fail_count+=1


    if total_count == 0:
        return 0
    else:
        accuracy = correct_count / total_count
        failures = fail_count / total_count
        return accuracy, failures



if __name__== '__main__':  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    # Load the OGBN_ARXIV dataset
    dataset = PygNodePropPredDataset(root=data_dir, name='ogbn-arxiv')
    data = dataset[0]
    #print stats about the dataset
    #print_dataset_stats(dataset)
    #lets print the label distribution
    #plot_label_distribution(data)

    #sample nodes and create the prompt for gpt 3.5-turbo
    random.seed(10)
    openai.api_key = os.environ["OPENAI_API_UMNKEY"]

    # ---- PARAMS --- #
    NO_OF_HOPS = [1,2]
    USE_EDGE_TEXT = [False]
    NO_OF_SAMPLED_NODES = [10]
    # ------------------
    
    rate_limit_pause = 1.2 # london uses 1.2; says we can use 1.0
    edge_format = "edgelist"
    if USE_EDGE_TEXT ==True:
        edge_format = "edgetext"
    
    for hops in NO_OF_HOPS:
        for use_edge in USE_EDGE_TEXT:
            for sampled_nodes in NO_OF_SAMPLED_NODES:
                print("****Starting generation for :", hops, " hops, ",use_edge, " using edgetext, ", sampled_nodes, " sample nodes****")
                filename = "./results/arxiv/ax_"+str(NO_OF_SAMPLED_NODES)+"nodes_"+str(NO_OF_HOPS)+"hop_"+edge_format+".csv"
                # get the y labels and the graph list (in this dataset we need to access the y labels in a special way)
                y_labels_dict, nx_ids, graph_list = generate_graphlist(sampled_nodes,hops,data)
                with open(filename,'w') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['GroundTruth', 'Parsed Value', 'Prompt', 'Response'])
                    error_count = 0
                    token_err_count = 0
                    for i, graph in enumerate(graph_list):
                        text, node_with_question_mark, ground_truth = generate_text_for_prompt(i, nx_ids, graph, y_labels_dict, use_edge)
                        prompt = f"""
                        Task : Node Label Prediction (Predict the label of the node marked with a ?, in the format "Label of Node = " : <predicted label>) given the edge connectivity and label information in the text enclosed in triple backticks.
                        ```{text}```
                        """
                        try:
                            response = get_completion(prompt)
                        except Exception as e:
                            error_count+=1
                            if error_count>5:
                                if isinstance(e, openai.error.RateLimitError):
                                    raise Exception("Rate limit exceeded too many times.") from e
                                elif isinstance(e, openai.error.ServiceUnavailableError):
                                    raise Exception("Service unavailable too many times.") from e
                                else:
                                    raise e
                        
                            if isinstance(e, openai.error.RateLimitError):
                                print(f"Rate limit exceeded. Pausing for {rate_limit_pause} seconds.")
                            elif isinstance(e, openai.error.ServiceUnavailableError):
                                print(f"Service unavailable; you likely paused and resumed. Pausing on our own for {rate_limit_pause} seconds to help reset things and then retrying.")
                            elif isinstance(e, openai.error.InvalidRequestError):
                                token_err_count+=1
                                print("Prompt tokens > context limit of 4097")
                                print(e)
                                #continue
                            else:
                                print(f"Type of error: {type(e)}")
                                print(f"Error: {e}")
                                print(f"Pausing for {rate_limit_pause} seconds.")
                            time.sleep(rate_limit_pause)
                            continue
                        
                        #print(text)

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
                accuracy, failures = compute_accuracy(filename)
                print(f"% of times LLM prompt was too large: {token_err_count/sampled_nodes}")
                print(f"Accuracy: {accuracy:.2%}")
                print(f"No of times LLM failed to predict a label: {failures:.2%}")