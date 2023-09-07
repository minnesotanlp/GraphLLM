import statistics
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
from utils import draw_graph,plot_label_distribution
from metrics import compute_accuracy,record_metrics
from response_parser import parse_response

def get_completion(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


# modified such that it only has <50 nodes and <50 edges
def generate_graphlist_constrained(num_nodes_to_sample, no_of_hops, data):
    # stores labels of each sub graph --> center node : {node: label}, ..
    y_labels_dict = {}
    # List to store sampled graphs
    graph_list = []

    # Convert the PyG graph to NetworkX graph
    nx_graph = to_networkx(data, to_undirected=True)
    sampled_nodes = set()

    while len(graph_list) < num_nodes_to_sample:
        # Choose a random node index
        center_node_index = random.randint(0, data.num_nodes - 1)
        center_node = int(center_node_index)

        if center_node in sampled_nodes:
            continue

        sampled_subgraph = nx.ego_graph(nx_graph, center_node, radius=no_of_hops, undirected=True)

        # Check the size of the subgraph before adding it
        if (
            sampled_subgraph.number_of_edges() < 100
        ):
            y_labels_dict[center_node] = {}  # Initialize dictionary for this center node
            for node in sampled_subgraph.nodes():
                y_labels_dict[center_node][node] = data.y[node].item()  # Store y label
            graph_list.append(sampled_subgraph)
            sampled_nodes.add(center_node)

    # Convert the list of center nodes to integers
    nx_ids = list(y_labels_dict.keys())

    return y_labels_dict, nx_ids, graph_list


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

    # Sample and yield subgraphs one at a time
    #for center_node in nx_ids:
    #    sampled_subgraph = nx.ego_graph(nx_graph, center_node, radius=no_of_hops, undirected=True)
    #    y_labels_dict[center_node] = {}  # Initialize dictionary for this center node
    #    for node in sampled_subgraph.nodes():
    #        y_labels_dict[center_node][node] = data.y[node].item()  # Store y label
    #    yield y_labels_dict[center_node], center_node, sampled_subgraph

   
    for center_node in nx_ids:
        sampled_subgraph = nx.ego_graph(nx_graph, center_node, radius=no_of_hops, undirected=True)
        y_labels_dict[center_node] = {}  # Initialize dictionary for this center node
        for node in sampled_subgraph.nodes():
            y_labels_dict[center_node][node] = data.y[node].item()  # Store y label
        graph_list.append(sampled_subgraph)
    return y_labels_dict, nx_ids, graph_list

def edge_list_to_adjacency_list(edge_list):
    adjacency_list = {}

    for edge in edge_list:
        u, v = edge

        # Add u to v's adjacency list
        if v in adjacency_list:
            adjacency_list[v].append(u)
        else:
            adjacency_list[v] = [u]

        # Add v to u's adjacency list (assuming it's an undirected graph)
        if u in adjacency_list:
            adjacency_list[u].append(v)
        else:
            adjacency_list[u] = [v]

    return adjacency_list

def generate_edgelist(graph):
    # Print out the labels associated with a graph
    edge_list = list(graph.edges())
    return edge_list

def generate_textual_edgelist2(edge_list):           
        # Convert the edge list information into text
        edgelist_converted = ''
        for edge in edge_list:
            source, target = edge
            edgelist_converted += f'Node {source} - Node {target}. '
        return edgelist_converted

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
        
def generate_text_for_prompt(i, nx_ids, graph, y_labels_dict, edge_text_flag, adjacency_flag):
    text = ""
    ground_truth = ""
    #text+= f"Edge Connectivity Information :"+"\n"
    center_node = nx_ids[i]
    edge_list = generate_edgelist(graph)

    if edge_text_flag:
        edge_list_converted = generate_textual_edgelist(edge_list)
        text+="Edge connections (source node - target node): "+str(edge_list_converted)+"\n"
        #text+="Edge list: "+str(edge_list_converted)+"\n"
    elif adjacency_flag:
        adjacency_list = edge_list_to_adjacency_list(edge_list)
        text+="Adjacency list: "+str(adjacency_list)+"\n"
    else:
        text+="Edge list: "+str(edge_list)+"\n"

    # Randomly choose a node to have a "?" label
    node_with_question_mark = random.choice(list(graph.nodes()))

    ground_truth, node_label_dict = generate_node_label_dict(graph, node_with_question_mark, center_node, y_labels_dict)

    text+=f"Node to Label Mapping : "+"\n"
    for node in node_label_dict:
        text+=f"Node {node}: Label {node_label_dict[node]}| "
    return text, node_with_question_mark, ground_truth



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
    NO_OF_SAMPLED_NODES = [10,50]
    RUN_COUNT = 2
    USE_ADJACENCY = True # in this case USE_EDGE_TEXT has to be set to false so that it generates an edgelist first.
    #model = "gpt-3.5-turbo"
    model = "gpt-4"
    rate_limit_pause = 1.2 # calculated as per rate limit policy

    # ------------------
    # this logs all the run metrics -- this needs to be changed everytime you run it
    metrics_filename = "./results/arxiv/edge/edgetext/metrics_test.csv"
    with open(metrics_filename, 'w') as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(["no of hops", "edgetext", "sampled nodes", "mean accuracy", "SD-accuracy","mean failure fraction","SD failure fraction"," mean token err frac", "SD token frac"])
    
    # This stores all error messages - So need to rename file for every run -- remove this bit later
    csv_filename = "./results/arxiv/edge/edgetext/invalid_request_errors_test.csv"
    # Open the CSV file in append mode?
    with open(csv_filename, 'w', newline='') as csvfilei:
        csv_writer_i = csv.writer(csvfilei)
        csv_writer_i.writerow(["setting", "nodes","edges", "error", "response if present"])

        for hops in NO_OF_HOPS:
            for use_edge in USE_EDGE_TEXT:
                if use_edge==True:
                    edge_format = "edgetext"
                elif USE_ADJACENCY == True :
                    edge_format = "adjacency"
                else :
                    edge_format = "edgelist"

                for sampled_nodes in NO_OF_SAMPLED_NODES:
                    acc_list = []
                    fail_list =[]
                    token_err_list = []
                    for run_count in range(0,RUN_COUNT):
                        start_time = time.time()
                        print("Run Count : ", run_count+1)
                        print("****Starting generation for :", hops, " hops, ",edge_format, " , ", sampled_nodes, " sample nodes****")
                        filename = "./results/arxiv/edge/edgetext/ax_"+str(sampled_nodes)+"nodes_"+str(hops)+"hop_"+edge_format+"_run"+str(run_count)+".csv"

                        # get the y labels and the graph list (in this dataset we need to access the y labels in a special way)
                        y_labels_dict, nx_ids, graph_list = generate_graphlist_constrained(sampled_nodes,hops,data)

                        with open(filename,'w') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow(['GroundTruth', 'Parsed Value', 'Prompt', 'Response'])
                            error_count = 0
                            token_err_count = 0
                            for i, graph in enumerate(graph_list):
                                #print("Graph ",i)
                                text, node_with_question_mark, ground_truth = generate_text_for_prompt(i, nx_ids, graph, y_labels_dict, use_edge, USE_ADJACENCY)
                                error = ""
                                prompt = f"""
                                Task : Node Label Prediction (Predict the label of the node marked with a ?, given the edge connectivity information and node-label mapping in the text enclosed in triple backticks. Response should be in the format "Label of Node = <predicted label>". If the predicted label cannot be determined, return "Label of Node = -1") 
                                ```{text}```
                                """
                                try:
                                    response = get_completion(prompt, model)
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
                                        error = str(e)                                   
                                        csv_writer_i.writerow([f'"{(hops,edge_format,sampled_nodes)}"', f'{graph.number_of_nodes()}', f'{graph.number_of_edges()}', f'{error}'])
                                    else:
                                        print(f"Type of error: {type(e)}")
                                        print(f"Error: {e}")
                                        print(f"Pausing for {rate_limit_pause} seconds.")
                                    time.sleep(rate_limit_pause)
                                    continue
                                
                                print(text)
                                delimiter_options = ['=', ':']  # You can add more delimiters if needed
                                parsed_value = None
                                for delimiter in delimiter_options: 
                                    parsed_value = parse_response(response, delimiter) # check for better rules here!
                                    if parsed_value is not None: # general checking for the delimiter responses
                                        csv_writer.writerow([ground_truth, parsed_value, f'"{prompt}"', f'"{response}"'])
                                        break
                                    else :
                                        print("Delimiter not found in response from the LLM")
                                        csv_writer.writerow([ground_truth, parsed_value, f'"{prompt}"', f'"{response}"','delimiter not found']) # remove delimiter not found part later
                                        csv_writer_i.writerow([f'"{(hops,edge_format,sampled_nodes)}"', f'{graph.number_of_nodes()}', f'{graph.number_of_edges()}','delimiter not found', f'"{response}"'])
                                        break
                                        
                                print("RESPONSE --> ", response)
                                print("Node with ?: ", node_with_question_mark, "Label: ",ground_truth)
                                print("="*30)
                        accuracy, failure_perc = compute_accuracy(filename) # check if value based on entries is same as sample nodes
                        token_err_perc = token_err_count/sampled_nodes
                        print(f"% of times LLM prompt was too large: {token_err_perc}")
                        print(f"Accuracy: {accuracy:.2%}")
                        print(f"No of times LLM failed to predict a label: {failure_perc:.2%}")
                        acc_list.append(accuracy)
                        token_err_list.append(token_err_perc)
                        fail_list.append(failure_perc)

                        #end_time = time.time()
                        #iteration_time = end_time - start_time
                        #print(f"Iteration took {iteration_time} seconds")

                    # Record average metrics in the metrics.csv file
                    mean_accuracy = statistics.mean(acc_list)
                    std_accuracy = statistics.stdev(acc_list)
                    mean_failure = statistics.mean(fail_list)
                    std_failure = statistics.stdev(fail_list)
                    mean_token_perc = statistics.mean(token_err_list)
                    std_token_perc = statistics.stdev(token_err_list)

                    # write the average metrics out
                    record_metrics(metrics_filename, hops, use_edge, sampled_nodes, mean_accuracy, std_accuracy, mean_failure, std_failure, mean_token_perc, std_token_perc)
                        
                        
                                        