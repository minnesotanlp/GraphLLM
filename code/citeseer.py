import os
import time
import statistics
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
from utils import draw_graph,plot_label_distribution,print_dataset_stats
from metrics import compute_accuracy,record_metrics
from response_parser import parse_response
from connection_information import generate_edgelist,generate_node_label_dict,generate_textual_edgelist,generate_textual_edgelist2,generate_graphlist,generate_graphlist_constrained,edge_list_to_adjacency_list
from prompt_generation import get_completion,generate_text_for_prompt

def load_dataset(data_dir, dataset_name):
    dataset = Planetoid(root=data_dir, name=dataset_name)
    return dataset


if __name__== '__main__':  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    dataset = load_dataset(data_dir,'Citeseer')
    # print_dataset_stats(dataset)
    data = dataset[0]

    #lets print the label distribution
    # plot_label_distribution(data)

    #lets print a subgraph
    # edge_index = data.edge_index.numpy()
    # print(edge_index.shape)
    # edge_example = edge_index[:, np.where(edge_index[0]==30)[0]]
    # draw_graph(edge_example)

    #sample nodes and create the prompt for gpt 3.5-turbo
    random.seed(10)
    openai.api_key = os.environ["OPENAI_API_UMNKEY"]

    # ---- PARAMS --- #
    NO_OF_HOPS = [2]
    USE_EDGE_TEXT = [False]
    NO_OF_SAMPLED_NODES = [50]
    RUN_COUNT = 2
    USE_ADJACENCY = True # in this case USE_EDGE_TEXT has to be set to false so that it generates an edgelist first.
    #model = "gpt-3.5-turbo"
    model = "gpt-4"
    rate_limit_pause = 1.2 # calculated as per rate limit policy

    # ------------------
    # this logs all the run metrics -- this needs to be changed everytime you run it
    metrics_filename = "./results/citeseer/edge/edge_connection/metrics_test2.csv"
    with open(metrics_filename, 'w') as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(["no of hops", "edgetext", "sampled nodes", "mean accuracy", "SD-accuracy","mean failure fraction","SD failure fraction"," mean token err frac", "SD token frac"])
    
    # This stores all error messages - So need to rename file for every run -- remove this bit later
    csv_filename = "./results/citeseer/edge/edge_connection/invalid_request_errors_test.csv"
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
                        filename = "./results/citeseer/edge/edge_connection/ci_"+str(sampled_nodes)+"nodes_"+str(hops)+"hop_"+edge_format+"_run"+str(run_count)+".csv"

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
                                Task : Node Label Prediction (Predict the label of the node marked with a ?, given the adjacency list information as a dictionary of type "node:node neighborhood"
                                and node-label mapping in the text enclosed in triple backticks. Response should be in the format "Label of Node = <predicted label>". If the predicted label cannot be determined, return "Label of Node = -1") 
                                ```{text}```
                                """
                                #prompt = f"""
                                #Task : Node Label Prediction (Predict the label of the node marked with a ?, given the edge connectivity information and node-label mapping in the text enclosed in triple backticks. Response should be in the format "Label of Node = <predicted label>". If the predicted label cannot be determined, return "Label of Node = -1") 
                                #```{text}```
                                #"""
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
                        
                        
                                        