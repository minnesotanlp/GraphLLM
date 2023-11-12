import networkx as nx
import numpy as np
import random
import json
import csv
import os
from torch_geometric.utils import to_networkx
from plotting import plot_graph_structure_community_colored


def create_result_location(result_location):
    os.makedirs(result_location, exist_ok=True)

def extract_columns_from_csv_dict(run_location, filename):
    extracted_data = []
    
    with open(f"{run_location}/{filename}", mode='r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            extracted_data.append({
                'graph_id': row['graph_id'],
                'ques_node_id': row['ques_node_id'],
                'label': row['label'],
                'sample_size': row['sample_size']
            })
            
    return extracted_data

def load_graph_node_json(json_file_path):
    """
    Load a JSON file containing node information and return it as a dictionary.
    
    :param json_file_path: str, the full path of the JSON file
    :return: dict, containing the loaded node information
    """
    try:
        # Open and load the JSON file
        with open(json_file_path, 'r') as file:
            node_data = json.load(file)
        
        # Returning the loaded data as a dictionary
        return node_data
    
    except FileNotFoundError:
        print(f"JSON file not found at path: {json_file_path}")
        return None
    
    except json.JSONDecodeError:
        print(f"Error decoding JSON file at path: {json_file_path}")
        return None


def load_edgelist(filename):
    #node_counts = 0    
    try:
        # Reading the edgelist and creating a graph
        G = nx.read_edgelist(filename)
        
        # Calculating the number of nodes
        #node_counts = len(G.nodes())
        
    except FileNotFoundError:
        print(f"Graph File not found.")
    
    return G



def create_graphs(input_location, setting, no_of_runs):
    ego_flag = False
    for run in range(0,no_of_runs): 
        run_location = os.path.join(input_location, f'run_{run}')
        # get the ground truth labels for the graphs in the setting
        ground_truth_filename = f'{setting}_run_{run}_graph_image_values.csv'
        ground_truth_info = extract_columns_from_csv_dict(run_location, ground_truth_filename)
        graph_info_location = os.path.join(run_location, f'{setting}')
        for graph in ground_truth_info:
            graph_id = graph['graph_id']
            size = graph['sample_size']
            ground_truth = graph['label']
            node_with_question_mark = str(graph['ques_node_id'])
            # Constructing the filename based on the graph_id
            edgelist_name = f"{graph_info_location}/{graph_id}_edgelist.txt"
            ylabelsjson_name = f"{graph_info_location}/{graph_id}_ylabels.json"
            G = load_edgelist(edgelist_name)
            y_labels = load_graph_node_json(ylabelsjson_name)
            # generate the image for the graph
            plot_graph_structure_community_colored(G, y_labels, node_with_question_mark, graph_id, f'{setting}_graphsize_{size}', graph_info_location, ego_flag=ego_flag)
            print(f'Image ID {graph_id} generated for setting {setting} and run {run}')
       
                    

def main():
    settings = ["ego", "ff"]
    no_of_runs = 3
    no_of_samples = 50
    dataset_name = "pubmed"
    input_location = f"./results/{dataset_name}/graph_images/sample_size_{no_of_samples}/"
    for setting in settings:
        create_graphs(input_location, setting, no_of_runs)
main()


