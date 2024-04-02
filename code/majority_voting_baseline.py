import networkx as nx
import numpy as np
import openai
import random
import json
import time
import csv
import os
from torch_geometric.utils import to_networkx
from utils import load_dataset, save_response, create_log_dir
from metrics import is_failure,is_accurate, get_token_limit_fraction
from collections import Counter

def majority_voting(neighbors, y_labels):
    labels = [y_labels[n] for n in neighbors if n in y_labels]
    if labels:
        most_common_label = Counter(labels).most_common(1)[0][0]
    else:
        most_common_label = None  # Handle the case where a node has no labeled neighbors
    return most_common_label

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
                'label': row['label']
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



def run_experiment(input_location, no_of_samples, no_of_runs, setting, log_dir, log_sub_dir):
    avg_accuracy_values = []
    avg_failure_values = []
    avg_inaccuracy_values = []
    avg_token_limit_fraction = []

    for run in range(no_of_runs):
        run_location = os.path.join(input_location, f'run_{run}')
        accurate_labels = 0
        failure_labels = 0

        ground_truth_filename = f'{setting}_run_{run}_graph_image_values.csv'
        ground_truth_info = extract_columns_from_csv_dict(run_location, ground_truth_filename)
        graph_info_location = os.path.join(run_location, f'{setting}')

        for graph in ground_truth_info:
            graph_id = graph['graph_id']
            ground_truth = graph['label']
            print("=====================================")
            print("ground_truth: ", ground_truth)
            node_with_question_mark = str(graph['ques_node_id'])
            edgelist_name = f"{graph_info_location}/{graph_id}_edgelist.txt"
            ylabelsjson_name = f"{graph_info_location}/{graph_id}_ylabels.json"

            G = load_edgelist(edgelist_name)
            y_labels = load_graph_node_json(ylabelsjson_name)

            neighbors = list(G.neighbors(node_with_question_mark))
            print("neighbors: ", neighbors)
            predicted_label = majority_voting(neighbors, y_labels)
            print("predicted_label: ", predicted_label)

            if int(predicted_label) == int(ground_truth):
                print("Accurate")
                accurate_labels += 1
            else:
                failure_labels += 1
                print("Failure")
            print("=====================================")
        # Calculate accuracy, inaccuracy, and failure rates for this run
        accuracy = accurate_labels / no_of_samples
        inaccuracy = failure_labels / no_of_samples # because here there is no notion of -1
        failure = 1 - (accuracy + inaccuracy)

        avg_accuracy_values.append(accuracy)
        avg_failure_values.append(failure)
        avg_inaccuracy_values.append(inaccuracy)

    return avg_accuracy_values, avg_inaccuracy_values, avg_failure_values




                    
            

with open('code/config/config_textencoder.json', 'r') as config_file:
    config = json.load(config_file)
dataset_name = config["dataset_name"]
input_location = config["input_location"]
no_of_samples = config["no_of_samples"]   
no_of_runs = config["no_of_runs"]
settings = config["settings"]
log_dir = config["log_dir"]

#model = config["model"]
#rate_limit_pause = config["rate_limit_pause"]
log_sub_dir = create_log_dir(log_dir)      
input_location += f'{dataset_name}/graph_images/sample_size_{no_of_samples}/'

def main():
    with open(f"{input_location}/majority_voting_across_runs_metrics.csv", mode='w') as metrics_file:
            csvwriterf = csv.writer(metrics_file)
            csvwriterf.writerow(['setting', 'mean_accuracy', 'std_accuracy', 'mean_inaccuracy', 'std_inaccuracy', 'mean_failure', 'std_failure'])
            for setting in settings:
                avg_accuracy_runs, avg_inaccuracy_runs, avg_failure_runs = run_experiment(input_location, no_of_samples, no_of_runs, setting, log_dir, log_sub_dir)
    
                # write the per run results to a csv file
                csvwriterf.writerow([setting, np.mean(avg_accuracy_runs), np.std(avg_accuracy_runs), np.mean(avg_inaccuracy_runs), np.std(avg_inaccuracy_runs), np.mean(avg_failure_runs), np.std(avg_failure_runs)])
                print("SETTING : ", setting)
                print("Average accuracy across runs:", np.mean(avg_accuracy_runs), "Standard deviation of accuracy across runs:", np.std(avg_accuracy_runs))
                print("Average Inaccuracy across runs:", np.mean(avg_inaccuracy_runs), "Standard deviation of inaccuracy across runs:   ", np.std(avg_inaccuracy_runs))
                print("Average failure across runs:", np.mean(avg_failure_runs), "Standard deviation of failure across runs:", np.std(avg_failure_runs))
    
                print("="*30)
main()


