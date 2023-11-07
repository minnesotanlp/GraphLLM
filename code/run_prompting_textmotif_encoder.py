import networkx as nx
import numpy as np
import openai
import random
import json
import csv
import os
from torch_geometric.utils import to_networkx
from utils import load_dataset, save_response, create_log_dir
from prompt_generation import get_prompt_network_only, get_completion_json, generate_text_motif_encoder
from response_parser import parse_response
from metrics import is_failure,is_accurate
from graph_assays import count_star_graphs, count_triangles_nx, find_3_cliques_connected_to_node, get_star_motifs_connected_to_node

def process_text(text_for_prompt, ground_truth, node_with_question_mark, log_dir, log_sub_dir, model, rate_limit_pause):
    error_count = 0
    prompt = get_prompt_network_only(text_for_prompt, flag = 2)
    try:
        response_json = get_completion_json(prompt, model)
        save_response(response_json, log_dir, log_sub_dir)
        response = response_json.choices[0].message["content"]
    except Exception as e:
        error_count = handle_openai_errors(e, error_count, rate_limit_pause, model)
        return error_count, 0, 0

    delimiter_options = ['=', ':']
    parsed_value = None
    for delimiter in delimiter_options: 
        parsed_value = parse_response(response, delimiter)
        if parsed_value is not None:
            break

    print("RESPONSE --> ", response)
    print("Node with ?: ", node_with_question_mark, "Label: ",ground_truth)
    print("="*30)

    accurate_labels = is_accurate(parsed_value, ground_truth)
    failure_labels = is_failure(parsed_value)
    return error_count, accurate_labels, failure_labels, prompt, response, parsed_value



def create_result_location(result_location):
    os.makedirs(result_location, exist_ok=True)

def handle_openai_errors(e, error_count, rate_limit_pause, model):
    error_count += 1
    if error_count > 5:
        if isinstance(e, openai.error.RateLimitError):
            raise Exception("Rate limit exceeded too many times.") from e
        elif isinstance(e, openai.error.ServiceUnavailableError):
            raise Exception("Service unavailable too many times.") from e
        else:
            raise e

    if isinstance(e, openai.error.RateLimitError):
        print(f"Rate limit exceeded. Pausing for {rate_limit_pause} seconds.")
    elif isinstance(e, openai.error.ServiceUnavailableError):
        print(f"Service unavailable; Pausing for {rate_limit_pause} seconds to help reset things and then retrying.")
    elif isinstance(e, openai.error.InvalidRequestError):
        print(f'Prompt tokens > context limit of {model}')
        print(e)
    else:
        print(f"Type of error: {type(e)}")
        print(f"Error: {e}")
        print(f"Pausing for {rate_limit_pause} seconds.")
    return error_count


def get_desired_sizes(average_2hop_size, num_samples_per_size = 1):
    sizes = [
        int(average_2hop_size * 0.25),  
        int(average_2hop_size * 0.5),  
        average_2hop_size,             
        int(average_2hop_size * 1.5),  
        int(average_2hop_size * 2),    
    ]
    final_desired_sizes = [size for size in sizes for _ in range(num_samples_per_size)]
    return final_desired_sizes
    #[size for size in sizes if size > 0]

def load_and_prepare_data(data_dir, dataset_name):
    dataset = load_dataset(data_dir, dataset_name)
    data = dataset[0]
    graph = to_networkx(data, to_undirected=True)
    return data, graph

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

def generate_assays(G, node_with_question_mark):
    assay_dictionary = dict()
    assay_dictionary['No of star motifs'] = count_star_graphs(G)
    assay_dictionary['No of triangle motifs'] = count_triangles_nx(G)
    assay_dictionary['Triangle motifs attached to ? node'] = find_3_cliques_connected_to_node(G, node_with_question_mark)
    assay_dictionary['Star motifs connected to ? node'] = get_star_motifs_connected_to_node(G, node_with_question_mark)
    return assay_dictionary

def run_experiment(input_location, no_of_samples, no_of_runs, setting, log_dir, log_sub_dir, model, rate_limit_pause):
    avg_accuracy_values = []
    avg_failure_values = []
    avg_inaccuracy_values = []

    for run in range(0,no_of_runs): 
        run_location = os.path.join(input_location, f'run_{run}')
        error_count = 0
        accurate_labels = 0
        failure_labels = 0      
        # get the ground truth labels for the graphs in the setting
        ground_truth_filename = f'{setting}_run_{run}_graph_image_values.csv'
        result_filename = f'{setting}_run_{run}_{setting}_text_motif_encoder_results.csv'
        ground_truth_info = extract_columns_from_csv_dict(run_location, ground_truth_filename)
        graph_info_location = os.path.join(run_location, f'{setting}')
        with open(f"{run_location}/{result_filename}", mode='w') as result_file:
            csvwriter = csv.writer(result_file)
            csvwriter.writerow(['setting', 'run', 'graph_id','node_with_question_mark', 'ground_truth', 'prompt', 'response', 'parsed_response'])
            for graph in ground_truth_info:
                graph_id = graph['graph_id']
                ground_truth = graph['label']
                node_with_question_mark = str(graph['ques_node_id'])
                # Constructing the filename based on the graph_id
                edgelist_name = f"{graph_info_location}/{graph_id}_edgelist.txt"
                ylabelsjson_name = f"{graph_info_location}/{graph_id}_ylabels.json"
                G = load_edgelist(edgelist_name)
                y_labels = load_graph_node_json(ylabelsjson_name)
                print("graph id: " , graph_id, "setting: ", setting,"run: ",  run)
                print("y_labels", y_labels)
                #now generate a text prompt based on this assay information + y label information
                assay_dictionary = generate_assays(G, node_with_question_mark)
                text_for_prompt = generate_text_motif_encoder(G, assay_dictionary, y_labels, node_with_question_mark)
                
                #prompt the                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        LLM for prediction
                error_count, acc, fail, prompt, response, parsed_response = process_text(text_for_prompt, ground_truth, node_with_question_mark, log_dir, log_sub_dir, model, rate_limit_pause)
                csvwriter.writerow([setting, run, graph_id, node_with_question_mark, ground_truth, prompt, response, parsed_response])
                #check if the parsed prediction is correct compared to ground truth
                accurate_labels += acc
                failure_labels += fail
            #compute the accuracy, inaccuracy and failure metrics
            accuracy = accurate_labels / no_of_samples
            failure = 0 if accuracy == 1.0 else failure_labels / (no_of_samples)
            inaccuracy = 1 - (accuracy + failure)
            avg_accuracy_values.append(accuracy)
            avg_inaccuracy_values.append(inaccuracy)
            avg_failure_values.append(failure)
    return avg_accuracy_values, avg_inaccuracy_values, avg_failure_values

     
openai.api_key = os.environ["OPENAI_API_UMNKEY"]
with open('code/config_textmotif_encoder.json', 'r') as config_file:
    config = json.load(config_file)
dataset_name = config["dataset_name"]
input_location = config["input_location"]
no_of_samples = config["no_of_samples"]   
no_of_runs = config["no_of_runs"]
settings = config["settings"]
log_dir = config["log_dir"]
model = config["model"]
rate_limit_pause = config["rate_limit_pause"]
log_sub_dir = create_log_dir(log_dir)      

def main():
    with open(f"{input_location}/text_motif_encoder_across_runs_metrics.csv", mode='w') as metrics_file:
            csvwriterf = csv.writer(metrics_file)
            csvwriterf.writerow(['setting', 'mean_accuracy', 'std_accuracy', 'mean_inaccuracy', 'std_inaccuracy', 'mean_failure', 'std_failure'])
            for setting in settings:
                avg_accuracy_runs, avg_inaccuracy_runs, avg_failure_runs = run_experiment(input_location, no_of_samples, no_of_runs, setting, log_dir, log_sub_dir, model, rate_limit_pause)
                # write the per run results to a csv file
                csvwriterf.writerow([setting, np.mean(avg_accuracy_runs), np.std(avg_accuracy_runs), np.mean(avg_inaccuracy_runs), np.std(avg_inaccuracy_runs), np.mean(avg_failure_runs), np.std(avg_failure_runs)])
                print("SETTING : ", setting)
                print("Average accuracy across runs:", np.mean(avg_accuracy_runs), "Standard deviation of accuracy across runs:", np.std(avg_accuracy_runs))
                print("Average Inaccuracy across runs:", np.mean(avg_inaccuracy_runs), "Standard deviation of inaccuracy across runs:   ", np.std(avg_inaccuracy_runs))
                print("Average failure across runs:", np.mean(avg_failure_runs), "Standard deviation of failure across runs:", np.std(avg_failure_runs))
                print("="*30)



# Example usage:

main()


