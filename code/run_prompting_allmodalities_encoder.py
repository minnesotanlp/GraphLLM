import networkx as nx
import numpy as np
import openai
import random
import json
import time
import csv
import base64
import os
from torch_geometric.utils import to_networkx
from utils import load_dataset, save_response, create_log_dir
from prompt_generation import generate_all_modality_encoder, get_image_completion_json
from response_parser import parse_response
from metrics import is_failure,is_accurate, get_token_limit_fraction
from graph_assays import count_star_graphs, count_triangles_nx, find_3_cliques_connected_to_node, get_star_motifs_connected_to_node


def generate_assays(G, node_with_question_mark):
    assay_dictionary = dict()
    assay_dictionary['No of star motifs'] = count_star_graphs(G)
    assay_dictionary['No of triangle motifs'] = count_triangles_nx(G)
    assay_dictionary['Triangle motifs attached to ? node'] = find_3_cliques_connected_to_node(G, node_with_question_mark)
    assay_dictionary['Star motifs connected to ? node'] = get_star_motifs_connected_to_node(G, node_with_question_mark)
    return assay_dictionary


def process_text(prompt, base64_image, detail, ground_truth, node_with_question_mark, log_dir, log_sub_dir, model, rate_limit_pause):
    error_count = 0
    while error_count <= 3: #this is no of attempts per prompt
        try:
            response_json = get_image_completion_json(prompt, model, base64_image, detail=detail)
            save_response(response_json, log_dir, log_sub_dir)
            usage = int(response_json.usage.total_tokens)
            response = response_json.choices[0].message["content"]
            break  # If the call was successful, break out of the loop
        except Exception as e:
            error_count = handle_openai_errors(e, error_count, rate_limit_pause, model)
            if error_count == -1:  # If error_count returned as -1, stop trying
                return error_count, 0, 0, None, None, None

    # If error_count has exceeded 3, this point would only be reached if the last attempt also failed
    if error_count > 3:
        return error_count, 0, 0, None, None, None

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
    token_labels = get_token_limit_fraction(usage, model)
    return error_count, accurate_labels, failure_labels, prompt, response, parsed_value, token_labels




def create_result_location(result_location):
    os.makedirs(result_location, exist_ok=True)

def handle_openai_errors(e, error_count, rate_limit_pause, model):
    error_count += 1
    if error_count > 3:
      if isinstance(e, openai.error.RateLimitError):
          raise Exception("Rate limit exceeded too many times.") from e
      elif isinstance(e, openai.error.ServiceUnavailableError):
          raise Exception("Service unavailable too many times.") from e
      else:
          raise e
      return -1

    if isinstance(e, openai.error.RateLimitError):
        print(f"Rate limit exceeded. Pausing for {rate_limit_pause} seconds.")
    elif isinstance(e, openai.error.ServiceUnavailableError):
        print(f"Service unavailable; Pausing for {rate_limit_pause} seconds to help reset things and then retrying.")
    elif isinstance(e, openai.error.InvalidRequestError):
        print(f'Prompt tokens > context limit of {model}.')
    else:
        print(f"Type of error: {type(e)}. Error: {e}")

    print(f"Pausing for {rate_limit_pause} seconds before retrying.")
    time.sleep(rate_limit_pause)  # Pausing before retrying

    return error_count

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

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

def run_experiment(input_location, no_of_samples, no_of_runs, setting, log_dir, log_sub_dir, model, rate_limit_pause, detail):
    avg_accuracy_values = []
    avg_failure_values = []
    avg_inaccuracy_values = []
    avg_token_limit_fraction = []

    for run in range(0,no_of_runs): 
        run_location = os.path.join(input_location, f'run_{run}')
        error_count = 0
        accurate_labels = 0
        failure_labels = 0  
        token_limit_fraction = 0  
        token_limit_fractions = []  
        # get the ground truth labels for the graphs in the setting
        ground_truth_filename = f'{setting}_run_{run}_graph_image_values.csv'
        result_filename = f'{setting}_run_{run}_{setting}_text+motif+image_encoder_results.csv'
        ground_truth_info = extract_columns_from_csv_dict(run_location, ground_truth_filename)
        graph_info_location = os.path.join(run_location, f'{setting}')
        with open(f"{run_location}/{result_filename}", mode='w') as result_file:
            csvwriter = csv.writer(result_file)
            csvwriter.writerow(['setting', 'run', 'graph_id','node_with_question_mark', 'ground_truth', 'prompt', 'response', 'parsed_response', 'token_limit_fraction'])
            for graph in ground_truth_info:
                graph_id = graph['graph_id']
                ground_truth = graph['label']
                node_with_question_mark = str(graph['ques_node_id'])
                # Constructing the filename based on the graph_id
                edgelist_name = f"{graph_info_location}/{graph_id}_edgelist.txt"
                ylabelsjson_name = f"{graph_info_location}/{graph_id}_ylabels.json"
                G = load_edgelist(edgelist_name)
                y_labels = load_graph_node_json(ylabelsjson_name)
                assay_dictionary = generate_assays(G, node_with_question_mark)
                image_path = f"{graph_info_location}/{graph_id}_new.png"
                base64_image = encode_image(image_path)
                
                text_for_prompt = generate_all_modality_encoder(G, y_labels, assay_dictionary, node_with_question_mark)
                #prompt the LLM for prediction
                error_count, acc, fail, prompt, response, parsed_response, token_limit_fraction = process_text(text_for_prompt, base64_image, detail, ground_truth, node_with_question_mark, log_dir, log_sub_dir, model, rate_limit_pause)

              
                
                csvwriter.writerow([setting, run, graph_id, node_with_question_mark, ground_truth, prompt, response, parsed_response, token_limit_fraction])
                #check if the parsed prediction is correct compared to ground truth
                accurate_labels += acc
                failure_labels += fail
                token_limit_fractions.append(token_limit_fraction) # for a single graph
            #compute the accuracy, inaccuracy and failure metrics
            accuracy = accurate_labels / no_of_samples
            failure = 0 if accuracy == 1.0 else failure_labels / (no_of_samples)
            inaccuracy = 1 - (accuracy + failure)

            avg_accuracy_values.append(accuracy)
            avg_inaccuracy_values.append(inaccuracy)
            avg_failure_values.append(failure)
            avg_token_limit_fraction.append(np.mean(token_limit_fractions)) # to calculate across all runs 
    return avg_accuracy_values, avg_inaccuracy_values, avg_failure_values, avg_token_limit_fraction




                    
            
openai.api_key = os.environ["OPENAI_KEY"] # organization api key

with open('code/config/config_all_encoder.json', 'r') as config_file:
    config = json.load(config_file)
dataset_name = config["dataset_name"]
input_location = config["input_location"]
no_of_samples = config["no_of_samples"]   
no_of_runs = config["no_of_runs"]
settings = config["settings"]
log_dir = config["log_dir"]
model = config["model"]
rate_limit_pause = config["rate_limit_pause"]
detail = config["detail"]
log_sub_dir = create_log_dir(log_dir)
input_location += f'{dataset_name}/graph_images/sample_size_{no_of_samples}/'

def main():
    with open(f"{input_location}/text+motif+image_encoder_across_runs_metrics.csv", mode='w') as metrics_file:
            csvwriterf = csv.writer(metrics_file)
            csvwriterf.writerow(['setting', 'mean_accuracy', 'std_accuracy', 'mean_inaccuracy', 'std_inaccuracy', 'mean_failure', 'std_failure', 'mean_token_limit_fraction', 'std_token_limit_fraction'])
            for setting in settings:
                avg_accuracy_runs, avg_inaccuracy_runs, avg_failure_runs, avg_tokenfraction_runs = run_experiment(input_location, no_of_samples, no_of_runs, setting, log_dir, log_sub_dir, model, rate_limit_pause, detail)
                # write the per run results to a csv file
                csvwriterf.writerow([setting, np.mean(avg_accuracy_runs), np.std(avg_accuracy_runs), np.mean(avg_inaccuracy_runs), np.std(avg_inaccuracy_runs), np.mean(avg_failure_runs), np.std(avg_failure_runs), np.mean(avg_tokenfraction_runs), np.std(avg_tokenfraction_runs)])
                print("SETTING : ", setting)
                print("Average accuracy across runs:", np.mean(avg_accuracy_runs), "Standard deviation of accuracy across runs:", np.std(avg_accuracy_runs))
                print("Average Inaccuracy across runs:", np.mean(avg_inaccuracy_runs), "Standard deviation of inaccuracy across runs:   ", np.std(avg_inaccuracy_runs))
                print("Average failure across runs:", np.mean(avg_failure_runs), "Standard deviation of failure across runs:", np.std(avg_failure_runs))
                print("Average token limit fraction across runs:", np.mean(avg_tokenfraction_runs), "Standard deviation of token fraction across runs:", np.std(avg_tokenfraction_runs))
                print("="*30)
main()


