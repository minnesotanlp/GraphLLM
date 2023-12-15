import networkx as nx
from torch_geometric.utils import to_networkx
import random
import openai
import json
import os
import csv
import numpy as np
import statistics

from metrics import is_failure,is_accurate
from utils import load_dataset, save_response, create_log_dir
from prompt_generation import get_completion_json, get_prompt, generate_textprompt_anygraph
from connection_information import get_y_labels_graph
from graph_sample_generation import generate_egograph_sample, generate_forestfire_sample
from response_parser import parse_response
from plotting import plot_graph_structure

# ---- PARAMS --- #
#-------------------------------------------------------------
random.seed(10)
openai.api_key = os.environ["OPENAI_KEY"]
# Load configuration from the JSON file
with open('code/config1.json', 'r') as config_file:
    config = json.load(config_file)
# Access parameters from the config dictionary
dataset_name = config["dataset_name"]
no_of_hops = config["NO_OF_HOPS"]
use_edge = config["USE_EDGE"]
num_samples = config["NO_OF_SAMPLED_NODES"] # no of ego graphs
no_of_runs = config["RUN_COUNT"]
USE_ADJACENCY = config["USE_ADJACENCY"]
compression = config['compression']
model = config["model"]
rate_limit_pause = config["rate_limit_pause"]
data_dir = config["data_dir"]
log_dir = config["log_dir"]
result_location = config["result_location"]

neighborhood_sampling_flag = config["neighborhood_sampling"]
average_2hop_size = 36
neighborhood_hop = 3
#------------------------------------------------
def create_result_location(result_location):
    os.makedirs(result_location, exist_ok=True)

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


def load_and_prepare_data(data_dir, dataset_name):
    dataset = load_dataset(data_dir, dataset_name)
    data = dataset[0]
    graph = to_networkx(data, to_undirected=True)
    return data, graph


def process_text(text, ground_truth, node_with_question_mark, log_dir, log_sub_dir, compression, model, rate_limit_pause):
    error_count = 0
    modification = 0 # add prompt instruction.
    prompt = get_prompt(text, compression, modification) 
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
    return 0, accurate_labels, failure_labels


def run_experiment(desired_sizes, data, graph, log_dir, log_sub_dir, compression, model, rate_limit_pause, num_samples, use_edge, USE_ADJACENCY, neighborhood_hop=2, neighborhood_sampling_flag=False):
    avg_accuracy_values_ego = []
    avg_accuracy_values_ff = []
    avg_failure_values_ego = []
    avg_failure_values_ff = []
    avg_inaccuracy_values_ego = []
    avg_inaccuracy_values_ff = []

    for run in range(3):
        error_count = 0
        accurate_labels_ego = 0
        failure_labels_ego = 0
        accurate_labels_ff = 0
        failure_labels_ff = 0
        inaccuracy_ego = []
        inaccuracy_ff = []

        for size in desired_sizes:
            ego_node, ego_graph = generate_egograph_sample(graph, no_of_hops, size)
            y_labels_egograph = get_y_labels_graph(data, ego_graph, True, ego_node=ego_node)
            ground_truth_ego = y_labels_egograph[ego_node][ego_node] #? is the ego node
            if neighborhood_sampling_flag:
                print("EG neighborhood activated!")
                # use 2 hop subgraph        
                ego_graph = nx.ego_graph(ego_graph, ego_node, radius=neighborhood_hop)
            text_ego = generate_textprompt_anygraph(ego_graph, ego_node, y_labels_egograph, ego_node, use_edge, USE_ADJACENCY, ego_flag=True)
            error_count, acc_ego, fail_ego = process_text(text_ego, ground_truth_ego, ego_node, log_dir, log_sub_dir, compression, model, rate_limit_pause)
            accurate_labels_ego += acc_ego
            failure_labels_ego += fail_ego

            ff_graph = generate_forestfire_sample(graph, size)
            y_labels_ffgraph = get_y_labels_graph(data, ff_graph, False)
            node_with_question_mark_ff = random.choice(list(ff_graph.nodes()))
            ground_truth_ff = y_labels_ffgraph[node_with_question_mark_ff]
            if neighborhood_sampling_flag:
                print("FF neighborhood activated!")
                # use 2 hop subgraph        
                ff_graph = nx.ego_graph(ff_graph, node_with_question_mark_ff, radius=neighborhood_hop)
            text_ff = generate_textprompt_anygraph(ff_graph, None, y_labels_ffgraph, node_with_question_mark_ff, use_edge, USE_ADJACENCY, ego_flag=False)
            error_count, acc_ff, fail_ff = process_text(text_ff, ground_truth_ff, node_with_question_mark_ff, log_dir, log_sub_dir, compression, model, rate_limit_pause)
            accurate_labels_ff += acc_ff
            failure_labels_ff += fail_ff

        accuracy_ego = accurate_labels_ego / num_samples
        accuracy_ff = accurate_labels_ff / num_samples
        failure_ego = 0 if accuracy_ego == 1.0 else failure_labels_ego / (num_samples)
        failure_ff = 0 if accuracy_ff == 1.0 else failure_labels_ff / (num_samples)

        inaccuracy_ego = 1 - (accuracy_ego + failure_ego)
        inaccuracy_ff = 1 - (accuracy_ff + failure_ff)

        #failure_ego = 0 if accuracy_ego == 1.0 else failure_labels_ego / (num_samples - accurate_labels_ego)
        #failure_ff = 0 if accuracy_ff == 1.0 else failure_labels_ff / (num_samples - accurate_labels_ff)

        avg_accuracy_values_ego.append(accuracy_ego)
        avg_accuracy_values_ff.append(accuracy_ff)
        avg_inaccuracy_values_ego.append(inaccuracy_ego)
        avg_inaccuracy_values_ff.append(inaccuracy_ff)
        avg_failure_values_ego.append(failure_ego)
        avg_failure_values_ff.append(failure_ff)

    return avg_accuracy_values_ego, avg_accuracy_values_ff, avg_failure_values_ego, avg_failure_values_ff, avg_inaccuracy_values_ego, avg_inaccuracy_values_ff



def main():
    log_sub_dir = create_log_dir(log_dir)
    create_result_location(result_location)
    data, graph = load_and_prepare_data(data_dir, dataset_name)
    desired_sizes = get_desired_sizes(average_2hop_size, num_samples_per_size = 4) # total sample size = 20
    
    avg_accuracy_values_ego, avg_accuracy_values_ff, avg_failure_values_ego, avg_failure_values_ff,avg_inaccuracy_values_ego, avg_inaccuracy_values_ff = run_experiment(desired_sizes, data, graph, log_dir, log_sub_dir, compression, model, rate_limit_pause, num_samples, use_edge, USE_ADJACENCY, neighborhood_hop, neighborhood_sampling_flag)

    print("Accuracy for Ego Graph", avg_accuracy_values_ego)
    print("Failure for Ego Graph", avg_failure_values_ego)
    print("InAccuracy for Ego Graph", avg_inaccuracy_values_ego)
    print("Average accuracy across runs for Ego:", np.mean(avg_accuracy_values_ego), "Standard deviation of accuracy for Ego:", np.std(avg_accuracy_values_ego))
    print("Average failure across runs for Ego:", np.mean(avg_failure_values_ego), "Standard deviation of failure for Ego:", np.std(avg_failure_values_ego))
    print("Average Inaccuracy across runs for Ego:", np.mean(avg_inaccuracy_values_ego), "Standard deviation of inaccuracy for Ego:", np.std(avg_inaccuracy_values_ego))
    
    print("Accuracy for FF Graph", avg_accuracy_values_ff)
    print("Failure for FF Graph", avg_failure_values_ff)
    print("InAccuracy for FF Graph", avg_inaccuracy_values_ff)

    print("Average accuracy across runs for FF:", np.mean(avg_accuracy_values_ff), "Standard deviation of accuracy for FF:", np.std(avg_accuracy_values_ff))
    print("Average failure across runs for FF:", np.mean(avg_failure_values_ff), "Standard deviation of failure for FF:", np.std(avg_failure_values_ff))
    print("Average Inaccuracy across runs for FF:", np.mean(avg_inaccuracy_values_ff), "Standard deviation of inaccuracy for FF:", np.std(avg_inaccuracy_values_ff))
    
            
if __name__ == "__main__":
    main()
           


        
        

        
