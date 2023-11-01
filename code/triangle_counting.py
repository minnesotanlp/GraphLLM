import networkx as nx
import numpy as np
import openai
import random
import json
import csv
import os
from torch_geometric.utils import to_networkx
from utils import load_dataset, save_response, create_log_dir
from connection_information import get_y_labels_graph
from graph_sample_generation import generate_egograph_sample, generate_forestfire_sample
from plotting import plot_graph_structure_community_colored
from prompt_generation import get_completion_json, get_prompt_assays, generate_textprompt_anygraph, generate_textprompt_anygraph_labelmaps
from response_parser import parse_response
from metrics import is_failure,is_accurate
from graph_assays import count_triangles_nx, find_3_cliques_connected_to_node, count_star_graphs

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


def process_text(text, assay_info, ground_truth, node_with_question_mark, log_dir, log_sub_dir, model, rate_limit_pause):
    error_count = 0
    default_flag = False #make it false for assays
    connection_flag = True # make it true if you want to use the adjacency list information
    prompt = get_prompt_assays(text, assay_info, default_flag, connection_flag)
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
    return 0, accurate_labels, failure_labels, prompt, response, parsed_value


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



def run_experiment(desired_sizes, no_of_runs, no_of_hops, data, graph, og_result_location, log_dir, log_sub_dir, model, rate_limit_pause,  use_edge, USE_ADJACENCY, neighborhood_hop=2, neighborhood_sampling_flag=False):
    avg_accuracy_values_ego = []
    avg_failure_values_ego = []
    avg_inaccuracy_values_ego = []
    avg_accuracy_values_ff = []
    avg_failure_values_ff = []
    avg_inaccuracy_values_ff = []

    for run in range(0, no_of_runs):
        error_count = 0
        accurate_labels_ego = 0
        failure_labels_ego = 0
        inaccuracy_ego = 0
        accurate_labels_ff = 0
        failure_labels_ff = 0
        inaccuracy_ff = 0
        result_location = os.path.join(og_result_location, f'run_{run}')
        create_result_location(result_location)  
        with open(os.path.join(result_location, f'ego_run_{run}_graph_assay_values.csv'), 'w') as csvfile, open(os.path.join(result_location, f'ff_run_{run}_graph_assay_values.csv'), 'w') as csvfile2:
            csvwriter = csv.writer(csvfile)
            csvwriter2 = csv.writer(csvfile2)
            csvwriter.writerow(['graph_id', 'size', 'node_id', 'ground_label', 'prompt', 'response', 'parsed_value'])
            csvwriter2.writerow(['graph_id', 'size', 'node_id', 'ground_label','prompt', 'response', 'parsed_value'])
            for index, size in enumerate(desired_sizes):
                
                ego_node, ego_graph = generate_egograph_sample(graph, no_of_hops, size)
                y_labels_egograph = get_y_labels_graph(data, ego_graph, ego_flag=True, ego_node=ego_node)
                node_with_question_mark = ego_node
                ground_truth_ego = y_labels_egograph[ego_node][node_with_question_mark] #? is the ego node
                if neighborhood_sampling_flag: # use only the 2 hop neighborhood -- go change config
                    # use 2 hop subgraph        
                    ego_graph = nx.ego_graph(ego_graph, ego_node, radius=neighborhood_hop)
                print(f"ego graph size: {len(ego_graph)}")

                #text_ego = generate_textprompt_anygraph_labelmaps(ego_graph, ego_node, y_labels_egograph, ego_node, use_edge, USE_ADJACENCY, ego_flag=True)
                text_ego = generate_textprompt_anygraph(ego_graph, ego_node, y_labels_egograph, ego_node, use_edge, USE_ADJACENCY, ego_flag=True)
                plot_graph_structure_community_colored(ego_graph, y_labels_egograph, node_with_question_mark, size, f'ego_{index}_graphsize_', result_location, ego_flag=True)
                
                assay_info = f'Number of triangles in the ego graph = {count_triangles_nx(ego_graph)}.'
                # Find and print all 3-cliques connected to node ?
                #connected_3_cliques = find_3_cliques_connected_to_node(ego_graph, node_with_question_mark)
                #assay_info += f"Triangles connected to node {node_with_question_mark}: {connected_3_cliques}"

                assay_info += f'Number of stars in the ego graph = {count_star_graphs(ego_graph)}.'
                

                error_count, acc_ego, fail_ego, prompt, response, parsed_response = process_text(text_ego, assay_info, ground_truth_ego, node_with_question_mark, log_dir, log_sub_dir, model, rate_limit_pause)
                csvwriter.writerow([index, size, node_with_question_mark, ground_truth_ego, prompt, response, parsed_response])
                accurate_labels_ego += acc_ego
                failure_labels_ego += fail_ego
                

                # Forest Fire samples
                ff_graph = generate_forestfire_sample(graph, size)
                y_labels_ffgraph = get_y_labels_graph(data, ff_graph, ego_flag=False)
                node_with_question_mark_ff = random.choice(list(ff_graph.nodes()))
                ground_truth_ff = y_labels_ffgraph[node_with_question_mark_ff]
                if neighborhood_sampling_flag:
                    # use 2 hop subgraph        
                    ff_graph = nx.ego_graph(ff_graph, node_with_question_mark_ff, radius=neighborhood_hop)
                print(f"ff graph size: {len(ff_graph)}")
                #print("LABELS " , y_labels_ffgraph)
                
        
                #text_ff = generate_textprompt_anygraph_labelmaps(ff_graph, node_with_question_mark_ff, y_labels_ffgraph, node_with_question_mark_ff, use_edge, USE_ADJACENCY, ego_flag=False)
                text_ff = generate_textprompt_anygraph(ff_graph, node_with_question_mark_ff, y_labels_ffgraph, node_with_question_mark_ff, use_edge, USE_ADJACENCY, ego_flag=False)
                plot_graph_structure_community_colored(ff_graph, y_labels_ffgraph, node_with_question_mark_ff, size, f'ff_{index}_graphsize_', result_location, ego_flag=False)
               
                #assay_info = f'Number of triangles in the graph = {count_triangles_nx(ff_graph)}'
                #connected_3_cliques = find_3_cliques_connected_to_node(ff_graph, node_with_question_mark_ff)
                #assay_info += f"triangles connected to node {node_with_question_mark_ff}: {connected_3_cliques}"
                assay_info = f'Number of stars in the graph = {count_star_graphs(ff_graph)}.'

                error_count, acc_ff, fail_ff, prompt, response, parsed_response = process_text(text_ff, assay_info, ground_truth_ff, node_with_question_mark_ff, log_dir, log_sub_dir, model, rate_limit_pause)
                csvwriter2.writerow([index, size, node_with_question_mark_ff, ground_truth_ff, prompt, response, parsed_response])
                accurate_labels_ff += acc_ff
                failure_labels_ff += fail_ff

            num_samples = len(desired_sizes)
            accuracy_ego = accurate_labels_ego / num_samples
            failure_ego = 0 if accuracy_ego == 1.0 else failure_labels_ego / (num_samples)
            inaccuracy_ego = 1 - (accuracy_ego + failure_ego)

            accuracy_ff = accurate_labels_ff / num_samples
            failure_ff = 0 if accuracy_ff == 1.0 else failure_labels_ff / (num_samples)
            inaccuracy_ff = 1 - (accuracy_ff + failure_ff)

            avg_accuracy_values_ego.append(accuracy_ego)
            avg_inaccuracy_values_ego.append(inaccuracy_ego)
            avg_failure_values_ego.append(failure_ego)

            avg_accuracy_values_ff.append(accuracy_ff)
            avg_inaccuracy_values_ff.append(inaccuracy_ff)
            avg_failure_values_ff.append(failure_ff)

    return avg_accuracy_values_ego, avg_failure_values_ego, avg_inaccuracy_values_ego, avg_accuracy_values_ff, avg_failure_values_ff, avg_inaccuracy_values_ff

     
    
#------------------------------------------------
random.seed(10)
openai.api_key = os.environ["OPENAI_API_UMNKEY"]
# Load configuration from the JSON file
with open('code/config3.json', 'r') as config_file:
    config = json.load(config_file)
# Access parameters from the config dictionary
dataset_name = config["dataset_name"]
no_of_hops = config["NO_OF_HOPS"]
no_of_runs = config["RUN_COUNT"]
USE_EDGE = config["USE_EDGE"]
USE_ADJACENCY = config["USE_ADJACENCY"]
neighborhood_sampling_flag = config["neighborhood_sampling"]
model = config["model"]
rate_limit_pause = config["rate_limit_pause"]
data_dir = config["data_dir"]
log_dir = config["log_dir"]
result_location = config["result_location"]
average_2hop_size = config["average_2hop_size"]
neighborhood_hop = config["neighborhood_hop_size"]

#------------------------------------------------

log_sub_dir = create_log_dir(log_dir)
data, graph = load_and_prepare_data(data_dir, dataset_name)
desired_sizes = get_desired_sizes(average_2hop_size, num_samples_per_size = 4)
avg_accuracy_values_ego, avg_failure_values_ego, avg_inaccuracy_values_ego,avg_accuracy_values_ff, avg_failure_values_ff, avg_inaccuracy_values_ff = run_experiment(desired_sizes, no_of_runs, no_of_hops, data, graph, result_location, log_dir, log_sub_dir, model, rate_limit_pause, USE_EDGE, USE_ADJACENCY, neighborhood_hop, neighborhood_sampling_flag)
#print("Accuracy for Ego Graph", avg_accuracy_values_ego)
#print("Failure for Ego Graph", avg_failure_values_ego)
#print("InAccuracy for Ego Graph", avg_inaccuracy_values_ego)
print("Average accuracy across runs for Ego:", np.mean(avg_accuracy_values_ego), "Standard deviation of accuracy for Ego:", np.std(avg_accuracy_values_ego))
print("Average failure across runs for Ego:", np.mean(avg_failure_values_ego), "Standard deviation of failure for Ego:", np.std(avg_failure_values_ego))
print("Average Inaccuracy across runs for Ego:", np.mean(avg_inaccuracy_values_ego), "Standard deviation of inaccuracy for Ego:", np.std(avg_inaccuracy_values_ego))
print("============================================")
#print("Accuracy for FF Graph", avg_accuracy_values_ff)
#print("Failure for FF Graph", avg_failure_values_ff)
#print("InAccuracy for FF Graph", avg_inaccuracy_values_ff)
print("Average accuracy across runs for FF:", np.mean(avg_accuracy_values_ff), "Standard deviation of accuracy for FF:", np.std(avg_accuracy_values_ff))
print("Average failure across runs for FF:", np.mean(avg_failure_values_ff), "Standard deviation of failure for FF:", np.std(avg_failure_values_ff))
print("Average Inaccuracy across runs for FF:", np.mean(avg_inaccuracy_values_ff), "Standard deviation of inaccuracy for FF:", np.std(avg_inaccuracy_values_ff))
       