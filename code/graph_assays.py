import random
import openai
import json
import os
import networkx as nx
from torch_geometric.utils import to_networkx
from utils import load_dataset, save_response, create_log_dir
from prompt_generation import get_completion_json, get_prompt, generate_textprompt_anygraph
from response_parser import parse_response
from metrics import is_failure,is_accurate
from connection_information import get_y_labels_graph
from graph_sample_generation import generate_egograph_sample, generate_forestfire_sample
from plotting import plot_graph_structure, plot_graph_structure_community_colored
# ---- PARAMS --- #
#-------------------------------------------------------------
random.seed(10)
openai.api_key = os.environ["OPENAI_API_UMNKEY"]
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
    modification = 2 # add prompt instruction.
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


def count_triangles(graph):
    triangle_count = 0
    for node in graph.nodes():
        neighbors = set(graph.neighbors(node))
        for neighbor1 in neighbors:
            for neighbor2 in neighbors:
                if neighbor1 != neighbor2 and graph.has_edge(neighbor1, neighbor2):
                    triangle_count += 1
    return triangle_count // 2

def calculate_triangles_given_subgraph(ego_graph, y_labels_egograph, ego_node, node_with_question_mark):
    triangle_count = 0
    ego_neighbors = set(ego_graph.neighbors(ego_node))
    
    for neighbor1 in ego_neighbors:
        for neighbor2 in ego_neighbors:
            if neighbor1 != neighbor2 and ego_graph.has_edge(neighbor1, neighbor2):
                triangle_count += 1
                print(f"Triangle: {ego_node} -> {neighbor1} -> {neighbor2}")

    print(f"Total triangles including '{node_with_question_mark}': {triangle_count}")

def run_experiment(desired_sizes, data, graph, log_dir, log_sub_dir, compression, model, rate_limit_pause, num_samples, use_edge, USE_ADJACENCY, neighborhood_hop=2, neighborhood_sampling_flag=False):
    avg_accuracy_values_ego = []
    avg_failure_values_ego = []
    avg_inaccuracy_values_ego = []
   

    for run in range(1):
        error_count = 0
        accurate_labels_ego = 0
        failure_labels_ego = 0
        inaccuracy_ego = []

        desired_sizes = [9]

        for index, size in enumerate(desired_sizes):
            ego_node, ego_graph = generate_egograph_sample(graph, no_of_hops, size)
            y_labels_egograph = get_y_labels_graph(data, ego_graph, ego_flag=True, ego_node=ego_node)
            node_with_question_mark = ego_node
            ground_truth_ego = y_labels_egograph[ego_node][node_with_question_mark] #? is the ego node
            if neighborhood_sampling_flag: # use only the 2 hop neighborhood -- go change config
                # use 2 hop subgraph        
                ego_graph = nx.ego_graph(ego_graph, ego_node, radius=neighborhood_hop)
            text_ego = generate_textprompt_anygraph(ego_graph, ego_node, y_labels_egograph, ego_node, use_edge, USE_ADJACENCY, ego_flag=True)
            #plot_graph_structure(ego_graph, y_labels_egograph, node_with_question_mark, size, f'ego_{index}_graphsize_', ego_flag=True)
            plot_graph_structure_community_colored(ego_graph, y_labels_egograph, node_with_question_mark, size, f'ego_{index}_graphsize_', ego_flag=True)
            #print(f'no of triangles in the ego graph {count_triangles(ego_graph)}')
            #calculate_triangles_given_subgraph(ego_graph, y_labels_egograph, ego_node, node_with_question_mark)


            # this calls the openapi completion api
            #error_count, acc_ego, fail_ego = process_text(text_ego, ground_truth_ego, ego_node, log_dir, log_sub_dir, compression, model, rate_limit_pause)
            #accurate_labels_ego += acc_ego
            #failure_labels_ego += fail_ego
        

        accuracy_ego = accurate_labels_ego / num_samples
        failure_ego = 0 if accuracy_ego == 1.0 else failure_labels_ego / (num_samples)
        inaccuracy_ego = 1 - accuracy_ego - failure_ego


        avg_accuracy_values_ego.append(accuracy_ego)
        avg_inaccuracy_values_ego.append(inaccuracy_ego)
        avg_failure_values_ego.append(failure_ego)


    return avg_accuracy_values_ego, avg_failure_values_ego, avg_inaccuracy_values_ego





def main():
    log_sub_dir = create_log_dir(log_dir)
    create_result_location(result_location)
    data, graph = load_and_prepare_data(data_dir, dataset_name)
    desired_sizes = get_desired_sizes(average_2hop_size, num_samples_per_size = 4) # total sample size = 20
    print("Desired sizes: ", desired_sizes)
    avg_accuracy_values_ego, avg_failure_values_ego, avg_inaccuracy_values_ego = run_experiment(desired_sizes, data, graph, log_dir, log_sub_dir, compression, model, rate_limit_pause, num_samples, use_edge, USE_ADJACENCY, neighborhood_hop, neighborhood_sampling_flag)


main()