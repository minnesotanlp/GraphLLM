import random
import json
import os
import csv
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
# Load configuration from the JSON file
with open('code/config_images.json', 'r') as config_file:
    config = json.load(config_file)

# Access parameters from the config dictionary
dataset_name = config["dataset_name"]
no_of_hops = config["NO_OF_HOPS"]
no_of_runs = config["RUN_COUNT"]
data_dir = config["data_dir"]
result_location = config["result_location"]
neighborhood_sampling_flag = config["neighborhood_sampling"]
average_2hop_size = config["average_2hop_size"]
neighborhood_hop = config["neighborhood_hop_size"]
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


def load_and_prepare_data(data_dir, dataset_name):
    dataset = load_dataset(data_dir, dataset_name)
    data = dataset[0]
    graph = to_networkx(data, to_undirected=True)
    return data, graph


def run_experiment(no_of_runs, no_of_hops, desired_sizes, data, graph, og_result_location, neighborhood_sampling_flag=False):
    for run in range(0,no_of_runs): 
        result_location = os.path.join(og_result_location, f'run_{run}')
        create_result_location(result_location)  
        with open(os.path.join(result_location, f'ego_run_{run}_graph_image_values.csv'), 'w') as csvfile, open(os.path.join(result_location, f'ff_run_{run}_graph_image_values.csv'), 'w') as csvfile2:
            csvwriter = csv.writer(csvfile)
            csvwriter2 = csv.writer(csvfile2)
            csvwriter.writerow(['graph_id', 'size', 'node_id', 'label'])
            csvwriter2.writerow(['graph_id', 'size', 'node_id', 'label'])
            for index, size in enumerate(desired_sizes):
                ego_node, ego_graph = generate_egograph_sample(graph, no_of_hops, size)
                y_labels_egograph = get_y_labels_graph(data, ego_graph, ego_flag=True, ego_node=ego_node)
                node_with_question_mark = ego_node
                ground_truth_ego = y_labels_egograph[ego_node][node_with_question_mark] #? is the ego node
                if neighborhood_sampling_flag: # use only the 2 hop neighborhood -- go change config
                    # use 2 hop subgraph        
                    ego_graph = nx.ego_graph(ego_graph, ego_node, radius=neighborhood_hop)
                csvwriter.writerow([index, size, node_with_question_mark, ground_truth_ego])
                plot_graph_structure_community_colored(ego_graph, y_labels_egograph, node_with_question_mark, size, f'ego_{index}_graphsize_', result_location, ego_flag=True)
    
                ff_graph = generate_forestfire_sample(graph, size)
                y_labels_ffgraph = get_y_labels_graph(data, ff_graph, ego_flag=False)
                node_with_question_mark_ff = random.choice(list(ff_graph.nodes()))
                ground_truth_ff = y_labels_ffgraph[node_with_question_mark_ff]
                if neighborhood_sampling_flag:
                    # use 2 hop subgraph        
                    ff_graph = nx.ego_graph(ff_graph, node_with_question_mark_ff, radius=neighborhood_hop)
                csvwriter2.writerow([index, size, node_with_question_mark_ff, ground_truth_ff])
                plot_graph_structure_community_colored(ff_graph, y_labels_ffgraph, node_with_question_mark_ff, size, f'ff_{index}_graphsize_', result_location, ego_flag=False)    


def main():
    data, graph = load_and_prepare_data(data_dir, dataset_name)
    #get the stratified sample 
    desired_sizes = get_desired_sizes(average_2hop_size, num_samples_per_size = 4) # total sample size = 20
    print("Desired sizes: ", desired_sizes)
    run_experiment(no_of_runs, no_of_hops, desired_sizes, data, graph, result_location, neighborhood_sampling_flag)


main()