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
from plotting import plot_graph_structure, plot_graph_structure_community_colored, plot_graphviz_graph
from connection_information import generate_edgelist, write_edgelist_to_file, write_labels_to_json


# ---- PARAMS --- #
#-------------------------------------------------------------
# Load configuration from the JSON file
# with open('config_images.json', 'r') as config_file:
#     config = json.load(config_file)

# Access parameters from the config dictionary
# fixed_seed = 42
# dataset_name = config["dataset_name"]
# no_of_hops = config["NO_OF_HOPS"]
# no_of_runs = config["RUN_COUNT"]
# data_dir = config["data_dir"]
# result_location = config["result_location"]
# neighborhood_sampling_flag = config["neighborhood_sampling"]
# average_2hop_size = config["average_2hop_size"]
# neighborhood_hop = config["neighborhood_hop_size"]
# number_of_samples = config["NO_OF_SAMPLED_NODES"]

#------------------------------------------------

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
            node_neighbors = list(G.adj[graph['ques_node_id']].keys())
            y_labels = load_graph_node_json(ylabelsjson_name)
            # generate the image for the graph
            G = nx.nx_agraph.to_agraph(G)
            plot_graphviz_graph(G, y_labels, node_with_question_mark, graph_id, f'{setting}_graphsize_{size}', graph_info_location, node_neighbors, ego_flag=ego_flag)
            print(f'Image ID {graph_id} generated for setting {setting} and run {run}')

def get_desired_sizes(average_hop_size, num_samples, seed = 0):
    random.seed(seed)
    # Calculate the lower and upper bounds for the sizes
    lower_bound = int(0.25 * average_hop_size)
    upper_bound = int(2 * average_hop_size)
    # Generate the list of sizes
    sizes = [random.randint(lower_bound, upper_bound) for _ in range(num_samples)]
    return sizes

def load_and_prepare_data(data_dir, dataset_name):
    dataset = load_dataset(data_dir, dataset_name)
    data = dataset[0]
    print(data)
    graph = to_networkx(data, to_undirected=True)
    # graph = nx.nx_agraph.to_agraph(graph)
    return data, graph

def run_experiment(no_of_runs, no_of_hops, desired_sizes, data, graph, og_result_location, neighborhood_sampling_flag=False):
    for run in range(0,no_of_runs): 
        result_location = os.path.join(og_result_location, f'run_{run}')
        create_result_location(result_location) # creates the result_location folder
        create_result_location(result_location+'/ego') # creates the result_location/run/ego folder
        create_result_location(result_location+'/ff') # creates the result_location/run/ff folder

        with open(os.path.join(result_location, f'ego_run_{run}_graph_image_values.csv'), 'w') as csvfile, open(os.path.join(result_location, f'ff_run_{run}_graph_image_values.csv'), 'w') as csvfile2:
            csvwriter = csv.writer(csvfile) # writes in the result_location/run/
            csvwriter2 = csv.writer(csvfile2) # writes in the result_location/run/
            csvwriter.writerow(['graph_id', 'sample_size', 'no_nodes', 'no_edges', 'ques_node_id', 'label'])
            csvwriter2.writerow(['graph_id', 'sample_size','no_nodes', 'no_edges', 'ques_node_id', 'label'])
            for index, size in enumerate(desired_sizes):
                #note : the y_labels would be more than graph edgelist as we are doing neighborhood sampling and choosing only the 2 hop neighborhood
                # EGO GRAPH SETTING
                setting = 'ego'
                ego_node, ego_graph = generate_egograph_sample(graph, no_of_hops, size)
                y_labels_egograph = get_y_labels_graph(data, ego_graph, ego_flag=True, ego_node=ego_node)
                node_with_question_mark = ego_node
                node_neighbors = list(graph.adj[ego_node].keys())
                ground_truth_ego = y_labels_egograph[ego_node][node_with_question_mark] #? is the ego node
                if neighborhood_sampling_flag: # use only the 2 hop neighborhood -- go change config
                    # use 2 hop subgraph        
                    ego_graph = nx.ego_graph(ego_graph, ego_node, radius=neighborhood_hop)

                no_nodes = ego_graph.number_of_nodes()
                no_edges = ego_graph.number_of_edges()
                graph_edge_path = result_location + f'/ego/'

                # record the ground truth, the edgelist and the y_labels
                csvwriter.writerow([index, size, no_nodes, no_edges, node_with_question_mark, ground_truth_ego])
                write_edgelist_to_file(generate_edgelist(ego_graph), os.path.join(graph_edge_path, f'{index}_edgelist.txt'))
                labels_dict = {int(key): int(value) for key, value in y_labels_egograph[ego_node].items()}
                write_labels_to_json(labels_dict, os.path.join(graph_edge_path, f'{index}_ylabels.json'))
                plot_graph_structure_community_colored(ego_graph, y_labels_egograph, node_with_question_mark, index, f'{setting}_graphsize_{size}', graph_edge_path, ego_flag=True)
                ego_graph = nx.nx_agraph.to_agraph(ego_graph)
                
                plot_graphviz_graph(ego_graph, y_labels_egograph, node_with_question_mark, index, f'{setting}_graphsize_{size}', graph_edge_path, node_neighbors, ego_flag=True)
                
                # FOREST FIRE SETTING
                setting = 'forest_fire'
                ff_graph = generate_forestfire_sample(graph, size)
                y_labels_ffgraph = get_y_labels_graph(data, ff_graph, ego_flag=False)
                node_with_question_mark_ff = random.choice(list(ff_graph.nodes()))
                ground_truth_ff = y_labels_ffgraph[node_with_question_mark_ff]
                if neighborhood_sampling_flag:
                    # use 2 hop subgraph        
                    ff_graph = nx.ego_graph(ff_graph, node_with_question_mark_ff, radius=neighborhood_hop)
                no_nodes = ff_graph.number_of_nodes()
                no_edges = ff_graph.number_of_edges()
                graph_edge_path = result_location + f'/ff/'

                # record the ground truth, the edgelist and the y_labels
                csvwriter2.writerow([index, size, no_nodes, no_edges, node_with_question_mark_ff, ground_truth_ff])
                write_edgelist_to_file(generate_edgelist(ff_graph), os.path.join(graph_edge_path, f'{index}_edgelist.txt'))
                labels_dict = {int(key): int(value) for key, value in y_labels_ffgraph.items()}
                write_labels_to_json(labels_dict, os.path.join(graph_edge_path, f'{index}_ylabels.json'))

                # plot_graph_structure_community_colored(ff_graph, y_labels_ffgraph, node_with_question_mark_ff, index, '.', graph_edge_path, ego_flag=False)
                ff_graph = nx.nx_agraph.to_agraph(ff_graph)
                plot_graphviz_graph(ff_graph, y_labels_ffgraph, node_with_question_mark_ff, index, f'{setting}_graphsize_{size}', graph_edge_path, node_neighbors, ego_flag=False)

def main():
    # data, graph = load_and_prepare_data(data_dir, dataset_name)
    # #get the stratified sample sizes (0.25x to 2x)
    # desired_sizes = get_desired_sizes(average_2hop_size, num_samples = number_of_samples, seed=fixed_seed)
    # print("Desired sizes: ", desired_sizes)
    # run_experiment(no_of_runs, no_of_hops, desired_sizes, data, graph, result_location, neighborhood_sampling_flag)
    settings = ["ego", "ff"]
    no_of_runs = 3
    no_of_samples = 50
    dataset_name = "citeseer"
    input_location = f"../results/{dataset_name}/graph_images/sample_size_{no_of_samples}/"
    for setting in settings:
        create_graphs(input_location, setting, no_of_runs)


main()