import networkx as nx
import numpy as np
import random 
from utils import load_dataset, save_response, create_log_dir
from torch_geometric.utils import to_networkx
from prompt_generation import get_completion_json, get_prompt, generate_textprompt_anygraph
from connection_information import get_y_labels_graph
from metrics import is_failure,is_accurate
from response_parser import parse_response
import openai
import csv
import os
import json


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
#-------------------------------------------------------------

log_sub_dir = create_log_dir(log_dir)
os.makedirs(result_location, exist_ok=True)

#------------- 

dataset = load_dataset(data_dir, dataset_name)
data = dataset[0]
# Convert the PyG graph to NetworkX graph
X = to_networkx(data, to_undirected=True)


avg_failure_values = []
avg_accuracy_values = []

node_list = list(X.nodes())
# Perform the experiment for N runs keeping hops and no of sampled ego graphs constant
for run in range(0,no_of_runs):
    edges_list = []
    res_filename = os.path.join(result_location,f'{dataset_name}_{no_of_hops}hops_{run}run_{num_samples}samples.csv')
    
    with open(res_filename, 'w', newline='') as f1:
        res_csv_writer = csv.writer(f1)        
        res_csv_writer.writerow(['GroundTruth', 'Parsed Value', 'Prompt', 'Response'])
        #f1.flush() 

        error_count = 0
        token_err_count = 0
        accurate_labels = 0
        failure_labels = 0

        for _ in range(num_samples):

            while True:
                # Randomly select a node from the graph as the ego node
                ego_node = np.random.choice(node_list) 
                # Extract the 2-hop ego subgraph around the ego node
                ego_graph = nx.ego_graph(X, ego_node, radius=no_of_hops)
                if ego_graph.number_of_edges() < 100:
                    break  # Exit the loop if the ego graph has fewer than 100 edges
            
          
            #get labels for the subgraph 
            y_labels_egograph = get_y_labels_graph(data, ego_graph, True, ego_node)

            # first subgraph is taken out. Let us now randomly assign one node as ? and extract the 2 hop subgraph if needed
            node_with_question_mark = random.choice(list(ego_graph.nodes()))
            ground_truth = y_labels_egograph[ego_node][node_with_question_mark]
            if neighborhood_sampling_flag:
                print("neighborhood activated!")
                # use 2 hop subgraph
                # Randomly choose a node to have a "?" label
                two_hop_ego_graph = nx.ego_graph(ego_graph, node_with_question_mark, radius=3)
                prompt_graph = two_hop_ego_graph  
            else:
                # use original subgraph here (? node will be the ego node)
                print("OG subgraph used")            
                prompt_graph = ego_graph
                
                
                
                
            text = generate_textprompt_anygraph(prompt_graph, ego_node, y_labels_egograph, node_with_question_mark, use_edge,  USE_ADJACENCY)
            error = ""
            prompt = get_prompt(text, compression)
            
            try:
                response_json = get_completion_json(prompt, model)
                save_response(response_json, log_dir, log_sub_dir)
                response = response_json.choices[0].message["content"]
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
                    print(f'Prompt tokens > context limit of {model}')
                    print(e)
                else:
                    print(f"Type of error: {type(e)}")
                    print(f"Error: {e}")
                    print(f"Pausing for {rate_limit_pause} seconds.")
                continue
            

            delimiter_options = ['=', ':']  # You can add more delimiters if needed
            parsed_value = None
            #attempt to parse the response
            for delimiter in delimiter_options: 
                parsed_value = parse_response(response, delimiter) # check for better rules here!
                if parsed_value is not None: # general checking for the delimiter responses
                    res_csv_writer.writerow([ground_truth, parsed_value, f'"{prompt}"', f'"{response}"'])
                    f1.flush()
                    break
                else :
                    print("BUG : Delimiter not found in response from the LLM, response written instead of parsed value")
                    res_csv_writer.writerow([ground_truth, response, f'"{prompt}"', f'"{response}"']) # this is handled in is accurate
                    f1.flush()
                    break

            print("RESPONSE --> ", response)
            print("Node with ?: ", node_with_question_mark, "Label: ",ground_truth)
            print("="*30)

            if is_accurate(parsed_value,ground_truth):
                accurate_labels +=1
            if is_failure(parsed_value):
                failure_labels +=1
        
        
        accuracy = accurate_labels/num_samples
        if accuracy == 1.0 :
            failure = 0
        else :
            failure = failure_labels/(num_samples-accurate_labels)

        avg_failure_values.append(failure)
        avg_accuracy_values.append(accuracy)


print("Accuracy", avg_accuracy_values)
print("Failure", avg_failure_values)


print("Average accuracy across runs:", np.mean(avg_accuracy_values)," Standard deviation of accuracy:", np.std(avg_accuracy_values))
print("Average failure across runs:", np.mean(avg_failure_values)," Standard deviation of failure:", np.std(avg_failure_values))


                    
            
            
    