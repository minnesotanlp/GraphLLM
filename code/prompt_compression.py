import networkx as nx
import numpy as np
import random 
from utils import load_dataset, save_response, create_log_dir
from torch_geometric.utils import to_networkx
from prompt_generation import generate_textprompt_anygraph, get_completion_json, get_prompt
from connection_information import get_y_labels_graph
from metrics import is_failure,is_accurate, token_compression_percentage, connection_info_compression_percentage
from connection_information import generate_graphlist_constrained
from response_parser import parse_response
import matplotlib.pyplot as plt
import openai
import time
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
metrics_filename = config["metrics_filename"]
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
avg_token_compr_values = []
avg_conn_compr_values = []

node_list = list(X.nodes())
# Perform the experiment for N runs keeping hops and no of sampled ego graphs constant
for run in range(0,no_of_runs):
    edges_list = []
    res_filename = result_location + f'{dataset_name}_{no_of_hops}hops_{run}run_{num_samples}samples.csv'
    met_filename = result_location + f'{dataset_name}_{no_of_hops}hops_{run}run_{num_samples}samples_metrics.csv'
    
    with open(res_filename, 'w', newline='') as f1, open(met_filename, 'w', newline='') as f2:
        res_csv_writer = csv.writer(f1)
        met_csv_writer = csv.writer(f2)
        
        res_csv_writer.writerow(['GroundTruth', 'Parsed Value', 'Prompt', 'Response'])
        f1.flush()
        met_csv_writer.writerow(['Input Text', 'Compressed Text', 'Token Compression Percentage', 'Connection Info Compression Percentage'])
        f2.flush()        
        error_count = 0
        token_err_count = 0
        accurate_labels = 0
        failure_labels = 0
        token_compr_values = []
        conn_compr_values = []
        for _ in range(num_samples):

            while True:
                # Randomly select a node from the graph as the ego node
                ego_node = np.random.choice(node_list) 
                # Extract the 2-hop ego subgraph around the ego node
                ego_graph = nx.ego_graph(X, ego_node, radius=no_of_hops)
                if ego_graph.number_of_edges() < 100:
                    break  # Exit the loop if the ego graph has fewer than 100 edges
            
            # Compute metrics on the ego graph

            y_labels_egograph = get_y_labels_graph(data, ego_graph, True, ego_node)
            text, node_with_question_mark, ground_truth = generate_textprompt_anygraph(ego_graph, ego_node, y_labels_egograph, ego_node, use_edge, USE_ADJACENCY, True)
            error = ""
            prompt = get_prompt(text, compression)
            input_text = prompt # ????
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
            
            # get compression metrics
            compr_prompt = response
            compressed_text = compr_prompt
            token_compression_perc = token_compression_percentage(input_text, compressed_text, model)
            token_compr_values.append(token_compression_perc)
            connection_info_compression_perc = connection_info_compression_percentage(input_text, compressed_text, model)
            conn_compr_values.append(connection_info_compression_perc)
            met_csv_writer.writerow([input_text, compressed_text, f'{token_compression_perc}', f'{connection_info_compression_perc}'])
            f2.flush()
            
            
            #Chain the response to do the task but using compressed prompt
            try:
                response_json = get_completion_json(compr_prompt, model)
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
                    res_csv_writer.writerow([ground_truth, parsed_value, f'"{compr_prompt}"', f'"{response}"'])
                    f1.flush()
                    break
                else :
                    print("BUG : Delimiter not found in response from the LLM, response written instead of parsed value")
                    res_csv_writer.writerow([ground_truth, response, f'"{compr_prompt}"', f'"{response}"']) # this is handled in is accurate
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
        avg_token_compr_values.append(np.mean([val for val in token_compr_values if val > 0]))
        avg_conn_compr_values.append(np.mean([val for val in conn_compr_values if val > 0]))
        avg_failure_values.append(failure)
        avg_accuracy_values.append(accuracy)


print("Accuracy", avg_accuracy_values)
print("Failure", avg_failure_values)


print("Average accuracy across runs:", np.mean(avg_accuracy_values)," Standard deviation of accuracy:", np.std(avg_accuracy_values))
print("Average failure across runs:", np.mean(avg_failure_values)," Standard deviation of failure:", np.std(avg_failure_values))
print("Average Token compression across runs:", np.mean(avg_token_compr_values)," Standard deviation of token compression:", np.std(avg_token_compr_values))
print("Average Connection compression across runs:", np.mean(avg_conn_compr_values)," Standard deviation of connection compression:", np.std(avg_conn_compr_values))


                    
            
            
    