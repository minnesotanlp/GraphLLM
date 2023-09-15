import networkx as nx
import numpy as np
import random 
from utils import load_dataset
from torch_geometric.utils import to_networkx
from prompt_generation import generate_textprompt_egograph, get_completion
from connection_information import generate_graphlist_constrained
from response_parser import parse_response
import matplotlib.pyplot as plt
import openai
import time
import csv
import os
import re

# Define your custom failure and accuracy functions
def is_failure(parsed_value):
    if str(parsed_value) == '-1' or str(parsed_value) == '?':
        return True
    else :
        return False

def is_accurate(parsed_value, ground_truth):
    ground_truth = str(ground_truth)
    parsed_value = str(parsed_value)
    matches = re.search(r'the label of node (\d+) is (\d+)', parsed_value, re.IGNORECASE)
    if matches:
        label_value = matches.group(1)
        if label_value == ground_truth:
            return True
        else :
            return False
    else:
        tokens = re.findall(r'\w+|[^\w\s]', parsed_value)
        if tokens[0] == ground_truth:
            return True
        else:
            return False
        
           
    

def get_y_labels_egograph(data, ego_graph, ego_node):
    y_labels_dict = {}
    y_labels_dict[ego_node] = {}   # Initialize dictionary for this ego graph   
    # Iterate over the nodes in the ego graph
    for node in ego_graph.nodes():
        # Get the label for the current node from the data
        label = data.y[node].item()     
        # Store the label in the y_labels_dict
        y_labels_dict[ego_node][node] = label
    return y_labels_dict



def get_prompt(connectivity_information, compression_flag):
    if compression_flag:
        text =f"""
        Task : Node Label Prediction (Predict the label of the node marked with a ?, given the adjacency list information as a dictionary of type "node:node neighborhood"
        and node-label mapping in the text enclosed in triple backticks. Response should be in the format "Label of Node = <predicted label>". If the predicted label cannot be determined, return "Label of Node = -1") 
        ```{connectivity_information}```
        """
        prompt = f"""
        Compress the following text in triple angular brackets '<<< >>>', into the size of a tweet such that you (GPT-4) can reconstruct the intention of the human who wrote text as close as possible to the original intention. Response format information given in the text needs to be retained in the reconstruction.This is for yourself. It does not need to be human readable or understandable. Abuse of language mixing, abbreviations, symbols (unicode and emoji), or any other encodings or internal representations is all permissible, as long as it, if pasted in a new inference cycle, will yield near-identical results as the original text.
        <<<{text}>>>
        """
    else:
        prompt = f"""
        Task : Node Label Prediction (Predict the label of the node marked with a ?, given the adjacency list information as a dictionary of type "node:node neighborhood"
        and node-label mapping in the text enclosed in triple backticks. Response should be in the format "Label of Node = <predicted label>". If the predicted label cannot be determined, return "Label of Node = -1") 
        ```{connectivity_information}```
        """
    return prompt

 
#-- params -- 
openai.api_key = os.environ["OPENAI_API_UMNKEY"]
data_dir = './data'
dataset_name = 'cora'
use_edge = False
USE_ADJACENCY = True
model = 'gpt-4'
rate_limit_pause = 1.2
no_of_hops = 1
no_of_runs = 3
result_location = f'./results/{dataset_name}/compression/'
# Define the number of ego graphs you want to sample
num_samples = 20

compression = True

os.makedirs(result_location, exist_ok=True)
random.seed(10)
#------------- 

dataset = load_dataset(data_dir, dataset_name)
data = dataset[0]
# Convert the PyG graph to NetworkX graph
X = to_networkx(data, to_undirected=True)


avg_failure_values = []
avg_accuracy_values = []

node_list = list(X.nodes())
# Perform the experiment
for run in range(0,no_of_runs):
    edges_list = []
    filename = result_location + f'{dataset_name}_{no_of_hops}hops_{run}run_{num_samples}samples_baseline.csv'
    with open(filename,'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['GroundTruth', 'Parsed Value', 'Prompt', 'Response'])
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
            
            # Compute metrics on the ego graph

            y_labels_egograph = get_y_labels_egograph(data, ego_graph, ego_node)
            text, node_with_question_mark, ground_truth = generate_textprompt_egograph(ego_graph, ego_node, y_labels_egograph, use_edge, USE_ADJACENCY)
            error = ""
            compression = False # we want the usual prompt
            prompt = get_prompt(text, compression)
            try:
                response = get_completion(prompt, model)
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
                    csv_writer.writerow([ground_truth, parsed_value, f'"{prompt}"', f'"{response}"'])
                    break
                else :
                    print("BUG : Delimiter not found in response from the LLM")
                    csv_writer.writerow([ground_truth, response, f'"{prompt}"', f'"{response}"']) # this is handled in is accurate
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


                       
    