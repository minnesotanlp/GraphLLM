import networkx as nx
import numpy as np
import random 
from utils import load_dataset
from torch_geometric.utils import to_networkx
from prompt_generation import generate_textprompt_anygraph, get_completion
from connection_information import get_y_labels_graph
from response_parser import parse_response
from plotting import visualize_ego_graph
import matplotlib.pyplot as plt
import openai
import time
import csv
import os

# Define your custom failure and accuracy functions
def is_failure(parsed_value):
    if str(parsed_value) == '-1' or str(parsed_value) == '?':
        return True
    else :
        return False

def is_accurate(parsed_value, ground_truth):
    parsed_value = str(parsed_value)
    ground_truth = str(ground_truth)

    if parsed_value == ground_truth:
        return True
    elif ground_truth in parsed_value:
        return True
    else:
        return False
    

def get_prompt(text):
    prompt = f"""
    Task : Node Label Prediction (Predict the label of the node marked with a ?, given the adjacency list information as a dictionary of type "node:node neighborhood"
    and node-label mapping in the text enclosed in triple backticks. Response should be in the format "Label of Node = <predicted label>". If the predicted label cannot be determined, return "Label of Node = -1") 
    ```{text}```
    """
    return prompt

 
#-- params -- 
openai.api_key = os.environ["OPENAI_API_UMNKEY"]
data_dir = './data'
dataset_name = 'pubmed'
use_edge = False
USE_ADJACENCY = True
model = 'gpt-4'
rate_limit_pause = 1.2
no_of_hops = 2
no_of_runs = 3
result_location = f'./results/{dataset_name}/edge_density/'
# Define the number of ego graphs you want to sample
num_samples = 20
# Define the number of nodes for each ego graph
num_nodes_per_ego = 60

os.makedirs(result_location, exist_ok=True)
random.seed(10)
#------------- 

dataset = load_dataset(data_dir, dataset_name)
data = dataset[0]
# Convert the PyG graph to NetworkX graph
X = to_networkx(data, to_undirected=True)

# Precompute a list of nodes satisfying the condition
eligible_nodes = [node for node in X.nodes() if nx.ego_graph(X, node, radius=no_of_hops, undirected=True).number_of_nodes() == num_nodes_per_ego]


average_edges_run = [] # stores the average number of edges every run
std_edges_run = []
avg_failure_values = []
avg_accuracy_values = []

# Perform the experiment
for run in range(0,no_of_runs):
    edges_list = []
    filename = result_location + f'{dataset_name}_{no_of_hops}hops_{run}run_{num_nodes_per_ego}nodesinego.csv'
    with open(filename,'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['GroundTruth', 'Parsed Value', 'Prompt', 'Response'])

        error_count = 0
        token_err_count = 0
        accurate_labels = 0
        failure_labels = 0
        for _ in range(num_samples):
            # Randomly select a node from the graph as the ego node
            ego_node = np.random.choice(eligible_nodes)
            #print("ego node selected", ego_node)
            # Extract the 2-hop ego subgraph around the ego node
            ego_graph = nx.ego_graph(X, ego_node, radius=no_of_hops)
            #print("ego graph selected")
            # Compute metrics on the ego graph
            num_edges = ego_graph.number_of_edges()
            edges_list.append(num_edges)
            y_labels_egograph = get_y_labels_graph(data, ego_graph, True, ego_node)
            #visualize_ego_graph(ego_graph, y_labels_egograph, ego_node)
            
            # HERE IS WHERE WE DO PROMPTING
            text, node_with_question_mark, ground_truth = generate_textprompt_anygraph(ego_graph, ego_node, y_labels_egograph, ego_node, use_edge, USE_ADJACENCY, True)
            error = ""
            #print(text)
            prompt = get_prompt(text)
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
                    #error = str(e)                                   
                    #csv_writer_i.writerow([f'"{(hops,edge_format,sampled_nodes)}"', f'{graph.number_of_nodes()}', f'{graph.number_of_edges()}', f'{error}'])
                else:
                    print(f"Type of error: {type(e)}")
                    print(f"Error: {e}")
                    print(f"Pausing for {rate_limit_pause} seconds.")
                continue


            delimiter_options = ['=', ':']  # You can add more delimiters if needed
            parsed_value = None
            #Write out the response
            for delimiter in delimiter_options: 
                parsed_value = parse_response(response, delimiter) # check for better rules here!
                if parsed_value is not None: # general checking for the delimiter responses
                    csv_writer.writerow([ground_truth, parsed_value, f'"{prompt}"', f'"{response}"'])
                    break
                else :
                    print("BUG : Delimiter not found in response from the LLM")
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
        average_edges_run.append(np.mean(edges_list))
        std_edges_run.append(np.std(edges_list))

print("Accuracy", avg_accuracy_values)
print("Failure", avg_failure_values)
print("Edges", average_edges_run)

print("Average accuracy across runs:", np.mean(avg_accuracy_values)," Standard deviation of accuracy:", np.std(avg_accuracy_values))
print("Average failure across runs:", np.mean(avg_failure_values)," Standard deviation of failure:", np.std(avg_failure_values))


# Sort average_edges_run and get the sorted indices
sorted_indices = np.argsort(average_edges_run)
sorted_average_edges_run = np.sort(average_edges_run)
sorted_avg_failure_values = np.array(avg_failure_values)[sorted_indices]
sorted_avg_accuracy_values = np.array(avg_accuracy_values)[sorted_indices]

 # Plot the trend of avg failure and accuracy with the avg number of edges
plt.figure(figsize=(10, 5))
plt.plot(sorted_average_edges_run, sorted_avg_failure_values, label='Failure',marker ='o', linestyle = '-')
plt.plot(sorted_average_edges_run, sorted_avg_accuracy_values, label='Accuracy', marker ='x', linestyle = '-')
plt.xlabel('Number of Edges')
plt.ylabel('Metric Value')
plt.title(f'Metrics vs. Average Number of Edges Run')
plt.grid(True)
plt.legend()
plt.show()

    

            
                    
            
            
    