
import openai
import os
import base64
import requests
import networkx as nx
import numpy as np
import json
import csv
from torch_geometric.utils import to_networkx
from utils import load_dataset, save_response, create_log_dir
from prompt_generation import get_image_completion_json
from response_parser import parse_response
from metrics import is_failure,is_accurate

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_text(prompt, base64_image, detail, ground_truth, node_with_question_mark, log_dir, log_sub_dir, model, rate_limit_pause):
    print(prompt)
    error_count = 0
    try:
        response_json = get_image_completion_json(prompt, model, base64_image, detail=detail)
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
    return error_count, accurate_labels, failure_labels, prompt, response, parsed_value


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


def run_experiment(input_location, no_of_samples, no_of_runs, setting, log_dir, log_sub_dir, model, rate_limit_pause, detail):
    avg_accuracy_values = []
    avg_failure_values = []
    avg_inaccuracy_values = []

    for run in range(0,no_of_runs): 
        run_location = os.path.join(input_location, f'run_{run}')
        error_count = 0
        accurate_labels = 0
        failure_labels = 0      
        # get the ground truth labels for the graphs in the setting
        ground_truth_filename = f'{setting}_run_{run}_graph_image_values.csv'
        result_filename = f'{setting}_run_{run}_{setting}_image_encoder_results.csv'
        ground_truth_info = extract_columns_from_csv_dict(run_location, ground_truth_filename)
        graph_info_location = os.path.join(run_location, f'{setting}')
        with open(f"{run_location}/{result_filename}", mode='w') as result_file:
            csvwriter = csv.writer(result_file)
            csvwriter.writerow(['setting', 'run', 'graph_id','node_with_question_mark', 'ground_truth', 'prompt', 'response', 'parsed_response'])
            for graph in ground_truth_info:
                graph_id = graph['graph_id']
                ground_truth = graph['label']
                node_with_question_mark = str(graph['ques_node_id'])

                image_path = f"{graph_info_location}/{graph_id}.png"
                # Getting the base64 string
                base64_image = encode_image(image_path)
                text_for_prompt = f'Your task is Node Label Prediction (Predict the label of the red node marked with a ?, given the graph structure information in the image). Response should be in the format "Label of Node = <predicted label>". If the predicted label cannot be determined, return "Label of Node = -1'
                #prompt the                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        LLM for prediction
                error_count, acc, fail, prompt, response, parsed_response = process_text(text_for_prompt, base64_image, detail, ground_truth, node_with_question_mark, log_dir, log_sub_dir, model, rate_limit_pause)
                csvwriter.writerow([setting, run, graph_id, node_with_question_mark, ground_truth, prompt, response, parsed_response])
                #check if the parsed prediction is correct compared to ground truth
                accurate_labels += acc
                failure_labels += fail
            #compute the accuracy, inaccuracy and failure metrics
            accuracy = accurate_labels / no_of_samples
            failure = 0 if accuracy == 1.0 else failure_labels / (no_of_samples)
            inaccuracy = 1 - (accuracy + failure)
            avg_accuracy_values.append(accuracy)
            avg_inaccuracy_values.append(inaccuracy)
            avg_failure_values.append(failure)
    return avg_accuracy_values, avg_inaccuracy_values, avg_failure_values

     
openai.api_key = os.environ["OPENAI_API_UMNKEY"]
with open('code/config_image_encoder.json', 'r') as config_file:
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

def main():
    with open(f"{input_location}/text_image_encoder_across_runs_metrics.csv", mode='w') as metrics_file:
            csvwriterf = csv.writer(metrics_file)
            csvwriterf.writerow(['setting', 'mean_accuracy', 'std_accuracy', 'mean_inaccuracy', 'std_inaccuracy', 'mean_failure', 'std_failure'])
            for setting in settings:
                avg_accuracy_runs, avg_inaccuracy_runs, avg_failure_runs = run_experiment(input_location, no_of_samples, no_of_runs, setting, log_dir, log_sub_dir, model, rate_limit_pause, detail)
                # write the per run results to a csv file
                csvwriterf.writerow([setting, np.mean(avg_accuracy_runs), np.std(avg_accuracy_runs), np.mean(avg_inaccuracy_runs), np.std(avg_inaccuracy_runs), np.mean(avg_failure_runs), np.std(avg_failure_runs)])
                print("SETTING : ", setting)
                print("Average accuracy across runs:", np.mean(avg_accuracy_runs), "Standard deviation of accuracy across runs:", np.std(avg_accuracy_runs))
                print("Average Inaccuracy across runs:", np.mean(avg_inaccuracy_runs), "Standard deviation of inaccuracy across runs:   ", np.std(avg_inaccuracy_runs))
                print("Average failure across runs:", np.mean(avg_failure_runs), "Standard deviation of failure across runs:", np.std(avg_failure_runs))
                print("="*30)



# Example usage:

main()



