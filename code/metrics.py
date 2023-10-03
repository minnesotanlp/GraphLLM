
import csv
import re
from prompt_generation import return_no_tokens

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



def connection_info_compression_percentage(input_text, compressed_text, model):
    # Tokenize the input and compressed texts and get no of tokens
    #num_input_tokens = return_no_tokens(model, input_text)
    #num_compressed_tokens = return_no_tokens(model, compressed_text)

    # Extract the content within curly braces in the input_text for token count
    curly_braces_pattern_input = re.compile(r'Adjacency list:\s*\{(.+?)\}', re.DOTALL)
    curly_braces_match_input = curly_braces_pattern_input.search(input_text)

    if not curly_braces_match_input:
        # the pattern was not matched in the input_text
        return -2
    
    # Tokenize the content within curly braces from input_text
    curly_braces_content_input = curly_braces_match_input.group(1)
    num_input_tokens = return_no_tokens(model, curly_braces_content_input)
    
    # Extract the content within curly braces in the compressed text
    curly_braces_pattern_compressed = re.compile(r'AdjList:\s*\{(.+?)\}', re.DOTALL)
    curly_braces_match_compressed = curly_braces_pattern_compressed.search(compressed_text)

    if curly_braces_match_compressed:
        # Tokenize the content within curly braces from compressed_text
        curly_braces_content_compressed = curly_braces_match_compressed.group(1)
        num_curly_braces_tokens = return_no_tokens(model, curly_braces_content_compressed)
        
        # Calculate the connection info compression percentage
        connection_info_compression = (num_curly_braces_tokens / num_input_tokens) * 100
    else:
        # the pattern was not matched in the compressed_text
        return -1
        
    return connection_info_compression

def token_compression_percentage(input_text, compressed_text, model):
    # Tokenize the input and compressed texts and get no of tokens
    num_input_tokens = return_no_tokens(model, input_text)
    num_compressed_tokens = return_no_tokens(model, compressed_text)
    
    # Calculate the token compression percentage
    compression_percentage = (num_compressed_tokens / num_input_tokens )* 100
    
    return compression_percentage
        

def compute_accuracy(csv_filename):
    # computes accuracy based on the no of records in the file
    total_count = 0
    correct_count = 0
    fail_count = 0
    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            total_count += 1
            ground_truth = row['GroundTruth']
            parsed_value = row['Parsed Value']
            if parsed_value == '-1' or parsed_value == '?':
                fail_count+=1
            if ground_truth == parsed_value:
                correct_count += 1


    if total_count == 0:
        return 0,0
    else:
        accuracy = correct_count / total_count
        if total_count == correct_count:
            failure_perc = 0
        else:
            failure_perc = fail_count / (total_count-correct_count)
        return accuracy, failure_perc
    
def token_limit_percent(response):
    return 100 * (response["usage"]["prompt_tokens"])/(8192)

# Function to record metrics in "metrics.csv"
def record_metrics(metrics_filename, hops, use_edge, sampled_nodes, mean_accuracy, std_accuracy, mean_failure, std_failure, mean_token_perc, std_token_perc, token_limit):
    with open(metrics_filename, 'a') as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow([hops, use_edge, sampled_nodes, mean_accuracy, std_accuracy, mean_failure, std_failure, mean_token_perc, std_token_perc, token_limit])

