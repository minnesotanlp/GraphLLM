
import json 
import os
from datetime import datetime

def get_directories(path):
    """
    Returns a list of directories at the specified path.

    Parameters:
    path (str): Path of the directory.

    Returns:
    list: List of directories in the specified path.
    """
    try:
        all_entries = os.listdir(path)
        directories = [d for d in all_entries if os.path.isdir(os.path.join(path, d))]
        return directories
    except FileNotFoundError:
        print(f"The specified path {path} was not found.")
        return []
    except PermissionError:
        print(f"Permission denied for accessing the path: {path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Define a function to calculate the cost for a single JSON file
def calculate_cost(json_file, pricing, pricing_id):
    with open(json_file, 'r') as file:
        data = json.load(file)

    no_of_prompt_tokens = data['usage']['prompt_tokens']
    no_of_completion_tokens = data['usage']['completion_tokens']
    pricing_id = data['model']

    cost = (
        no_of_prompt_tokens * pricing[pricing_id][PROMPT_PRICE_PER_TOKEN_KEY] +
        no_of_completion_tokens * pricing[pricing_id][COMPLETION_PRICE_PER_TOKEN_KEY]
    )

    return cost



GPT4_ID = "gpt-4-0613" # chat only
CHAT_GPT_ID = "gpt-3.5-turbo"
GPT3_ID = "text-davinci-003"
GPT3_INSTRUCT_ID= "gpt-3.5-turbo-instruct"
model = "gpt-4"

PROMPT_PRICE_PER_TOKEN_KEY = "prompt_price_per_token"
COMPLETION_PRICE_PER_TOKEN_KEY = "completion_price_per_token"
GENERAL_PRICE_PER_TOKEN_KEY = "price_per_token"
pricing = {GPT3_ID: {GENERAL_PRICE_PER_TOKEN_KEY: 0.02 / 1000},
           GPT4_ID: {PROMPT_PRICE_PER_TOKEN_KEY: 0.03 / 1000, COMPLETION_PRICE_PER_TOKEN_KEY: 0.06 / 1000}, 
           CHAT_GPT_ID: {PROMPT_PRICE_PER_TOKEN_KEY: 0.002 / 1000, COMPLETION_PRICE_PER_TOKEN_KEY: 0.002 / 1000},
           GPT3_INSTRUCT_ID: {PROMPT_PRICE_PER_TOKEN_KEY: 0.00015 / 1000, COMPLETION_PRICE_PER_TOKEN_KEY: 0.002 / 1000}        
           }

# Directory containing JSON files from today's usage
today_date = datetime.now().strftime("%m-%d-%y")
log_directory = f'./logs/{today_date}/'

# for a range of dates
list_log_dir_path = f'./logs/'
directories = get_directories(list_log_dir_path)
print("Dates we're calculating for: ", directories)

# Initialize the total cost
total_cost = 0

# Iterate through JSON files in the directory
for log_directory in directories:
    log_path = os.path.join(list_log_dir_path,log_directory)
    for filename in os.listdir(log_path):
        if filename.endswith('.json'):
            json_file_path = os.path.join(log_path, filename)
            cost = calculate_cost(json_file_path, pricing, GPT4_ID)
            total_cost += cost

print(f'Total cost of our experiments today: {total_cost}')
