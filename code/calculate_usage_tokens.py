
import json 
import os
from datetime import datetime

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
# yesterday and today
list_log_dir = [f'./logs/09-25-23/', f'./logs/{today_date}/']
# Initialize the total cost
total_cost = 0

# Iterate through JSON files in the directory
for log_directory in list_log_dir:
    for filename in os.listdir(log_directory):
        if filename.endswith('.json'):
            json_file_path = os.path.join(log_directory, filename)
            cost = calculate_cost(json_file_path, pricing, GPT4_ID)
            total_cost += cost

print(f'Total cost of our experiments today: {total_cost}')
