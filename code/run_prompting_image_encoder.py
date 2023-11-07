
import openai
#from openai import OpenAI
import os
import base64
import requests
from prompt_generation import get_image_completion_json


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

openai.api_key = os.environ["OPENAI_API_UMNKEY"]
# Path to your image
image_path = "code/example_plots/case1.png"

# Getting the base64 string
base64_image = encode_image(image_path)
model = "gpt-4-vision-preview"
prompt = f'Your task is Node Label Prediction (Predict the label of the red node marked with a ?, given the graph structure information in the image). Response should be in the format "Label of Node = <predicted label>". If the predicted label cannot be determined, return "Label of Node = -1'
response_json = get_image_completion_json(prompt, model, base64_image, detail="low")
print(response_json)
