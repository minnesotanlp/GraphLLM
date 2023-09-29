import openai
import random
import networkx as nx
import json
import re
from connection_information import generate_edgelist,generate_node_label_dict,generate_textual_edgelist,generate_textual_edgelist2,generate_graphlist,generate_graphlist_constrained,edge_list_to_adjacency_list,generate_GML
import tiktoken


# uses a fast a fast BPE tokenizer specifically designed for OpenAI models
def return_no_tokens(model, text): 
    encoding = tiktoken.encoding_for_model(model)
    token_count = len(encoding.encode(text))
    return token_count

def get_completion(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_completion_json(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response

#move to prompt generation
def get_prompt(connectivity_information, compression_flag):
    if compression_flag:
        text =f"""
        Task : Node Label Prediction (Predict the label of the node marked with a ?, given the adjacency list information as a dictionary of type "node:node neighborhood" and node-label mapping in the text enclosed in triple backticks. Response should be in the format "Label of Node = <predicted label>". If the predicted label cannot be determined, return "Label of Node = -1") 
        ```{connectivity_information}```
        """
        prompt = f"""
        Compress the following text in triple angular brackets '<<< >>>', into the size of a tweet such that you (GPT-4) can reconstruct the intention of the human who wrote text as close as possible to the original intention. Response format information given in the text needs to be retained in the reconstruction.This is for yourself. It does not need to be human readable or understandable. Abuse of language mixing, abbreviations, symbols (unicode and emoji), or any other encodings or internal representations is all permissible, as long as it, if pasted in a new inference cycle, will yield near-identical results as the original text.
        <<<{text}>>>
        """
    else:
        prompt = f"""
        Task : Node Label Prediction (Predict the label of the node marked with a ?, given the adjacency list information as a dictionary of type "node:node neighborhood" and node-label mapping in the text enclosed in triple backticks. Response should be in the format "Label of Node = <predicted label>". If the predicted label cannot be determined, return "Label of Node = -1") 
        ```{connectivity_information}```
        """
    return prompt


def generate_textprompt_anygraph(graph, center_node, y_labels_dict, node_with_question_mark, edge_text_flag, adjacency_flag):
    text = ""
    ground_truth = ""

    edge_list = generate_edgelist(graph)

    if edge_text_flag:
        edge_list_converted = generate_textual_edgelist(edge_list)
        text+="Edge connections (source node - target node): "+str(edge_list_converted)+"\n"

    elif adjacency_flag:
        adjacency_list = edge_list_to_adjacency_list(edge_list)
        text+="Adjacency list: "+str(adjacency_list)+"\n"
    else:
        text+="Edge list: "+str(edge_list)+"\n"

    ground_truth, node_label_dict = generate_node_label_dict(graph, node_with_question_mark, center_node, y_labels_dict)

    text+=f"Node to Label Mapping : "+"\n"
    for node in node_label_dict:
        text+=f"Node {node}: Label {node_label_dict[node]}| "
    return text


def generate_textprompt_egograph(graph, center_node, y_labels_dict, edge_text_flag, adjacency_flag):
    text = ""
    ground_truth = ""

    edge_list = generate_edgelist(graph)

    if edge_text_flag:
        edge_list_converted = generate_textual_edgelist(edge_list)
        text+="Edge connections (source node - target node): "+str(edge_list_converted)+"\n"

    elif adjacency_flag:
        adjacency_list = edge_list_to_adjacency_list(edge_list)
        text+="Adjacency list: "+str(adjacency_list)+"\n"
    else:
        text+="Edge list: "+str(edge_list)+"\n"

    # Randomly choose a node to have a "?" label
    node_with_question_mark = random.choice(list(graph.nodes()))

    ground_truth, node_label_dict = generate_node_label_dict(graph, node_with_question_mark, center_node, y_labels_dict)

    text+=f"Node to Label Mapping : "+"\n"
    for node in node_label_dict:
        text+=f"Node {node}: Label {node_label_dict[node]}| "
    return text, node_with_question_mark, ground_truth



def generate_text_for_prompt(i, nx_ids, graph, y_labels_dict, edge_text_flag, adjacency_flag):
    text = ""
    ground_truth = ""
    #text+= f"Edge Connectivity Information :"+"\n"
    center_node = nx_ids[i]
    edge_list = generate_edgelist(graph)

    if edge_text_flag:
        edge_list_converted = generate_textual_edgelist(edge_list)
        text+="Edge connections (source node - target node): "+str(edge_list_converted)+"\n"
        #text+="Edge list: "+str(edge_list_converted)+"\n"
    elif adjacency_flag:
        adjacency_list = edge_list_to_adjacency_list(edge_list)
        text+="Adjacency list: "+str(adjacency_list)+"\n"
    else:
        text+="Edge list: "+str(edge_list)+"\n"

    # Randomly choose a node to have a "?" label
    node_with_question_mark = random.choice(list(graph.nodes()))

    ground_truth, node_label_dict = generate_node_label_dict(graph, node_with_question_mark, center_node, y_labels_dict)

    text+=f"Node to Label Mapping : "+"\n"
    for node in node_label_dict:
        text+=f"Node {node}: Label {node_label_dict[node]}| "
    return text, node_with_question_mark, ground_truth

def generate_text_for_prompt_GML(i, nx_ids, graph, y_labels_dict, dataset_name):
    text = ""
    ground_truth = ""
    center_node = nx_ids[i]
    gml = generate_GML(graph)
    text+="GraphML: "+gml+"\n"

    # Randomly choose a node to have a "?" label
    node_with_question_mark = random.choice(list(graph.nodes()))

    text = re.sub("\"", "", text)

    # # For textual Labels
    # with open('label_data/' + dataset_name + '_labels.json', 'r') as label_file:
    #     labels = json.load(label_file)

    ground_truth, node_label_dict = generate_node_label_dict(graph, node_with_question_mark, center_node, y_labels_dict)

    # For GML Format
    ids = (re.compile("label (.*)")).findall(text)
    edges = (re.compile("source (.*)")).findall(text) + (re.compile("target (.*)")).findall(text)
    for i in range(len(ids)):
        text = re.sub("id " + str(i) + "\n", "id " + ids[i] + "\n", text)
        text = re.sub("label " + ids[i] + "\n", "label " + str(node_label_dict[int(ids[i])]) + "\n", text)
    for j in edges:
        text = re.sub("source " + j + "\n", "source " + ids[int(j)] + "\n", text)
        text = re.sub("target " + j + "\n", "target " + ids[int(j)] + "\n", text)

    # # For GraphML Format
    # ground_truth = labels[ground_truth]["label"]
    # regex = re.compile("id=(.*) />")
    # for id in regex.findall(text):
    #     if (id != str(node_with_question_mark)):
    #         # For no label names
    #         # text = re.sub("id=" + id, "id=" + id + " label=" + str(node_label_dict[int(id)]), text)
    #         # For label names
    #         text = re.sub("id=" + id, "id=" + id + " label=" + labels[node_label_dict[int(id)]]["label"], text)
    #     else:
    #         text = re.sub("id=" + str(node_with_question_mark), 'id=' + str(node_with_question_mark) + " label=?" , text)

    return text, node_with_question_mark, ground_truth
