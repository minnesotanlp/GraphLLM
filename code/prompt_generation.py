import openai
import random
from connection_information import generate_edgelist,generate_node_label_dict,generate_textual_edgelist,generate_textual_edgelist2,generate_graphlist,generate_graphlist_constrained,edge_list_to_adjacency_list

def get_completion(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

        
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
