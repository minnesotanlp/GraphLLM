import random
from torch_geometric.utils import to_networkx
import networkx as nx
import torch 
import json

# get the y labels for the graph passed
def get_y_labels_graph(data, graph_passed, ego_flag, ego_node = None):
    if ego_flag:
        y_labels_dict = {}
        y_labels_dict[ego_node] = {}   # Initialize dictionary for this ego graph   
        # Iterate over the nodes in the ego graph
        for node in graph_passed.nodes():
            # Get the label for the current node from the data
            label = data.y[node].item()     
            # Store the label in the y_labels_dict
            y_labels_dict[ego_node][node] = label
    else:
        y_labels_dict = {} 
        # Iterate over the nodes in the ego graph
        for node in graph_passed.nodes():
            # Get the label for the current node from the data
            label = data.y[node].item()     
            # Store the label in the y_labels_dict
            y_labels_dict[node] = label
    return y_labels_dict


# This function generates a edgelist from the graph
def generate_edgelist(graph):
    # Print out the labels associated with a graph
    edge_list = list(graph.edges())
    return edge_list

# This function writes an egelist to file
def write_edgelist_to_file(edge_list, filename):
    with open(filename, 'w') as file:
        for edge in edge_list:
            file.write(f"{edge[0]} {edge[1]}\n")

#This function writes the y_labels to a json
def write_labels_to_json(labels_dict, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(labels_dict, file, ensure_ascii=False, indent=4)


# This function generates a compressed textual version (Node A - Node B) of the edgelist from the graph
def generate_textual_edgelist2(edge_list):           
        # Convert the edge list information into text
        edgelist_converted = ''
        for edge in edge_list:
            source, target = edge
            edgelist_converted += f'Node {source} - Node {target}. '
        return edgelist_converted

# This function generates a textual version (Node A is connected to Node B) of the edgelist from the graph
def generate_textual_edgelist(edge_list):           
        # Convert the edge list information into text
        edgelist_converted = ''
        for edge in edge_list:
            source, target = edge
            edgelist_converted += f'Node {source} is connected to Node {target}. '
        return edgelist_converted


# generates graph list which has graphs <100 edges
def generate_graphlist_constrained(num_nodes_to_sample, no_of_hops, data):
    # stores labels of each sub graph --> center node : {node: label}, ..
    y_labels_dict = {}
    # List to store sampled graphs
    graph_list = []

    # Convert the PyG graph to NetworkX graph
    nx_graph = to_networkx(data, to_undirected=True)
    sampled_nodes = set()

    # num_nodes_to_sample : tells how many nodes in the graph list while sampled_subgraph.number_of_nodes is how many nodes per ego graph
    while len(graph_list) < num_nodes_to_sample:
        # Choose a random node index
        center_node_index = random.randint(0, data.num_nodes - 1)
        center_node = int(center_node_index)

        if center_node in sampled_nodes:
            continue

        sampled_subgraph = nx.ego_graph(nx_graph, center_node, radius=no_of_hops, undirected=True)

        # Check the size of the subgraph before adding it, the and condition is likely to fail with very small graphs
        if (

            sampled_subgraph.number_of_edges() < 100
        ):
            y_labels_dict[center_node] = {}  # Initialize dictionary for this center node
            for node in sampled_subgraph.nodes():
                y_labels_dict[center_node][node] = data.y[node].item()  # Store y label
            graph_list.append(sampled_subgraph)
            sampled_nodes.add(center_node)

    # Convert the list of center nodes to integers
    nx_ids = list(y_labels_dict.keys())

    return y_labels_dict, nx_ids, graph_list

# generates graphlist which can have any no of edges
def generate_graphlist(num_nodes_to_sample,no_of_hops,data):
    # stores labels of each sub graph --> center node : {node: label}, ..
    y_labels_dict= {}
    # List to store sampled graphs
    graph_list = []
    #choose some random nodes to sample
    sampled_indices = torch.randint(0, data.num_nodes, (num_nodes_to_sample,))
    # Convert the PyG graph to NetworkX graph
    nx_graph = to_networkx(data, to_undirected=True)

    # Convert sampled indices to integers, list of center nodes
    nx_ids = [int(node_id.item()) for node_id in sampled_indices]
   
    for center_node in nx_ids:
        sampled_subgraph = nx.ego_graph(nx_graph, center_node, radius=no_of_hops, undirected=True)
        y_labels_dict[center_node] = {}  # Initialize dictionary for this center node
        for node in sampled_subgraph.nodes():
            y_labels_dict[center_node][node] = data.y[node].item()  # Store y label
        graph_list.append(sampled_subgraph)
    return y_labels_dict, nx_ids, graph_list

# This function converts the edge list to adjacency list
def edge_list_to_adjacency_list(edge_list):
    adjacency_list = {}

    for edge in edge_list:
        u, v = edge

        # Add u to v's adjacency list
        if v in adjacency_list:
            adjacency_list[v].append(u)
        else:
            adjacency_list[v] = [u]

        # Add v to u's adjacency list (assuming it's an undirected graph)
        if u in adjacency_list:
            adjacency_list[u].append(v)
        else:
            adjacency_list[u] = [v]
    return adjacency_list

# This function generates a gml or graph ML format of the graph
def generate_GML(graph, format):
    if format == "graphml":
        #For XML GML Format
        graphml_string = ""
        for line in nx.generate_graphml(graph, prettyprint=True):
            graphml_string += line + "\n"
        return graphml_string
    else:
        # For Normal GML Format
        gml = nx.generate_gml(graph)
        return "\n".join(gml)

     
    
#G = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)])

# Generate an edge list from the graph
#edge_list = generate_edgelist(G)

# Write the edge list to a file
#write_edgelist_to_file(edge_list, 'code/plots/edgelist.txt')
    