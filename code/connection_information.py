import random
from torch_geometric.utils import to_networkx
import networkx as nx
import torch 

def generate_edgelist(graph):
    # Print out the labels associated with a graph
    edge_list = list(graph.edges())
    return edge_list

def generate_textual_edgelist2(edge_list):           
        # Convert the edge list information into text
        edgelist_converted = ''
        for edge in edge_list:
            source, target = edge
            edgelist_converted += f'Node {source} - Node {target}. '
        return edgelist_converted

def generate_textual_edgelist(edge_list):           
        # Convert the edge list information into text
        edgelist_converted = ''
        for edge in edge_list:
            source, target = edge
            edgelist_converted += f'Node {source} is connected to Node {target}. '
        return edgelist_converted

def generate_node_label_dict(graph, node_with_question_mark, center_node, y_labels_dict): 
    ground_truth =  y_labels_dict[center_node][node_with_question_mark]
    node_label_dict= {} # node: label
    for node in graph.nodes():
        if node == node_with_question_mark:
            label = "?"
        else:
            label = y_labels_dict[center_node][node]  # Extract node label
        node_label_dict[node]=label
    return ground_truth, node_label_dict

# modified such that it only has <50 nodes and <50 edges
def generate_graphlist_constrained(num_nodes_to_sample, no_of_hops, data):
    # stores labels of each sub graph --> center node : {node: label}, ..
    y_labels_dict = {}
    # List to store sampled graphs
    graph_list = []

    # Convert the PyG graph to NetworkX graph
    nx_graph = to_networkx(data, to_undirected=True)
    sampled_nodes = set()

    while len(graph_list) < num_nodes_to_sample:
        # Choose a random node index
        center_node_index = random.randint(0, data.num_nodes - 1)
        center_node = int(center_node_index)

        if center_node in sampled_nodes:
            continue

        sampled_subgraph = nx.ego_graph(nx_graph, center_node, radius=no_of_hops, undirected=True)

        # Check the size of the subgraph before adding it
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

    # Sample and yield subgraphs one at a time
    #for center_node in nx_ids:
    #    sampled_subgraph = nx.ego_graph(nx_graph, center_node, radius=no_of_hops, undirected=True)
    #    y_labels_dict[center_node] = {}  # Initialize dictionary for this center node
    #    for node in sampled_subgraph.nodes():
    #        y_labels_dict[center_node][node] = data.y[node].item()  # Store y label
    #    yield y_labels_dict[center_node], center_node, sampled_subgraph

   
    for center_node in nx_ids:
        sampled_subgraph = nx.ego_graph(nx_graph, center_node, radius=no_of_hops, undirected=True)
        y_labels_dict[center_node] = {}  # Initialize dictionary for this center node
        for node in sampled_subgraph.nodes():
            y_labels_dict[center_node][node] = data.y[node].item()  # Store y label
        graph_list.append(sampled_subgraph)
    return y_labels_dict, nx_ids, graph_list

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

def generate_GML(graph):
    # For Normal GML Format
    gml = nx.generate_gml(graph)
    return "\n".join(gml)

    # For XML GML Format
    # graphml_string = ""
    # for line in nx.generate_graphml(graph, prettyprint=True):
    #     graphml_string += line + "\n"
    # return graphml_string

    