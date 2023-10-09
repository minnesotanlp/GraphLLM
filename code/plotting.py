import matplotlib.pyplot as plt
import networkx as nx

# This function plots the given graph structure with its labels 
def plot_graph_structure(graph, y_labels_graph, ego_node, size, title, ego_flag):
    # Setup plot environment
    plt.figure(figsize=(12, 6))
    if ego_flag : # if its a ego graph the y_labels structure is {ego node: {node: label, node: label}}
        ego_labels = y_labels_graph[ego_node]  # Extract inner dictionary of labels
        ego_colors = [ego_labels[node] for node in graph.nodes()]  # Map labels to nodes
        #plt.subplot(121)
        nx.draw(graph, node_color=ego_colors, with_labels=True, font_weight='bold', node_size=700, cmap=plt.cm.Blues)
        plt.title(f'{title} {size}')
    else: # if its a normal graph the y_labels structure is {node: label, node: label}
        ff_colors = [y_labels_graph.get(node, 'y') for node in graph.nodes()]  # Map labels to nodes, use a default color if node not found
        #plt.subplot(122)
        nx.draw(graph, node_color=ff_colors, with_labels=True, font_weight='bold', node_size=700, cmap=plt.cm.Reds)
        plt.title(f'{title} {size}')
        # Show the plot
        plt.show()

# think this does the same as the above function
def visualize_ego_graph(ego_graph, y_labels, ego_node):
    y_labels_egograph = y_labels[ego_node]
    # Create a subgraph to include only the ego graph
    subgraph = ego_graph.subgraph(list(y_labels_egograph.keys()))
    
    # Get node labels from y_labels_egograph
    node_labels = {node: str(label) for node, label in y_labels_egograph.items()}
    
    # Get node colors based on labels (assuming labels are integers)
    node_colors = [y_labels_egograph[node] for node in subgraph.nodes()]
    
    # Draw the ego graph with labels and colors
    pos = nx.spring_layout(subgraph)  # Layout for node positioning
    nx.draw(subgraph, pos, with_labels=True, labels=node_labels,
            node_color=node_colors, cmap=plt.cm.jet, node_size=200)
    
    # Set plot title
    plt.title(f'Ego Graph Visualization (Ego Node: {ego_node})')
    
    # Show the plot
    plt.show()