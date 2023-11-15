import matplotlib.pyplot as plt
import networkx as nx

# This function plots the given graph structure with its labels 
def plot_graph_structure(graph, y_labels_graph, node_with_question_mark, size, title, ego_flag):
    # Setup plot environment
    plt.figure(figsize=(12, 6))
    if ego_flag : # if its a ego graph the y_labels structure is {ego node: {node: label, node: label}}
        # Creating new labels for display
        display_labels = {node: ('?' if node == node_with_question_mark else y_labels_graph[node_with_question_mark][node]) for node in graph.nodes()}
        
        # Setting color of the ego_node to red and others to skyblue
        colors = ['red' if node == node_with_question_mark else 'skyblue' for node in graph.nodes()]
        
    else: # if its a normal graph the y_labels structure is {node: label, node: label}

         # Creating new labels for display
        display_labels = {node: ('?' if node == node_with_question_mark else y_labels_graph[node]) for node in graph.nodes()}
        
        # Setting color of the node_with_question_mark to red and others to skyblue
        colors = ['red' if node == node_with_question_mark else 'skyblue' for node in graph.nodes()]
    
    nx.draw(graph, labels=display_labels, node_color=colors, with_labels=True, font_weight='bold', node_size=700)
    plt.title(f'{title}_{size}')
    
    plt.savefig(f'code/plots/{title}{size}.png') 
    plt.close()
        
# plot the graph structure where you're coloring the graphs of the same label with the same color
def plot_graph_structure_community_colored(graph, y_labels_graph, node_with_question_mark, index, title, result_location, ego_flag):
    # Define a set of colors for nodes
    colors_list = ['blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'grey',
                   'lime', 'navy', 'gold', 'maroon', 'turquoise', 'olive', 'indigo', 'lightcoral',
                   'darkgreen', 'chocolate', 'magenta', 'lightseagreen', 'black']
    
    
    # Setup plot environment
    plt.figure(figsize=(12, 6))
    
    # Color mapping based on labels
    color_mapping = {}
    current_color_index = 0
    
    colors = []
    for node in graph.nodes():
        if ego_flag:
            label = '?' if node == node_with_question_mark else y_labels_graph[node_with_question_mark][node]
        else:
            label = '?' if node == node_with_question_mark else y_labels_graph[node]
            
        if label == '?':
            colors.append('red')  # Node with '?' label will be red
        else:
            if label not in color_mapping:
                color_mapping[label] = colors_list[current_color_index]
                current_color_index += 1  # Move to the next color for a new label

            colors.append(color_mapping[label])
    
    # Creating new labels for display
    display_labels = {}
    for node in graph.nodes():
        if node == node_with_question_mark:
            display_labels[node] = '?'
        else :
            display_labels[node] = y_labels_graph[node_with_question_mark][node] if ego_flag else y_labels_graph[node]

    nx.draw(graph, labels=display_labels, node_color=colors, with_labels=True, font_weight='bold', node_size=700)
    plt.title(f'{title}')
    
    plt.savefig(f'{result_location}/{index}.png')
    plt.close()

def plot_graphviz_graph(graph, y_labels_graph, node_with_question_mark, index, title, result_location, node_neighbors, ego_flag):
    # Define a set of colors for nodes
    colors_list = ['blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'grey',
                   'lime', 'navy', 'gold', 'maroon', 'turquoise', 'olive', 'indigo', 'lightcoral',
                   'darkgreen', 'chocolate', 'magenta', 'lightseagreen', 'black']
    
    dark_colors = ['blue', 'purple', 'navy', 'indigo', 'black', 'darkgreen']
    
    # Color mapping based on labels
    color_mapping = {}
    current_color_index = 0
    
    graph.graph_attr["overlap"] = False
    graph.graph_attr['splines'] = True
    graph.graph_attr['sep'] = '+2'
    graph.graph_attr['esep'] = '+1'

    for node in graph.nodes():
        node.attr["fontsize"] = "25pt"
        node.attr["style"] = "filled"
        if ego_flag:
            label = '?' if str(node) == node_with_question_mark else y_labels_graph[node_with_question_mark][str(node)]
        else:
            label = '?' if str(node) == node_with_question_mark else y_labels_graph[str(node)]
        
        if label == '?':
            node.attr["color"] = "red"
        else:
            if label not in color_mapping:
                color_mapping[label] = colors_list[current_color_index]
                current_color_index += 1  # Move to the next color for a new label
            node.attr["color"] = color_mapping[label]
            if color_mapping[label] in dark_colors:
                node.attr["fontcolor"] = 'white'

    # Set Labels in GraphViz
    for node in graph.nodes():
        if str(node) == node_with_question_mark:
            node.attr["label"] = '?'
            node.attr['fontsize'] = '35pt'
        else:
            node.attr["label"] = y_labels_graph[node_with_question_mark][str(node)] if ego_flag else y_labels_graph[str(node)]

        if str(node) in node_neighbors:
            node.attr['fontsize'] = '35pt'

    graph.draw(f'{result_location}/{index}_new.png', prog="neato")


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