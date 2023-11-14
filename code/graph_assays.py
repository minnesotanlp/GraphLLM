import networkx as nx
# Function to count triangles in the graph
def count_triangles_nx(graph):
    triangles = list(nx.enumerate_all_cliques(graph))
    triangle_count = sum(1 for clique in triangles if len(clique) == 3)
    return triangle_count

# Function to find all 3-cliques connected to a given node
def find_3_cliques_connected_to_node(graph, node):
    # Create a subgraph induced by the neighbors of the given node
    neighbors = set(graph.neighbors(node))
    subgraph = graph.subgraph(neighbors | {node})

    # Enumerate all 3-cliques in the subgraph
    triangles = list(nx.enumerate_all_cliques(subgraph))
    return [clique for clique in triangles if len(clique) == 3]

# Function to find Hamiltonian cycles in the graph
def find_hamiltonian_cycles(graph):
    hamiltonian_cycles = list(nx.simple_cycles(graph))
    return [cycle for cycle in hamiltonian_cycles if len(cycle) == graph.number_of_nodes() and cycle[0] == cycle[-1]]

def count_star_graphs(G):
    # Function to check if a vertex forms a star graph
    def is_star(G, vertex):
        neighbors = list(G.neighbors(vertex))
        if len(neighbors) < 2:  # A star graph must have at least 2 neighbors
            return False
        for neighbor in neighbors:
            if len(list(G.neighbors(neighbor))) != 1:  # Neighbors must only connect to the central vertex
                return False
        return True

    # Count the star graphs
    star_graph_count = 0
    for vertex in G.nodes:
        if is_star(G, vertex):
            star_graph_count += 1

    return star_graph_count


def find_star_graphs(G, specific_vertex):
    # Function to check if a vertex forms a star graph
    def is_star(G, vertex):
        neighbors = list(G.neighbors(vertex))
        if len(neighbors) < 2:  # A star graph must have at least 2 neighbors
            return False
        for neighbor in neighbors:
            if len(list(G.neighbors(neighbor))) != 1:  # Neighbors must only connect to the central vertex
                return False
        return True
    
    # Finding the star graphs
    star_graphs = []
    if is_star(G, specific_vertex):
        star_graph = [specific_vertex] + list(G.neighbors(specific_vertex))
        star_graphs.append(star_graph)
    
    return star_graphs


# Function to get star motifs that a node is part of
def get_star_motifs_connected_to_node(graph, node):
    star_motifs = []
    neighbors = list(graph.neighbors(node))

    # If the node has more than two neighbors and no edges between those neighbors,
    # then it is the center of a star motif
    if len(neighbors) > 2 and all(not graph.has_edge(neighbors[i], neighbors[j]) 
                                   for i in range(len(neighbors)) 
                                   for j in range(i + 1, len(neighbors))):
        star_motifs.append([node] + neighbors)
    
    # If the node is not the center, check if it's part of a star motif
    for neighbor in neighbors:
        # Get the neighbors of 'neighbor', excluding 'node', and make a list out of it
        sub_neighbors_list = list(set(graph.neighbors(neighbor)) - {node})
        
        # Check if 'neighbor' is the center of a star motif
        if len(sub_neighbors_list) > 1 and all(not graph.has_edge(sub_neighbors_list[i], sub_neighbors_list[j]) 
                                           for i in range(len(sub_neighbors_list)) 
                                           for j in range(i + 1, len(sub_neighbors_list))):
            star_motifs.append([neighbor] + sub_neighbors_list + [node])

    return star_motifs

def get_count_and_cliques_of_node(graph, node):
    # Find all maximal cliques in the graph
    all_cliques = nx.find_cliques(graph)
    # Filter cliques that are larger than size 4 and contain the specified node
    relevant_cliques = [clique for clique in all_cliques if node in clique and len(clique) > 3]

    # Count of such cliques
    count = len(relevant_cliques)

    return count, relevant_cliques

def is_node_attached_to_clique(graph, clique, node):
    # Count the number of connections node has with nodes in the clique
    connections = sum(1 for member in clique if graph.has_edge(member, node))
    # A node is attached to a clique if it is connected to all but at most one of the clique's nodes
    return connections >= len(clique) - 1

def find_cliques_connected_node(graph, node):
    # Find all maximal cliques in the graph
    cliques_connected = []
    c_flag = False
    all_cliques = nx.find_cliques(graph)
    # Check if the node is attached to any of these cliques
    for clique in all_cliques:
        if len(clique) > 3 and is_node_attached_to_clique(graph, clique, node):
            c_flag = True
            cliques_connected.append(clique) # The node is attached to at least one clique
    if c_flag:
        return cliques_connected
    else:
        return [] # The node is not attached to any clique

# Example usage
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (3, 5)])
print(find_cliques_connected_node(G, 5)) 

# Example usage:
G = nx.Graph()
edges1 = [(1, 5), (2, 5), (3, 5), (7, 5), (6, 5), (4, 5)]
G.add_edges_from(edges1)
print(find_cliques_connected_node(G, 5))  # Output: 0, []

G = nx.Graph()
edges1 = [(1, 2), (1,4), (1,5),(2,1),(2,4),(2, 3),(2,5), (3,2), (4,1),(4,2),(4, 5), (5,1),(5,2),(5,4)]
G.add_edges_from(edges1)
print(find_cliques_connected_node(G, 3))  # Output: 0, []

edges = [(1, 2), (1, 3), (1, 4), (2, 3),(2, 4), (3, 4)]
G.add_edges_from(edges)
print(find_cliques_connected_node(G, 3))
#print(get_star_motifs(G, 5))  # Output: [[5, 1, 2, 3, 7, 6, 4]]

#G = nx.Graph()
#edges2 = [(1, 5), (2, 5), (3, 5), (7, 5), (6, 5), (4, 5), (5,8), (8,9), (8,10)]
#G.add_edges_from(edges2)
#print(get_star_motifs(G, 8))  # Output: [[8, 5, 1, 2, 3, 7, 6, 4]]

#G = nx.Graph()
#edges3 = [(1, 2), (2, 3), (4, 2), (4,6), (5, 6), (7, 6), (8, 6)]
#G.add_edges_from(edges3)
#print(get_star_motifs(G, 4))  # Output: [[6, 5, 7, 8]]

# Test the function with the provided examples
#G = nx.Graph()

#edges1 = [(1, 5), (2, 5), (3, 5), (7, 5), (6, 5), (4, 5)]
#G.add_edges_from(edges1)
#print(count_star_graphs(G))  # Output: 1

#G = nx.Graph()
#edges2 = [(1, 2), (2, 3), (4, 2), (5, 6), (7, 6), (8, 6)]
#G.add_edges_from(edges2)
#print(count_star_graphs(G))  # Output: 2

#G=nx.Graph()
#edges3 = [(8,7), (1, 5), (2, 5), (3, 5), (7, 5), (6, 5), (4, 5), (1, 8)]
#G.add_edges_from(edges3)
#print(count_star_graphs(G))

#G=nx.Graph()
#edges4 = [(1,2), (1, 3), (1, 5), (1, 6)]
#G.add_edges_from(edges4)
#print(count_star_graphs(G))

#G=nx.Graph()
#edges5 = [(1,2), (1, 3), (1, 5), (1, 6), (1,7), (7,8), (1, 8), (8,9), (8, 10)]
#G.add_edges_from(edges5)
#print(count_star_graphs(G))

#G=nx.Graph()
#edges6 = [(1,2), (1, 3), (1, 4), (1, 5)]
#G.add_edges_from(edges6)
#print(count_star_graphs(G))
