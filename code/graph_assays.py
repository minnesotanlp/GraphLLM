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

import networkx as nx

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

# Example usage:
G = nx.Graph()
edges1 = [(1, 5), (2, 5), (3, 5), (7, 5), (6, 5), (4, 5)]
G.add_edges_from(edges1)
print(find_star_graphs(G, 5))  # Output: [[5, 1, 2, 3, 7, 6, 4]]

G = nx.Graph()
edges2 = [(1, 2), (2, 3), (4, 2), (4,6), (5, 6), (7, 6), (8, 6)]
G.add_edges_from(edges2)
print(find_star_graphs(G, 4))  # Output: [[6, 5, 7, 8]]




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
