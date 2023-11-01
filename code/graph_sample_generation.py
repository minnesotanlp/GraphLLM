from littleballoffur import ForestFireSampler, RandomWalkSampler, CommonNeighborAwareRandomWalkSampler
import numpy as np
import networkx as nx
import time

def generate_CNAware_randomwalk_sample(graph, sample_size):
    sampler = CommonNeighborAwareRandomWalkSampler(number_of_nodes=sample_size)
    cnrw_graph = sampler.sample(graph)
    return cnrw_graph

def generate_randomwalk_sample(graph, sample_size):
    sampler = RandomWalkSampler(number_of_nodes=sample_size)
    rw_graph = sampler.sample(graph)
    return rw_graph

def generate_egograph_sample(graph, hops, sample_size):
    node_list = list(graph.nodes())
    while True:
        # Randomly select a node from the graph as the ego node
        ego_node = np.random.choice(node_list) 
        # Extract the 2-hop ego subgraph around the ego node
        ego_graph = nx.ego_graph(graph, ego_node, radius=hops)
        if ego_graph.number_of_nodes() == sample_size:
            break
    return ego_node, ego_graph
    

def generate_forestfire_sample(graph, sample_size):
    current_time = int(time.time())
    sampler = ForestFireSampler(number_of_nodes=sample_size, p=0.3, seed=current_time)
    ff_graph = sampler.sample(graph)
    return ff_graph
