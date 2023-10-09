import networkx as nx
from torch_geometric.utils import to_networkx
from utils import load_dataset
import random
import openai
import json
import os
import numpy as np
import statistics
import matplotlib.pyplot as plt
from graph_sample_generation import generate_egograph_sample, generate_forestfire_sample, generate_randomwalk_sample, generate_CNAware_randomwalk_sample


# ---- PARAMS --- #
#-------------------------------------------------------------
#random.seed(10)
openai.api_key = os.environ["OPENAI_API_UMNKEY"]
# Load configuration from the JSON file
with open('code/config1.json', 'r') as config_file:
    config = json.load(config_file)
# Access parameters from the config dictionary
dataset_name = config["dataset_name"]
no_of_hops = config["NO_OF_HOPS"]
use_edge = config["USE_EDGE"]
num_samples = config["NO_OF_SAMPLED_NODES"] # no of ego graphs
no_of_runs = config["RUN_COUNT"]
USE_ADJACENCY = config["USE_ADJACENCY"]
compression = config['compression']
model = config["model"]
rate_limit_pause = config["rate_limit_pause"]
data_dir = config["data_dir"]
log_dir = config["log_dir"]
result_location = config["result_location"]
neighborhood_sampling_flag = config["neighborhood_sampling"]

average_2hop_size = 36
#-------------------------------------------------------------

dataset = load_dataset(data_dir, dataset_name)
data = dataset[0]
# Convert the PyG graph to NetworkX graph
graph = to_networkx(data, to_undirected=True)



# ranges of samples 
desired_sizes = [
    int(average_2hop_size * 0.25),  # 25% of average size
    int(average_2hop_size * 0.5),  # 50% of average size
    average_2hop_size,             # 100% of average size
    int(average_2hop_size * 1.5),  # 150% of average size
    int(average_2hop_size * 2),    # 200% of average size
]
#Initialize overall data storage
#data = {'ego': {'edges': []}, 'ff': {'edges': []}, 'rw': {'edges': []}, 'cnrw': {'edges': []}}
data = {'ego': {'edges': []}, 'ff': {'edges': []}}
# Validate sizes (e.g., ensuring they are not zero or negative)
desired_sizes = [size for size in desired_sizes if size > 0]
assert len(desired_sizes)==5

for run in range(5): 
    ego_edges_sample = []
    ff_edges_sample = []
    rw_edges_sample = []
    cnrw_edges_sample = []

    for size in desired_sizes: # runs 5 times
        # Generate ego graph sample and count edges
        ego_node, ego_graph = generate_egograph_sample(graph, no_of_hops, size)
        ego_edges_sample.append(ego_graph.number_of_edges())
        
        # Generate forest fire graph sample and count edges (this should be a deterministic value )
        ff_graph = generate_forestfire_sample(graph, size)
        ff_edges_sample.append(ff_graph.number_of_edges())

        # Generate random walk graph sample and count edges 
        #rw_graph = generate_randomwalk_sample(graph, size)
        #rw_edges_sample.append(rw_graph.number_of_edges())

        # Generate common neighbor aware random walk graph sample and count edges 
        # The random walker is biased to visit neighbors that have a lower number of common neighbors. This way the sampling procedure is able to escape tightly knit communities and visit new ones.
        #cnrw_graph = generate_CNAware_randomwalk_sample(graph, size)
        #cnrw_edges_sample.append(cnrw_graph.number_of_edges())

    # Calculate and print averages and standard deviations for N runs
    avg_ego_edges = statistics.mean(ego_edges_sample)
    std_ego_edges = statistics.stdev(ego_edges_sample)
    avg_ff_edges = statistics.mean(ff_edges_sample)
    std_ff_edges = statistics.stdev(ff_edges_sample)
    #avg_rw_edges = statistics.mean(rw_edges_sample)
    #std_rw_edges = statistics.stdev(rw_edges_sample)
    #avg_cnrw_edges = statistics.mean(cnrw_edges_sample)
    #std_cnrw_edges = statistics.stdev(cnrw_edges_sample)
    
    # Store the generated data
    data['ego']['edges'].append(ego_edges_sample)
    data['ff']['edges'].append(ff_edges_sample)
    #data['rw']['edges'].append(rw_edges_sample)
    #data['cnrw']['edges'].append(cnrw_edges_sample)

    print(f'Run count #:',run)
    print(f"Ego Graph Samples: Avg Edges = {avg_ego_edges:.2f} (std = {std_ego_edges:.2f})")
    print(f"Forest Fire Graph Samples: Avg Edges = {avg_ff_edges:.2f} (std = {std_ff_edges:.2f})")
    #print(f"Random Walk Graph Samples: Avg Edges = {avg_rw_edges:.2f} (std = {std_rw_edges:.2f})")
    #print(f"CNA Random Walk Graph Samples: Avg Edges = {avg_cnrw_edges:.2f} (std = {std_cnrw_edges:.2f})")
    


# Visualization
# Extracting data
ego_edges_data = np.array(data['ego']['edges'])
ff_edges_data = np.array(data['ff']['edges'])
#rw_edges_data = np.array(data['rw']['edges'])
#cnrw_edges_data = np.array(data['cnrw']['edges'])

# Set up the subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Edges Distribution for Various Graph Types')

# Plotting for any number of runs
number_of_runs = len(ego_edges_data)
for run in range(number_of_runs):
    axs[0, 0].plot(desired_sizes, ego_edges_data[run, :], marker='o', label=f'Run {run+1}')
    axs[0, 1].plot(desired_sizes, ff_edges_data[run, :], marker='o', label=f'Run {run+1}')
    #axs[1, 0].plot(desired_sizes, rw_edges_data[run, :], marker='o', label=f'Run {run+1}')
    #axs[1, 1].plot(desired_sizes, cnrw_edges_data[run, :], marker='o', label=f'Run {run+1}')

# Adding labels and legends
axs[0, 0].set_title('Ego Graph Edges')
axs[0, 0].set_xlabel('Sample Size')
axs[0, 0].set_ylabel('Number of Edges')
axs[0, 0].legend()

axs[0, 1].set_title('Forest Fire Graph Edges')
axs[0, 1].set_xlabel('Sample Size')
axs[0, 1].set_ylabel('Number of Edges')
axs[0, 1].legend()
"""

axs[1, 0].set_title('Random Walk Graph Edges')
axs[1, 0].set_xlabel('Sample Size')
axs[1, 0].set_ylabel('Number of Edges')
axs[1, 0].legend()

axs[1, 1].set_title('CN Random Walk Graph Edges')
axs[1, 1].set_xlabel('Sample Size')
axs[1, 1].set_ylabel('Number of Edges')
axs[1, 1].legend()
"""
# Show the plot
plt.show()