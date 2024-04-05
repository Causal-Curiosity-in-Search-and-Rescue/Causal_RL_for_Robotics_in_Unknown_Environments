import numpy as np
import pandas as pd
import networkx as nx
from causalnex.structure.notears import from_pandas_lasso
from cdt.metrics import precision_recall, SHD
import matplotlib.pyplot as plt
import pdb

np.random.seed(42)  # Ensure reproducibility

def generate_data(n_samples):
    """Generates dataset based on specified relationships."""
    texture_1 = np.random.randint(0, 2, n_samples)
    texture_2 = np.random.randint(0, 2, n_samples)
    movability = (texture_1 & texture_2)
    return pd.DataFrame({'Texture_1': texture_1, 'Texture_2': texture_2, 'Movability': movability})

def generate_true_graph():
    """Generates the true graph structure as a NetworkX undirected graph."""
    G = nx.Graph()
    G.add_nodes_from(["Texture_1", "Texture_2", "Movability"])
    G.add_edge("Texture_1", "Movability")
    G.add_edge("Texture_2", "Movability")
    return G

def calculate_metrics(inferred_graph, true_graph, node_names):
    """Calculates SHD and precision between the true graph and the inferred graph."""
    # Ensure that both true_graph and inferred_graph are nx.Graph() and include all nodes
    true_adj_matrix = nx.to_numpy_array(true_graph, nodelist=node_names)
    inferred_adj_matrix = nx.to_numpy_array(inferred_graph, nodelist=node_names)

    # Symmetrize the adjacency matrices for undirected graph comparison
    true_adj_matrix = np.maximum(true_adj_matrix, true_adj_matrix.T)
    inferred_adj_matrix = np.maximum(inferred_adj_matrix, inferred_adj_matrix.T)

    # Calculate SHD
    shd = SHD(true_adj_matrix, inferred_adj_matrix)

    # Calculate precision and recall using cdt.metrics.precision_recall
    # Flatten the adjacency matrices to compare edge presence as binary arrays
    true_edges = true_adj_matrix.flatten()
    inferred_edges = inferred_adj_matrix.flatten()
    precision, recall = precision_recall(inferred_edges.astype(int), true_edges.astype(int))

    return shd, precision

# Hyperparameters
w_threshold = 0.8
beta = 0.01
max_iter = 50

# Generate the true graph for comparison
true_graph = generate_true_graph()

# Sample sizes to iterate over
sample_sizes = list(range(1,30))

# Results placeholder
results = []

# Loop over sample sizes
for n_samples in sample_sizes:
    shd_values = []
    precision_values = []
    # Generate 10 different samples for each sample size
    for _ in range(10):
        df = generate_data(n_samples)
        # For each sample, perform 10 trials of graph inference
        for _ in range(10):
            sm = from_pandas_lasso(df, w_threshold=w_threshold, beta=beta, max_iter=max_iter)
            # Create an inferred graph and ensure all nodes are present
            inferred_graph = nx.Graph()
            inferred_graph.add_nodes_from(['Texture_1', 'Texture_2', 'Movability'])
            if len(sm.edges()) > 0:
                inferred_graph.add_edges_from(sm.edges())
            if n_samples > 10:
                pdb.set_trace()

            shd, precision = calculate_metrics(inferred_graph, true_graph, node_names=['Texture_1', 'Texture_2', 'Movability'])
            shd_values.append(shd)
            precision_values.append(precision)

    # Calculate average and standard deviation of SHD and precision
    results.append({
        'n_samples': n_samples,
        'shd_avg': np.mean(shd_values),
        'shd_std': np.std(shd_values),
        'precision_avg': np.mean(precision_values),
        'precision_std': np.std(precision_values)
    })

# Convert results to a DataFrame for display
results_32_df = pd.DataFrame(results)
pdb.set_trace()
