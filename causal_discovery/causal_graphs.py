# -*- coding: utf-8 -*-
"""Causal_graphs_GDP_060424.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cyPVEcqFcnmWAdg3xwoaxV7LSEEywGLm

# Install Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# pip install causalnex

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# pip install cdt

import warnings
#warnings.filterwarnings('ignore')

"""# Three variables

## One affects
"""

import numpy as np
import pandas as pd
import networkx as nx
from causalnex.structure.notears import from_pandas_lasso
from cdt.metrics import precision_recall
import matplotlib.pyplot as plt

np.random.seed(42)  # Ensure reproducibility

def generate_data(n_samples):
    """Generates dataset based on specified relationships."""
    texture_1 = np.random.randint(0, 2, n_samples)
    texture_2 = np.random.randint(0, 2, n_samples)
    movability = texture_1
    return pd.DataFrame({'Texture_1': texture_1, 'Texture_2': texture_2, 'Movability': movability})

def generate_true_graph():
    """Generates the true graph structure as a NetworkX undirected graph."""
    G = nx.Graph()
    G.add_nodes_from(["Texture_1", "Texture_2", "Movability"])
    G.add_edge("Texture_1", "Movability")
    return G

def manual_shd_calculation(inferred_graph, true_graph):
    """Calculates SHD manually for undirected graphs."""
    true_edges = set(true_graph.edges())
    inferred_edges = set(inferred_graph.edges())
    # Calculate symmetric difference to find edges that are not shared
    symmetric_diff = true_edges.symmetric_difference(inferred_edges)
    shd = len(symmetric_diff)
    return shd

def calculate_metrics(inferred_graph, true_graph, node_names):
    """Calculates SHD and precision between the true graph and the inferred graph."""
    shd = manual_shd_calculation(inferred_graph, true_graph)

    # Flatten the adjacency matrices for precision and recall calculation
    true_adj_matrix = nx.to_numpy_array(true_graph, nodelist=node_names)
    inferred_adj_matrix = nx.to_numpy_array(inferred_graph, nodelist=node_names)
    true_edges = true_adj_matrix.flatten()
    inferred_edges = inferred_adj_matrix.flatten()
    precision, recall = precision_recall(true_edges.astype(int), inferred_edges.astype(int))

    return shd, precision

# Hyperparameters
w_threshold = 0.8
beta = 0.01
max_iter = 50

# Generate the true graph for comparison
true_graph = generate_true_graph()

# Sample sizes to iterate over
sample_sizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

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
            inferred_graph = nx.Graph()
            inferred_graph.add_nodes_from(['Texture_1', 'Texture_2', 'Movability'])
            inferred_graph.add_edges_from(sm.edges())

            shd, precision = calculate_metrics(inferred_graph, true_graph, node_names=['Texture_1','Texture_2', 'Movability'])
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
results_31_df = pd.DataFrame(results)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# First y-axis for SHD
color_shd = 'tab:red'
ax1.set_xlabel('Number of Samples')
ax1.set_ylabel('SHD', color=color_shd)
ax1.errorbar(results_31_df['n_samples'], results_31_df['shd_avg'], yerr=results_31_df['shd_std'],
             fmt='o-', color=color_shd, ecolor='lightgray', capsize=5, label='SHD')
ax1.tick_params(axis='y', labelcolor=color_shd)
ax1.set_ylim(-1, 4)  # Set the limit for SHD axis

# Second y-axis for Precision
ax2 = ax1.twinx()
color_precision = 'tab:blue'
ax2.set_ylabel('Precision', color=color_precision)
ax2.errorbar(results_31_df['n_samples'], results_31_df['precision_avg'], yerr=results_31_df['precision_std'],
             fmt='o-', color=color_precision, ecolor='lightgray', capsize=5, label='Precision')
ax2.tick_params(axis='y', labelcolor=color_precision)
ax2.set_ylim(-0.5, 1.5)  # Set the limit for Precision axis

# Title and legend
fig.tight_layout()
plt.title('SHD and Precision over Number of Samples')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))  # Adjust legend position

plt.show()

print(results_31_df)

"""## Both affects"""

import numpy as np
import pandas as pd
import networkx as nx
from causalnex.structure.notears import from_pandas_lasso
from cdt.metrics import precision_recall
import matplotlib.pyplot as plt

np.random.seed(42)  # Ensure reproducibility

def generate_data(n_samples):
    """Generates dataset based on specified relationships."""
    texture_1 = np.random.randint(0, 2, n_samples)
    texture_2 = np.random.randint(0, 2, n_samples)
    movability = texture_1 & texture_2
    return pd.DataFrame({'Texture_1': texture_1, 'Texture_2': texture_2, 'Movability': movability})

def generate_true_graph():
    """Generates the true graph structure as a NetworkX undirected graph."""
    G = nx.Graph()
    G.add_nodes_from(["Texture_1", "Texture_2", "Movability"])
    G.add_edge("Texture_1", "Movability")
    G.add_edge("Texture_2","Movability")
    return G

def manual_shd_calculation(inferred_graph, true_graph):
    """Calculates SHD manually for undirected graphs."""
    true_edges = set(true_graph.edges())
    inferred_edges = set(inferred_graph.edges())
    # Calculate symmetric difference to find edges that are not shared
    symmetric_diff = true_edges.symmetric_difference(inferred_edges)
    shd = len(symmetric_diff)
    return shd

def calculate_metrics(inferred_graph, true_graph, node_names):
    """Calculates SHD and precision between the true graph and the inferred graph."""
    shd = manual_shd_calculation(inferred_graph, true_graph)

    # Flatten the adjacency matrices for precision and recall calculation
    true_adj_matrix = nx.to_numpy_array(true_graph, nodelist=node_names)
    inferred_adj_matrix = nx.to_numpy_array(inferred_graph, nodelist=node_names)
    true_edges = true_adj_matrix.flatten()
    inferred_edges = inferred_adj_matrix.flatten()
    precision, recall = precision_recall(true_edges.astype(int), inferred_edges.astype(int))

    return shd, precision

# Hyperparameters
w_threshold = 0.8
beta = 0.01
max_iter = 50

# Generate the true graph for comparison
true_graph = generate_true_graph()

# Sample sizes to iterate over
sample_sizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

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
            inferred_graph = nx.Graph()
            inferred_graph.add_nodes_from(['Texture_1', 'Texture_2', 'Movability'])
            inferred_graph.add_edges_from(sm.edges())

            shd, precision = calculate_metrics(inferred_graph, true_graph, node_names=['Texture_1','Texture_2', 'Movability'])
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

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# First y-axis for SHD
color_shd = 'tab:red'
ax1.set_xlabel('Number of Samples')
ax1.set_ylabel('SHD', color=color_shd)
ax1.errorbar(results_32_df['n_samples'], results_32_df['shd_avg'], yerr=results_32_df['shd_std'],
             fmt='o-', color=color_shd, ecolor='lightgray', capsize=5, label='SHD')
ax1.tick_params(axis='y', labelcolor=color_shd)
ax1.set_ylim(-1, 4)  # Set the limit for SHD axis

# Second y-axis for Precision
ax2 = ax1.twinx()
color_precision = 'tab:blue'
ax2.set_ylabel('Precision', color=color_precision)
ax2.errorbar(results_32_df['n_samples'], results_32_df['precision_avg'], yerr=results_32_df['precision_std'],
             fmt='o-', color=color_precision, ecolor='lightgray', capsize=5, label='Precision')
ax2.tick_params(axis='y', labelcolor=color_precision)
ax2.set_ylim(-0.5, 1.5)  # Set the limit for Precision axis

# Title and legend
fig.tight_layout()
plt.title('SHD and Precision over Number of Samples')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))  # Adjust legend position

plt.show()

print(results_32_df)

"""## None affects"""

import numpy as np
import pandas as pd
import networkx as nx
from causalnex.structure.notears import from_pandas_lasso
from cdt.metrics import precision_recall
import matplotlib.pyplot as plt

np.random.seed(42)  # Ensure reproducibility

def generate_data(n_samples):
    """Generates dataset based on specified relationships."""
    texture_1 = np.random.randint(0, 2, n_samples)
    texture_2 = np.random.randint(0, 2, n_samples)
    movability = np.random.randint(0, 2, n_samples)
    return pd.DataFrame({'Texture_1': texture_1, 'Texture_2': texture_2,'Movability': movability})

def generate_true_graph():
    """Generates the true graph structure as a NetworkX undirected graph."""
    G = nx.Graph()
    G.add_nodes_from(["Texture_1", 'Texture_2', "Movability"])
    return G

def manual_shd_calculation(inferred_graph, true_graph):
    """Calculates SHD manually for undirected graphs."""
    true_edges = set(true_graph.edges())
    inferred_edges = set(inferred_graph.edges())
    # Calculate symmetric difference to find edges that are not shared
    symmetric_diff = true_edges.symmetric_difference(inferred_edges)
    shd = len(symmetric_diff)
    return shd

def calculate_metrics(inferred_graph, true_graph, node_names):
    """Calculates SHD and precision between the true graph and the inferred graph."""
    shd = manual_shd_calculation(inferred_graph, true_graph)

    # Flatten the adjacency matrices for precision and recall calculation
    true_adj_matrix = nx.to_numpy_array(true_graph, nodelist=node_names)
    inferred_adj_matrix = nx.to_numpy_array(inferred_graph, nodelist=node_names)
    true_edges = true_adj_matrix.flatten()
    inferred_edges = inferred_adj_matrix.flatten()
    if true_edges.sum() == 0 and inferred_edges.sum() == 0:
        precision = 1.0  # If there are no true edges and no predicted edges, precision is defined as 1.
    else:
        precision, recall = precision_recall(true_edges.astype(int), inferred_edges.astype(int))

    return shd, precision

# Hyperparameters
w_threshold = 0.8
beta = 0.01
max_iter = 50

# Generate the true graph for comparison
true_graph = generate_true_graph()

# Sample sizes to iterate over
sample_sizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

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
            inferred_graph = nx.Graph()
            inferred_graph.add_nodes_from(['Texture_1', 'Movability'])
            inferred_graph.add_edges_from(sm.edges())

            shd, precision = calculate_metrics(inferred_graph, true_graph, node_names=['Texture_1', 'Movability'])
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
results_30_df = pd.DataFrame(results)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# First y-axis for SHD
color_shd = 'tab:red'
ax1.set_xlabel('Number of Samples')
ax1.set_ylabel('SHD', color=color_shd)
ax1.errorbar(results_30_df['n_samples'], results_30_df['shd_avg'], yerr=results_30_df['shd_std'],
             fmt='o-', color=color_shd, ecolor='lightgray', capsize=5, label='SHD')
ax1.tick_params(axis='y', labelcolor=color_shd)
ax1.set_ylim(-1, 4)  # Set the limit for SHD axis

# Second y-axis for Precision
ax2 = ax1.twinx()
color_precision = 'tab:blue'
ax2.set_ylabel('Precision', color=color_precision)
ax2.errorbar(results_30_df['n_samples'], results_30_df['precision_avg'], yerr=results_30_df['precision_std'],
             fmt='o-', color=color_precision, ecolor='lightgray', capsize=5, label='Precision')
ax2.tick_params(axis='y', labelcolor=color_precision)
ax2.set_ylim(-0.5, 1.5)  # Set the limit for Precision axis

# Title and legend
fig.tight_layout()
plt.title('SHD and Precision over Number of Samples')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))  # Adjust legend position

plt.show()

print(results_30_df)

"""# Two variables

## Affect
"""

import numpy as np
import pandas as pd
import networkx as nx
from causalnex.structure.notears import from_pandas_lasso
from cdt.metrics import precision_recall
import matplotlib.pyplot as plt

np.random.seed(42)  # Ensure reproducibility

def generate_data(n_samples):
    """Generates dataset based on specified relationships."""
    texture_1 = np.random.randint(0, 2, n_samples)
    movability = texture_1
    return pd.DataFrame({'Texture_1': texture_1, 'Movability': movability})

def generate_true_graph():
    """Generates the true graph structure as a NetworkX undirected graph."""
    G = nx.Graph()
    G.add_nodes_from(["Texture_1", "Movability"])
    G.add_edge("Texture_1", "Movability")
    return G

def manual_shd_calculation(inferred_graph, true_graph):
    """Calculates SHD manually for undirected graphs."""
    true_edges = set(true_graph.edges())
    inferred_edges = set(inferred_graph.edges())
    # Calculate symmetric difference to find edges that are not shared
    symmetric_diff = true_edges.symmetric_difference(inferred_edges)
    shd = len(symmetric_diff)
    return shd

def calculate_metrics(inferred_graph, true_graph, node_names):
    """Calculates SHD and precision between the true graph and the inferred graph."""
    shd = manual_shd_calculation(inferred_graph, true_graph)

    # Flatten the adjacency matrices for precision and recall calculation
    true_adj_matrix = nx.to_numpy_array(true_graph, nodelist=node_names)
    inferred_adj_matrix = nx.to_numpy_array(inferred_graph, nodelist=node_names)
    true_edges = true_adj_matrix.flatten()
    inferred_edges = inferred_adj_matrix.flatten()
    precision, recall = precision_recall(true_edges.astype(int), inferred_edges.astype(int))

    return shd, precision

# Hyperparameters
w_threshold = 0.8
beta = 0.01
max_iter = 50

# Generate the true graph for comparison
true_graph = generate_true_graph()

# Sample sizes to iterate over
sample_sizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

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
            inferred_graph = nx.Graph()
            inferred_graph.add_nodes_from(['Texture_1', 'Movability'])
            inferred_graph.add_edges_from(sm.edges())

            shd, precision = calculate_metrics(inferred_graph, true_graph, node_names=['Texture_1', 'Movability'])
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
results_21_df = pd.DataFrame(results)

print(results_21_df)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# First y-axis for SHD
color_shd = 'tab:red'
ax1.set_xlabel('Number of Samples')
ax1.set_ylabel('SHD', color=color_shd)
ax1.errorbar(results_21_df['n_samples'], results_21_df['shd_avg'], yerr=results_21_df['shd_std'],
             fmt='o-', color=color_shd, ecolor='lightgray', capsize=5, label='SHD')
ax1.tick_params(axis='y', labelcolor=color_shd)
ax1.set_ylim(-1, 4)  # Set the limit for SHD axis

# Second y-axis for Precision
ax2 = ax1.twinx()
color_precision = 'tab:blue'
ax2.set_ylabel('Precision', color=color_precision)
ax2.errorbar(results_21_df['n_samples'], results_21_df['precision_avg'], yerr=results_21_df['precision_std'],
             fmt='o-', color=color_precision, ecolor='lightgray', capsize=5, label='Precision')
ax2.tick_params(axis='y', labelcolor=color_precision)
ax2.set_ylim(-0.5, 1.5)  # Set the limit for Precision axis

# Title and legend
fig.tight_layout()
plt.title('SHD and Precision over Number of Samples')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))  # Adjust legend position

plt.show()

"""## Does not affect"""

import numpy as np
import pandas as pd
import networkx as nx
from causalnex.structure.notears import from_pandas_lasso
from cdt.metrics import precision_recall
import matplotlib.pyplot as plt

np.random.seed(42)  # Ensure reproducibility

def generate_data(n_samples):
    """Generates dataset based on specified relationships."""
    texture_1 = np.random.randint(0, 2, n_samples)
    movability = np.random.randint(0, 2, n_samples)
    return pd.DataFrame({'Texture_1': texture_1, 'Movability': movability})

def generate_true_graph():
    """Generates the true graph structure as a NetworkX undirected graph."""
    G = nx.Graph()
    G.add_nodes_from(["Texture_1", "Movability"])
    return G

def manual_shd_calculation(inferred_graph, true_graph):
    """Calculates SHD manually for undirected graphs."""
    true_edges = set(true_graph.edges())
    inferred_edges = set(inferred_graph.edges())
    # Calculate symmetric difference to find edges that are not shared
    symmetric_diff = true_edges.symmetric_difference(inferred_edges)
    shd = len(symmetric_diff)
    return shd

def calculate_metrics(inferred_graph, true_graph, node_names):
    """Calculates SHD and precision between the true graph and the inferred graph."""
    shd = manual_shd_calculation(inferred_graph, true_graph)

    # Flatten the adjacency matrices for precision and recall calculation
    true_adj_matrix = nx.to_numpy_array(true_graph, nodelist=node_names)
    inferred_adj_matrix = nx.to_numpy_array(inferred_graph, nodelist=node_names)
    true_edges = true_adj_matrix.flatten()
    inferred_edges = inferred_adj_matrix.flatten()
    if true_edges.sum() == 0 and inferred_edges.sum() == 0:
        precision = 1.0  # If there are no true edges and no predicted edges, precision is defined as 1.
    else:
        precision, recall = precision_recall(true_edges.astype(int), inferred_edges.astype(int))

    return shd, precision

# Hyperparameters
w_threshold = 0.8
beta = 0.01
max_iter = 50

# Generate the true graph for comparison
true_graph = generate_true_graph()

# Sample sizes to iterate over
sample_sizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

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
            inferred_graph = nx.Graph()
            inferred_graph.add_nodes_from(['Texture_1', 'Movability'])
            inferred_graph.add_edges_from(sm.edges())

            shd, precision = calculate_metrics(inferred_graph, true_graph, node_names=['Texture_1', 'Movability'])
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
results_20_df = pd.DataFrame(results)

print(results_20_df)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# First y-axis for SHD
color_shd = 'tab:red'
ax1.set_xlabel('Number of Samples')
ax1.set_ylabel('SHD', color=color_shd)
ax1.errorbar(results_20_df['n_samples'], results_20_df['shd_avg'], yerr=results_20_df['shd_std'],
             fmt='o-', color=color_shd, ecolor='lightgray', capsize=5, label='SHD')
ax1.tick_params(axis='y', labelcolor=color_shd)
ax1.set_ylim(-1, 4)  # Set the limit for SHD axis

# Second y-axis for Precision
ax2 = ax1.twinx()
color_precision = 'tab:blue'
ax2.set_ylabel('Precision', color=color_precision)
ax2.errorbar(results_20_df['n_samples'], results_20_df['precision_avg'], yerr=results_20_df['precision_std'],
             fmt='o-', color=color_precision, ecolor='lightgray', capsize=5, label='Precision')
ax2.tick_params(axis='y', labelcolor=color_precision)
ax2.set_ylim(-0.5, 1.5)  # Set the limit for Precision axis

# Title and legend
fig.tight_layout()
plt.title('SHD and Precision over Number of Samples')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))  # Adjust legend position

plt.show()