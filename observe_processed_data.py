#%%
from custom_dataset import ToxDataset
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.data import DataLoader
# %%
train_set = ToxDataset(root="data/", filename='toxicity-train-resampled.csv')
test_set = ToxDataset(root="data/", filename='toxicity-test.csv', test=True)

# %%
print('======================')
print(f'Number of graphs: {len(test_set)}')
print(f'Number of features: {test_set.num_features}')
data = test_set[232]  # Get a graph object.
print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
# %% Observe the data into networkx
G = to_networkx(data, to_undirected=True)
nx.draw(G, with_labels=True)

