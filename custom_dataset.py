#%% imports
import pandas as pd
import numpy as np
import os
import torch
import torch_geometric
from torch_geometric.data import Dataset
from tqdm import tqdm
import deepchem as dc
# %% Creating dataset
class ToxDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):        
        self.test = test
        self.filename = filename
        super(ToxDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return[f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return[f'data_{i}.pt' for i in list(self.data.index)]
    
    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            f = featurizer.featurize(mol['smiles'])
            data = f[0].to_pyg_graph()
            data.y = self._get_label(mol['labels'])
            data.smiles = mol['smiles']
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f'data_test_{index}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype = torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

        return data

# %%
