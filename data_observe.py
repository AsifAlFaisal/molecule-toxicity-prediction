#%% Load dataframe
import pandas as pd
data = pd.read_csv("data/raw/toxicity-train.csv")
data.head()
# %% Check labels
print(data.shape)
print(data['labels'].value_counts())
# Train data is highly imbalanced
# %% Working with rdkit
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

sample_smiles = data['smiles']
sample_mols = [Chem.MolFromSmiles(mol) for mol in sample_smiles]
sample_mols[232]
