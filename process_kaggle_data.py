#%% Imports
import pandas as pd
# %% helper function to merge smiles and labels
def process_kaggle(filepath):
    smiles = pd.read_csv(filepath+'names_smiles.csv', names=['muid','smiles'])
    labels = pd.read_csv(filepath+'names_labels.csv', names=['muid','labels'])
    merged = pd.merge(smiles, labels, on='muid')
    return merged
# %% Merging smiles and labels
train_path = "Kaggle downloaded data/NR-ER-train/"
test_path = "Kaggle downloaded data/NR-ER-test/"

# Discard list for few problematic rows
discard_train_muid = [87, 225, 334, 1260, 1513, 1536, 2487, 4678, 5223, 5903, 6374, 6880, 7335]
discard_test_muid = [169]

# Getting the dataframes
train_data = process_kaggle(train_path).drop(discard_train_muid).reset_index(drop=True)
test_data = process_kaggle(test_path).drop(discard_test_muid).reset_index(drop=True)

# Saving the dataframes into data/raw directory
train_data.to_csv('data/raw/toxicity-train.csv', index=False)
test_data.to_csv('data/raw/toxicity-test.csv', index=False)
