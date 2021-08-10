#%% Import
import pandas as pd
from sklearn.utils import shuffle
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# %%
over_sampler = RandomOverSampler(sampling_strategy=0.6, random_state=0)
under_sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=0)
# %%
data = pd.read_csv("data/raw/toxicity-train.csv")
data = shuffle(data)
print(data['label'].value_counts())
print(len(data))
X = data.iloc[:,:2]
y = data.pop('label')
# %% Resampling
X, y = under_sampler.fit_resample(X, y)
print(Counter(y))
X, y = over_sampler.fit_resample(X, y)
print(Counter(y))

# %% Outputting new data
X['label'] = y
df = shuffle(X)
df.to_csv('data/raw/toxicity-train-oversampled.csv', index=False)
# %%
