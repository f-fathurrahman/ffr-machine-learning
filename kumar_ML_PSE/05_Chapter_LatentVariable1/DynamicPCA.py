# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chapter: Dimension Reduction and Latent Variable Methods (Part 1)
#
#
# # Topic: Dynamic PCA

# %%
# import required packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
# fetch data
data = pd.read_excel('proc1a.xls', skiprows = 1,usecols = 'C:AI')

# separate train data
data_train = data.iloc[0:69,]

# %%
# augment training data
lag = 5
N = data_train.shape[0]
m = data_train.shape[1]

data_train_augmented = np.zeros((N-lag,(lag+1)*m))

for sample in range(lag, N):
    dataBlock = data_train.iloc[sample-lag:sample+1,:].values # converting from pandas dataframe to numpy array
    data_train_augmented[sample-lag,:] = np.reshape(dataBlock, (1,-1), order = 'F')

# %%
# scale data
scaler = StandardScaler()
data_train_augmented_normal = scaler.fit_transform(data_train_augmented)

# PCA
pca = PCA()
score_train = pca.fit_transform(data_train_augmented_normal)

# %%
