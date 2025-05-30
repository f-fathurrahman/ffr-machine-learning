# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chapter: Data Preprocessing
#
# # Topic: Wrapper Method: Backward SFS

# %%
# read data
import numpy as np
VSdata = np.loadtxt('VSdata.csv', delimiter=',')

# %%
# separate X and y
y = VSdata[:,0]
X = VSdata[:,1:]

# %%
# scale data
from sklearn.preprocessing import StandardScaler
xscaler = StandardScaler()
X_scaled = xscaler.fit_transform(X)

yscaler = StandardScaler()
y_scaled = yscaler.fit_transform(y[:,None])

# %%
# SFS-based variable selection
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

BSFS = SequentialFeatureSelector(LinearRegression(), n_features_to_select=10, direction='backward', cv=5).fit(X_scaled, y_scaled)

# %%
# check selected inputs
print('Inputs selected: ', BSFS.get_support(indices=True)+1) # returns integer index of the features selected

# %%
# reduce X to only top relevant inputs
X_relevant = BSFS.transform(X)
