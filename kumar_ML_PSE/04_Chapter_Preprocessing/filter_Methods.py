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
# # Topic: Filter Methods for Variable Selection

# %%
# read data
import numpy as np
VSdata = np.loadtxt('VSdata.csv', delimiter=',')

# %%
# separate X and y
y = VSdata[:,0]
X = VSdata[:,1:]

# %%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##           Linear correlation-based variable selection
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%
# compute linear correlation based scores 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

VSmodel_Correlation = SelectKBest(f_regression, k=10).fit(X, y)
input_scores = VSmodel_Correlation.scores_

# find the top ranked inputs
top_k_inputs_Correlation = np.argsort(input_scores)[::-1][:10] + 1#  [::-1] reverses the array returned by argsort() and [:n] gives that last n elements
print(top_k_inputs_Correlation)

# reduce X to only top relevant inputs
X_relevant = VSmodel_Correlation.transform(X)

# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##           MI-based variable selection
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%
# compute linear correlation based scores 
from sklearn.feature_selection import mutual_info_regression

VSmodel_MI = SelectKBest(mutual_info_regression, k=10).fit(X, y)
input_scores = VSmodel_MI.scores_

# find the top ranked inputs
top_k_inputs_MI = np.argsort(input_scores)[::-1][:10] #  [::-1] reverses the array returned by argsort() and [:n] gives that last n elements
print(top_k_inputs_MI)

# reduce X to only top relevant inputs
X_relevant = VSmodel_MI.transform(X)

# %%
