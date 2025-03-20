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
# # Topic: Embedded Method: Lasso

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
# fit Lasso model 
from sklearn.linear_model import LassoCV
Lasso_model = LassoCV(cv=5).fit(X_scaled, y_scaled.ravel())

# %%
# cfind the relevant inputs using model coefficients
top_k_inputs = np.argsort(abs(Lasso_model.coef_))[::-1][:10] + 1
print('Relevant inputs: ', top_k_inputs)
