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
# # Chapter: Best Practices
#
# # Topic: Feature Engineering (one-hot encoding)

# %%
import numpy as np
from sklearn.preprocessing import OneHotEncoder

x = np.array([['type A'],
              ['type C'],
              ['type B'],
              ['type C']])
ohe = OneHotEncoder(sparse=False) # sparse=False returns array
X_encoded = ohe.fit_transform(x) # x in numerical form

print(X_encoded)
print(ohe.categories_)
