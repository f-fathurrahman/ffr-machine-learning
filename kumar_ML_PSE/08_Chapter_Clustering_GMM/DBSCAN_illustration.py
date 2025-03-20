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
# # Chapter: Finding Groups in Process Data: Clustering & Mixture Modeling
#
# # Topic: DBSCAN algorithm

# %%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          DBSCAN algorithm for the illustration example
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%
# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# %%
# generate ellipsoidal shaped data
from sklearn.datasets import make_blobs

n_samples = 1500
X, y = make_blobs(n_samples=n_samples, random_state=100)

plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.show()

rotation_matrix = [[0.60, -0.70], [-0.5, 0.7]]
X_transformed = np.dot(X, rotation_matrix)

plt.figure()
plt.scatter(X_transformed[:,0], X_transformed[:,1])
plt.show()

# %%
# fit DBSCAN model
db = DBSCAN(eps = 0.5, min_samples = 5).fit(X_transformed) # talk about impact of parameters
cluster_label = db.labels_

plt.figure()
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c = cluster_label, s=20, cmap='viridis')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')

print('Cluster labels: ', np.unique(cluster_label))
