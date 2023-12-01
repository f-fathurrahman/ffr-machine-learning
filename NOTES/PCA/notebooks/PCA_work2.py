# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# <h1 style="text-align: center;">Principal component analysis</h1>

# %% [markdown]
# # Kode program

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# import IPython
# IPython.display.set_matplotlib_formats("svg")
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

# %% [markdown]
# Generate data:

# %%
np.random.seed(1234)

# %%
Y_1 = np.random.randn(20,2)
Y_2 = np.random.randn(20,2) + 5.0
Y_3 = np.random.randn(20,2) - 5.0

# %%
plt.scatter(Y_1[:,0], Y_1[:,1], marker="*", color="blue")
plt.scatter(Y_2[:,0], Y_2[:,1], color="green")
plt.scatter(Y_3[:,0], Y_3[:,1], color="red")

# %%
Y = np.concatenate( (Y_1, Y_2, Y_3), axis=0 );

# %%
Y.shape

# %% [markdown]
# Add random dimensions:

# %%
Ndata = Y.shape[0]
Y = np.concatenate( (Y, np.random.randn(Ndata,5)), axis=1)

# %%
Y.shape

# %%
labels = np.concatenate( ([0]*20, [1]*20, [2]*20) )

# %%
plt.clf()
markers = ["o", "s", "*"]
for i in range(3):
    idx = labels==i
    plt.scatter(Y[idx,0], Y[idx,1], marker=markers[i])
plt.grid()

# %%
plt.clf()
markers = ["o", "s", "*"]
for i in range(3):
    idx = labels==i
    plt.scatter(Y[idx,1], Y[idx,5], marker=markers[i])
plt.grid()

# %%
ybar = np.mean(Y,axis=0)
ybar

# %%
Yshifted = Y - ybar

# %%
np.mean(Yshifted,axis=0)

# %% [markdown]
# Covariance matrix:

# %%
C = Yshifted.transpose() @ Yshifted
C

# %%
C.shape

# %%
λ, w = np.linalg.eig(C)
idx_sorted = np.argsort(λ)[::-1]
λ = λ[idx_sorted]
w = w[:,idx_sorted]

# %%
λ

# %%
λ[0]

# %%
w[0]

# %%
λ[1], w[1]

# %%
np.dot(w[1], w[0])

# %%
w.T @ w

# %%
plt.bar(range(7), λ)
plt.grid()

# %%
w[0]

# %% [markdown]
# Projected data to two first projection dimensions:

# %%
Yproj = Y @ w[:,0:2]

# %%
Y.shape

# %%
Yproj.shape

# %%
plt.clf()
markers = ["o", "s", "*"]
for i in range(3):
    idx = labels==i
    plt.scatter(Yproj[idx,0], Yproj[idx,1], marker=markers[i])
plt.xlabel("1st orig dim")
plt.ylabel("2nd orig dim")
plt.grid()

# %%
Yproj_1 = Y @ w[:,0:1]
Yproj_1.shape

# %%
plt.plot(Yproj_1, np.ones(Yproj_1.shape[0]), marker="o", lw=0)
plt.xlabel("proj dim 1st")
plt.grid()

# %%
markers = ["o", "s", "*"]
for i in range(3):
    idx = labels==i
    plt.plot(Yproj_1[idx], np.ones(Yproj_1[idx].shape[0]), marker=markers[i], lw=0)

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# Pair plot:

# %%
import pandas as pd

# %%
import seaborn as sns
sns.set()

# %%
labelsString = []
for i in range(len(labels)):
    labelsString.append("class"+str(labels[i]))

# %%
labelsDF = pd.DataFrame(labelsString, columns=["class"])
labelsDF.head()

# %%
Ypd.head()

# %%
YDF = pd.merge(Ypd, labelsDF, left_index=True, right_index=True)
YDF.head()

# %%
columns = ["desc"+str(i) for i in range(1,8)]
columns

# %%
Ypd = pd.DataFrame(Y, columns=columns)

# %%
Ypd.head()

# %%
sns.pairplot(YDF, hue="class")

# %%
YDF.head()

# %%
YDF.loc[:,:"desc7"]

# %%
