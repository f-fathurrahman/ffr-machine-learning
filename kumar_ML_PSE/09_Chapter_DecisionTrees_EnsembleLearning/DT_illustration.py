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
# # Chapter: Decision Trees and Ensemble Learning
#
#
# # Topic: DT illustration for quadratic fitting

# %%
# import
import numpy as np
np.random.seed(1)

# %%
# generate data
x = np.linspace(-1, 1, 50)[:, None]
y = x*x + 0.25 + np.random.normal(0, 0.15, (50,1))

# %%
# plot raw data
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x, y, edgecolor="black", c="darkorange")
plt.xlabel('x'), plt.ylabel('y')
plt.show()

# %%
# fit regularized DT model
from sklearn import tree
model = tree.DecisionTreeRegressor(max_depth=3).fit(x, y)

# %%
# predict and plot
y_pred = model.predict(x)

plt.figure()
plt.scatter(x, y, edgecolor="black", c="darkorange", label="raw data")
plt.plot(x, y_pred, 'yellowgreen', label="predictions")
plt.xlabel('x'), plt.ylabel('y')
plt.legend()
plt.show()

# %%
# plot tree
plt.figure(figsize=(20,8))
tree.plot_tree(model, feature_names=['x'], filled=True, rounded=True)
plt.show()
