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
# # Chapter: Control Charts for Statistical Process Control
#
#
# # Topic: Shewhart Control Chart

# %%
# import required packages
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
np.random.seed(10)

# %%
# generate data
# NOC data
N = 250
x0 = np.random.normal(loc=10, scale=2, size=N)

# faulty data
N = 50
x1 = np.random.normal(loc=11, scale=2, size=N)

# combine data
x = np.hstack((x0,x1))


# %%
# fit Shewhart model and plot chart for NOC data
mu, sigma = np.mean(x0), np.std(x0)
UCL, LCL = mu + 3*sigma, mu - 3*sigma

plt.figure(figsize=(10,3))
plt.plot(x0,'--',marker='o', markersize=4, color='teal')
plt.plot([1,len(x0)],[UCL,UCL], color='red'), plt.plot([1,len(x0)],[LCL,LCL], color='red')
plt.plot([1,len(x0)],[mu,mu], '--', color='maroon')
plt.xlabel('sample #'), plt.ylabel('x')
plt.grid()
plt.show()

# %%
# control chart for combined data
plt.figure(figsize=(10,3))
plt.plot(x,'--',marker='o', markersize=4, color='teal')
plt.plot([1,len(x)],[UCL,UCL], color='red')
plt.plot([1,len(x)],[LCL,LCL], color='red')
plt.plot([1,len(x)],[mu,mu], '--', color='maroon')
plt.xlabel('sample #'), plt.ylabel('x')
plt.grid()
plt.show()
