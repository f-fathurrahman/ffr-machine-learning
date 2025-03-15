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
# # Chapter: Exploratory Analysis and Visualization of Dynamic Dataset
#
# ## Topic: PSD Illustration

# %%
# import packages 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

# %%
# read data
data = np.loadtxt('simpleInputOutput.csv', delimiter=',')
u = data[:,0]; y = data[:,1]

# time-plot
# plot y
plt.figure(figsize=(6,1.5))
plt.plot(y, 'g', linewidth=0.8)
plt.ylabel('y'), plt.xlabel('k'), plt.xlim(0)
plt.show()

# plot x
plt.figure(figsize=(6,1.5))
plt.plot(u, 'steelblue', linewidth=0.8)
plt.ylabel('u'), plt.xlabel('k'), plt.xlim(0)
plt.show()

# %%
# Periodograms
# input PSD
freq, PSD = signal.welch(u)

plt.figure(figsize=(5,3)), plt.plot(freq, PSD, 'darkviolet', linewidth=0.8)
plt.ylabel('input PSD'), plt.xlabel('frequency [Hz]'), plt.xlim(0)
plt.show()

# output PSD
freq, PSD  = signal.welch(y)

plt.figure(figsize=(5,3)), plt.plot(freq, PSD, 'darkviolet', linewidth=0.8)
plt.ylabel('output PSD'), plt.xlabel('frequency [Hz]'), plt.xlim(0)
plt.show()
