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
# # Topic: De-noising Process Signals

# %%
# read data
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# %%
# plots
from matplotlib import pyplot as plt

# %%
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

# %%
import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})

# %%
noisy_signal = np.loadtxt('noisy_flow_signal.csv', delimiter=',')

# %%
noisy_signal.shape

# %% [markdown]
# SMA filter:

# %%
windowSize = 15
smoothed_signal_MA = pd.DataFrame(noisy_signal).rolling(windowSize).mean().values

# %% [markdown]
# Savgol filter:

# %%
smoothed_signal_SG = savgol_filter(noisy_signal, window_length = 15, polyorder = 2)

# %%
plt.figure(figsize=(11,3))
plt.plot(noisy_signal, alpha=0.5, label='Noisy signal')
plt.plot(smoothed_signal_MA, label='SMA smoothed signal')
plt.plot(smoothed_signal_SG, label='SG smoothed signal')
plt.xlabel('Sample #')
plt.ylabel('Value')
plt.legend();
