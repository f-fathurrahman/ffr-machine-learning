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
# ## Topic: ACF and PACF Illustration

# %%
# import packages 
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from matplotlib.ticker import MaxNLocator

# %%
# read data
y = np.loadtxt('simpleTimeSeries.csv', delimiter=',')

# time-plot
plt.figure(figsize=(6,3))
plt.plot(y, 'g', linewidth=0.8)
plt.ylabel('y'), plt.xlabel('k'), plt.xlim(0)
plt.show()

# %%
# generate ACF plot
conf_int = 2/np.sqrt(len(y))

plot_acf(y, lags= 20, alpha=None) # alpha=None avoids plot_acf's inbuilt confidence interval plotting
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.xlabel('lag')
plt.show()

# %%
# generate PACF plot
conf_int = 2/np.sqrt(len(y))

plot_pacf(y, lags= 20, alpha=None) # alpha=None avoids plot_acf's inbuilt confidence interval plotting
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.xlabel('lag')
plt.show()
