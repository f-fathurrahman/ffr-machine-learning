# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

#import matplotlib
#matplotlib.style.use("dark_background")
#matplotlib.rcParams.update({
#    "axes.grid" : True,
#    "grid.color": "gray"
#})

# %%
pd.options.mode.chained_assignment = None

# %%
df = pd.read_csv('../data/jj.csv')
df.head()

# %%
df.tail()

# %% [markdown]
# # Plot data with train/test split 

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(df['date'], df['data'])
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(80, 83, color="blue", alpha=0.3)

plt.xticks(np.arange(0, 81, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])

fig.autofmt_xdate()
plt.tight_layout()

# %% [markdown]
# # Split to train/test 

# %%
train = df[:-4]
test = df[-4:]

# %% [markdown]
# # Predict historical mean 

# %%
historical_mean = np.mean(train['data'])
historical_mean

# %%
test.loc[:, 'pred_mean'] = historical_mean
test


# %% [markdown]
# Mean absolute percentage error

# %%
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# %%
mape_hist_mean = mape(test['data'], test['pred_mean'])
mape_hist_mean

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(train['date'], train['data'], '-.', label='Train')
ax.plot(test['date'], test['data'], '-', label='Test')
ax.plot(test['date'], test['pred_mean'], '--', label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(80, 83, color='blue', alpha=0.3)
ax.legend(loc=2)

plt.xticks(np.arange(0, 85, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])

fig.autofmt_xdate()
plt.tight_layout()

# %% [markdown]
# # Predict last year mean 

# %%
last_year_mean = np.mean(train['data'][-4:])
last_year_mean

# %%
test.loc[:, 'pred__last_yr_mean'] = last_year_mean
test

# %%
mape_last_year_mean = mape(test['data'], test['pred__last_yr_mean'])
mape_last_year_mean

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(train['date'], train['data'], '-.', label='Train')
ax.plot(test['date'], test['data'], '-', label='Test')
ax.plot(test['date'], test['pred__last_yr_mean'], '--', label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(80, 83, color='blue', alpha=0.3)
ax.legend(loc=2)

plt.xticks(np.arange(0, 85, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])

fig.autofmt_xdate()
plt.tight_layout()

# %% [markdown]
# # Predict last know value 

# %%
last = train['data'].iloc[-1]
last

# %%
test.loc[:, 'pred_last'] = last
test

# %%
mape_last = mape(test['data'], test['pred_last'])
mape_last

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(train['date'], train['data'], '-.', label='Train')
ax.plot(test['date'], test['data'], '-', label='Test')
ax.plot(test['date'], test['pred_last'], '--', label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(80, 83, color='blue', alpha=0.3)
ax.legend(loc=2)

plt.xticks(np.arange(0, 85, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])

fig.autofmt_xdate()
plt.tight_layout()

# %% [markdown]
# # Naive seasonal forecast 

# %%
test.loc[:, 'pred_last_season'] = train['data'][-4:].values
test

# %%
mape_naive_seasonal = mape(test['data'], test['pred_last_season'])
mape_naive_seasonal

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(train['date'], train['data'], '-.', label='Train')
ax.plot(test['date'], test['data'], '-', label='Test')
ax.plot(test['date'], test['pred_last_season'], '--', label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(80, 83, color='blue', alpha=0.3)
ax.legend(loc=2)

plt.xticks(np.arange(0, 85, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])

fig.autofmt_xdate()
plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(5,4))

x = ['hist_mean', 'last_year_mean', 'last', 'naive_seasonal']
y = [70.00, 15.60, 30.46, 11.56]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Baselines')
ax.set_ylabel('MAPE (%)')
ax.set_ylim(0, 75)

for index, value in enumerate(y):
    plt.text(x=index, y=value + 1, s=str(value), ha='center')

plt.tight_layout()

# %%
