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

# %%
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})

# %%
pd.options.mode.chained_assignment = None

# %% [markdown]
# ## GOOGL - April 28, 2020 to April 27, 2021

# %%
df = pd.read_csv('../data/GOOGL.csv')
df.head()

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(df['Date'], df['Close'])
ax.set_xlabel('Date')
ax.set_ylabel('Closing price (USD)')

plt.xticks(
    [4, 24, 46, 68, 89, 110, 132, 152, 174, 193, 212, 235], 
    ['May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan 2021', 'Feb', 'Mar', 'April'])

fig.autofmt_xdate()
plt.tight_layout()

# %% [markdown]
# ## 3.1 The random walk process 

# %%
np.random.seed(42)
steps = np.random.standard_normal(1000)
steps[0] = 0.0
random_walk = np.cumsum(steps)

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(random_walk)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.tight_layout()


# %% [markdown]
# ### 3.2.2 Testing for stationarity 

# %%
def simulate_process(is_stationary: bool) -> np.array:
    np.random.seed(42)
    process = np.empty(400)
    if is_stationary:
        alpha = 0.5
        process[0] = 0
    else:
        alpha = 1
        process[0] = 10
    for i in range(400):
        if i+1 < 400:
            process[i+1] = alpha*process[i] + np.random.standard_normal()
        else:
            break
    return process


# %%
stationary = simulate_process(True)
non_stationary = simulate_process(False)

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(stationary, linestyle='-', label='stationary')
ax.plot(non_stationary, linestyle='--', label='non-stationary')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
ax.legend(loc=2)
plt.tight_layout()


# %%
def mean_over_time(process: np.array) -> np.array:
    mean_func = []
    for i in range(1,len(process)): # start from 1
        mean_func.append(np.mean(process[:i]))
    return mean_func


# %%
stationary_mean = mean_over_time(stationary)
non_stationary_mean = mean_over_time(non_stationary)

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(stationary_mean, label='stationary')
ax.plot(non_stationary_mean, linestyle='--', label='non-stationary')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Mean')
ax.legend(loc=1)
plt.tight_layout()


# %%
def var_over_time(process: np.array) -> np.array:
    var_func = []
    for i in range(1,len(process)): # start from 1
        var_func.append(np.var(process[:i]))
    return var_func


# %%
stationary_var = var_over_time(stationary)
non_stationary_var = var_over_time(non_stationary)

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(stationary_var, label='stationary')
ax.plot(non_stationary_var, linestyle='--', label='non-stationary')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Variance')
ax.legend(loc=2)

plt.tight_layout()

# %% [markdown]
# ### 3.2.4 Putting it all together 

# %%
ADF_result = adfuller(random_walk)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# %%
fig, ax = plt.subplots(figsize=(6,3))
plot_acf(random_walk, lags=20, ax=ax)
plt.ylim(-1.1, 1.1) # make it look nicer a bit
plt.tight_layout()

# %%
diff_random_walk = np.diff(random_walk, n=1)

# %%
plt.figure(figsize=(6,3))
plt.plot(diff_random_walk)
plt.title('Differenced Random Walk')
plt.xlabel('Timesteps')
plt.ylabel('Value')
plt.tight_layout()

# %%
ADF_result = adfuller(diff_random_walk)

print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# %%
fig, ax = plt.subplots(figsize=(6,3))
plot_acf(diff_random_walk, lags=20, ax=ax)
plt.ylim(-1.1, 1.1)
plt.tight_layout()

# %% [markdown]
# ### 3.2.5 Is GOOGL a random walk? 

# %%
GOOGL_ADF_result = adfuller(df['Close'])

print(f'ADF Statistic: {GOOGL_ADF_result[0]}')
print(f'p-value: {GOOGL_ADF_result[1]}')

# %%
diff_close = np.diff(df['Close'], n=1)

# %%
GOOGL_diff_ADF_result = adfuller(diff_close)

print(f'ADF Statistic: {GOOGL_diff_ADF_result[0]}')
print(f'p-value: {GOOGL_diff_ADF_result[1]}')

# %%
fig, ax = plt.subplots(figsize=(6,3))
plot_acf(diff_close, lags=20, ax=ax);
plt.ylim(-1.1, 1.1)
plt.tight_layout()

# %% [markdown]
# ## 3.3 Forecasting a random walk
# ### 3.3.1 Forecasting on a long horizon

# %%
df = pd.DataFrame({'value': random_walk})
train = df[:800]
test = df[800:]

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(random_walk)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
ax.axvspan(800, 1000, color='blue', alpha=0.3)
plt.tight_layout()

# %%
mean = np.mean(train.value)
test.loc[:, 'pred_mean'] = mean
test.head()

# %%
last_value = train.iloc[-1].value
test.loc[:, 'pred_last'] = last_value
test.head()

# %%
deltaX = 800 - 1
deltaY = last_value - 0
drift = deltaY / deltaX
x_vals = np.arange(801, 1001, 1)
pred_drift = drift * x_vals
test.loc[:, 'pred_drift'] = pred_drift
test.head()

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(train.value, '-')
ax.plot(test['value'], '-', label="True value")
ax.plot(test['pred_mean'], '-.', label='Mean')
ax.plot(test['pred_last'], '--', label='Last value')
ax.plot(test['pred_drift'], ':', label='Drift')
ax.axvspan(800, 1000, color='blue', alpha=0.3)
ax.legend(loc=2)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.tight_layout()

# %%
from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(test['value'], test['pred_mean'])
mse_last = mean_squared_error(test['value'], test['pred_last'])
mse_drift = mean_squared_error(test['value'], test['pred_drift'])

print(mse_mean, mse_last, mse_drift)

# %%
fig, ax = plt.subplots(figsize=(6,3))

x = ['mean', 'last_value', 'drift']
y = [mse_mean, mse_last, mse_drift]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Methods')
ax.set_ylabel('MSE')
ax.set_ylim(0, 500)

for index, value in enumerate(y):
    plt.text(x=index, y=value+5, s=str(round(value, 2)), ha='center')

plt.tight_layout()

# %% [markdown]
# ### 3.3.2 Forecasting the next timestep 

# %%
df_shift = df.shift(periods=1)
df_shift.head()

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(df, '-', label='actual')
ax.plot(df_shift, '-.', label='forecast')
ax.legend(loc=2)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.tight_layout()

# %%
mse_one_step = mean_squared_error(test['value'], df_shift[800:])
mse_one_step

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(df, '-', label='actual')
ax.plot(df_shift, '-.', label='forecast')
ax.legend(loc=2)
ax.set_xlim(900, 1000)
ax.set_ylim(15, 28)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.tight_layout()

# %%
