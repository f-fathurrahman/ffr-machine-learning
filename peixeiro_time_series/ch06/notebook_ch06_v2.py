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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller
from tqdm.notebook import tqdm
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
from typing import Union

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
# # Identifying a stationary ARMA process 

# %%
np.random.seed(42)
ar1 = np.array([1, -0.33])
ma1 = np.array([1, 0.9])
ARMA_1_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)

# %%
fig, ax = plt.subplots(figsize=(6,3))
ax.plot(ARMA_1_1)
plt.tight_layout()

# %%
ADF_result = adfuller(ARMA_1_1)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# %%
fig, ax = plt.subplots(figsize=(6,3))
plot_acf(ARMA_1_1, lags=20, ax=ax)
plt.ylim(-1.1, 1.1)
plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(6,3))
plot_pacf(ARMA_1_1, lags=20, ax=ax, method="ywm")
plt.ylim(-1.1, 1.1)
plt.tight_layout()

# %% [markdown]
# # Selecting the best model 

# %%
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
print(order_list)


# %%
def optimize_ARMA(endog: Union[pd.Series, list], order_list: list) -> pd.DataFrame:
    results = []    
    for order in tqdm(order_list):
        try: 
            model = SARIMAX(endog, order=(order[0], 0, order[1]), simple_differencing=False).fit(disp=False)
        except:
            continue
        aic = model.aic
        results.append([order, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


# %%
result_df = optimize_ARMA(ARMA_1_1, order_list)
result_df

# %% [markdown]
# ### 6.4.3 Performing residuals analysis 

# %% [markdown]
# #### Qualitative analysis: studying the Q-Q plot

# %%
from statsmodels.graphics.gofplots import qqplot

# %%
fig, ax = plt.subplots(figsize=(4,4))
gamma = np.random.default_rng().standard_gamma(shape=2, size=1000)
qqplot(gamma, line='45', ax=ax);

# %%
fig, ax = plt.subplots(figsize=(4,4))
normal = np.random.normal(size=1000)
qqplot(normal, line='45', ax=ax)

# %% [markdown]
# ### 6.4.4 Performing residuals analysis 

# %%
model = SARIMAX(ARMA_1_1, order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
residuals = model_fit.resid

# %%
fig, ax = plt.subplots(figsize=(3,3))
qqplot(residuals, line='45', ax=ax)

# %%
model_fit.plot_diagnostics(figsize=(8, 6));

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox

# %% [markdown]
# Keluaran dari `acorr_ljungbox` adalah sebuah `DataFrame`

# %%
res = acorr_ljungbox(residuals, np.arange(1, 11, 1))

# %%
res

# %% [markdown]
# ## 6.5 Applying the general modeling procedure 

# %%
df = pd.read_csv('../data/bandwidth.csv')
df.head()

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(df['hourly_bandwidth'])
ax.set_xlabel('Time')
ax.set_ylabel('Hourly bandwith usage (MBps)')

plt.xticks(
    np.arange(0, 10000, 730), 
    ['Jan 2019', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan 2020', 'Feb'])

fig.autofmt_xdate()
plt.tight_layout()

# %%
ADF_result = adfuller(df['hourly_bandwidth'])

print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# %%
bandwidth_diff = np.diff(df.hourly_bandwidth, n=1)

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(bandwidth_diff)
ax.set_xlabel('Time')
ax.set_ylabel('Hourly bandwith usage - diff (MBps)')

plt.xticks(
    np.arange(0, 10000, 730), 
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '2020', 'Feb'])

fig.autofmt_xdate()
plt.tight_layout()

# %%
ADF_result = adfuller(bandwidth_diff)

print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# %%
fig, ax = plt.subplots(figsize=(6,3))
plot_acf(bandwidth_diff, lags=20, ax=ax)
plt.ylim(-1.1, 1.1)
plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(6,3))
plot_pacf(bandwidth_diff, lags=20, ax=ax, method="ywm")
plt.ylim(-1.1, 1.1)
plt.tight_layout()

# %%
df_diff = pd.DataFrame({'bandwidth_diff': bandwidth_diff})
train = df_diff[:-168]
test = df_diff[-168:]
print(len(train))
print(len(test))

# %%
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(7, 6))

ax1.plot(df['hourly_bandwidth'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Hourly bandwidth usage (MBps)')
ax1.axvspan(9831, 10000, color='blue', alpha=0.3)

ax2.plot(df_diff['bandwidth_diff'])
ax2.set_xlabel('Time')
ax2.set_ylabel('Hourly bandwidth - diff (MBps)')
ax2.axvspan(9830, 9999, color='blue', alpha=0.3)

plt.xticks(
    np.arange(0, 10000, 730), 
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '2020', 'Feb'])

fig.autofmt_xdate()
plt.tight_layout()


# %%
def optimize_ARMA(endog: Union[pd.Series, list], order_list: list) -> pd.DataFrame:
    
    results = []
    
    for order in tqdm(order_list):
        try: 
            model = SARIMAX(endog, order=(order[0], 0, order[1]), simple_differencing=False).fit(disp=False)
        except:
            continue
            
        aic = model.aic
        results.append([order, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


# %%
ps = range(0, 4, 1)
qs = range(0, 4, 1)

order_list = list(product(ps, qs))

# %%
result_df = optimize_ARMA(train['bandwidth_diff'], order_list)
result_df

# %%
model = SARIMAX(train['bandwidth_diff'], order=(2,0,2), simple_differencing=False)
model_fit = model.fit(disp=False)
print(model_fit.summary())

# %%
model_fit.plot_diagnostics(figsize=(10, 8))

# %%
residuals = model_fit.resid
lbvalue, pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(pvalue)


# %% [markdown]
# ## 6.6 Forecasting bandwidth usage

# %%
def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, method: str) -> list:
    
    total_len = train_len + horizon
    end_idx = train_len
    
    if method == 'mean':
        pred_mean = []
        
        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
            
        return pred_mean

    elif method == 'last':
        pred_last_value = []
        
        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
            
        return pred_last_value
    
    elif method == 'ARMA':
        pred_ARMA = []
        
        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order=(2,0,2))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_ARMA.extend(oos_pred)
            
        return pred_ARMA


# %%
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2

pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_last_value = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'last')
pred_ARMA = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'ARMA')

test.loc[:, 'pred_mean'] = pred_mean
test.loc[:, 'pred_last_value'] = pred_last_value
test.loc[:, 'pred_ARMA'] = pred_ARMA

test.head()

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(df_diff['bandwidth_diff'])
ax.plot(test['bandwidth_diff'], '-', label='actual')
ax.plot(test['pred_mean'], ':', label='mean')
ax.plot(test['pred_last_value'], '-.', label='last')
ax.plot(test['pred_ARMA'], '--', label='ARMA(2,2)')

ax.legend(loc=2)

ax.set_xlabel('Time')
ax.set_ylabel('Hourly bandwidth - diff (MBps)')

ax.axvspan(9830, 9999, color='blue', alpha=0.3)

ax.set_xlim(9800, 9999)

plt.xticks(
    [9802, 9850, 9898, 9946, 9994],
    ['2020-02-13', '2020-02-15', '2020-02-17', '2020-02-19', '2020-02-21'])

fig.autofmt_xdate()
plt.tight_layout()

# %%
mse_mean = mean_squared_error(test['bandwidth_diff'], test['pred_mean'])
mse_last = mean_squared_error(test['bandwidth_diff'], test['pred_last_value'])
mse_ARMA = mean_squared_error(test['bandwidth_diff'], test['pred_ARMA'])

print(mse_mean, mse_last, mse_ARMA)

# %%
fig, ax = plt.subplots(figsize=(3,3))

x = ['mean', 'last_value', 'ARMA(2,2)']
y = [mse_mean, mse_last, mse_ARMA] 

ax.bar(x, y, width=0.4)
ax.set_xlabel('Methods')
ax.set_ylabel('MSE')
ax.set_ylim(0, 7)

for index, value in enumerate(y):
    plt.text(x=index, y=value+0.25, s=str(round(value, 2)), ha='center')

plt.tight_layout()

# %%
df['pred_bandwidth'] = pd.Series()
df['pred_bandwidth'][9832:] = df['hourly_bandwidth'].iloc[9832] + test['pred_ARMA'].cumsum()

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(df['hourly_bandwidth'])
ax.plot(df['hourly_bandwidth'], '-', label='actual')
ax.plot(df['pred_bandwidth'], '--', label='ARMA(2,2)')

ax.legend(loc=2)
ax.set_xlabel('Time')
ax.set_ylabel('Hourly bandwith usage (MBps)')
ax.axvspan(9831, 10000, color='blue', alpha=0.3)
ax.set_xlim(9800, 9999)

plt.xticks(
    [9802, 9850, 9898, 9946, 9994],
    ['2020-02-13', '2020-02-15', '2020-02-17', '2020-02-19', '2020-02-21'])

fig.autofmt_xdate()
plt.tight_layout()

# %%
mae_ARMA_undiff = mean_absolute_error(df['hourly_bandwidth'][9832:], df['pred_bandwidth'][9832:])
print(mae_ARMA_undiff)

# %%
