# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# %%
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})

pd.options.mode.chained_assignment = None

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
plt.rcParams['figure.figsize'] = (10, 7.5)
plt.rcParams['axes.grid'] = False

# %% [markdown]
# ## 19.3 Basic forecasting with Prophet 

# %%
df = pd.read_csv('../data/daily_min_temp.csv')
df.head()

# %%
df.tail()

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(df['Temp'])
ax.set_xlabel('Date')
ax.set_ylabel('Minimum temperature (deg C)')

plt.xticks(np.arange(0, 3649, 365), np.arange(1981, 1991, 1))

fig.autofmt_xdate()
plt.tight_layout()

# %% [markdown]
# ## Basic forecasting with Prophet

# %%
df.columns = ['ds', 'y']
df.head()

# %%
train = df[:-365]
test = df[-365:]

# %%
m = Prophet()

# %%
m.fit(train);

# %%
future = m.make_future_dataframe(periods=365)

# %%
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-365:-360]

# %%
forecast.tail()

# %%
test[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']]
test.head()

# %%
test['baseline'] = train['y'][-365:].values
test.head()

# %%
from sklearn.metrics import mean_absolute_error

prophet_mae = mean_absolute_error(test['y'], test['yhat'])
baseline_mae = mean_absolute_error(test['y'], test['baseline'])

print(prophet_mae)
print(baseline_mae)

# %%
test.iloc[59]

# %%
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(train['y'])
ax.plot(test['y'], 'b-', label='Actual')
ax.plot(test['yhat'], color='darkorange', ls='--', lw=3, label='Predictions')
ax.plot(test['baseline'], 'r:', label='Baseline')

ax.set_xlabel('Date')
ax.set_ylabel('Minimum temperature (deg C)')

ax.axvspan(3285, 3649, color='#808080', alpha=0.1)

ax.legend(loc='best')

plt.xticks(
    [3224, 3254, 3285, 3316, 3344, 3375, 3405, 3436, 3466, 3497, 3528, 3558, 3589, 3619],
    ['Nov', 'Dec', 'Jan 1990', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.fill_between(x=test.index, y1=test['yhat_lower'], y2=test['yhat_upper'], color='lightblue')
plt.xlim(3200, 3649)

fig.autofmt_xdate()
plt.tight_layout()

# %% [markdown]
# ## 19.4 Exploring Prophet's advanced functionalities

# %% [markdown]
# ### 19.4.1 Visualization capabilities

# %%
fig1 = m.plot(forecast)

# %% [markdown]
# #### Plot components 

# %%
fig2 = m.plot_components(forecast)

# %% [markdown]
# #### Show trend changepoints 

# %%
from prophet.plot import add_changepoints_to_plot

fig3 = m.plot(forecast)
a = add_changepoints_to_plot(fig3.gca(), m, forecast)

# %% [markdown]
# #### Plot seasonal components 

# %%
from prophet.plot import plot_yearly, plot_weekly
fig4 = plot_yearly(m)

# %%
fig5 = plot_weekly(m)

# %%
m2 = Prophet(yearly_seasonality=20).fit(train)
fig6 = plot_yearly(m2)

# %% [markdown]
# ### 19.4.2 Cross-validation and performance metrics

# %%
from prophet.diagnostics import cross_validation
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days', parallel='processes')
df_cv.head()

# %%
from prophet.diagnostics import performance_metrics
df_perf = performance_metrics(df_cv, rolling_window=0)
df_perf.head()

# %%
from prophet.plot import plot_cross_validation_metric
fig7 = plot_cross_validation_metric(df_cv, metric='mae')

# %% [markdown]
# ### 19.4.3 Hyperparameter tuning 

# %%
from itertools import product

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
}

all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

maes = []

for params in all_params:
    m = Prophet(**params).fit(train)
    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days', parallel='processes')
    df_p = performance_metrics(df_cv, rolling_window=1)
    maes.append(df_p['mae'].values[0])
    
tuning_results = pd.DataFrame(all_params)
tuning_results['mae'] = maes

tuning_results

# %%
best_params = all_params[np.argmin(maes)]
print(best_params)

# %% [markdown]
# ## 19.5 Implementing a robust forecasting process with Prophet 

# %% [markdown]
# ### 19.5.1 Forecasting project: Predicting the popularity of "chocolate" searches on Google 

# %% [markdown]
# Source: https://trends.google.com/trends/explore?date=all&geo=US&q=chocolate

# %%
df = pd.read_csv('../data/monthly_chocolate_search_usa.csv')
df.head()

# %%
df.tail()

# %%
fig, ax = plt.subplots()

ax.plot(df['chocolate'])
ax.set_xlabel('Date')
ax.set_ylabel('Proportion of searches using with the keyword "chocolate"')

plt.xticks(np.arange(0, 215, 12), np.arange(2004, 2022, 1))

fig.autofmt_xdate()
plt.tight_layout()

# %%
df.columns = ['ds', 'y']
df.head()

# %%
from pandas.tseries.offsets import MonthEnd
df['ds'] = pd.to_datetime(df['ds']) + MonthEnd(1)
df.head()

# %%
train = df[:-12]
test = df[-12:]

# %%
train.tail()

# %%
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
}

params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

mses = []

cutoffs = pd.date_range(start='2009-01-31', end='2020-01-31', freq='12M')

for param in params:
    m = Prophet(**param)
    m.add_country_holidays(country_name='US')
    m.fit(train)
    
    df_cv = cross_validation(model=m, horizon='365 days', cutoffs=cutoffs)
    df_p = performance_metrics(df_cv, rolling_window=1)
    mses.append(df_p['mse'].values[0])
    
tuning_results = pd.DataFrame(params)
tuning_results['mse'] = mses

# %%
best_params = params[np.argmin(mses)]
print(best_params)

# %%
m = Prophet(**best_params)
m.add_country_holidays(country_name='US')
m.fit(train);

# %%
future = m.make_future_dataframe(periods=12, freq='M')

# %%
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)

# %%
test[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']]
test.head()

# %%
test['baseline'] = train['y'][-12:].values
test.head()

# %%
prophet_mae = mean_absolute_error(test['y'], test['yhat'])
baseline_mae = mean_absolute_error(test['y'], test['baseline'])

print(prophet_mae)
print(baseline_mae)

# %%
fig, ax = plt.subplots()

ax.plot(train['y'])
ax.plot(test['y'], 'b-', label='Actual')
ax.plot(test['baseline'], 'k:', label='Baseline')
ax.plot(test['yhat'], color='darkorange', ls='--', lw=3, label='Predictions')

ax.set_xlabel('Date')
ax.set_ylabel('Proportion of searches using with the keyword "chocolate"')

ax.axvspan(204, 215, color='#808080', alpha=0.1)

ax.legend(loc='best')

plt.xticks(np.arange(0, 215, 12), np.arange(2004, 2022, 1))
plt.fill_between(x=test.index, y1=test['yhat_lower'], y2=test['yhat_upper'], color='lightblue')
plt.xlim(180, 215)

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH19_F15_peixeiro.png', dpi=300)

# %%
prophet_components_fig = m.plot_components(forecast)

plt.savefig('figures/CH19_F16_peixeiro.png', dpi=300)

# %% [markdown]
# ### 19.5.2 Experiment: Can SARIMA do better? 

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm_notebook
from itertools import product
from typing import Union

# %%
ad_fuller_result = adfuller(df['y'])

print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

# %%
y_diff = np.diff(df['y'], n=1)

ad_fuller_result = adfuller(y_diff)

print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')


# %% [markdown]
# $d=1$, $D=0$ and $m=12$

# %%
def optimize_SARIMAX(endog: Union[pd.Series, list], exog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int) -> pd.DataFrame:
    
    results = []
    
    for order in tqdm_notebook(order_list):
        try: 
            model = SARIMAX(
                endog,
                exog,
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False).fit(disp=False)
        except:
            continue
            
        aic = model.aic
        results.append([order, model.aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


# %%
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)

order_list = list(product(ps, qs, Ps, Qs))

d = 1
D = 0
s = 12

# %%
SARIMA_result_df = optimize_SARIMAX(train['y'], None, order_list, d, D, s)
SARIMA_result_df

# %%
SARIMA_model = SARIMAX(train['y'], order=(1,1,1), seasonal_order=(1,0,1,12), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)

print(SARIMA_model_fit.summary())

# %%
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));

plt.savefig('figures/CH19_F17_peixeiro.png', dpi=300)

# %%
residuals = SARIMA_model_fit.resid

lbvalue, pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))

print(pvalue)

# %%
test

# %%
SARIMA_pred = SARIMA_model_fit.get_prediction(204, 215).predicted_mean

test['SARIMA_pred'] = SARIMA_pred

test

# %%
SARIMA_mae = mean_absolute_error(test['y'], test['SARIMA_pred'])

print(SARIMA_mae)

# %%
