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
# # Import packages

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% [markdown]
# ## Setup plot

# %%
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})

# %% [markdown]
# Menghindari warning pada `pandas`:

# %%
pd.options.mode.chained_assignment = None

# %% [markdown]
# # Data penjualan widget harian

# %%
df = pd.read_csv("../data/widget_sales.csv")
df.head()

# %%
len(df)

# %% [markdown]
# Data waktu tidak diberikan dalam hari, namun tidak ada informasi tanggal atau bulan.

# %% [markdown]
# Berikut ini adalah ticks position dan label yang digunakan di buku.

# %%
xticks_pos = [0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409, 439, 468, 498]
xticks_label = ['Jan 2019', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
                'Jan 2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(df['widget_sales'])
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales (k$)')
plt.xticks(xticks_pos, xticks_label)
fig.autofmt_xdate()
plt.tight_layout()

# %% [markdown]
# Cek apakah time series ini stasioner atau tidak dengan menggunakan uji augmented Dickey-Fuller (ADF):

# %%
ADF_result = adfuller(df['widget_sales'])

print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# %% [markdown]
# Karena nilai $p$ dan statistik ADF negatif cukup dekat ke nol, disimpulkan bahwa time series ini tidak stasioner.

# %% [markdown]
# Karena tidak stasioner, kita lakukan diferensiasi pada time series ini:

# %%
widget_sales_diff = np.diff(df['widget_sales'], n=1)

# %%
fig, ax = plt.subplots(figsize=(8,3))

ax.plot(widget_sales_diff)
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales - diff (k$)')
plt.xticks(xticks_pos, xticks_label)
fig.autofmt_xdate()
plt.tight_layout()

# %% [markdown]
# Sepertinya time series ini sudah stasioner. Lakukan uji ADF untuk mendapatkan hasil kuantitatif:

# %%
ADF_result = adfuller(widget_sales_diff)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# %% [markdown]
# Diperoleh nilai statistik ADF negatif yang cukup jauh dari nol dan nilai $p$ yang sangat kecil, artinya `widget_sales_diff` adalah time series yang stasioner.

# %% [markdown]
# Plot fungsi autokorelasi (autocorrelation function):

# %%
fig, ax = plt.subplots(figsize=(6,3))
plot_acf(widget_sales_diff, lags=30, ax=ax);
plt.tight_layout()

# %%
df_diff = pd.DataFrame({'widget_sales_diff': widget_sales_diff})
# split train
train = df_diff[:int(0.9*len(df_diff))]
test = df_diff[int(0.9*len(df_diff)):]
print("number of train data = ", len(train))
print("number of test data = ", len(test))

# %% [markdown]
# Jumlah total `df_diff` kurang 1 dibanding `df`.
#
# 50 data terakhir sebagai data uji, data lain sebagai data latih.

# %%
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,4))

ax1.plot(df['widget_sales'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Widget sales (k$)')
ax1.axvspan(450, 500, color='blue', alpha=0.5)

ax2.plot(df_diff['widget_sales_diff'])
ax2.set_xlabel('Time')
ax2.set_ylabel('Widget sales - diff (k$)')
ax2.axvspan(449, 498, color='blue', alpha=0.5)

plt.xticks(xticks_pos, xticks_label)
fig.autofmt_xdate()
plt.tight_layout()

# %% [markdown]
# Perhatikan gambar plot bagian bawah (`df_diff`). Sekarang kita ingin melakukan forecasting untuk nilai yang diarsir, diberikan informasi nilai dari waktu-waktu sebelumnya.

# %% [markdown]
# Tampilkan sebagian data:

# %%
fig, ax = plt.subplots(figsize=(5,3))
ax.plot(df_diff['widget_sales_diff'])
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales - diff (k$)')
ax.axvspan(449, 498, color='blue', alpha=0.5)
plt.xticks(xticks_pos, xticks_label)
ax.set_xlim(350, 498)
fig.autofmt_xdate()
plt.tight_layout()

# %%
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2
for i in range(TRAIN_LEN, TRAIN_LEN+HORIZON, WINDOW):
    print(i)

# %%
model = SARIMAX(df_diff[:TRAIN_LEN], order=(0,0,2)) # p,d,q
res = model.fit(disp=False)

# %%
TRAIN_LEN

# %%
TRAIN_LEN+WINDOW-1

# %%
predictions = res.get_prediction(0, TRAIN_LEN + WINDOW - 1)
predictions

# %%
predictions.predicted_mean.iloc[-WINDOW:]

# %%
fig, ax = plt.subplots(figsize=(5,3))
ax.plot(df_diff['widget_sales_diff'])
ax.plot(predictions.predicted_mean.iloc[-WINDOW:], marker="o")
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales - diff (k$)')
ax.axvspan(449, 498, color='blue', alpha=0.5)
plt.xticks(xticks_pos, xticks_label)
ax.set_xlim(350, 498)
fig.autofmt_xdate()
plt.tight_layout()

# %%
#oos_pred = predictions.predicted_mean.iloc[-window:]
#pred_MA.extend(oos_pred)

# %%

# %%

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX

def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, method: str) -> list:
    total_len = train_len + horizon
    #
    if method == 'mean':
        pred_mean = []
        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
        return pred_mean
    #
    elif method == 'last':
        pred_last_value = []   
        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value
    #
    elif method == 'MA':
        pred_MA = []   
        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order=(0,0,2))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_MA.extend(oos_pred)
        return pred_MA
    else:
        raise RuntimeError(f"Unknown method={method}")


# %%
pred_df = test.copy()

TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2

pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_last_value = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'last')
pred_MA = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'MA')

pred_df['pred_mean'] = pred_mean
pred_df['pred_last_value'] = pred_last_value
pred_df['pred_MA'] = pred_MA

pred_df.head()

# %%
fig, ax = plt.subplots(figsize=(8,4))

ax.plot(df_diff['widget_sales_diff'])
ax.plot(pred_df['widget_sales_diff'], '-', label='actual')
ax.plot(pred_df['pred_mean'], ':', label='mean')
ax.plot(pred_df['pred_last_value'], '-.', label='last')
ax.plot(pred_df['pred_MA'], '--', label='MA(2)')

ax.legend(loc=2)

ax.set_xlabel('Time')
ax.set_ylabel('Widget sales - diff (k$)')

ax.axvspan(449, 498, color="blue", alpha=0.3)

ax.set_xlim(430, 500)

plt.xticks(
    [439, 468, 498], 
    ['Apr', 'May', 'Jun'])

fig.autofmt_xdate()
plt.tight_layout()

# %%
from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_mean'])
mse_last = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_last_value'])
mse_MA = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_MA'])

print("mse_mean = ", mse_mean)
print("mse_last = ", mse_last)
print("mse_MA = ", mse_MA)

# %%
fig, ax = plt.subplots(figsize=(4,3))

x = ['mean', 'last_value', 'MA(2)']
y = [mse_mean, mse_last, mse_MA]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Methods')
ax.set_ylabel('MSE')
ax.set_ylim(0, 5)

for index, value in enumerate(y):
    plt.text(x=index, y=value+0.25, s=str(round(value, 2)), ha='center')

plt.tight_layout()

# %% [markdown]
# Lakukan integrasi

# %%
df['pred_widget_sales'] = pd.Series()
df['pred_widget_sales'][450:] = df['widget_sales'].iloc[450] + pred_df['pred_MA'].cumsum()

# %%
fig, ax = plt.subplots(figsize=(4,3))

ax.plot(df['widget_sales'], '-', label='actual')
ax.plot(df['pred_widget_sales'], '--', label='MA(2)')

ax.legend(loc=2)

ax.set_xlabel('Time')
ax.set_ylabel('Widget sales (K$)')

ax.axvspan(450, 500, color='blue', alpha=0.2)

ax.set_xlim(400, 500)

plt.xticks(
    [409, 439, 468, 498], 
    ['Mar', 'Apr', 'May', 'Jun'])

fig.autofmt_xdate()
plt.tight_layout()

# %%
from sklearn.metrics import mean_absolute_error
mae_MA_undiff = mean_absolute_error(df['widget_sales'].iloc[450:], df['pred_widget_sales'].iloc[450:])
print(mae_MA_undiff)

# %%
