from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})

pd.options.mode.chained_assignment = None



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


# Data penjualan widget harian
df = pd.read_csv("../data/widget_sales.csv")

xticks_pos = [0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409, 439, 468, 498]
xticks_label = ['Jan 2019', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
                'Jan 2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

widget_sales_diff = np.diff(df['widget_sales'], n=1)
df_diff = pd.DataFrame({'widget_sales_diff': widget_sales_diff})
# split train
train = df_diff[:int(0.9*len(df_diff))]
test = df_diff[int(0.9*len(df_diff)):]
print("number of train data = ", len(train))
print("number of test data = ", len(test))


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

plt.close("all")
fig, ax = plt.subplots(figsize=(8,4))
#
ax.plot(df_diff['widget_sales_diff'])
ax.plot(pred_df['widget_sales_diff'], '-', label='actual')
ax.plot(pred_df['pred_mean'], ':', label='mean')
ax.plot(pred_df['pred_last_value'], '-.', label='last')
ax.plot(pred_df['pred_MA'], '--', label='MA(2)')
#
ax.legend(loc=2)
#
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales - diff (k$)')
ax.axvspan(449, 498, color="blue", alpha=0.3)
ax.set_xlim(430, 500)
plt.xticks(
    [439, 468, 498], 
    ['Apr', 'May', 'Jun'])
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("IMG_compare_01.png", dpi=150)


from sklearn.metrics import mean_squared_error
mse_mean = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_mean'])
mse_last = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_last_value'])
mse_MA = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_MA'])
print("mse_mean = ", mse_mean)
print("mse_last = ", mse_last)
print("mse_MA = ", mse_MA)

# Lakukan integrasi
df['pred_widget_sales'] = pd.Series()
df['pred_widget_sales'][450:] = df['widget_sales'].iloc[450] + pred_df['pred_MA'].cumsum()

plt.close("all")
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


mae_MA_undiff = mean_absolute_error(df['widget_sales'].iloc[450:], df['pred_widget_sales'].iloc[450:])
print(mae_MA_undiff)


