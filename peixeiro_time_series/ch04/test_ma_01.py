from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})

pd.options.mode.chained_assignment = None

# Baca data
df = pd.read_csv("../data/widget_sales.csv")

xticks_pos = [0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409, 439, 468, 498]
xticks_label = ["Jan 2019", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                "Jan 2020", "Feb", "Mar", "Apr", "May", "Jun"]


widget_sales_diff = np.diff(df["widget_sales"], n=1)
df_diff = pd.DataFrame({'widget_sales_diff': widget_sales_diff})

# split train
train = df_diff[:int(0.9*len(df_diff))]
test = df_diff[int(0.9*len(df_diff)):]

TRAIN_LEN = len(train)
TEST_LEN = len(test)
WINDOW = 2
TOTAL_LEN = TRAIN_LEN + TEST_LEN

ifig = 1
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(df_diff['widget_sales_diff'])
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales - diff (k$)')
ax.axvspan(449, 498, color='blue', alpha=0.5)
ax.set_xlim(430, 500)
plt.xticks(
    [439, 468, 498], 
    ['Apr', 'May', 'Jun'])
fig.autofmt_xdate()
plt.tight_layout()
for i in range(TRAIN_LEN, TOTAL_LEN, WINDOW):
    model = SARIMAX(df_diff[:i], order=(0,0,2)) # p,d,q
    res = model.fit(disp=False)
    preds_df = res.get_prediction(0, i + WINDOW - 1).predicted_mean
    #
    #plt.clf()
    ax.plot(preds_df.iloc[-WINDOW:], marker="o", color="yellow", linewidth=0)
    plt.savefig(f"IMG_step_{ifig:02d}.png", dpi=150)
    print(f"ifig = {ifig} is done")
    ifig += 1

