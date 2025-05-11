from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
plt.style.use("ggplot")

import numpy as np
import pandas as pd

df = pd.read_csv("../data/GOOGL.csv")
df.head()

fig, ax = plt.subplots()
ax.plot(df["Date"], df["Close"])
ax.set_xlabel("Date")
ax.set_ylabel("Closing price (USD)")

plt.xticks(
    [4, 24, 46, 68, 89, 110, 132, 152, 174, 193, 212, 235], 
    ['May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan 2021', 'Feb', 'Mar', 'April']
)

plt.grid(True)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("IMG_googl_01.png", dpi=150)
