from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
plt.style.use("ggplot")

import numpy as np

np.random.seed(42)

steps = np.random.randn(1000)
steps[0] = 0 # initialize to 0

random_walk = np.cumsum(steps)

plt.figure(figsize=(8,6))

plt.plot(random_walk)
plt.xlabel("Timesteps")
plt.ylabel("Value")

plt.tight_layout()
plt.savefig("IMG_rand_walk_01.png", dpi=150)

