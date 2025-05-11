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

ADF_result = adfuller(random_walk)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

plot_acf(random_walk, lags=20)
plt.tight_layout()
plt.ylim(-1.1, 1.1)
plt.savefig("IMG_acf_01.png", dpi=150)
plt.show()


diff_random_walk = np.diff(random_walk, n=1)

plt.clf()
plt.plot(diff_random_walk)
plt.title('Differenced Random Walk')
plt.xlabel('Timesteps')
plt.ylabel('Value')
plt.tight_layout()
plt.savefig('IMG_diff_01.png', dpi=150)
plt.show()


ADF_result = adfuller(diff_random_walk)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

plot_acf(diff_random_walk, lags=20);
plt.tight_layout()
plt.savefig('IMG_acf_diff_01.png', dpi=150)
plt.show()