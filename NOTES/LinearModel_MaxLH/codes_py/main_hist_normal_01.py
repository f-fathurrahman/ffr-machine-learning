import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

np.random.seed(1234)

σ_1 = 0.5
μ_1 = 1.1

σ_2 = 1.5
μ_2 = -1.0

Nsamples = 10000

#data1 = μ_1 + σ_1*np.random.randn(Nsamples)
data1 = np.random.randn(Nsamples) # no transformation
data2 = μ_2 + σ_2*np.random.randn(Nsamples)

plt.clf()
plt.hist(data1, bins=40, label="data1", alpha=0.8, edgecolor="None")
plt.hist(data2, bins=40, label="data2", alpha=0.8, edgecolor="None")
plt.legend()
plt.grid(True)
plt.savefig("IMG_hist_normal_01.pdf")
