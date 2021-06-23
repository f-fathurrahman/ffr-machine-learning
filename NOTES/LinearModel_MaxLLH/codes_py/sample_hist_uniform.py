import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

np.random.seed(1234)

Nsamples = 10000

data1 = np.random.rand(Nsamples) # no transformation

a = -0.5
b =  0.5
data2 = 3*np.random.rand(Nsamples) + 1.5

plt.clf()
plt.hist(data1, bins=40, label="data1", alpha=0.8, edgecolor="None")
plt.hist(data2, bins=40, label="data2", alpha=0.8, edgecolor="None")
plt.legend()
plt.grid()
plt.savefig("IMG_sample_hist_uniform.pdf")
