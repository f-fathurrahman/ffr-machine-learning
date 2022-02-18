import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

r = 0.5
N = 10
k = np.arange(0,N+1,10)
k = np.arange(0,N+1,1)
P = scipy.stats.binom.pmf(k, N, r)

plt.clf()
plt.bar(k, P)
plt.grid(True)
plt.savefig("IMG_binom_01.pdf")

