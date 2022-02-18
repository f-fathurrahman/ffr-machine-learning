import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

r = np.linspace(0.0, 1.0, 500)

N = 10; k = 6
P = scipy.stats.binom.pmf(k, N, r)
plt.plot(r, P, label="data1")

N = 100; k = 70
P = scipy.stats.binom.pmf(k, N, r)
plt.plot(r, P, label="data2")

plt.legend()
plt.savefig("IMG_binom_02.pdf")

