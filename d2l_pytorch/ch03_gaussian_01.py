import numpy as np
import d2l_torch as d2l

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("default")
plt.rcParams.update({"text.usetex": True})


def normal_distrib(x, μ, σ):
    p = 1/np.sqrt(2 * np.pi * σ**2) # normalization factor
    return p * np.exp(-0.5/σ**2 * (x - μ)**2)


x = np.arange(-7.0, 7.0, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]

d2l.plot(x, [normal_distrib(x, μ, σ) for μ, σ in params], xlabel="x",
    ylabel="p(x)", figsize=(4.5, 2.5),
    legend=[f"mean {μ}, std {σ}" for μ, σ in params])

plt.savefig("IMG_ch03_gaussian.pdf")