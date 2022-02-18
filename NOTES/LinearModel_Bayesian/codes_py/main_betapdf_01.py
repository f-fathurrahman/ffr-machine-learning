import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

#betapdf = scipy.stats.beta.pdf # shortcut
# or betapdf(r, α, β)

r = np.linspace(0.0, 1.0, 500)
plt.clf()
#
α = 6; β = 4
mydist = scipy.stats.beta(α, β)
plt.plot(r, mydist.pdf(r), label="data1")
#
α = 70; β = 30
mydist = scipy.stats.beta(α, β)
plt.plot(r, mydist.pdf(r), label="data2")
plt.legend()
plt.grid(True)
plt.savefig("IMG_betapdf_01.pdf")

