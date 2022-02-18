import numpy as np
import scipy.special

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

gammafunc = scipy.special.gamma

def mybetapdf(r, α, β):
    res1 = r**(α - 1) * (1 - r)**(β - 1)
    normalization = gammafunc(α + β)/( gammafunc(α)*gammafunc(β) )
    return normalization*res1

r = np.linspace(0.0, 1.0, 500)

plt.clf()
α = 6; β = 4
plt.plot(r, mybetapdf(r, α, β), label="data1")
α = 70; β = 30
plt.plot(r, mybetapdf(r, α, β), label="data2")
plt.legend()
plt.grid(True)
plt.savefig("IMG_betapdf_02.pdf")

