import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

def calc_likelihood(t, x, w, σ2):
    σ = σ2**0.5
    μ = np.dot(w, x)
    return np.exp(-0.5/σ2*(t - μ)**2)/(σ*np.sqrt(2*np.pi))

σ2 = 0.05 # assumption
# Model parameter, from LSQ solution, vary the parameter
w_0 = 36.596455902505334
w_1 = -0.013330885710962845
w = np.asarray([w_0, w_1])
x = np.asarray([1.0, 1980.0])

plt.clf()
t_grid = np.linspace(9.0, 11.0, 200)
p_grid = calc_likelihood(t_grid, x, w, σ2)
plt.plot(t_grid, p_grid)

tA = 9.8 # estimate, from the Fig 2.10
pA = calc_likelihood(tA, x, w, σ2)
plt.plot([tA,tA], [0.0,pA], alpha=0.5, color="black", linestyle="--")
plt.plot([9.0,tA], [pA,pA], alpha=0.5, color="black", linestyle="--")
plt.plot([tA], [pA], marker="o", label="A") # the point

tB = 10.1 # estimate, from the Fig 2.10
pB = calc_likelihood(tB, x, w, σ2)
plt.plot([tB,tB], [0.0,pB], alpha=0.5, color="black", linestyle="--")
plt.plot([9.0,tB], [pB,pB], alpha=0.5, color="black", linestyle="--")
plt.plot([tB], [pB], marker="o", label="B")

# actual data
tC = 10.25
pC = calc_likelihood(tC, x, w, σ2)
plt.plot([tC,tC], [0.0,pC], alpha=0.5, color="black", linestyle="--")
plt.plot([9.0,tC], [pC,pC], alpha=0.5, color="black", linestyle="--")
plt.plot([tC], [pC], marker="o", label="C (actual data)")

plt.grid(True)
plt.xlim(9.0, 11.0)
plt.xlabel("t")
plt.ylabel(r"$p(t|w,x,\sigma^2)$")
plt.legend()
plt.savefig("IMG_likelihood_1980_v2.pdf")
