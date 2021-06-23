import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

def calc_likelihood(t, x, w, σ2):
    σ = σ2**0.5
    μ = np.dot(w, x)
    return np.exp(-0.5/σ2*(t - μ)**2)/(σ*np.sqrt(2*np.pi))

σ2 = 0.05 # assumption
# Model parameter, from LSQ solution
w_0 = 36.416455902505334
w_1 = -0.013330885710962845
w = np.asarray([w_0, w_1])
x = np.asarray([1.0, 1980.0])

plt.clf()
t_grid = np.linspace(9.0, 11.0, 200)
p_grid = calc_likelihood(t_grid, x, w, σ2)
plt.plot(t_grid, p_grid)

tA = 9.8 # estimate, from the Fig 2.10
plt.plot([tA], calc_likelihood(tA, x, w, σ2), marker="o", label="A")

tB = 10.1 # estimate, from the Fig 2.10
plt.plot([tB], calc_likelihood(tB, x, w, σ2), marker="o", label="B")

# actual data
tC = 10.25
plt.plot([tC], calc_likelihood(tC, x, w, σ2), marker="o", label="C (actual data)")

plt.grid(True)
plt.xlabel("t")
plt.ylabel(r"$p(t|w,x,\sigma^2)$")
plt.legend()
plt.savefig("IMG_likelihood_1980.pdf")
