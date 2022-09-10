import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import matplotlib
matplotlib.style.use("dark_background")

print("Drawing some number from normal distribution")
μ = 0.0
σ = 1.0
X = stats.norm(μ, σ) # random number with normal distribution
x = X.rvs(3)
print(x)


def do_plot_normal_pdf(μ, σ, filename):
    plt.clf()

    x = np.linspace(-7.0, 7.0, 200)
    y = stats.norm(μ, σ).pdf(x)
    plt.plot(x, y)
    plt.plot([], label="μ = {:3.2f}\nσ = {:3.2f}".format(μ, σ), alpha=0)
    plt.legend(loc="upper right")
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.savefig(filename, dpi=150)


μ_params = [-1.0, 0.0, 1.0]
σ_params = [0.5, 1.0, 1.5]

for μ in μ_params:
    for σ in σ_params:
        filename = "IMG_normal_pdf_{:3.2f}_{:3.2f}.png".format(μ, σ)
        do_plot_normal_pdf(μ, σ, filename)
