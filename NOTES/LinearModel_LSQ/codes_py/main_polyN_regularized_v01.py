import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

from linear_model_polynomial import *

np.random.seed(1234)
Ntrain = 6 # training data
x = np.linspace(0.0, 1.0, Ntrain)
y = 2*x - 3
NoiseVar = 0.1
noise = np.sqrt(NoiseVar)*np.random.randn(x.shape[0])
t = y + noise # add some noise

Npoly = 5
plt.clf()
plt.plot(x, t, marker="o", label="data")
x_eval = np.linspace(x[0], x[-1], 100)
for λ in [0.0, 1e-6, 1e-4, 1e-1]:
    #
    X, w = fit_polynomial_ridge(x, t, Npoly, λ=λ)
    t_eval = predict_polynomial(w, x_eval)
    #
    plt.plot(x_eval, t_eval, label="$\\lambda$={:8.1e}".format(λ))

plt.xlim(-0.05, 1.05)
plt.ylim(-3.8, -0.4)
plt.grid()
plt.legend()
plt.grid(True)
plt.savefig("IMG_reg_fit_poly" + str(Npoly) + "_synth.pdf")
