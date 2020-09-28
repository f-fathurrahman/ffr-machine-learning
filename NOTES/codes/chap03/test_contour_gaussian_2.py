import numpy as np
import matplotlib.pyplot as plt

import scipy.stats

μ = np.array([0.0, 0.1])
Σ = np.array([[5.0, 0], [0.0, 5.0]])

Nx = 51
Ny = 51
x = np.linspace(-0.5, 0.5, Nx)
y = np.linspace(-0.5, 0.5, Ny)
Z = np.zeros((Nx,Ny))

mvn = scipy.stats.multivariate_normal(μ, Σ)

# Evaluate by loop
for i in range(Nx):
    for j in range(Ny):
        Z[i,j] = mvn.pdf([x[i], y[j]])

plt.clf()
plt.contour(x, y, np.transpose(Z), 20)
plt.savefig("IMG_test_contour_2.pdf")

