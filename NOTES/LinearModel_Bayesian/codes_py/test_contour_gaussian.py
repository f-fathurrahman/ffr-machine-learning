import numpy as np
import matplotlib.pyplot as plt

def eval_gauss2d(μ, Σ, x, y):
    D = np.linalg.det(Σ)
    Sinv = np.linalg.inv(Σ)
    w = np.array([x, y])
    wmu = w - μ
    C1 = np.matmul(Sinv, wmu)
    C2 = -0.5*np.dot(wmu, C1)
    return np.exp(C2)/(2*np.pi*np.sqrt(D))

μ = np.array([0.0, 0.1])
Σ = np.array([[5.0, 0], [0.0, 5.0]])

Nx = 51
Ny = 51
x = np.linspace(-0.5, 0.5, Nx)
y = np.linspace(-0.5, 0.5, Ny)
Z = np.zeros((Nx,Ny))

# Evaluate by loop
for i in range(Nx):
    for j in range(Ny):
        Z[i,j] = eval_gauss2d(μ, Σ, x[i], y[j])

plt.clf()
plt.contour(x, y, np.transpose(Z), 20)
plt.savefig("IMG_test_contour.pdf")

