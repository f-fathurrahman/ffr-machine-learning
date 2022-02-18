import numpy as np
import matplotlib.pyplot as plt

import scipy.stats

def eval_gauss2d(μ, Σ, x, y):
    D = np.linalg.det(Σ)
    Sinv = np.linalg.inv(Σ)
    w = np.array([x, y])
    wmu = w - μ
    C1 = np.matmul(Sinv, wmu)
    C2 = -0.5*np.dot(wmu, C1)
    return np.exp(C2)/(2*np.pi*np.sqrt(D))

μ = np.array([0.0, 0.1])
Σ = np.array([[1.0, 0], [0.0, 5.0]])
x = 1.1
y = 2.2

mvn = scipy.stats.multivariate_normal(μ, Σ)
print("From mvn          = %18.10f" % mvn.pdf([x,y]))
print("from eval_gauss2d = %18.10f" % eval_gauss2d(μ, Σ, x, y))
