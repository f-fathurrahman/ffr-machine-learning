import numpy as np
import matplotlib.pyplot as plt

## Illustrate power iterations
X = np.random.randn(1000,100)
U, S, VT = np.linalg.svd(X, full_matrices=0)
S = np.arange(1, 0, -0.01)
X = U @ np.diag(S) @ VT

plt.plot(S, "o-", color="k", linewidth=1, label="SVD")

Y = X
for q in range(1,6):
    Y = X.T @ Y
    Y = X @ Y
    Uq, Sq, VTq = np.linalg.svd(Y,full_matrices=0)
    plt.plot(Sq, "-o", linewidth=1, label="randomized_SVD, q = "+str(q))

plt.legend()
plt.savefig("IMG_power_iter.png", dpi=150)
plt.show()
