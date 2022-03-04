import scipy.io
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("default")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)


# Load data
mat_data = scipy.io.loadmat("../../../DATA/logregdata.mat")
t = mat_data["t"].flatten()
X = mat_data["X"]

Nfeatures = X.shape[1]
Ndata = X.shape[0]

w = np.zeros(Nfeatures) # guess: all zeros
w_prev = np.copy(w)

TOL = 1e-6 # Stopping tolerance
NiterMax = 100

σ2 = 10.0 # Prior variance on the parameters of w
change = np.inf

for iterOpt in range(1,NiterMax+1):
    P = 1.0/(1 + np.exp(-X @ w))
    # gradient
    grad_log_g = -w/σ2
    # Hessian
    hessian = -np.eye(Nfeatures)/σ2
    # Use loop
    for n in range(Ndata):
        xn = X[n,:] # take one row of X
        grad_log_g += xn * (t[n] - P[n])
        hessian -= np.outer(xn, xn) * P[n] * (1 - P[n])

    # Update w
    w = w - np.linalg.inv(hessian) @ grad_log_g.T
    Δw = w - w_prev
    error_w = np.dot(Δw,Δw)
    print("iterOpt = %3d " % iterOpt, end="")
    print("error_w = %10.5e" % error_w)
    if error_w < TOL:
        print("Converged!")
        break
    # Not converged
    # Next iteration will be executed, save old value of w
    w_prev = np.copy(w)

print("w = ", w)

#for iterOpt in range(1,NiterMax+1)

#plt.clf()
#idx0 = t == 0
#plt.scatter(X[idx0,0], X[idx0,1], marker="o")
#idx1 = t == 1
#plt.scatter(X[idx1,0], X[idx1,1], marker="s")
#plt.savefig("IMG_logreg_data_02.pdf")