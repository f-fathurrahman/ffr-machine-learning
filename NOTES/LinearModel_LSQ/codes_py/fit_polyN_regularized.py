import numpy as np
import matplotlib.pyplot as plt
import math

import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

np.random.seed(1234)
Ntrain = 6 # training data
x = np.linspace(0.0, 1.0, Ntrain)
y = 2*x - 3
NoiseVar = 0.1
noise = math.sqrt(NoiseVar)*np.random.randn(x.shape[0])
t = y + noise # add some noise

def do_fit(x, t, Npoly, λ=0.0):
    Ndata = len(x)
    # Npoly is degree of the polynomial
    X = np.zeros( (Ndata,Npoly+1) )
    X[:,0] = 1.0
    for i in range(1,Npoly+1):
        X[:,i] = np.power( x, i )
    XtX = X.transpose() @ X + Ndata*λ*np.eye(Ndata)
    XtXinv = np.linalg.inv(XtX)
    w = XtXinv @ X.transpose() @ t
    return X, w

def do_predict(w, x_eval):
    Npoly = w.shape[0] - 1
    Ndata_eval = x_eval.shape[0]
    # Build X matrix for new input
    X_eval = np.zeros( (Ndata_eval,Npoly+1) )
    X_eval[:,0] = 1.0
    for i in range(1,Npoly+1):
        X_eval[:,i] = np.power( x_eval, i )
    # evaluate
    t_eval = X_eval @ w
    return t_eval

Npoly = 5
plt.clf()
plt.plot(x, t, marker="o", label="data")
x_eval = np.linspace(x[0], x[-1], 100)
for λ in [0.0, 1e-6, 1e-4, 1e-1]:
    X, w = do_fit(x, t, Npoly, λ=λ)
    #
    t_eval = do_predict(w, x_eval)
    plt.plot(x_eval, t_eval, label="$\\lambda$={:8.1e}".format(λ))

plt.xlim(-0.05, 1.05)
plt.ylim(-3.8, -0.4)
plt.grid()
plt.legend()
plt.savefig("IMG_reg_fit_poly" + str(Npoly) + "_synth.pdf")
