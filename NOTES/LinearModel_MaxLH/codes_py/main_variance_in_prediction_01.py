import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

def maxLH_fit_polyN(x, t, N):
    Ndata = x.shape[0]
    # Build the input matrix
    X = np.zeros((Ndata,N+1))
    X[:,0] = 1.0
    for i in range(1,N+1):
        X[:,i] = x**i
    # Calculate model parameters
    XtX = X.T @ X
    XtXinv = np.linalg.inv(XtX)
    w = XtXinv @ X.T @ t
    # Calculate variance
    σ2 = (t.T @ t - t.T @ X @ w)/Ndata

    return w, σ2

def maxLH_predict_polyN(x, w, σ2):
    Ndata = x.shape[0]
    N = w.shape[0] - 1 # polynomial order
    # Build X matrix for xgrid
    X = np.zeros((Ndata,N+1))
    X[:,0] = 1.0
    for i in range(1,N+1):
        X[:,i] = x**i
    # Evaluate the model
    t = X @ w
    σ2_test = np.zeros(Ndata)
    XtXinv = np.linalg.inv(X.T @ X)
    for i in range(Ndata):
        xnew = X[i,:]
        σ2_test[i] = σ2 * xnew.T @ XtXinv @ xnew
    #
    return t, σ2_test


def do_main(x, t, N):

    assert(N > 0)

    w, σ2 = maxLH_fit_polyN(x, t, N)

    NptsPlot = 40
    x_test = np.linspace(-5.5, 5.5, NptsPlot)
    μ_test, σ2_test = maxLH_predict_polyN(x_test, w, σ2)

    plt.clf()
    plt.plot(x, t, marker="o", linewidth=0)
    plt.errorbar(x_test, μ_test, yerr=σ2_test,
        capsize=2.0, label="test data", alpha=0.7, color="red")
    plt.grid(True)
    plt.title("Polynom order N = " + str(N))
    plt.legend(loc=2)
    plt.xlim(-5.6,5.6)
    plt.ylim(-1000,1000)
    plt.savefig("IMG_ex_pred_var_" + str(N) + ".png", dpi=150)
    plt.savefig("IMG_ex_pred_var_" + str(N) + ".pdf")


# Generate random numbers, using uniform distribution
Ndata = 150 # number of data
x = np.sort( 10.0*np.random.rand(Ndata) - 5.0 )
t = 5*x**3 - 2*x**2 + x # true model
σ2_true = 500.0 # true noise parameter
t = t + np.random.randn(Ndata)*np.sqrt(σ2_true) # μ of noise is 0

# Only use some set of data
idx_use = (x < 0.0) | (x > 2.0)
x = x[idx_use]
t = t[idx_use]

for N in range(1,10):
    do_main(x, t, N)
