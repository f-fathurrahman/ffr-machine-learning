import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from numpy.linalg import inv

import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

np.random.seed(1234)

# NOTE: plt.clf() and plt.savefig() should be called manually
def plot_gauss_contour(μ, Σ, xlim, ylim, Nx=101, Ny=101):
    #
    x = np.linspace(xlim[0], xlim[1], Nx)
    y = np.linspace(ylim[0], ylim[1], Ny)
    Z = np.zeros((Nx,Ny))
    mvn = scipy.stats.multivariate_normal(μ, Σ)
    # Evaluate by loop
    for i in range(Nx):
        for j in range(Ny):
            Z[i,j] = mvn.pdf( [x[i], y[j]])
    plt.contour(x, y, np.transpose(Z), 20)
    plt.xlabel("$w_0$")
    plt.ylabel("$w_1$")

# x: input/feature vector
# Norder: polynomial order
def build_input_matrix(x, Norder):
    #
    assert Norder > 0
    Ndata = len(x)
    #
    X = np.zeros( (Ndata,Norder+1) )
    X[:,0] = np.ones(Ndata)
    for i in range(1,Norder+1):
        X[:,i] = np.power(x,i)
    return X


FILEDATA = "../../../DATA/olympic100m.txt"

data = np.loadtxt(FILEDATA, delimiter=",")

t = data[:,1] # Target
# Rescale the data to avoid numerical problems with large numbers
x = data[:,0]
x = x - x[0]
x = 0.25*x

# Define the prior distribution
μ_0 = np.array([0.0, 0.0])
Σ_0 = np.array([ [5.0, 0.0], [0.0, 5.0] ])
σ2 = 0.1 # Vary this to see the effect on the posterior samples

# Draw some functions from the prior
W = np.random.multivariate_normal(μ_0, Σ_0, 10)
X = build_input_matrix(x, 1)

# Plot the data and the function
plt.clf()
plt.plot(x, t, marker="o", linestyle="None")

# Save xlim and ylim
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

# drawing different lines using different values of W
for i in range(10):
    w = W[i,:]
    plt.plot(x, np.matmul(X, w))
#
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig("IMG_sample_01.pdf")


# Add the data 3 points at a time
NsamplesTry = np.arange(3,27+1,3)
for Nsample in NsamplesTry:
    #
    Xsub = X[:Nsample,:]
    tsub = t[:Nsample]
    #
    Σ_w = inv( (1.0/σ2) * np.matmul(Xsub.transpose(), Xsub) + inv(Σ_0) )
    #
    tmp1 = np.matmul(Xsub.transpose(), tsub)
    tmp2 = np.matmul(inv(Σ_0), μ_0)
    tmp3 = (1.0/σ2) * tmp1 + tmp2
    μ_w = np.matmul(Σ_w, tmp3)
    #
    plt.clf()
    plt.plot(x[Nsample:], t[Nsample:], marker="o", linestyle="None", label="Unused data")
    plt.plot(x[:Nsample], t[:Nsample], marker="o", linestyle="None", label="Used data")
    #
    print("\nNsample = ", Nsample)
    print("μ_w = "); print(μ_w)
    print("Σ_w = "); print(Σ_w)
    #
    W = np.random.multivariate_normal(μ_w, Σ_w, 10)
    for i in range(10):
        w = W[i,:]
        # plot the line
        plt.plot(x, np.matmul(X, w), color="gray")
    #
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.savefig("IMG_olymp_" + str(Nsample) + ".pdf")
    plt.savefig("IMG_olymp_" + str(Nsample) + ".png", dpi=150)
    #
    # Plot the contour plot
    #
    plt.clf()
    plot_gauss_contour(μ_0, Σ_0, [9, 13], [-0.2, 0.2])
    plot_gauss_contour(μ_w, Σ_w, [9, 13], [-0.2, 0.2])
    plt.savefig("IMG_olymp_contour_" + str(Nsample) + ".png", dpi=150)

