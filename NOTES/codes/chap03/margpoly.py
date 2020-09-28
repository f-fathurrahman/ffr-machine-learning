import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.style.use("ggplot")
matplotlib.rc("text", usetex=True)

from numpy.linalg import inv, det

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


# Generate uniform distribution
N = 100
x = np.random.rand(N)*10.0 - 5.0
x = np.sort(x)
σ2_true = 200

t = 5*x**3 - x**2 + x
t = t + np.random.randn(N)*np.sqrt(σ2_true)

plt.clf()
plt.plot(x, t, marker="o", linestyle="None", label="data")
plt.legend()
plt.savefig("IMG_data_margpoly.pdf")

NorderMax = 8
log_marg = np.zeros(NorderMax)
# Fit
for Norder in range(1,NorderMax+1):
    μ_0 = np.zeros(Norder+1)
    Σ_0 = np.eye(Norder+1)

    X = build_input_matrix(x, Norder)

    xtest = np.linspace(-6.0, 6.0, 1001)
    Xtest = build_input_matrix(xtest, Norder)

    Σ_w = inv( (1.0/σ2_true) * np.matmul(X.transpose(), X) + inv(Σ_0) )
    #
    tmp1 = np.matmul(X.transpose(), t)
    tmp2 = np.matmul(inv(Σ_0), μ_0)
    tmp3 = (1.0/σ2_true) * tmp1 + tmp2
    μ_w = np.matmul(Σ_w, tmp3)

    plt.clf()
    plt.plot(x, t, marker="o", linestyle="None", label="data")
    plt.plot(xtest, np.matmul(Xtest, μ_w))
    plt.legend()
    plt.xlim(-10,10)
    plt.ylim(-1500,1500)
    plt.savefig("IMG_margpoly_Noder_" + str(Norder) + ".png", dpi=150)

    # Compute the marginal likelihood
    XΣ_0 = np.matmul(X, Σ_0)
    margcov = σ2_true*np.eye(N) + np.matmul(XΣ_0, X.transpose())
    margmu = np.matmul(X, μ_0)
    log_marg[Norder-1] = -(N/2)*np.log(2*np.pi) - 0.5*np.log(det(margcov))
    tt = t - margmu
    mat1 = np.matmul(tt.transpose(), inv(margcov))
    mat2 = np.matmul(mat1, tt)
    log_marg[Norder-1] = log_marg[Norder-1] - 0.5*mat2
    print("Norder = %d, log_marg = %f" % (Norder, log_marg[Norder-1]))

print()
LH = np.exp(log_marg)
max_power = -np.max(np.log10(LH))
print("max_power = ", max_power)
for i in range(len(LH)):
    print("Norder = %d LH = %e" % (i+1, 10**max_power * LH[i]))

LH = 10**max_power*LH
plt.clf()
plt.bar(np.arange(1,Norder+1), LH)
plt.savefig("IMG_margpoly_LH.png", dpi=150)
