import numpy as np
import matplotlib.pyplot as plt

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

def maxLH_predict_polyN(x, w):
    Ndata = x.shape[0]
    N = w.shape[0] - 1 # polynomial order
    # Build X matrix for xgrid
    X = np.zeros((Ndata,N+1))
    X[:,0] = 1.0
    for i in range(1,N+1):
        X[:,i] = x**i
    # Evaluate the model
    return X @ w

# Load the data
DATAPATH = "../../../DATA/olympic100m.txt"
data = np.loadtxt(DATAPATH, delimiter=",")

Ndata = len(data) # data.shape[0]

x = data[:,0]
t = data[:,1]

# Preprocess/transform the input
x = x - np.min(x)
x = x/4
print("Transformed")
print(x)

plt.clf()
plt.plot(x, t, marker="o", linewidth=0, label="data")

for N in [1,2,3,9]:
    print("\nUsing polynomial of order ", N)
    w, σ2 = maxLH_fit_polyN(x, t, N)
    print("Model parameters:")
    print(w)
    print("σ2 = ", σ2)
    #
    NptsPlot = 100
    # make it slightly outside the original datarange
    xgrid = np.linspace(np.min(x), np.max(x), NptsPlot)
    ygrid = maxLH_predict_polyN(xgrid, w)
    plt.plot(xgrid, ygrid, label="order-"+str(N))

plt.grid(True)
plt.legend()
plt.savefig("IMG_olympic100m_polyreg.pdf")

