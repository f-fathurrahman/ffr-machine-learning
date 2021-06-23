import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

# Load the data
DATAPATH = "../../../DATA/olympic100m.txt"
data = np.loadtxt(DATAPATH, delimiter=",")

Ndata = len(data) # data.shape[0]

x = data[:,0]
t = data[:,1]

# Build the input matrix
X = np.zeros((Ndata,2))
X[:,0] = 1.0
X[:,1] = data[:,0]

# Calculate model parameters
XtX = X.T @ X
XtXinv = np.linalg.inv(XtX)
w = XtXinv @ X.T @ t

print("Model parameters:")
print("w0 = %18.10e" % w[0])
print("w1 = %18.10e" % w[1])

# Calculate variance
σ2 = (t.T @ t - t.T @ X @ w)/Ndata
print("σ2 = ", σ2)

t_pred = X @ w
