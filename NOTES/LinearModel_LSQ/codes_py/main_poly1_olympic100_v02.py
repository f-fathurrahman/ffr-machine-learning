import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
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

# Calculate the solution
XtX = X.T @ X
XtXinv = np.linalg.inv(XtX)
w = XtXinv @ X.T @ t

print("Model parameters:")
print("w0 = %18.10e" % w[0])
print("w1 = %18.10e" % w[1])

t_pred = X @ w

plt.clf()
plt.plot(x, t, marker="o", linewidth=0, label="data")
plt.plot(x, t_pred, marker="x", label="linear-fit")
plt.grid(True)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Time (seconds)")
plt.savefig("IMG_fit_linear_olympic100.pdf")
