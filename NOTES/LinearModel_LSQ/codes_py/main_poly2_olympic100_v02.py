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

# Rescale the data to avoid numerical problems with large numbers
x = data[:,0]
x = x - x[0]
x = 0.25*x

# Build matrix X
X = np.zeros( (Ndata,3) )
X[:,0] = 1.0
X[:,1] = x
X[:,2] = np.power( x, 2 )

t = data[:,1] # target

XtX = X.transpose() @ X
XtXinv = np.linalg.inv(XtX)
w = XtXinv @ X.transpose() @ t

print("Model parameters:")
print("w0 = %18.10e" % w[0])
print("w1 = %18.10e" % w[1])
print("w2 = %18.10e" % w[2])

t_pred = X @ w

x = data[:,0]

plt.clf()
plt.plot(x, t, marker="o", linewidth=0, label="data")
plt.plot(x, t_pred, marker="x", label="linear-fit")
plt.grid(True)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Time (seconds)")
plt.savefig("IMG_fit_quadratic_olympic100.pdf")
