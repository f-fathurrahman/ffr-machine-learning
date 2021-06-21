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

# Calculate parameters
tbar = np.sum(t)/Ndata # np.average also can be used
xbar = np.sum(x)/Ndata
xtbar = np.sum(x*t)/Ndata
x2bar = np.sum(x**2)/Ndata

w1 = (xtbar - xbar*tbar)/(x2bar - xbar**2)
w0 = tbar - w1*xbar

print("Model parameters:")
print("w0 = %18.10e" % w0)
print("w1 = %18.10e" % w1)

t_pred = w0 + w1*x

plt.clf()
plt.plot(x, t, marker="o", linewidth=0, label="data")
plt.plot(x, t_pred, marker="x", label="linear-fit")
plt.grid(True)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Time (seconds)")
plt.savefig("IMG_fit_linear_olympic100_simple.pdf")
