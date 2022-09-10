import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

# Load the data
DATA_PATH = "../../../DATA/olympic100m.txt"
data = np.loadtxt(DATA_PATH, delimiter=",")

x = data[:,0]
t = data[:,1]

# Calculate the parameters
tbar = np.average(t)
xbar = np.average(x)
xtbar = np.average(x*t)
x2bar = np.average(x**2)

w1 = (xtbar - xbar*tbar)/(x2bar - xbar**2)
w0 = tbar - w1*xbar

print("Model parameters:")
print("w0 = %18.10f" % w0)
print("w1 = %18.10f" % w1)

t_pred = w0 + w1*x

plt.clf()
plt.plot(x, t, marker="o", linewidth=0, label="data")
plt.plot(x, t_pred, marker="x", label="linear-fit")
plt.grid(True)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Winning time (seconds)")
plt.savefig("IMG_fit_linear_olympic100_simple.pdf")
