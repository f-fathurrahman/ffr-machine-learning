import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

from linear_model_polynomial import *

# Load the data
DATAPATH = "../../../DATA/olympic100m.txt"
data = np.loadtxt(DATAPATH, delimiter=",")

Ndata = len(data) # data.shape[0]

x = data[:,0]
t = data[:,1]

# Preprocess/transform the input
x = x - np.min(x)
x = x/4

plt.clf()
plt.plot(x, t, marker="o", linewidth=0, label="data")

for N in [1,2,3,9]:
    w, σ2 = fit_polynomial_maxLH(x, t, N)
    print("%3d %10.5f" % (N, σ2))
    #
    NptsPlot = 100
    # make it slightly outside the original datarange
    xgrid = np.linspace(np.min(x), np.max(x), NptsPlot)
    ygrid = predict_polynomial(w, xgrid)
    plt.plot(xgrid, ygrid, label="order-"+str(N))

plt.xlabel("Year (shifted and scaled)")
plt.grid(True)
plt.legend()
plt.savefig("IMG_olympic100.pdf")

