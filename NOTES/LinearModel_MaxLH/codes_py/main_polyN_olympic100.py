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

# Preprocess/transform the input
x = x - np.min(x)
x = x/4

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
plt.savefig("IMG_olympic100.pdf")
