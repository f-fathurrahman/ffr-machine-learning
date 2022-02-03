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

t = data[:,1] # Target
# Rescale the data to avoid numerical problems with large numbers
x = data[:,0]
x = x - x[0]
x = 0.25*x

for Npoly in [3]:
    X, w = fit_polynomial(x, t, Npoly)
    # Define new input from first x to last x where the model will be evaluated
    NptsPlot = 200
    x_eval = np.linspace(x[0], x[-1], NptsPlot)
    # Build X matrix for new input
    X_eval = np.zeros( (NptsPlot,Npoly+1) )
    X_eval[:,0] = 1.0
    for i in range(1,Npoly+1):
        X_eval[:,i] = np.power( x_eval, i )
    
    t_eval = X_eval @ w
    t_pred = X @ w

    plt.clf()
    plt.plot(x, t, marker="o", label="data")
    plt.plot(x_eval, t_eval, label="fitted")
    plt.grid(True)
    plt.legend()
    plt.ylim(9.5, 12.2) # use fixed limits
    plt.xlim(-1, 30)
    plt.xlabel("Year (scaled and shifted)")
    plt.ylabel("Time (seconds)")
    plt.savefig("IMG_fit_poly" + str(Npoly) + "_olympic100.pdf")
    
    print("Npoly = %2d is done" % (Npoly))
