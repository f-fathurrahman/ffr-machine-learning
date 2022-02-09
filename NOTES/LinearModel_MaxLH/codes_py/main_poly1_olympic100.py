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

from linear_model_polynomial import *

w, σ2 = fit_polynomial_maxLH(x, t, 1)

print("Model parameters:")
print("w0 = %18.10e" % w[0])
print("w1 = %18.10e" % w[1])
print("σ2 = ", σ2)
