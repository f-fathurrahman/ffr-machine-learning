import scipy.io
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("default")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)


# Load data
mat_data = scipy.io.loadmat("../../../DATA/logregdata.mat")
t = mat_data["t"].flatten()
X = mat_data["X"]

plt.clf()
idx0 = t == 0
plt.scatter(X[idx0,0], X[idx0,1], marker="o")
idx1 = t == 1
plt.scatter(X[idx1,0], X[idx1,1], marker="s")
plt.savefig("IMG_logreg_data_02.pdf")