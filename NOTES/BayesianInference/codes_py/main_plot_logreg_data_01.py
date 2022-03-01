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
plt.scatter(X[:,0], X[:,1], c=t, edgecolors="k")
plt.savefig("IMG_logreg_data_01.pdf")