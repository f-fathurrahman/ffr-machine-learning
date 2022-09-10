import numpy as np
import scipy.io
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("default")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

mat_data = scipy.io.loadmat("../ORIG_m/KNN_data1.mat")
X = mat_data["x"]
t = mat_data["t"].flatten()

plt.clf()
idx0 = t == 0
plt.scatter(X[idx0,0], X[idx0,1])
idx1 = t == 1
plt.scatter(X[idx1,0], X[idx1,1])
#plt.savefig("IMG_kNN_v02_data.png", dpi=150)

# Meshgrid, for plotting decision line
xmin = np.min(X[:,0])
xmax = np.max(X[:,0])

ymin = np.min(X[:,1])
ymax = np.max(X[:,1])

xgrid = np.arange(xmin, xmax, 0.1)
ygrid = np.arange(ymin, ymax, 0.1)
Xnew, Ynew = np.meshgrid(xgrid, ygrid)

Nx = xgrid.shape[0]
Ny = ygrid.shape[0]

k = 1
xp = np.zeros(2)
for i in range(Nx):
    for j in range(Ny):
        xp[0] = xgrid[i]
        xp[1] = ygrid[j]
