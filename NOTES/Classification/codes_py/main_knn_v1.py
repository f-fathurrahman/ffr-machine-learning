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

#idx0 = t == 0
#plt.scatter(X[idx0,0], X[idx0,1], marker="o")
#idx1 = t == 1
#plt.scatter(X[idx1,0], X[idx1,1], marker="s")
#plt.savefig("IMG_kNN_data1.pdf")

# Meshgrid, for plotting decision line
OFFSET = 0.2
xmin = np.min(X[:,0]) - OFFSET
xmax = np.max(X[:,0]) + OFFSET

ymin = np.min(X[:,1]) - OFFSET
ymax = np.max(X[:,1]) + OFFSET

xgrid = np.arange(xmin, xmax, 0.1)
ygrid = np.arange(ymin, ymax, 0.1)
Xnew, Ynew = np.meshgrid(xgrid, ygrid)

k = 59
C_new = np.zeros(Xnew.shape) # new classes
NpointsData = len(X)
# Preallocate
distances = np.zeros(NpointsData)
xp = np.zeros(2)
Nx = len(xgrid)
Ny = len(ygrid)
# Using itertools.product?
for i in range(Nx):
    for j in range(Ny):
        xp[0] = xgrid[i]
        xp[1] = ygrid[j]
        distances[:] = np.linalg.norm(X - xp, axis=1)
        idx_kNN = distances.argsort()[:k]
        classes_kNN = t[idx_kNN]
        #print("xp = ", xp)
        #print("Xnew = ", Xnew[j,i])
        #print("Ynew = ", Ynew[j,i])
        C_new[j,i] = scipy.stats.mode(classes_kNN).mode
        #print("classes_kNN = ", classes_kNN)
        #print("classes     = ", classes[i,j])

plt.clf()
# Contour
ax = plt.gca()
CS = ax.contourf(Xnew, Ynew, C_new)
ax.set_aspect("equal", "box")

idx0 = t == 0
plt.scatter(X[idx0,0], X[idx0,1], marker="o")
idx1 = t == 1
plt.scatter(X[idx1,0], X[idx1,1], marker="s")

plt.tight_layout()
plt.savefig("IMG_kNN_contour_k_" + str(k) + ".png", dpi=150)

