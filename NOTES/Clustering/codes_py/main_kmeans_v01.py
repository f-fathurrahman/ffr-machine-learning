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

mat_data = scipy.io.loadmat("../../../DATA/kmeansdata.mat")
X = mat_data["X"]

#plt.clf()
#plt.scatter(X[:,0], X[:,1], marker="s")
#plt.savefig("IMG_kmeans_data1.png", dpi=150)

K = 4    # The number of clusters
Ndim = 2
cluster_means = np.random.rand(K,Ndim)*10 - 5

# Iteratively update the means and assignments
converged = False
Ndata = X.shape[0]
#cluster_assignments = np.zeros((Ndata,K))
cluster_assignments = np.zeros(Ndata)
distances = np.zeros((Ndata,K))
#colors = ["r", "g", "b"]

for iterCl in range(10):

    for k in range(K):
        distances[:,k] = np.linalg.norm(X - cluster_means[k,:], axis=1)

    old_cluster_assigments = np.copy(cluster_assignments)
    for i in range(Ndata):
        cluster_assignments[i] = distances[i,:].argmin()
    #print(cluster_assignments)

    plt.clf()
    # TODO: plot cluster_means
    for k in range(K):
        idx_k = cluster_assignments == k
        plt.scatter(X[idx_k,0], X[idx_k,1])
    plt.savefig("IMG_kmeans_01_iter" + str(iterCl) + ".png", dpi=150)

    Δ = np.sum(np.abs(cluster_assignments - old_cluster_assigments))
    print("Δ = ", Δ)
    if Δ <= 0.0:
        break

    for k in range(K):
        idx_k = cluster_assignments == k
        if np.sum(idx_k) == 0: # empty cluster
            cluster_means[k,:] = np.random.rand(Ndim)*10 - 5 #
        else:
            cluster_means[k,:] = np.mean(X[idx_k], axis=0)

    print("Cluster means = ")
    print(cluster_means)

