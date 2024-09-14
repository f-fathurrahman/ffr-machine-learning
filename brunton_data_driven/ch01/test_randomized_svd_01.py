import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os

plt.rcParams["figure.figsize"] = [16,6]
plt.rcParams.update({"font.size": 18})

# Define randomized SVD function
def randomized_SVD(X, r, q, p):
    # Step 1: Sample column space of X with P matrix
    ny = X.shape[1]
    P = np.random.randn(ny,r+p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z, mode="reduced")

    # Step 2: Compute SVD on projected Y = Q.T @ X
    Y = Q.T @ X
    print("shape of Y = ", Y.shape)
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY

    return U, S, VT


A = imread(os.path.join("..", "ORIG_", "DATA", "jupiter.jpg"))
X = np.mean(A, axis=2) # Convert RGB -> grayscale

import time

# Deterministic SVD
start_time = time.perf_counter()
U, S, VT = np.linalg.svd(X, full_matrices=0) 
end_time = time.perf_counter()
print("Elapsed time for deterministic SVD = {:4f} s".format((end_time - start_time)))


r = 400 # Target rank
q = 1   # Power iterations
p = 5   # Oversampling parameter

start_time = time.perf_counter()
rU, rS, rVT = randomized_SVD(X, r, q, p)
end_time = time.perf_counter()
print("Elapsed time for randomized SVD = {:4f} s".format((end_time - start_time)))

## Reconstruction
XSVD = U[:,:(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1),:] # SVD approximation
errSVD = np.linalg.norm(X-XSVD,ord=2) / np.linalg.norm(X,ord=2)
print("errSVD (deterministic) = {:.5e}".format(errSVD))

XrSVD = rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:] # SVD approximation
errSVD = np.linalg.norm(X - XrSVD, ord=2) / np.linalg.norm(X,ord=2)
print("errSVD (randomized) = {:.5e}".format(errSVD))

## Plot
fig, axs = plt.subplots(1,3)
plt.set_cmap("gray")
axs[0].imshow(X)
axs[0].axis("off")
axs[1].imshow(XSVD)
axs[1].axis("off")
axs[2].imshow(XrSVD)
axs[2].axis("off")
plt.savefig("IMG_randomized_SVD_01.png")
plt.show()

