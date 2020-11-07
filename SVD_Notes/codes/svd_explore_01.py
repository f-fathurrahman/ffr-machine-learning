import numpy as np

# Shapes of the matrix
n = 5
m = 6

X = np.random.rand(n,m)
print("X = ")
print(X)

print("Full SVD:")
U, Σ, Vt = np.linalg.svd(X, full_matrices=True)

print("U = ")
print(U)

print("Σ = ")
print(Σ)

print("Vt = ")
print(Vt)

# Check the decomposition
Σ_full = np.zeros((n,m))
idx_min = min(n,m)
Σ_full[:idx_min,:idx_min] = np.diag(Σ)
x_tmp = np.matmul(U, Σ_full)
X_recons = np.matmul(x_tmp, Vt)

print("X_recons")
print(X_recons)

print("X_recons - X")
print(X_recons - X)

