import numpy as np

n = 5
m = 3

X = np.random.rand(n,m)
print("X = ")
print(X)

print("Economy SVD:")
U, Σ, Vt = np.linalg.svd(X, full_matrices=False)

print("U = ")
print(U)

print("Σ = ")
print(Σ)

print("Vt = ")
print(Vt)

# Check the decomposition
x_tmp = np.matmul(U, np.diag(Σ))
X_recons = np.matmul(x_tmp, Vt)

print("X_recons")
print(X_recons)

print("X_recons - X")
print(X_recons - X)

