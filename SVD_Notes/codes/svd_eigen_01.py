import numpy as np

# Shapes of the matrix
n = 5
m = 6

X = np.random.rand(n,m)
print("X = ")
print(X)

U, Σ, Vt = np.linalg.svd(X)
print("Σ = ")
print(Σ)

XXt = np.matmul(X,np.transpose(X))
λ1, UU = np.linalg.eig(XXt)
print("sqrt(abs(λ1)) = ")
print(np.sqrt(np.abs(λ1)))

XtX = np.matmul(np.transpose(X), X)
λ2, VV = np.linalg.eig(XtX)
print("sqrt(abs(λ2)) = ")
print(np.sqrt(np.abs(λ2)))

print("U = ")
print(U)
print("UU = ")
print(UU)
print("Vt = ")
print(Vt)
print("VV = ")
print(VV)