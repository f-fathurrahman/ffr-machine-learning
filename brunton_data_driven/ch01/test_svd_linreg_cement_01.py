import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams["figure.figsize"] = [8, 8]
plt.rcParams.update({"font.size": 18})

# Load dataset
DATA_PATH = os.path.join("..", "ORIG_", "DATA")
A = np.loadtxt(os.path.join(DATA_PATH,"hald_ingredients.csv"),delimiter=",")
b = np.loadtxt(os.path.join(DATA_PATH,"hald_heat.csv"),delimiter=",")

# Solve Ax=b using SVD
U, S, VT = np.linalg.svd(A,full_matrices=0)
x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

plt.plot(b, color="k", linewidth=2, label="Heat Data") # True relationship
plt.plot(A@x, "-o", color="r", linewidth=1.5, markersize=6, label="Regression")
plt.legend()
plt.show()

# Alternative 2:
x = np.linalg.pinv(A)@b