import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Random dataset
rng = np.random.RandomState(1)
X = np.sort(5*rng.rand(80,1), axis=0)
y = np.sin(X).ravel() # ravel is required to convert to 1d
y[::5] += 3 * (0.5 - rng.rand(16))


