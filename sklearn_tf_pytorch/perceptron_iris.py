import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:,(2,3)] # only choose limited features
y = (iris.target == 0).astype(np.int)

# Initialize classifier
clf = Perceptron()
clf.fit(X,y)

# Evaluate model at one arbitrary point
y_pred = clf.predict([[2,0.5]])
print("y_pred = ", y_pred)
