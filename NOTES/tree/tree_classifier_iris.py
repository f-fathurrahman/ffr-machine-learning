import numpy as np
import matplotlib.pyplot as plt

import matplotlib
#matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

# Load dataset
iris = load_iris()

pairidx = 0
pair = [0,1]

#for pairidx, pair, in enumerate([ [0,1], [0,2], [0,3],
#                                  [1,2], [1,3], [2,3]]):

# Take only two features
X = iris.data[:,pair]
y = iris.target

# Train
#clf = DecisionTreeClassifier(max_depth=4).fit(X, y)
clf = DecisionTreeClassifier().fit(X, y)

print(clf)

x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, plot_step),
    np.arange(y_min, y_max, plot_step)
)

#plt.subplot(2, 3, pairidx+1)
plt.clf()
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

plt.xlabel(iris.feature_names[pair[0]])
plt.ylabel(iris.feature_names[pair[1]])

# Plot the training points (data)
for i, color in zip(range(n_classes), plot_colors):
    print("class=%d color=%s target=%s" % (i, color, iris.target_names[i]))
    idx = np.where(y == i)
    plt.scatter(X[idx,0], X[idx,1], c=color, label=iris.target_names[i], 
        cmap=plt.cm.RdYlBu, edgecolor="black", s=15)


plt.savefig("IMG_iris_01.pdf")
