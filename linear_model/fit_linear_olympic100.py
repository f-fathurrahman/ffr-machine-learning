import numpy as np
import matplotlib.pyplot as plt

data = np.matrix( np.loadtxt("../DATA/olympic100m.txt", delimiter=",") )

Ndata = len(data)

X = np.matrix( np.zeros( (Ndata,2) ) )
X[:,0] = np.ones( (Ndata,1) )
X[:,1] = data[:,0]

t = np.copy(data[:,1])

XtX = X.transpose() * X
XtXinv = np.linalg.inv(XtX)
w = XtXinv * X.transpose() * t
print(w)

t_pred = X*w

x = data[:,0]
plt.clf()
plt.plot(x, t, marker="o", label="data")
plt.plot(x, t_pred, marker="o", label="linear-fit")
plt.grid()
plt.legend()
plt.savefig("TEMP_fit_linear_olympic100.png", dpi=150)
