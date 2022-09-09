import numpy as np

# Load the data
DATAPATH = "../../../DATA/olympic100m.txt"
data = np.loadtxt(DATAPATH, delimiter=",")

Ndata = len(data) # data.shape[0]

x = data[:,0]
t = data[:,1]

from sklearn import linear_model
model = linear_model.LinearRegression()
# x = x[:,np.newaxis]
# x = x.reshape(-1,1)
# x = x.reshape(len(x),1)
x = x[:,np.newaxis]
model.fit(x, t)

print("Model parameters:")
print("w0 = %18.10e" % model.intercept_)
print("w1 = %18.10e" % model.coef_[0])

