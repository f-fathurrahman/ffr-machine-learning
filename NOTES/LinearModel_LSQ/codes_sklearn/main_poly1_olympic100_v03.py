import numpy as np

# Load the data
DATA_PATH = "../../../DATA/olympic100m.txt"
data = np.loadtxt(DATA_PATH, delimiter=",")

Ndata = len(data) # data.shape[0]

x = data[:,0]
t = data[:,1]

print("Before: x.shape = ", x.shape)

# We need to reshape the input to be of the size (Ndata,Nfeatures)
# x = x[:,np.newaxis]
# x = x.reshape(-1,1)
# x = x.reshape(len(x),1)
x = x[:,np.newaxis]

print("After: x.shape = ", x.shape)

from sklearn import linear_model
model = linear_model.LinearRegression()

model.fit(x, t)

print("Model parameters:")
print("w0 = %18.10f" % model.intercept_)
print("w1 = %18.10f" % model.coef_[0])

