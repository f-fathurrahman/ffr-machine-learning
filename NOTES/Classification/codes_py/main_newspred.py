import numpy as np
import scipy.io

mat_data = scipy.io.loadmat("../../../DATA/newsgroups.mat")

# Change to 1d array
X = mat_data["X"]
t = mat_data["t"]

X_test = mat_data["Xt"]
t_test = mat_data["testt"]

classnames = mat_data["classnames"]


# Vocabulary size
M = X.shape[1]
Nclass = 20
q = np.zeros((Nclass,M))
α = 2.0 # Smoothing parameter
for c in range(0,Nclass):
    idx = (t == (c+1)).flatten()
    q[c,:] = (α - 1 + np.sum(X[idx,:],axis=0)) / ( M*(α-1) + np.sum(X[idx,:]) )


Nt = X_test.shape[0]
testP = np.zeros((Nt,Nclass))

lqT = np.log(q.T)
for i in range(0,Nt):
    print("Data i = ", i)
    testP[i,:] = X_test[i,:] * lqT

# Slower version
#for c in range(0,Nclass):
#    print("Class %d" % (c+1))
#    # This is slow but save memory
#    #for i in range(0,Nt):
#    #    testP[i,c] = (X_test[i,:] * np.log(q[c,:]))[0]
#    testP[:,c] = X_test[1]

