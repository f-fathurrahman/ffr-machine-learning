import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

def do_fit(x, t, Npoly):
    Ndata = len(x)
    # Npoly is degree of the polynomial
    X = np.zeros( (Ndata,Npoly+1) )
    X[:,0] = 1
    for i in range(1,Npoly+1):
        X[:,i] = np.power( x, i )
    XtX = X.transpose() @ X
    XtXinv = np.linalg.inv(XtX)
    w = XtXinv @ X.transpose() @ t
    return X, w

def do_predict(w, x_eval):
    Npoly = w.shape[0] - 1
    Ndata_eval = x_eval.shape[0]
    # Build X matrix for new input
    X_eval = np.zeros( (Ndata_eval,Npoly+1) )
    X_eval[:,0] = 1.0
    for i in range(1,Npoly+1):
        X_eval[:,i] = np.power( x_eval, i )
    # evaluate
    t_eval = X_eval @ w
    return t_eval

# Load the data
DATAPATH = "../../../DATA/olympic100m.txt"
data = np.loadtxt(DATAPATH, delimiter=",")

t_full = data[:,1] # Target
x_full = data[:,0]
# Data indices for validation and training data
idx_val = x_full > 1979
idx_train = x_full <= 1979
#
x_val = x_full[idx_val]
t_val = t_full[idx_val]
#
x = x_full[idx_train]
t = t_full[idx_train]

# Shift and rescale the data to avoid numerical problems with large numbers
x = x - x_full[0]
x = 0.25*x
# also do this for validation input
x_val = x_val - x_full[0]
x_val = 0.25*x_val

for Npoly in range(1,9):
    X, w = do_fit(x, t, Npoly)
    t_val_pred = do_predict(w, x_val)
    loss = np.sum( (t_val_pred - t_val)**2/len(t_val) )
    print("Npoly = %2d   loss = %10.5f" % (Npoly, loss))  
