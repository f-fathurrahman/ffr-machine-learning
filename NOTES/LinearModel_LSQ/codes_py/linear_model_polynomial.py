import numpy as np

def fit_polynomial(x, t, Npoly):
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

def fit_polynomial_ridge(x, t, Npoly, λ=0.0):
    Ndata = len(x)
    # Npoly is degree of the polynomial
    X = np.zeros( (Ndata,Npoly+1) )
    X[:,0] = 1.0
    for i in range(1,Npoly+1):
        X[:,i] = np.power( x, i )
    XtX = X.transpose() @ X + Ndata*λ*np.eye(Ndata)
    XtXinv = np.linalg.inv(XtX)
    w = XtXinv @ X.transpose() @ t
    return X, w

def predict_polynomial(w, x_eval):
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
