import numpy as np

def fit_polynomial_maxLH(x, t, Npoly):
    Ndata = len(x)
    # Npoly is degree of the polynomial
    X = np.zeros( (Ndata,Npoly+1) )
    X[:,0] = 1
    for i in range(1,Npoly+1):
        X[:,i] = x**i
    XtX = X.T @ X
    XtXinv = np.linalg.inv(XtX)
    # The parameters
    w = XtXinv @ X.T @ t
    # The variance
    σ2 = (t.T @ t - t.T @ X @ w)/Ndata
    #
    return w, σ2

def predict_polynomial(w, x_eval):
    Npoly = w.shape[0] - 1
    Ndata_eval = x_eval.shape[0]
    # Build X matrix for new input
    X_eval = np.zeros( (Ndata_eval,Npoly+1) )
    X_eval[:,0] = 1.0
    for i in range(1,Npoly+1):
        X_eval[:,i] = x_eval**i
    # evaluate
    t_eval = X_eval @ w
    return t_eval
