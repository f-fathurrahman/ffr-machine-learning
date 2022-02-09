import numpy as np

from linear_model_polynomial import *

# Eq. 2.31
def calc_logLH(σ2, t, t_pred):
    N = len(t)
    σ = np.sqrt(σ2)    
    ss = np.sum( (t - t_pred)**2 )
    term1 = -0.5*N*np.log(2*np.pi) - N*np.log(σ)
    logLH = term1 - 1/(2*σ2)*ss
    return logLH

# Load the data
DATAPATH = "../../../DATA/olympic100m.txt"
data = np.loadtxt(DATAPATH, delimiter=",")

Ndata = len(data) # data.shape[0]

x = data[:,0]
t = data[:,1]

# Preprocess/transform the input
x = x - np.min(x)
x = x/4

# Higher order polynomial will result in numerical instability
for Npoly in range(1,9):
    #print("\nUsing Npoly = ", Npoly)
    w, σ2 = fit_polynomial_maxLH(x, t, Npoly)
    #print("w = ", w)
    #print("σ2 = ", σ2)
    t_pred = predict_polynomial(w, x)
    #print("t_pred = ", t_pred)
    logLH = calc_logLH(σ2, t, t_pred)
    print("%3d %10.5f %10.5f" % (Npoly, σ2, logLH))
