import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(1234)

x = np.matrix( np.linspace(0.0, 1.0, 6) ).transpose()
y = 2*x - 3
NoiseVar = 0.1
noise = np.matrix( math.sqrt(NoiseVar)*np.random.randn(x.shape[0],1) )
t = y + noise

Ndata = x.shape[0]

def do_fit(Npoly, reg_param=0.0):
    X = np.matrix( np.zeros( (Ndata,Npoly+1) ) )
    X[:,0] = np.ones( (Ndata,1) )
    for i in range(1,Npoly+1):
        X[:,i] = np.power( x, i )

    XtX = X.transpose() * X + Ndata*reg_param*np.matrix( np.eye(Npoly+1) )
    XtXinv = np.linalg.inv(XtX)
    w = XtXinv * X.transpose() * t

    NptsPlot = 200
    x_eval = np.matrix( np.linspace(x[0,0], x[-1,0], NptsPlot) ).transpose()
    X_eval = np.matrix( np.zeros( (NptsPlot,Npoly+1) ) )
    X_eval[:,0] = np.ones( (NptsPlot,1) )
    for i in range(1,Npoly+1):
        X_eval[:,i] = np.power( x_eval, i )
    
    t_eval = X_eval*w
    t_pred = X*w

    return t_pred, x_eval, t_eval

Npoly = 5
plt.clf()
plt.plot(x, t, marker="o", label="data", color="black")
for r_param in [0.0, 1e-6, 1e-4, 1e-1]:
    t_pred, x_eval, t_eval = do_fit(Npoly, reg_param=r_param)
    #plt.plot(x, t_pred, marker="o", lw=0, color="blue")
    plt.plot(x_eval, t_eval, label="lambda="+str(r_param))
plt.xlim(-0.05, 1.05)
plt.ylim(-3.8, -0.4)
plt.grid()
plt.legend()
plt.savefig("TEMP_reg_fit_poly" + str(Npoly) + "_synth.png", dpi=150)

