import numpy as np
import matplotlib.pyplot as plt

data = np.matrix( np.loadtxt("../DATA/olympic100m.txt", delimiter=",") )

t = data[:,1] # Target
# Rescale the data to avoid numerical problems with large numbers
x = data[:,0]
x = x - x[0]
x = 0.25*x

Ndata = len(data)

def do_fit(Npoly):
    X = np.matrix( np.zeros( (Ndata,Npoly+1) ) )
    X[:,0] = np.ones( (Ndata,1) )
    for i in range(1,Npoly+1):
        X[:,i] = np.power( x, i )

    XtX = X.transpose() * X
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

    plt.clf()
    plt.plot(x, t, marker="o", label="data", color="black")
    plt.plot(x, t_pred, marker="o", lw=0, color="blue")
    plt.plot(x_eval, t_eval, label="fitted", color="blue")
    plt.grid()
    plt.legend()
    plt.savefig("TEMP_fit_poly" + str(Npoly) + "_olympic100.png", dpi=150)

for n in range(1,11):
    do_fit(n)
    print("n = %d is done" % (n))
