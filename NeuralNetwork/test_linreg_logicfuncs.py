# Demonstration of the Perceptron and Linear Regressor on the basic logic functions

import numpy as np
import linreg

def threshold_values(x):
    xx = np.copy(x)
    for i in range(len(x)):
        if x[i] < 0.5:
            xx[i] = 0.0
        else:
            xx[i] = 1.0
    return xx

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
testin = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
print(testin)

# AND data
ANDtargets = np.array([[0],[0],[0],[1]])
# OR data
ORtargets = np.array([[0],[1],[1],[1]])
# XOR data
XORtargets = np.array([[0],[1],[1],[0]])

print("AND data")
ANDbeta = linreg.linreg(inputs,ANDtargets)
ANDout = np.dot(testin,ANDbeta)
print(threshold_values(ANDout))

print("OR data")
ORbeta = linreg.linreg(inputs,ORtargets)
ORout = np.dot(testin,ORbeta)
print(threshold_values(ORout))

print("XOR data")
XORbeta = linreg.linreg(inputs,XORtargets)
XORout = np.dot(testin,XORbeta)
print(threshold_values(XORout))
