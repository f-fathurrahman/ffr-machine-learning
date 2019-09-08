import numpy as np
from Perceptron import Perceptron

# Demonstration of the Perceptron and Linear Regressor on the basic logic functions
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])

# AND data
ANDtargets = np.array([[0],[0],[0],[1]])

# OR data
ORtargets = np.array([[0],[1],[1],[1]])

# XOR data
XORtargets = np.array([[0],[1],[1],[0]])

#print("AND logic function")
#pAND = Perceptron(inputs, ANDtargets)
#pAND.train(inputs, ANDtargets, 0.25, 6)

print("OR logic function")
pOR = Perceptron(inputs,ORtargets)
pOR.train(inputs, ORtargets, 0.25,6)
inp = inputs = np.concatenate((inputs, -np.ones((pOR.nData,1))), axis=1)
print("inp")
print(inp)
activation = pOR.forward(inp)
print("activation")
print(activation)

#print("XOR logic function")
#pXOR = Perceptron(inputs,XORtargets)
#pXOR.train(inputs,XORtargets,0.25,6)
