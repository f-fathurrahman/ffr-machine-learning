import numpy as np
from Perceptron import *

# Run AND and XOR logic functions
a = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
b = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])

p = Perceptron(a[:,0:2],a[:,2:])
print(p)
p.train(a[:,0:2],a[:,2:],0.25,10)
p.confusion_matrix(a[:,0:2],a[:,2:])

#q = Perceptron(b[:,0:2], b[:,2:])
#q.train(b[:,0:2], b[:,2:], 0.25, 10)
#q.confusion_matrix(b[:,0:2], b[:,2:])

