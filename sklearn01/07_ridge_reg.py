from sklearn.linear_model import Ridge
import numpy as np

Nsamples = 10
Nfeatures = 5

rng = np.random.RandomState(0)
y = rng.randn(Nsamples)
X = rng.randn(Nsamples,Nfeatures)

model = Ridge( alpha=1.0 )
print( model.fit(X, y) )
print( model.score(X, y) )

#import matplotlib.pyplot as plt
#plt.clf()
#plt.
