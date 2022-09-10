import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

N_samples = 300

np.random.seed(0)

# spherical data, centered on (20,20)
shifted_gaussian = np.random.randn(N_samples, 2) + np.array([20,20])

# zero centered stretched Gaussian data
C = np.array( [ [0.0, -0.7], [3.5, 0.7] ] ) # covariance matrix
stretched_gaussian = np.dot( np.random.randn( n_samples, 2 ), C )

# concatenate the two datasets into final training set
X_train = np.vstack( [shifted_gaussian, stretched_gaussian] )

# fit a Gaussian mixture model with two components
model = mixture.GaussianMixture( n_components=2, covariance_type="full" )
model.fit(X_train)

# display predicted scores by the model as a contout plot






