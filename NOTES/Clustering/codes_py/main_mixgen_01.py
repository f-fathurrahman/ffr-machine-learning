import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("default")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

# Define the mixture components
mixture_means = np.array([
    [3.0,  3.0],
    [1.0, -3.0]
]);
mixture_covs(:,:,1) = [1 0; 0 2];
mixture_covs(:,:,2) = [2 0; 0 1];
priors = [0.7 0.3];