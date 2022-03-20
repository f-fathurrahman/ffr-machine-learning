import numpy as np
import scipy.stats

def posterior_dist(θ, T, α=1, β=1):
    if 0 <= θ <= 1:
        prior = scipy.stats.beta(α, β).pdf(θ)
        likelihood = scipy.stats.bernoulli(θ).pmf(Y).prod()
        prob = likelihood * prior
    else:
        prob = -np.inf
    #
    return prob

Y = scipy.stats.bernoulli(0.7).rvs(20)
print(Y)
