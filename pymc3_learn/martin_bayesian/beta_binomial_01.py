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

Y = scipy.stats.bernoulli(0.5).rvs(50)
print(Y)

Niters = 5000
can_sd = 0.05
α = 1
β = 1
θ = 0.5
trace = {"θ": np.zeros(Niters)}
p2 = posterior_dist(θ, Y, α, β)

for i in range(Niters):
    θ_can = scipy.stats.norm(θ, can_sd).rvs(1)
    p1 = posterior_dist(θ_can, Y, α, β)
    pa = p1/p2
    if pa > scipy.stats.uniform(0, 1).rvs(1):
        θ = θ_can
        p2 = p1
    trace["θ"][i] = θ

import matplotlib.pyplot as plt

_, axes = plt.subplots(1, 2, sharey=True)
axes[0].plot(trace["θ"])
#axes[1].hist(trace["θ"], color="0.5", orientation="horizontal", density=True)
axes[1].hist(trace["θ"], orientation="horizontal", density=True)
plt.tight_layout()
plt.savefig("IMG_try1.png", dpi=150)

import arviz as az
print(az.summary(trace, kind="stats", round_to=2))
