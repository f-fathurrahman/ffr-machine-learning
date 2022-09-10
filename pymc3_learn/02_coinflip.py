import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import pymc3 as pm
import arviz as az

np.random.seed(123)

Ntrials = 200
θ_true = 0.35
data = stats.bernoulli.rvs(p=θ_true, size=Ntrials)
print(data)

with pm.Model() as first_model:
    # prior
    θ = pm.Beta('θ', alpha=1.0, beta=1.0)
    # likelihood
    y = pm.Bernoulli('y', p=θ, observed=data)
    trace = pm.sample(1000, random_seed=123)

#
az.plot_trace(trace)
plt.savefig("IMG_02_trace_Ntrials_" + str(Ntrials) + ".pdf")

print(az.summary(trace))

az.plot_posterior(trace)
plt.savefig("IMG_02_posterior_Ntrials_" + str(Ntrials) + ".pdf")

