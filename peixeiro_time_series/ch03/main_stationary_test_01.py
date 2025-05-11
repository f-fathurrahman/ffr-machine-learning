from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
plt.style.use("ggplot")

import numpy as np

def simulate_process(is_stationary: bool) -> np.array:
    np.random.seed(42)
    process = np.empty(400)
    if is_stationary:
        alpha = 0.5
        process[0] = 0.0
    else:
        alpha = 1
        process[0] = 10.0
    #
    for i in range(400):
        if i + 1 < 400:
            process[i+1] = alpha*process[i] + np.random.randn()
        else:
            break
    return process

stationary = simulate_process(is_stationary=True)
non_stationary = simulate_process(is_stationary=False)

fig, ax = plt.subplots()

ax.plot(stationary, linestyle="-", label="stationary")
ax.plot(non_stationary, linestyle="--", label="non-stationary")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Value")
ax.legend(loc=2)
plt.grid(True)
plt.tight_layout()
plt.savefig("IMG_stat_vs_nonstat_01.png", dpi=150)


def mean_over_time(process: np.array) -> np.array:
    mean_func = []    
    for i in range(1,len(process)):
        mean_func.append(np.mean(process[:i]))
    return mean_func

mean_stationary = mean_over_time(stationary)
mean_non_stationary = mean_over_time(non_stationary)

fig, ax = plt.subplots()

ax.plot(mean_stationary, label="stationary")
ax.plot(mean_non_stationary, label="non-stationary")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Mean")
ax.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.savefig("IMG_mean_over_time_01.png", dpi=150)


def var_over_time(process: np.array) -> np.array:
    var_func = []
    for i in range(1,len(process)):
        var_func.append(np.var(process[:i]))
    return var_func

var_stationary = var_over_time(stationary)
var_non_stationary = var_over_time(non_stationary)

fig, ax = plt.subplots()

ax.plot(var_stationary, label="stationary")
ax.plot(var_non_stationary, label="non-stationary")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Mean")
ax.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.savefig("IMG_var_over_time_01.png", dpi=150)