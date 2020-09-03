import numpy as np
import matplotlib.pyplot as plt

import matplotlib.style
matplotlib.style.use("ggplot")

def eval_LLH(t, x, w, σ2):
    σ = σ2**0.5
    μ = np.dot(w, x)
    return np.exp(-0.5/σ2*(t - μ)**2)/(σ*np.sqrt(2*np.pi))

σ2 = 0.05
w_0 = 36.416455902505334
w_1 = -0.013330885710962845
w = np.asarray([w_0, w_1])
x = np.asarray([1.0, 1980.0])

plt.clf()
t_grid = np.linspace(9.0, 11.0, 200)
p_grid = eval_LLH(t_grid, x, w, σ2)
plt.plot(t_grid, p_grid)
plt.grid(True)
plt.xlabel("t")
plt.ylabel("p(t|w,x,σ2)")
plt.savefig("IMG_example_LLH.pdf")
