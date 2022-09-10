import numpy as np
import matplotlib.pyplot as plt

def my_func(a, x):
    # a is treated as parameter
    return np.exp(-a*x**2)

# Sample random parameters
Nsample = 10
a_set = 1.0 + 0.25*np.random.randn(Nsample)

# Grid points for plotting functions
x = np.linspace(-2.0, 2.0, 100)

plt.clf()
for i in range(Nsample):
    plt.plot(x, my_func(a_set[i], x))
    plt.xlabel("x")
    plt.ylabel("f(x)")
plt.savefig("IMG_sample_func_01_a.pdf")


# Sample random parameters, now with big number of samples
Nsample = 50000
a_set = 1.0 + 0.25*np.random.randn(Nsample)
x = 1.0 # Fixed x

plt.clf()
Nbins = 40
#f1 = np.exp(-a_set*x**2)
f1 = np.exp(-a_set*x**2)
c = plt.hist(f1, bins=Nbins)
plt.xlim(0.0, 1.0)
plt.savefig("IMG_sample_func_01_b.pdf")

p = c[0]/np.max(c[0])
plt.clf()
plt.plot(c[1][:Nbins], p)
plt.xlabel("f(x)")
plt.ylabel("Pr(f(x))")
plt.savefig("IMG_sample_func_01_c.pdf")
