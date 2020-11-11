import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)

x = 3.0
a = np.linspace(-2.0, 2.0, 9)
b = a*x + np.random.randn(np.size(a))

plt.clf()
plt.plot(a, x*a, marker="o", label="true")
plt.plot(a, b, marker="*", label="data")

U, Σ, Vt = np.linalg.svd(a[:,np.newaxis], full_matrices=False)
xx = np.matmul( np.transpose(Vt), np.linalg.inv(np.diag(Σ)) )
xx = np.matmul( xx, np.transpose(U) )
xtilde = np.matmul(xx, b)

plt.plot(a, xtilde*a, label="linreg")
plt.legend()
plt.grid()
plt.savefig("IMG_svd_linreg_01.pdf")
