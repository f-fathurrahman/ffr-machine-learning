import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def f(x):
    return 2*x + 1.0

Nsamples = 10
x_min = 0.0
x_max = 5.0

x = np.linspace(x_min, x_max, Nsamples)

rng = np.random.RandomState(1234)
y = f(x) + rng.randn(Nsamples)

X = x[:,np.newaxis]

NptsPlot = 200
x_plot = np.linspace(x_min, x_max, NptsPlot)
X_plot = x_plot[:,np.newaxis]

plt.clf()
plt.scatter(x, y, marker="o", color="red")

for deg in range(1,11):
    model = make_pipeline( PolynomialFeatures(deg), Ridge() )
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, label=("deg = %d" % deg))

plt.legend()
plt.grid()
plt.savefig("12_polynomial_ridge.pdf")

