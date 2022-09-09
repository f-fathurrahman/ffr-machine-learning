import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

def true_model(x):
    return 2.0*x**2 + 20*x + 10.0

np.random.seed(1234)

Npoints = 40
NOISE_AMP = 0.0

x = np.linspace(-10.0, 10.0, Npoints)
y_no_noise = true_model(x)
y = y_no_noise + NOISE_AMP*np.random.randn(Npoints)

#plt.clf()
#plt.plot(x, y, marker="o", label="(data) noisy")
#plt.plot(x, y_no_noise, marker="o", label="no noise")
#plt.grid(True)
#plt.legend()
#plt.savefig("IMG_poly2_synth_v01_DATA.pdf")

# For sklearn input
x = x.reshape(-1,1)

from sklearn.preprocessing import PolynomialFeatures

poly2 = PolynomialFeatures(2)
X = poly2.fit_transform(x)

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)

print("coef = ", model.coef_)
print("intercept = ", model.intercept_)

# For evaluating model
xnew = np.linspace(-10.0, 10.0, Npoints).reshape(-1,1)
Xnew = poly2.fit_transform(xnew)
ynew = model.predict(Xnew)

#plt.clf()
#plt.plot(xnew, ynew, marker="o", label="predict")
#plt.plot(x, y, marker="o", label="data")
#plt.grid(True)
#plt.legend()
#plt.savefig("IMG_poly2_synth_v01.pdf")

