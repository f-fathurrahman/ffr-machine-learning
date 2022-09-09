import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "savefig.dpi": 150
})

def true_model(x):
    return 2.0*x**2 + 20*x + 10.0


def fit_polyN(x, y, N):
    polyN = PolynomialFeatures(N)
    X = polyN.fit_transform(x)
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model



def predict_polyN(model, xnew, N):
    polyN = PolynomialFeatures(N)
    Xnew = polyN.fit_transform(xnew)
    ynew = model.predict(Xnew)
    return ynew



np.random.seed(1234)

Npoints = 50
NOISE_AMP = 50.0

x = np.linspace(-10.0, 10.0, Npoints)
y_no_noise = true_model(x)
y = y_no_noise + NOISE_AMP*np.random.randn(Npoints)

# Choose some region for the data
idx_use = (x < -2.0) | (x > 3.0)
x = x[idx_use]
y = y[idx_use]

print("Ndata = ", len(x))

# For sklearn input
x = x.reshape(-1,1)

for npoly in range(1,10):

    model = fit_polyN(x, y, npoly)
    print("\nnpoly = ", npoly)
    print("coef = ", model.coef_)
    print("intercept = ", model.intercept_)

    # For evaluating model
    # Use range data outside the range used to generate the data
    xnew = np.linspace(-12.0, 12.0, Npoints).reshape(-1,1)
    ynew = predict_polyN(model, xnew, npoly)

    plt.clf()
    plt.title("npoly = " + str(npoly))
    plt.scatter(x, y, marker="o", label="data", color="red")
    plt.plot(xnew, ynew, label="predict")
    plt.plot(xnew, true_model(xnew), label="true")
    plt.ylim(-180,650)
    plt.grid(True)
    plt.legend()
    plt.savefig("IMG_v02_polyN_" + str(npoly) + "_synth.png")

