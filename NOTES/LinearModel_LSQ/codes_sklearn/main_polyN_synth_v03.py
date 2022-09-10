# Using train_test_split
# Using model.score

import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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


def score_polyN(model, xtest, ytest, N):
    polyN = PolynomialFeatures(N)
    Xtest = polyN.fit_transform(xtest)
    score = model.score(Xtest, ytest)
    return score



np.random.seed(1234)

Npoints = 40
NOISE_AMP = 50.0

x = np.linspace(-10.0, 10.0, Npoints)
y_no_noise = true_model(x)
y = y_no_noise + NOISE_AMP*np.random.randn(Npoints)

# For sklearn input
x = x.reshape(-1,1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Score: coef of determination, best possible is 1 (R2)
print("npoly       score        loss_train       loss_test")
print("---------------------------------------------------")
for npoly in range(1,10):
    #
    model = fit_polyN(x_train, y_train, npoly)
    #
    y_pred = predict_polyN(model, x_train, npoly)
    loss_train = mean_squared_error(y_train, y_pred)
    #
    y_pred = predict_polyN(model, x_test, npoly)
    loss_test = mean_squared_error(y_test, y_pred)
    #
    score = score_polyN(model, x_test, y_test, npoly)
    #
    print("%3d \t %10.5f \t %10.5f \t %10.5f" % (npoly, score, loss_train, loss_test))


