# Using train_test_split
# Using model.score

import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipe_polyN(N):
    pipe = Pipeline([
        ("std_scaler", StandardScaler()),
        ("polyN", PolynomialFeatures(N)),
        ("linreg", linear_model.LinearRegression())
    ])
    return pipe



DATA_PATH = "../../../DATA/olympic100m.txt"
data = np.loadtxt(DATA_PATH, delimiter=",")
x = data[:,0]
y = data[:,1]

# For sklearn input
x = x.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Score: coef of determination, best possible is 1 (R2)
print("npoly       score        loss_train       loss_test")
print("---------------------------------------------------")
for npoly in range(1,10):
    #
    pipe = create_pipe_polyN(npoly)
    pipe.fit(x_train, y_train)
    #
    y_pred = pipe.predict(x_train)
    loss_train = mean_squared_error(y_train, y_pred)
    #
    y_pred = pipe.predict(x_test)
    loss_test = mean_squared_error(y_test, y_pred)
    #
    score = pipe.score(x_test, y_test)
    #
    #y_new = pipe.predict([[2012], [2016]])
    #print("y_new = ", y_new)
    print("%3d \t %10.5f \t %10.5f \t %10.5f" % (npoly, score, loss_train, loss_test))


