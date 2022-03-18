import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

from my_utils import save_fig, plot_digits

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
print("Data loading finished")

X = mnist["data"]
y = mnist["target"]

some_digit = X[0]

y = y.astype(np.uint8)

X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)
print("Done training the model")

res = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(res)

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

