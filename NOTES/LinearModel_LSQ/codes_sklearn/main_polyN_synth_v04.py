# Using Pipeline

from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("poly2", PolynomialFeatures(2)),
    ("linreg", linear_model.LinearRegression())
])
