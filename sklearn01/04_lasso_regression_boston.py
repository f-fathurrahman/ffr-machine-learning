import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection as modsel
from sklearn import linear_model

import matplotlib.pyplot as plt
plt.style.use('dark_background')

RANDSEED = 42

# Set random seed
np.random.seed(RANDSEED)

boston = datasets.load_boston()

print('dir(boston) = ', dir(boston))
print('boston data shape = ', boston.data.shape)
print('boston target shape = ', boston.target.shape)

# Prepare for lasso regression
lassoreg = linear_model.Lasso()

# split dataset, 10 percent for test
print('Splitting the dataset ...')
X_train, X_test, y_train, y_test = modsel.train_test_split(
    boston.data, boston.target, test_size=0.1, random_state=RANDSEED
)

# do the training (fitting)
print('Training ...')
lassoreg.fit(X_train, y_train)

# mean-squred error
mse = metrics.mean_squared_error(y_train, lassoreg.predict(X_train))
print('mse on training data = %18.10f' % mse)

# r2 scoring from lassoreg
r2 = lassoreg.score(X_train, y_train)
print('r2 on training data  = %18.10f' % r2)

# predicted on test data
y_pred = lassoreg.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print('mse on testing data = %18.10f' % mse)

plt.clf()
plt.figure(figsize=(10,6))
plt.plot(y_test, linewidth=3, label='ground truth')
plt.plot(y_pred, linewidth=3, label='predicted')
plt.legend(loc='best')
plt.xlabel('test data points')
plt.ylabel('target value')
plt.savefig('04_truth_vs_predicted.png', dpi=300)


plt.clf()
plt.figure(figsize=(10,6))
plt.plot(y_test, y_pred, 'o')
plt.plot([-10,60], [-10,60], '--')
plt.axis([-10, 60, -10, 60])
plt.xlabel('ground truth')
plt.ylabel('predicted')

scorestr = r'$R^2$ = %.3f' % lassoreg.score(X_test, y_test)
errstr = 'MSE = %.3f' % metrics.mean_squared_error(y_test, y_pred)
plt.text(-5,50, scorestr, fontsize=12)
plt.text(-5,45, errstr, fontsize=12)

plt.savefig('04_truth_vs_predicted_v2.png', dpi=300)