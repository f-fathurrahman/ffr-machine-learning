import numpy as np
from sklearn import metrics

# Set random seed
np.random.seed(1234)

x1 = 0.0
x2 = 10.0
Npoints = 100

x = np.linspace(x1,x2,Npoints)

y_true = np.sin(x) + np.random.rand(Npoints) - 0.5

y_pred = np.sin(x)

import matplotlib.pyplot as plt
#plt.style.use('ggplot')
plt.style.use('dark_background')

plt.clf()
plt.plot(x, y_pred, linewidth=2, label='model')
plt.plot( x, y_true, 'o', label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower left')

plt.savefig('02_scoring_regressors.png', dpi=300)

# mean-squeared error
mse = metrics.mean_squared_error(y_true, y_pred)

# fraction of variance unexplained
fvu = metrics.explained_variance_score(y_true, y_pred)

# R2 or coefficient of determination
r2 = metrics.r2_score(y_true, y_pred)

print("mse = %18.10f" % mse)
print("fvu = %18.10f" % fvu)
print("r2  = %18.10f" % r2)
