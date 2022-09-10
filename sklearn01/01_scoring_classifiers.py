import numpy as np
import sys

# Set random seed
np.random.seed(1234)

# Generate random labels: integer number 0 or 1, with size 5
Npoints = int( sys.argv[1] )
y_true = np.random.randint(0, 2, size=Npoints)

# predicted labels
y_pred = np.ones(Npoints, dtype=np.int32)

from sklearn import metrics

# calculate accuracy, precision, and recall scores
acc = metrics.accuracy_score(y_true, y_pred)
prec = metrics.precision_score(y_true, y_pred)

recall = metrics.recall_score(y_true, y_pred)

print("y_true  y_pred")
for i in range(Npoints):
    print(" %3d    %3d" % (y_true[i], y_pred[i]))

print('')
print('accuracy_score  = %18.10f' % acc)
print('precision_score = %18.10f' % prec)
print('recall_score    = %18.10f' % recall)
