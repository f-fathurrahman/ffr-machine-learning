from SVM import SVM, polynomial_kernel
from SVM_utils import *

X1, y1, X2, y2 = gen_non_lin_separable_data()

X = np.vstack( (X1, X2) )
y = np.hstack( (y1, y2) )

model = SVM(polynomial_kernel)
model.fit(X, y)

y_predict = model.predict(X)
correct = np.sum(y_predict == y)
print("%d out of %d predictions correct" % (correct, len(y_predict)))

plot_contour(X[y==1], X[y==-1], model)
