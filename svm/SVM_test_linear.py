from SVM import SVM
from SVM_utils import *

X1, y1, X2, y2 = gen_lin_separable_data()
X_train, y_train = split_train(X1, y1, X2, y2)
X_test, y_test = split_test(X1, y1, X2, y2)

model = SVM()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
correct = np.sum(y_predict == y_test)
print("%d out of %d predictions correct" % (correct, len(y_predict)))

plot_margin(X_train[y_train==1], X_train[y_train==-1], model)