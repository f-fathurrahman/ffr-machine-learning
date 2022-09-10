from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection as modsel
import cv2

# create data samples
Nsamples = 100
X, y = datasets.make_classification(
    n_samples=Nsamples, n_features=2, n_redundant=0, n_classes=2,
    random_state=7816
)

plt.clf()
plt.scatter(X[:,0], X[:,1], c=y, s=100)
plt.xlabel('x values')
plt.ylabel('y values')
plt.savefig('06_dataset.png', dpi=300)

# preprocessing for OpenCV
X = X.astype(np.float32)  # feature values as 32-bit floating point numbers
y = y*2 - 1  # target labels must be either -1 or +1

print("Splitting dataset ...")
X_train, X_test, y_train, y_test = modsel.train_test_split(
    X, y, test_size=0.2, random_state=42
)


def plot_decision_boundary(svm, X_test, y_test, pltname):
    x_min = X_test[:,0].min() - 1
    x_max = X_test[:,0].max() + 1
    #
    y_min = X_test[:,1].min() - 1
    y_max = X_test[:,1].max() + 1
    #
    h = 0.02  # stepsize
    #
    xx, yy = np.meshgrid( np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    X_hypo = np.c_[xx.ravel().astype(np.float32),
                   yy.ravel().astype(np.float32)]
    # ravel -> flatten multidim array
    # c_ -> transpose to column
    
    # pass X_hypo to `predict` method
    _, zz = svm.predict(X_hypo)
    #
    zz = zz.reshape(xx.shape)
    #
    plt.clf()
    plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, s=200)
    plt.savefig(pltname, dpi=300)


kernels = [ cv2.ml.SVM_LINEAR, cv2.ml.SVM_INTER,
            cv2.ml.SVM_SIGMOID, cv2.ml.SVM_RBF ]
pltname_suffix = ['linear', 'inter', 'sigmoid', 'rbf']


# do over various kernels
for i in range(len(kernels)):

    # prepare for SVM
    svm = cv2.ml.SVM_create()

    # Use SVM to partition the data with a straight line
    svm.setKernel(kernels[i])

    # train
    print("Training SVM ...")
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

    # predict label from test data
    _, y_pred = svm.predict(X_test)

    # score the classifier
    acc_score = metrics.accuracy_score(y_test, y_pred)
    print("acc_score = %18.10f" % acc_score)

    plot_decision_boundary(svm, X_test, y_test,
                '06_boundary_decision_' + pltname_suffix[i] + '.png')
    
