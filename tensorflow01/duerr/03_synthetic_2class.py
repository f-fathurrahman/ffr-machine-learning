import numpy as np
from scipy.stats import multivariate_normal

μ1 = [-3.0, -2.0]
μ2 = [3.0, 2.0]

rv1 = multivariate_normal(mean=μ1)
rv2 = multivariate_normal(mean=μ2)

Ndata1 = 50
Ndata2 = 50
Ndata = Ndata1 + Ndata2

dataset = np.zeros( (Ndata,3) )
for i in range(Ndata1):
    xy = rv1.rvs()
    dataset[i,0] = xy[0]
    dataset[i,1] = xy[1]
    dataset[i,2] = 0

for i in range(Ndata1,Ndata):
    xy = rv2.rvs()
    dataset[i,0] = xy[0]
    dataset[i,1] = xy[1]
    dataset[i,2] = 1

# Random permutation
idx = np.random.permutation(Ndata)
dataset = dataset[idx,:]

# Feature and target
X = dataset[:,:2]
Y = dataset[:,2]

import matplotlib.pyplot as plt

#idx_one = np.where(Y==1)
#idx_two = np.where(Y==2)
#plt.scatter(X[idx_one,0], X[idx_one,1], marker="o")
#plt.scatter(X[idx_two,0], X[idx_two,1], marker="^")
#plt.savefig("IMG_03_dataset.pdf")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

model = Sequential()
model.add( Dense(10, batch_input_shape=(None,2), activation="sigmoid") )
model.add( Dense(2, activation='softmax'))

sgd = optimizers.SGD(lr=0.15)

model.compile(
    loss="categorical_crossentropy",
    optimizer=sgd,
    metrics=["accuracy"]
)

print(model.summary())

from tensorflow.keras.utils import to_categorical
Y_c = to_categorical(Y, 2)

# training of the model␣
history = model.fit(
    X, Y_c, epochs=400,
    batch_size=128,
    verbose=1
)


def plot_decision_boundary(X, Y, model, t):
    print("Making grid ...", end="")
    x1 = np.linspace( np.min(X[:,0])-2, np.max(X[:,0])+2, 50 )
    x2 = np.linspace( np.min(X[:,1])-2, np.max(X[:,1])+2, 50 )
    X1grid, X2grid = np.meshgrid(x1, x2)
    print("... done")

    print("Doing prediction ...", end="", flush=True)
    p = np.zeros( (50,50) )
    in1 = np.zeros( (1,2) )
    for j in range(50):
        print(j, " is done")
        for i in range(50):
            in1[0,0] = x1[i]
            in1[0,1] = x2[j]
            p[i,j] = model.predict(in1)[0,0] # pick class 1
    print("... done", flush=True)

    print("Shape of p: ", end=" ", flush=True)
    print(p.shape, flush=True)
    
    plt.clf()
    plt.figure(figsize=(16,4))

    plt.subplot(1, 2, (1))
    cp = plt.contourf(X1grid, X2grid, np.transpose(p), cmap="viridis")
    plt.colorbar(cp)
    plt.title(t)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.subplot(1, 2, (2))
    cp = plt.contourf(X1grid, X2grid, np.transpose(p), cmap="viridis")
    plt.colorbar(cp)
    idx_f = [np.where(Y==1)]
    idx_r = [np.where(Y==0)]
    plt.scatter(X[idx_r,0], X[idx_r,1], alpha=1.0, marker="^", edgecolor="black")
    plt.scatter(X[idx_f,0], X[idx_f,1], alpha=1.0, marker="o", edgecolor="black")
    plt.title(t)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

plot_decision_boundary(X, Y, model, "fcnn separation layer")
plt.savefig("IMG_03_decision_boundary.pdf")