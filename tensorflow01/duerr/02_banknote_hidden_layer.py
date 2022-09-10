import numpy as np
import matplotlib.pyplot as plt

datafile = "../DATA/data_banknote_authentication.txt"

dataset = np.loadtxt(datafile, delimiter=",")
print(dataset.shape)

X = dataset[:,[1,3]]
Y = dataset[:,4]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers

# Definition of the network
model = Sequential()
model.add( Dense(8, batch_input_shape=(None,2), activation="sigmoid") )
model.add( Dense(2, activation='softmax'))
sgd = optimizers.SGD(lr=0.15)

model.compile(
    loss="categorical_crossentropy",
    optimizer=sgd,
    metrics=["accuracy"]
)

print(model.summary())

# Transform Y=0 to (1,0) and Y=1 to (0,1)
Y_c = to_categorical(Y, 2)

# training of the model␣
history = model.fit(
    X, Y_c,epochs=400,
    batch_size=128,
    verbose=1
)

plt.clf()
plt.plot(history.history["accuracy"])
plt.title("Model Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train"], loc="lower right")
plt.savefig("IMG_02_training_accuracy.pdf")

plt.clf()
plt.plot(history.history["loss"])
plt.title("Model Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train"], loc="lower right")
plt.savefig("IMG_02_training_loss.pdf")

def plot_decision_boundary(X, Y, model, t):
    print("Making grid ...", end="")
    x1 = np.linspace( np.min(X[:,0])-2, np.max(X[:,0])+2, 50 )
    x2 = np.linspace( np.min(X[:,1])-2, np.max(X[:,1])+2, 50 )
    X1grid, X2grid = np.meshgrid(x1, x2)
    print("... done")

    print("Doing prediction ...", end="", flush=True)
    in1 = np.zeros( (50*50,2) )
    ip = 0
    for j in range(50):
        for i in range(50):
            in1[ip,0] = x1[i]
            in1[ip,1] = x2[j]
            ip = ip + 1
    p = model.predict(in1)[:,1] # pick class 1
    p = np.reshape(p, X1grid.shape)
    print("... done", flush=True)

    print("Shape of p: ", end=" ", flush=True)
    print(p.shape, flush=True)

    params = {"mathtext.default": "regular" } #Nicer Plotting
    plt.rcParams.update(params)
    
    plt.clf()
    plt.figure(figsize=(16,4))

    plt.subplot(1, 2, (1))
    cp = plt.contourf(X1grid, X2grid, p, cmap="viridis")
    plt.colorbar(cp)
    plt.title(t)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.subplot(1, 2, (2))
    cp = plt.contourf(X1grid, X2grid, p, cmap="viridis")
    plt.colorbar(cp)
    idx_f = [np.where(Y==1)]
    idx_r = [np.where(Y==0)]
    plt.scatter(X[idx_r,0], X[idx_r,1], alpha=1.0, marker="^", edgecolor="black")
    plt.scatter(X[idx_f,0], X[idx_f,1], alpha=1.0, marker="o", edgecolor="black")
    plt.title(t)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

plot_decision_boundary(X, Y, model, "fcnn separation hidden layer")
plt.savefig("IMG_02_decision_boundary.pdf")

