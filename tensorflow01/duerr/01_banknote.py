import numpy as np
import matplotlib.pyplot as plt

datafile = "../DATA/data_banknote_authentication.txt"

dataset = np.loadtxt(datafile, delimiter=",")
print(dataset.shape)

X = dataset[:,[1,3]]
Y = dataset[:,4]

# Plotting  stuffs
#idx_fake = [np.where(Y==1)]
#idx_real = [np.where(Y==0)]
#params = {"mathtext.default" : "regular"}
#plt.rcParams.update(params)
#plt.scatter(X[idx_real,0], X[idx_real,1], alpha=1.0, marker="^", edgecolor="black")
#plt.scatter(X[idx_fake,0], X[idx_fake,1], alpha=1.0, marker="o", edgecolor="black")
#plt.title("Real and faked banknotes")
#plt.xlabel("$x_1$")
#plt.ylabel("$x_2$")
#plt.legend( ("faked", "real"), loc="lower left", fontsize=10)
#plt.savefig("IMG_01_banknote.pdf")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

model = Sequential()
model.add( Dense(1, batch_input_shape=(None,2), activation="sigmoid") )
sgd = optimizers.SGD(lr=0.15)

model.compile(
    loss="binary_crossentropy",
    optimizer=sgd,
    metrics=["accuracy"]
)

print(model.summary())

# training of the model␣
history = model.fit(
    X, Y,epochs=400,
    batch_size=128,
    verbose=1
)

#plt.clf()
#plt.plot(history.history["accuracy"])
#plt.title("Model Accuracy")
#plt.xlabel("epoch")
#plt.ylabel("accuracy")
#plt.legend(["train"], loc="lower right")
#plt.savefig("IMG_01_training_accuracy.pdf")

#plt.clf()
#plt.plot(history.history["loss"])
#plt.title("Model Loss")
#plt.xlabel("epoch")
#plt.ylabel("loss")
#plt.legend(["train"], loc="lower right")
#plt.savefig("IMG_01_training_loss.pdf")

def plot_decision_boundary(X, Y, model, t):
    print("Making grid ...", end="")
    x1 = np.linspace( np.min(X[:,0])-2, np.max(X[:,0])+2, 50 )
    x2 = np.linspace( np.min(X[:,1])-2, np.max(X[:,1])+2, 50 )
    X1grid, X2grid = np.meshgrid(x1, x2)
    print("... done")

    print("Doing prediction ...", end="", flush=True)
    p = np.array([
        model.predict( np.reshape(np.array([l1,l2]), (1,2)) ) for l1,l2 in \
        zip( np.ravel(X1grid), np.ravel(X2grid) )
    ])
    print("... done")

    print(p.shape)
    if len(p.shape) == 3 and p.shape[2] == 2:
        p = p[:,:,1] # pick p for class 1 if there are more than 2 classes
    p = np.reshape(p,X1grid.shape)

    params = {"mathtext.default": "regular" } #Nicer Plotting
    plt.rcParams.update(params)
    
    plt.clf()
    plt.figure(figsize=(16,4))

    plt.subplot(1, 2, (1))
    cp = plt.contourf(X1grid, X2grid, p,cmap="viridis")
    plt.colorbar(cp)
    plt.title(t)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.subplot(1, 2, (2))
    cp = plt.contourf(X1grid, X2grid, p,cmap="viridis")
    plt.colorbar(cp)
    idx_f = [np.where(Y==1)]
    idx_r = [np.where(Y==0)]
    plt.scatter(X[idx_r,0],X[idx_r,1], alpha=1.0,marker="^",edgecolor="black")
    plt.scatter(X[idx_f,0],X[idx_f,1], alpha=1.0,marker="o",edgecolor="black")
    plt.title(t)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

plot_decision_boundary(X, Y, model, "fcnn separation without hidden layer")
plt.savefig("IMG_01_decision_boundary.pdf")

