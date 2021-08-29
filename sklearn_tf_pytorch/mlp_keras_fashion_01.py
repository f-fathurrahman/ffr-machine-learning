#import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

print("TensorFlow version = ", tf.__version__)
print("Keras version      = ", keras.__version__)

# Load data using Keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Create validation data and scale it
X_val, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_val, y_train = y_train_full[:5000], y_train_full[5000:]

# Class names (hard-coded)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Plot an image
#idx_data = 10
#import matplotlib.pyplot as plt
#plt.imshow(X_val[idx_data,:,:])
#plt.title(class_names[y_val[idx_data]])
#plt.savefig("IMG_fashion_01.pdf")

# Create model
#model = keras.models.Sequential()
#model.add(keras.layers.Flatten(input_shape=[28,28]))
#model.add(keras.layers.Dense(300, activation="relu"))
#model.add(keras.layers.Dense(100, activation="relu"))
#model.add(keras.layers.Dense(10, activation="softmax"))

# Alternative
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()

# Each layers can be inspected
# model.layers[0].name
# model.layers[0].get_weights()
# ...etc

# Compile
model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer="sgd",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train, y_train, epochs=30, 
    validation_data=(X_val, y_val)
)


# Plot
#import pandas as pd
#import matplotlib.pyplot as plt
#
#pd.DataFrame(history.history).plot(figsize=(8,5))
#plt.grid(True)
#plt.gca().set_ylim(0,1)
#plt.savefig("IMG_mlp_keras_fashion_01.pdf")
