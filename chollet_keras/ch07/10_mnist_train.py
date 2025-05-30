import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tf_keras as keras
from tf_keras import layers

import tensorflow as tf
keras.backend.clear_session()
tf.random.set_seed(1234)

from tf_keras.datasets import mnist

def get_mnist_model():
    inputs = keras.Input(shape=(28*28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

(images, labels), (test_images, test_labels) = mnist.load_data()

# Reshape is done here, no need for Flatten layer in the model
images = images.reshape((60000, 28*28)).astype("float32")/255
test_images = test_images.reshape((10000, 28*28)).astype("float32")/255

# Split into train and val
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
train_hist = model.fit(train_images, train_labels, epochs=3)

test_metrics = model.evaluate(test_images, test_labels)

print("test_metrics = ", test_metrics)

predictions = model.predict(test_images)


