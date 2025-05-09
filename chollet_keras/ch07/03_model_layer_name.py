import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tf_keras as keras
from tf_keras import layers

import tensorflow as tf
keras.backend.clear_session()
tf.random.set_seed(1234)

model = keras.Sequential([
    layers.Dense(64, activation="relu", name="layer1"),
    layers.Dense(10, activation="softmax", name="layer2")
])

# Build the model by specifying input_shape
model.build((None,3))

model.summary()
