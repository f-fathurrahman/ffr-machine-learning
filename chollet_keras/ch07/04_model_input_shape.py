import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tf_keras as keras
from tf_keras import layers

import tensorflow as tf
keras.backend.clear_session()
tf.random.set_seed(1234)

model = keras.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(64, activation="relu", name="layer1"),
    layers.Dense(10, activation="softmax", name="layer2")
])
# Shape argument for Input layer must be the shape of each sample,
# not the shape of one batch

model.summary()
