# Example of creating a Functional model that includes a subclassed model

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tf_keras as keras
from tf_keras import layers

import tensorflow as tf
keras.backend.clear_session()
tf.random.set_seed(1234)

class Classifier(keras.Model):

    def __init__(self, num_classes=2):
        super().__init__()
        if num_classes == 2:
            num_units = 1
            activation = "sigmoid"
        else:
            num_units = num_classes
            activation = "softmax"
        # Create the layer
        self.dense = layers.Dense(num_units, activation=activation)
    
    def call(self, inputs):
        return self.dense(inputs)

# Test
inputs = keras.Input(shape=(3,))
features = layers.Dense(64, activation="relu")(inputs)
outputs = Classifier(num_classes=10)(features)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

