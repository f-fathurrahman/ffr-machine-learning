# Example of creating a subclassed model that includes a Functional model

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tf_keras as keras
from tf_keras import layers

import tensorflow as tf
keras.backend.clear_session()
tf.random.set_seed(1234)

# Create a Functional model
inputs = keras.Input(shape=(64,))
outputs = layers.Dense(1, activation="sigmoid")(inputs)
binary_classifier = keras.Model(inputs=inputs, outputs=outputs)

# The following subclassed Model will use binary_classifier, which is
# created by Functional model
class MyModel(keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        self.dense = layers.Dense(64, activation="relu")
        self.classifier = binary_classifier # XXX capture a global variable?
    
    def call(self, inputs):
        features = self.dense(inputs)
        return self.classifier(features)


model = MyModel()
model.build(input_shape=(None,64))
model.summary()

