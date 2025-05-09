import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tf_keras as keras
from tf_keras import layers

import tensorflow as tf
keras.backend.clear_session()
tf.random.set_seed(1234)

# Let's say you’re building a system to rank customer support
# tickets by priority and route them to the appropriate department.
# Your model has three inputs:
# - The title of the ticket (text input)
# - The text body of the ticket (text input)
# - Any tags added by the user (categorical input, assumed here to be one-hot encoded)

# We can encode the text inputs as arrays of ones and zeros of size
# vocabulary_size (see chapter 11 for detailed information about text encoding techniques).
# Your model also has two outputs:
# - The priority score of the ticket, a scalar between 0 and 1 (sigmoid output)
# - The department that should handle the ticket (a softmax over the set of departments)

vocabulary_size = 10_000
num_tags = 100
num_departments = 4

title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu")(features)

priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department =layers.Dense(num_departments, activation="softmax", name="department")(features)

model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department]
)

model.summary()

import numpy as np
num_samples = 1280

# Dummy input data
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# Dummy target data
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

model.compile(
    optimizer="rmsprop",
    loss=["mean_squared_error", "categorical_crossentropy"],
    metrics=[ ["mean_absolute_error"], ["accuracy"] ]
)

model.fit(
    [title_data, text_body_data, tags_data],
    [priority_data, department_data],
    epochs=1
)

# Score
model.evaluate(
    [title_data, text_body_data, tags_data],
    [priority_data, department_data]
)

# Prediction
priority_pred, department_pred = model.predict(
    [title_data, text_body_data, tags_data]
)
