import tensorflow as tf
import tensorflow_datasets as tdfs
import numpy as np

# You can use pip to install tensorflow_datasets package

# Import or generate dataset
# Need h5py installed. It is better to use conda instead of pip
dtset = tdfs.load("iris", split="train")


batch_size = 32
input_size = 4
output_size = 3
learning_rate = 1e-3
Nepochs = 1000

# Transform and normalize data
# Normalized data: mean=0 and stdev=1
for batch in dtset.batch(batch_size, drop_remainder=True):
    labels = tf.one_hot(batch["label"], 3)
    X = batch["features"]
    X = (X - np.mean(X))/np.std(X)


# Intialize variables
weights = tf.Variable(
    tf.random.normal(shape=(input_size,output_size), dtype=tf.float32)
)
biases = tf.Variable(
    tf.random.normal(shape=(output_size,), dtype=tf.float32)
)

# Define model structure
# This is a logistic regression model
logits = tf.add(tf.matmul(X, weights), biases)

# Declare the loss functions
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels, logits)
)

# Initialize and train the model
optimizer = tf.optimizers.SGD(learning_rate)
with tf.GradientTape() as tape:
    logits = tf.add(tf.matmul(X, weights), biases)
    # Huh? declare this here again?
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    )
gradients = tape.gradient(loss, [weights, biases])
optimizer.apply_gradients(zip(gradients, [weights, biases]))

# Evaluate the model
print(f"Final loss is: {loss.numpy():.3f}")
preds = tf.math.argmax( tf.add(tf.matmul(X,weights), biases), axis=1)
ground_truth = tf.math.argmax(labels, axis=1)
for y_true, y_pred in zip(ground_truth.numpy(), preds.numpy()):
    print(f"Real label: {y_true} fitted: {y_pred}")

