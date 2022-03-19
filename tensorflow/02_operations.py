import tensorflow as tf
import numpy as np

x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x_data = tf.Variable(x_vals, dtype=tf.float64)
# tf.float64 is default if we want to convert it from NumPy array with dtype float64

m_const = tf.constant(2.0, dtype=tf.float64)
# need to specify tf.float64

# Multiplication
res = tf.multiply(x_data, m_const)
print(res.numpy())