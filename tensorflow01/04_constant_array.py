import tensorflow as tf
import numpy as np

t = tf.constant([
    [1.0, 2.0, 3.0],
    [4.0, 3.0, 4.5]
])

print("t = ")
print(t)

t2 = tf.constant(np.random.rand(3, 3, 3))
print("t2 = ")
print(t2)

print(t[:,1:])

print(t2[...,1,tf.newaxis])

print(t + 10)

print(tf.square(t))

# Matrix mult
print( t @ tf.transpose(t) )

