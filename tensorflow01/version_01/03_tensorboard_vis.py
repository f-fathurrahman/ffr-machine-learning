import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # tf.float32 by default
total = a + b
print(a)
print(b)
print(total)

writer = tf.summary.FileWriter(".")
writer.add_graph( tf.get_default_graph() )
writer.close()

