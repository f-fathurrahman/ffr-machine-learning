import numpy as np
import tensorflow as tf

a = tf.constant(3.1, dtype=tf.float32)
b = tf.constant(4.3) # tf.float32 by default
total = a + b
print(a)
print(b)
print(total)

writer = tf.summary.FileWriter(".")
writer.add_graph( tf.get_default_graph() )

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1

sess = tf.Session(config=config)
print("total = ", sess.run(total))


