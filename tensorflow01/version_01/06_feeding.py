from __future__ import print_function

import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1

sess = tf.Session(config=config)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

print( "z = ", sess.run(z, feed_dict={x: 3, y:4.3}) )

print( "z = ", sess.run(z, feed_dict={x: [3,1.2], y:[4.3, 3.1]}) )

