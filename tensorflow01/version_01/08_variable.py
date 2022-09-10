import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

sess = tf.Session()
sess.run( x.initializer )
sess.run( y.initializer )
result = ress.run( f )

print(result)

sess.close()