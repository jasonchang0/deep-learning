import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)

print(result)

sess = tf.Session()
print(sess.run(result))
sess.close()

# Without manual closure
with tf.Session() as sess:
    print(sess.run(result))

# Computation graph vs Python variable
with tf.Session() as sess:
    output = sess.run(result)
    print(output)

print(output)

