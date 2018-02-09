import tensorflow as tf
import input_data
import numpy as np




x = tf.placeholder(shape=[2,4],dtype=tf.int32)
y = tf.placeholder(shape=[2,4],dtype=tf.int32)
z1 = tf.argmax(x,axis=1)
z2 = tf.argmax(y,axis=1)
eq = tf.cast(tf.equal(z1,z2),tf.float32)
eq = tf.reduce_mean(eq)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
x1 = np.array([[1,2,3,4],[2,3,4,5]])
y1 = np.array([[3,4,5,6],[4,5,6,7]])

x2,y2,z3,z4,eq2 = sess.run([x,y,z1,z2,eq],feed_dict={x:x1,y:y1})

print(x2)
print(y2)
print(z3)
print(z4)
print(eq2)