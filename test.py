import tensorflow as tf
import input_data
import numpy as np
# dir = "/Users/aria/MyDocs/cat_vs_dogs/train/cat.0.jpg"
# image_content = tf.read_file(dir)
# image = tf.image.decode_jpeg(image_content,channels=3)
# label = [1]
# array = [image,label]
#
# sess = tf.Session()
# a = sess.run(image)
#
# print(type(a))
# print(len(a))
# print(len(a[0]))
# print(len(a[0][0]))
#
# file = open("./array.txt","w")
# file.write(str(a))
# file.close()


#***************************#

x = tf.placeholder(dtype=tf.float32,shape=[4,3])
y = tf.placeholder(dtype=tf.float32,shape=[1])

initial = tf.truncated_normal(shape=[3,1],dtype=tf.float32,stddev=0.1)
weight = tf.Variable(initial)
bias = tf.constant(1,dtype=tf.float32,shape=[4,1])
out = tf.matmul(x,weight) + bias

sess = tf.Session()
sess.run(tf.global_variables_initializer())

temp = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]],dtype=float)
print(type(temp))
a = sess.run(out,feed_dict={x:[[1,2,3],[2,3,4],[3,4,5],[4,5,6]]})
print(type(a))
# return tf.Variable(initial)