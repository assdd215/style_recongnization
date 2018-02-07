#encoding=utf-8
import tensorflow as tf
import input_data
import numpy as np
import model

train_dir = '/Users/aria/MyDocs/cat_vs_dogs/train1/'
test_dir = '/Users/aria/MyDocs/cat_vs_dogs/test/'
train_logs_dir = './logs/train/'
val_logs_dir = './logs/val'

N_CLASSES = 2
IMG_W = 208     # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
RATIO = 0.2     # take 20% of dataset as validation data
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001

def weight_variable(shape,stddev = 0.1):  #stddev  表示标准差
    initial = tf.truncated_normal(shape=shape,dtype=tf.float32,stddev=stddev)
    return tf.Variable(initial)

def biases_variable(shape,value=0.1):

    initial = tf.constant(value,shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def max_pool(x, shape = [1, 3, 3, 1], strides = [1, 2, 2, 1]):
    return tf.nn.max_pool(x,ksize=shape,strides=strides,padding="SAME")

def conv2d(x,weight):
    return tf.nn.conv2d(x,weight,padding="SAME",strides=[1,1,1,1])

def lrn(x):
    return tf.nn.lrn(x,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75)


#获取数据
train, train_label, val, val_label = input_data.get_files(train_dir, RATIO)
train_batch, train_label_batch = input_data.get_batch(train,
                                                  train_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE,
                                                  CAPACITY)
val_batch, val_label_batch = input_data.get_batch(val,
                                                  val_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE,
                                                  CAPACITY)

x = tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,IMG_W,IMG_H,3])
y_ = tf.placeholder(dtype=tf.int32,shape=[BATCH_SIZE])

#第一层卷积层
conv1_weight = weight_variable([3,3,3,16])
conv1_biases = biases_variable([16])
# conv1 = tf.nn.relu(conv2d(x,conv1_weight) + conv1_biases)
conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x,conv1_weight),conv1_biases))
#第一层池化
pool1 = max_pool(conv1)
norm1 = lrn(pool1)

#第二层卷积层
conv2_weight = weight_variable([3,3,16,16])
conv2_biases = biases_variable([16])
# conv2 = tf.nn.relu(conv2d(norm1,conv2_weight) + conv2_biases)
conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(norm1,conv2_weight),conv2_biases))
#第二层池化
norm2 = lrn(conv2)
pool2 = max_pool(norm2, strides=[1, 1, 1, 1])

#第一层全连接层
pool2_flat = tf.reshape(pool2,[BATCH_SIZE,-1])
fc1_weight = weight_variable([pool2_flat.get_shape()[1].value,128],stddev=0.005)
fc1_biases = biases_variable([128])
fc1 = tf.nn.relu(tf.matmul(pool2_flat,fc1_weight) + fc1_biases)  #[64,128]

#第二层全连接层
fc2_weight = weight_variable([128,128],stddev=0.005)
fc2_biases = biases_variable([128])
fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weight) + fc2_biases)  #[64，128]

#softmax层  [64,2]
soft_weight = weight_variable([128,2])
soft_biases = biases_variable([2])
softmax = tf.matmul(fc2, soft_weight) + soft_biases
# softmax = model.inference(x, BATCH_SIZE, N_CLASSES)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax, labels=y_)
loss = tf.reduce_mean(cross_entropy)

global_step = tf.Variable(0, name='global_step', trainable=False)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
correct = tf.nn.in_top_k(softmax, y_, 1)
correct = tf.cast(correct,tf.float16)
acc = tf.reduce_mean(correct)
# acc = model.evaluation(softmax, y_)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

try:
    tra_images, tra_labels = sess.run([train_batch, train_label_batch])

    y = sess.run(softmax, feed_dict={x: tra_images})
    print(y)
    # for step in np.arange(MAX_STEP):
    #     if coord.should_stop():
    #         break
        # _,tra_loss,tra_acc = sess.run([train,loss,acc],feed_dict={x:tra_images,y_:tra_labels})
        # sess.run(train,feed_dict={x:tra_images,y_:tra_labels})
        # if step % 50 == 0:
        #     tra_loss,tra_acc = sess.run([loss,acc],feed_dict={x:tra_images,y_:tra_labels})
        #     print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
        # if step % 200 == 0 or (step + 1) == MAX_STEP:
        #     val_images, val_labels = sess.run([val_batch, val_label_batch])
        #     val_loss, val_acc = sess.run([loss, acc],
        #                                  feed_dict={x: val_images, y_: val_labels})
        #     print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc * 100.0))
# except tf.errors.OutOfRangeError:
#     print('Done training -- epoch limit reached')
# finally:
#     coord.request_stop()
# coord.join(threads)



finally:
    print("end")