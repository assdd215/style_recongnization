#encoding=utf-8
import tensorflow as tf
import input_data

train_dir = '/Users/aria/MyDocs/cat_vs_dogs/train1/'
test_dir = '/Users/aria/MyDocs/cat_vs_dogs/test/'
train_logs_dir = './logs/train/'
val_logs_dir = './logs/val'

N_CLASSES = 2
IMG_W = 256     # resize the image, if the input image is too large, training will be very slow.
IMG_H = 256
RATIO = 0.2     # take 20% of dataset as validation data
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.001 # with current parameters, it is suggested to use learning rate<0.0001

def weight_variable(shape,name="weight",stddev = 0.1):  #stddev  表示标准差
    initial = tf.truncated_normal(shape=shape,dtype=tf.float32,stddev=stddev)
    return tf.Variable(initial,name=name)

def biases_variable(shape,name="bias",value=0.1):

    initial = tf.constant(value,shape=shape,dtype=tf.float32)
    return tf.Variable(initial,name=name)

def max_pool(x, name="maxPool",shape = [1, 2, 2, 1], strides = [1, 2, 2, 1]):
    return tf.nn.max_pool(x,ksize=shape,strides=strides,padding="SAME",name=name)

def conv2d(x,weight,name="conv2d"):
    return tf.nn.conv2d(x,weight,padding="SAME",strides=[1,1,1,1])

def lrn(x,name="lrn"):
    return tf.nn.lrn(x,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75)


train,train_label,test_img,test_label = input_data.get_img_files()
train_batch,train_label_batch = input_data.get_img_batch(train,train_label,w=IMG_W,h = IMG_H,batch_size=BATCH_SIZE,capacity=CAPACITY)
test_batch,test_label_batch = input_data.get_img_batch(test_img,test_label,w=IMG_W,h = IMG_H,batch_size=BATCH_SIZE,capacity=CAPACITY)



with tf.name_scope('inpout'):
    x = tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,IMG_W,IMG_H,3],name="imgs_batch")
    y_ = tf.placeholder(dtype=tf.int32,shape=[BATCH_SIZE,4],name="label_batch")

#第一层卷积层
with tf.name_scope("conv1"):
    conv1_weight = weight_variable([3,3,3,16],name="conv1_W")
    conv1_biases = biases_variable([16],name='conv1_b')
    # conv1 = tf.nn.relu(conv2d(x,conv1_weight) + conv1_biases)
    conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x,conv1_weight),conv1_biases),name="conv1")
#第一层池化
with tf.name_scope("pool1"):
    pool1 = max_pool(conv1)
    norm1 = lrn(pool1)

#第二层卷积层
with tf.name_scope('conv2'):
    conv2_weight = weight_variable([3,3,16,16],name="conv2_W")
    conv2_biases = biases_variable([16],name="conv2_b")
    # conv2 = tf.nn.relu(conv2d(norm1,conv2_weight) + conv2_biases)
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(norm1,conv2_weight),conv2_biases),name='conv2')
#第二层池化
with tf.name_scope('pool2'):
    pool2 = max_pool(conv2)
    # norm2 = lrn(conv2)
    # pool2 = max_pool(norm2)

#第一层全连接层
with tf.name_scope('fc1'):
    pool2_flat = tf.reshape(pool2,[BATCH_SIZE,-1])
    fc1_weight = weight_variable([pool2_flat.get_shape()[1].value,128],stddev=0.005,name='fc1_W')
    fc1_biases = biases_variable([128],name="fc1_b")
    fc1 = tf.nn.relu(tf.matmul(pool2_flat,fc1_weight) + fc1_biases)  #[64,128]

#第二层全连接层
with tf.name_scope('fc2'):
    fc2_weight = weight_variable([128,128],stddev=0.005,name='fc2_W')
    fc2_biases = biases_variable([128],name='fc2_b')
    fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weight) + fc2_biases)  #[64，128]

#softmax层  [64,2]
with tf.name_scope('softmax'):
    soft_weight = weight_variable([128,4],name="softmax_W")
    soft_biases = biases_variable([4],name="softmax_b")
    softmax = tf.matmul(fc2, soft_weight) + soft_biases

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=softmax, labels=y_)
    loss = tf.reduce_mean(cross_entropy,name="loss")

with tf.name_scope("train"):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

with tf.name_scope("acc"):
    temp = tf.argmax(softmax,axis=1)
    correct = tf.equal(tf.argmax(softmax,axis=1), tf.argmax(y_,axis=1))
    correct = tf.cast(correct,tf.float32)
    acc = tf.reduce_mean(correct)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("graphs/",sess.graph)
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)



try:
    for i in range(MAX_STEP):
        tra_images, tra_labels = sess.run([train_batch, train_label_batch])
        _,tra_loss,tra_acc = sess.run([train_op,loss,acc],feed_dict={x:tra_images,y_:tra_labels})
        if i % 50 == 0 or i + 1 == MAX_STEP:
            print("Step: %d, tra loss = %.2f,train acc = %.2f%%" % (i,tra_loss,tra_acc * 100))
        if i % 100 == 0 or i + 1 == MAX_STEP:
            writer = open("./logs.txt",'a')
            val_img,val_labels = sess.run([test_batch,test_label_batch])
            val_result,val_loss,val_acc = sess.run([softmax,loss,acc],feed_dict={x:val_img,y_:val_labels})
            print("Step: %d,  val loss = %.2f,val acc = %.2f%%"%(i,val_loss,val_acc * 100))
            result = sess.run(merged,feed_dict={x:val_img,y_:val_labels})
            writer.add_summary(result,i)

finally:
    print("end")