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
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001

def weight_variable(shape,stddev = 0.1):  #stddev  表示标准差

    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))

    # initial = tf.truncated_normal(shape=shape,dtype=tf.float32,stddev=stddev)
    # return tf.Variable(initial)

def weigth_xavier_variable(shape):
    return tf.get_variable(shape=shape,initializer=tf.contrib.layers.xavier_initializer())

def biases_variable(shape,value=0.1):

    initial = tf.constant(value,shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def max_pool(x, shape = [1, 2, 2, 1], strides = [1, 2, 2, 1]):
    return tf.nn.max_pool(x,ksize=shape,strides=strides,padding="SAME")

def conv2d(x,weight):
    return tf.nn.conv2d(x,weight,padding="SAME",strides=[1,1,1,1])

def lrn(x):
    return tf.nn.lrn(x,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75)


train,train_label,test_img,test_label = input_data.get_img_files()
train_batch,train_label_batch = input_data.get_img_batch(train,train_label,w=IMG_W,h = IMG_H,batch_size=BATCH_SIZE,capacity=CAPACITY)
test_batch,test_label_batch = input_data.get_img_batch(test_img,test_label,w=IMG_W,h = IMG_H,batch_size=BATCH_SIZE,capacity=CAPACITY)


x = tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,IMG_W,IMG_H,3],name="image_batch")
y_ = tf.placeholder(dtype=tf.int32,shape=[BATCH_SIZE,4],name="label_batch")

#第一层卷积层
conv1_weight = weight_variable([3,3,3,16])
conv1_biases = biases_variable([16])
# conv1 = tf.nn.relu(conv2d(x,conv1_weight) + conv1_biases)
conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x,conv1_weight),conv1_biases))
#第一层池化
pool1 = max_pool(conv1)
norm1 = lrn(pool1)

#第二层卷积层
conv2_weight = weight_variable([5,5,16,16])
conv2_biases = biases_variable([16])
conv2 = tf.nn.relu(conv2d(norm1,conv2_weight) + conv2_biases)
# conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(conv1,conv2_weight),conv2_biases))
#第二层池化
pool2 = max_pool(conv2)
norm2 = lrn(pool2)
# pool2 = max_pool(norm2)

#第一层全连接层
pool2_flat = tf.reshape(norm2,[BATCH_SIZE,-1])
fc1_weight = weight_variable([pool2_flat.get_shape()[1].value,128],stddev=0.005)
fc1_biases = biases_variable([128])
fc1 = tf.nn.relu(tf.matmul(pool2_flat,fc1_weight) + fc1_biases)  #[64,128]

#第二层全连接层
# fc2_weight = weight_variable([128,128],stddev=0.005)
# fc2_biases = biases_variable([128])
# fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weight) + fc2_biases)  #[64，128]

#softmax层  [64,2]
soft_weight = weight_variable([128,4])
soft_biases = biases_variable([4])
# softmax = tf.matmul(fc1, soft_weight) + soft_biases
softmax = tf.add(tf.matmul(fc1, soft_weight),soft_biases,name="softmax")


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=softmax, labels=y_)

# cross_entropy =

loss = tf.reduce_mean(cross_entropy)

global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
temp = tf.argmax(softmax,axis=1)
correct = tf.equal(tf.argmax(softmax,axis=1), tf.argmax(y_,axis=1))
correct = tf.cast(correct,tf.float32)
acc = tf.reduce_mean(correct,name="acc")

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)



try:
    for i in range(MAX_STEP):
        tra_images, tra_labels = sess.run([train_batch, train_label_batch])
        _,tra_loss,tra_acc = sess.run([train_op,loss,acc],feed_dict={x:tra_images,y_:tra_labels})
        if i % 50 == 0 or i + 1 == MAX_STEP:
            print("Step: %d, tra loss = %.2f,train acc = %.2f%%" % (i,tra_loss,tra_acc * 100))
        if i % 200 == 0 or i + 1 == MAX_STEP:
            writer = open("./logs.txt",'a')
            val_img,val_labels = sess.run([test_batch,test_label_batch])
            val_result,val_loss,val_acc = sess.run([softmax,loss,acc],feed_dict={x:val_img,y_:val_labels})
            print("Step: %d,  val loss = %.2f,val acc = %.2f%%"%(i,val_loss,val_acc * 100))
            # writer.write("=================" + str(i) + "====================")
            # writer.write(str(val_labels) + "\n")
            # writer.write(str(val_result) + "\n")
            # writer.write("acc:" + str(val_acc)+"\n")
            # writer.close()
    saver.save(sess,'./result_model/MyModel',global_step = MAX_STEP)

finally:
    coord.request_stop()
    print("end")
coord.join()
