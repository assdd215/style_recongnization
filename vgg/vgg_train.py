#encoding=utf-8
import tensorflow as tf
import numpy as np
import input_data

learning_rate = 0.0005

class Vgg16:
    def __init__(self,vgg_npy_path = None):
        self.data_dict = np.load(vgg_npy_path,encoding="latin1").item()
        self.x = tf.placeholder(tf.float32,[None,224,224,3])
        self.y_ = tf.placeholder(tf.float32,[None,4])
        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(self.x, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        self.flatten = tf.reshape(pool5,[-1,7*7*512])
        with tf.name_scope(name='fc6'):
            self.fc6 = tf.layers.dense(self.flatten,256,tf.nn.relu,name='fc6')

        with tf.name_scope(name='softmax'):
            softmax_W = self.weight_variable([256,4],name='softmax_W')
            softmax_b = self.biases_variable([4],name='softmax_b')
            tf.summary.histogram('softmax/W',softmax_W)
            tf.summary.histogram('softmax/b',softmax_b)

            self.softmax = tf.add(tf.matmul(self.fc6,softmax_W),softmax_b,name='softmax')

        with tf.name_scope(name='loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax,labels=self.y_)
            self.loss = tf.reduce_mean(cross_entropy,name='loss')
            tf.summary.scalar('loss',self.loss)


        with tf.name_scope(name='train_op'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        with tf.name_scope(name='acc'):
            correct = tf.equal(tf.argmax(self.softmax,axis=1),tf.argmax(self.y_,axis=1))
            correct = tf.cast(correct,tf.float32)
            self.acc = tf.reduce_mean(correct,name='acc')
            tf.summary.scalar('acc',self.acc)

        self.merged = tf.summary.merge_all()



    def conv_layer(self,bottom,name):
        with tf.name_scope(name=name):
            conv = tf.nn.conv2d(bottom,self.data_dict[name][0],[1,1,1,1],padding='SAME',name=name)
            return tf.nn.relu(tf.nn.bias_add(conv,self.data_dict[name][1]))

    def max_pool(self,bottom,name):
        return tf.nn.max_pool(bottom,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)

    def weight_variable(self,shape,name="weight",stddev = 0.1):  #stddev  表示标准差
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initializer(shape),name=name)

    def biases_variable(self,shape, name="bias", value=0.1):
        initial = tf.constant(value, shape=shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)

IMG_W = 224
IMG_H = 224
BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 5000

# npyPath = "/Users/aria/MyDocs/npy/vgg16.npy"
npyPath = "D:\\train_data\\npy\\vgg16.npy"
def train():
    train, train_label, test_img, test_label = input_data.get_img_files()
    train_batch, train_label_batch = input_data.get_img_batch(train, train_label, w=IMG_W, h=IMG_H,
                                                              batch_size=BATCH_SIZE, capacity=CAPACITY)
    test_batch, test_label_batch = input_data.get_img_batch(test_img, test_label, w=IMG_W, h=IMG_H,
                                                            batch_size=BATCH_SIZE, capacity=CAPACITY)

    vgg = Vgg16(npyPath)
    sess = tf.Session()
    writer = tf.summary.FileWriter('graph/', sess.graph)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(MAX_STEP):
            tra_images,tra_labels = sess.run([train_batch,train_label_batch])
            _,tra_loss,tra_acc = sess.run([vgg.train_op,vgg.loss,vgg.acc],feed_dict={vgg.x:tra_images,vgg.y_:tra_labels})
            if step % 50 == 0:
                print("step:%d , tra loss = %.2f, tra acc = %.2f%%"%(step,tra_loss,tra_acc * 100))
                merged = sess.run(vgg.merged,feed_dict={vgg.x:tra_images,vgg.y_:tra_labels})
                writer.add_summary(merged,step)
            if step % 100 == 0 or step + 1 == MAX_STEP:
                val_imgs_batch, val_label_batch = sess.run([test_batch,test_label_batch])
                val_loss,val_acc = sess.run([vgg.loss,vgg.acc],
                                                   feed_dict={vgg.x:val_imgs_batch,vgg.y_:val_label_batch})
                print("Step:%d, val loss = %.2f, val acc = %.2f%%" % (step,val_loss,val_acc * 100))
    except Exception as e:
        print(e)
        coord.request_stop()
    finally:
        coord.request_stop()
    coord.join(threads)

if __name__=='__main__':
    train()
