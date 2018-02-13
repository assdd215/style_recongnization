#encoding=utf-8
import tensorflow as tf
import numpy as np
import os


baseFilePath = '/Users/aria/MyDocs/cat_vs_dogs/'

isMac = True
def get_img_files(train_size = 6250):
    cat_imgs = []
    cat_labels = []
    dog_imgs = []
    dog_labels = []
    train_labels = []
    train_imgs = []
    test_labels = []
    test_imgs = []

    for file in os.listdir(os.path.join(baseFilePath,"train")):
        if file == '.DS_Store':
            continue
        if file.split('.')[0] == 'cat':
            cat_imgs.append(os.path.join(baseFilePath, "train", file))
            cat_labels.append([1,0])
        else:
            dog_imgs.append(os.path.join(baseFilePath,"train",file))
            dog_labels.append([0,1])

    for i in range(0,train_size):
        train_imgs.append(cat_imgs[i])
        train_labels.append(cat_labels[i])
        train_imgs.append(dog_imgs[i])
        train_labels.append(dog_labels[i])
    for i in range(train_size,len(cat_imgs)):
        test_imgs.append(cat_imgs[i])
        test_labels.append(cat_labels[i])
        test_imgs.append(dog_imgs[i])
        test_labels.append(dog_labels[i])
    return train_imgs,train_labels,test_imgs,test_labels

def get_img_batch(imgs,labels,w = 224,h = 224,batch_size = 32,capacity = 2000):
    image = tf.cast(imgs,dtype=tf.string)
    label = tf.convert_to_tensor(labels,dtype=tf.int16)
    input_queue = tf.train.slice_input_producer([image,label],shuffle=True)#这个函数的功能还是不太懂
    label = input_queue[1]
    image_str = input_queue[0]
    image_content = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_content,channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image,w,h)
    image_str_batch,image_batch,label_batch = tf.train.shuffle_batch([image_str,image,label],
                                                                     batch_size=batch_size,num_threads=64,
                                                                     capacity=capacity,min_after_dequeue=capacity - 1)
    image_batch = tf.cast(image_batch,tf.float32)

    return image_batch,label_batch

def get_one_img(sess,imgPath,w = 224,h = 224):
    image_content = tf.read_file(imgPath)
    image = tf.image.decode_jpeg(image_content,channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image,w,h)
    image = tf.cast(image,tf.float32)
    image = tf.image.per_image_standardization(image)
    img = sess.run(image)
    img = np.reshape(img,[-1,w,h,3])
    return img

def get_one_img(sess,imgPath,w = 224,h = 224):
    image_content = tf.read_file(imgPath)
    image = tf.image.decode_jpeg(image_content,channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image,w,h)
    image = tf.cast(image,tf.float32)
    image = tf.image.per_image_standardization(image)
    img = sess.run(image)
    img = np.reshape(img,[-1,w,h,3])
    return img