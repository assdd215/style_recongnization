#encoding=utf-8
import tensorflow as tf
import numpy as np
import os
import math

# you need to change this to your data directory
train_dir = '/Users/aria/MyDocs/cat_vs_dogs/'



def get_files(file_dir, ratio):
    """
    Args:
        file_dir: file directory
        ratio:ratio of validation datasets
    Returns:
        list of images and labels
    """
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = n_sample - n_val # number of trainning samples

    tra_images = all_image_list[0:int(n_train)]
    tra_labels = all_label_list[0:int(n_train)]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[int(n_train):-1]
    val_labels = all_label_list[int(n_train):-1]
    val_labels = [int(float(i)) for i in val_labels]
    return tra_images,tra_labels,val_images,val_labels


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)#裁剪图片到指定尺寸

    # if you want to test the generated batches of images, you might want to comment the following line.
    # image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = capacity)
    #you can also use shuffle_batch
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch       #一个批量的图片batch和label batch  图片为[-1,width,height,channels]  label为[batch_size,1]

# baseFilePath = '/Users/aria/MyDocs/pics/'

baseFilePath = 'D:\\train_data\\'
def load_style_and_path(imgs,labels,filePath,shape):
    # filePath = filePath + "/"
    filePath = filePath + "\\"
    for file in os.listdir(filePath):
        if file == '.DS_Store':
            continue
        imgs.append(filePath + file)
        labels.append(shape)

def get_img_files():
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []

    #制作
    # load_style_and_path(train_imgs, train_labels, str(baseFilePath + "train/2_狂野"), [1, 0, 0, 0])
    # load_style_and_path(train_imgs, train_labels, str(baseFilePath + "train/4_甜美"), [0, 1, 0, 0])
    # load_style_and_path(train_imgs,train_labels,str(baseFilePath + "train/5_小清新"),[0,0,1,0])
    # load_style_and_path(train_imgs,train_labels,baseFilePath + "train/6_冷艳",[0,0,0,1])

    load_style_and_path(train_imgs, train_labels, str(baseFilePath + "train\\2_狂野"), [1, 0, 0, 0])
    load_style_and_path(train_imgs, train_labels, str(baseFilePath + "train\\4_甜美"), [0, 1, 0, 0])
    load_style_and_path(train_imgs,train_labels,str(baseFilePath + "train\\5_小清新"),[0,0,1,0])
    load_style_and_path(train_imgs,train_labels,baseFilePath + "train\\6_冷艳",[0,0,0,1])

    result_img = np.array(train_imgs)
    result_labels = np.array(train_labels)

    ##制作测试集
    # load_style_and_path(test_imgs, test_labels, baseFilePath + "test/2_狂野", [1, 0, 0, 0])
    # load_style_and_path(test_imgs, test_labels, baseFilePath + "test/4_甜美", [0, 1, 0, 0])
    # load_style_and_path(test_imgs, test_labels, baseFilePath + "test/5_小清新", [0, 0, 1, 0])
    # load_style_and_path(test_imgs, test_labels, baseFilePath + "test/6_冷艳", [0, 0, 0, 1])

    load_style_and_path(test_imgs, test_labels, baseFilePath + "test\\2_狂野", [1, 0, 0, 0])
    load_style_and_path(test_imgs, test_labels, baseFilePath + "test\\4_甜美", [0, 1, 0, 0])
    load_style_and_path(test_imgs, test_labels, baseFilePath + "test\\5_小清新", [0, 0, 1, 0])
    load_style_and_path(test_imgs, test_labels, baseFilePath + "test\\6_冷艳", [0, 0, 0, 1])

    result_img_test = np.array(test_imgs)
    result_labels_test = np.array(test_labels)

    return result_img,result_labels,result_img_test,result_labels_test

def get_img_batch(imgs,labels,w = 256,h = 256,batch_size = 32,capacity = 2000):
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

