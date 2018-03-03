#encoding=utf-8
import tensorflow as tf
from scipy import misc
import numpy as np
import os


baseFilePath = '/Users/aria/MyDocs/cat_vs_dogs/' # 猫狗大战的训练数据

isMac = True


# 获取训练数据下的图片与对应的label。 训练集用6250张是因为猫狗大战的总训练集是12500张，从中随机抓6250张做训练集 剩下的6250张用于做准确率测试
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
    return train_imgs,train_labels,test_imgs,test_labels  # cat [1,0]  dog [0,1]


# 根据传入图片和label整合成batch返回。batch的作用是每次从训练集中取出batch_size个图片与label投入训练
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


# 获取一个图片并将它转换成矩阵数组便于投入训练
def get_one_img(sess,imgPath,w = 224,h = 224):
    image_content = tf.read_file(imgPath)
    image = tf.image.decode_jpeg(image_content,channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image,w,h)
    image = tf.cast(image,tf.float32)
    image = tf.image.per_image_standardization(image)
    img = sess.run(image)
    img = np.reshape(img,[-1,w,h,3])
    return img


# 输入要进行预测的图片的文件夹，返回图片矩阵和标签矩阵
def get_predict_files(imgpath):
    if os.path.exists(imgpath) == False:
        print("error path")
        return 
    files = os.listdir(imgpath)
    imgs = []
    labels = []
    for file in files:
        temp = file.split('.')
        if temp[-1] != 'jpg' and temp[-1] != 'png' and temp[-1] != 'jpeg':
            continue
        img = misc.imread(os.path.join(imgpath, file), mode='RGB')
        img = misc.imresize(img,(224,224),interp='bilinear')
        imgs.append(img)
        labels.append(file)
    print("图片加载完毕")
    return np.array(imgs),np.array(labels)
