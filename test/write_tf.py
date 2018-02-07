#encoding=utf-8
import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np


def writeImgsToTfRecord(path, labels, outputFile, w, h):
    writer = tf.python_io.TFRecordWriter(outputFile)
    for index, name in enumerate(labels):
        label_path = path + name + "/"
        for img_name in os.listdir(label_path):
            if img_name == '.DS_Store':
                continue
            img = Image.open(label_path + img_name)
            img = img.resize((w, h))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })) #example对象对label和image数据进行封装

            writer.write(example.SerializeToString())
    writer.close()

def readDataFromTfRecord(fileName):
    fileName_queue = tf.train.string_input_producer([fileName])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(fileName_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'],tf.uint8)
    image = tf.reshape(image,[128,128,3])
    label = tf.cast(features['label'],tf.int32)
    return image,label

cwd='/Users/aria/MyDocs/cat_vs_dogs/tfrecord/'
classes={'1','2'} #人为 设定 2 类
writeImgsToTfRecord(cwd,classes,"myTfRecords",128,128)



image,label = readDataFromTfRecord("myTfRecords")

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(20):
        example,l = sess.run([image,label])
        print(l)
    coord.request_stop()
    coord.join(threads)

# with tf.Session() as sess: #开始一个会话
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     coord=tf.train.Coordinator()
#     threads= tf.train.start_queue_runners(coord=coord)
#     for i in range(20):
#         example, l = sess.run([image,label])#在会话中取出image和label
#         img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
#         img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
#         print(example, l)
#     coord.request_stop()
#     coord.join(threads)