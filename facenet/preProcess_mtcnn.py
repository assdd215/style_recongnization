#encoding=utf-8
from scipy import misc
from PIL import Image
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import os
import shutil
import types


model = "/Users/aria/PycharmProjects/style_recongnize/facenet/model/20170512-110547.pb"
base_img_path = "/Users/aria/MyDocs/pics/anchors"
save_path = "/Users/aria/MyDocs/pics/resize_anchors"
image_size = 160
margin = 44
gpu_memory_fraction = 1.0


def load_imgs(img_path = base_img_path,use_to_save = True):
    minsize = 20
    threshold = [0.6,0.7,0.7]
    factor = 0.709
    result = {}
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    img_paths = os.listdir(img_path)
    for image in img_paths:
        if image == '.DS_Store':
            continue
        aligned = mtcnn(os.path.join(img_path, image),minsize,pnet,rnet,onet,threshold,factor)
        if aligned == None:
            img_paths.remove(image)
            continue
        if use_to_save:
            result[image.split('.')[0]] = aligned
        else:
            prewhitened = facenet.prewhiten(aligned)  # 图片进行白化
            result[image.split('.')[0]] = prewhitened
    return result


def load_img(files):
    minsize = 20
    threshold = [0.6,0.7,0.7]
    factor = 0.709
    result = {}
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    for image in files:
        if image == '.DS_Store':
            continue
        aligned = mtcnn(image,minsize,pnet,rnet,onet,threshold,factor)
        if aligned == None:
            files.remove(image)
            continue
        prewhitened = facenet.prewhiten(aligned)  # 图片进行白化
        result[image.split('.')[0]] = prewhitened
    return result


def mtcnn(img_path,minsize, pnet, rnet, onet, threshold,factor):
    img = misc.imread(img_path, mode='RGB')  # 读取图片将图片转换成矩阵
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                      factor)  # 利用dect_face检测人脸
    # 这里的bounding_boxes实质上是指四个点 四个点连起来构成一个框
    if len(bounding_boxes) < 1:
        print("can't detect face, remove ", img_path)  # 当识别不到脸型的时候,不保留
        return None
        # bounding_boxes = np.array([[0, 0, img_size[0], img_size[1]]])
    det = np.squeeze(bounding_boxes[0, 0:4])
    # 这里是为检测到的人脸框加上边界
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    # 根据人脸框截取img得到cropped
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    # 这里是resize成适合输入到模型的尺寸
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    return aligned


def save_imgs():
    result = load_imgs()
    print("load pic end. result size: %d"%len(result))
    for image_path in result:
        img = Image.fromarray(result[image_path].astype('uint8'))
        img.save(os.path.join(save_path,image_path))
    print("save end")

def get_processed_imgs():
    img_path = os.listdir(save_path)
    result = {}
    for image in img_path:
        if image == '.DS_Store':
            continue
        img = misc.imread(os.path.join(save_path, image), mode='RGB')
        prewhitened = facenet.prewhiten(img)  # 白化的目的是去除输入数据的冗余信息。
        result[image.split('.')[0]] = prewhitened
    print("loaded data")
    return result


def embedPic(pics):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")  # 将图片矩阵映射为128的特征向量
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            feed_dict = {images_placeholder:pics,phase_train_placeholder:False}
            emb = sess.run(embeddings,feed_dict=feed_dict)
            return emb


def check_all_in_database(img_path):
    unembed = []
    database_labels = os.listdir(base_img_path)
    if type(img_path) is types.ListType:
        for image_path in img_path:
            image = image_path.split('/')[-1]
            if image in database_labels:
                print("image in database:%s"% image)
            else:
                unembed.append(image_path)
                shutil.copyfile(img_path, os.path.join(base_img_path, img_path.split("/")[-1]))
    elif os.path.isfile(img_path):
        image = img_path.split('/')[-1]
        if image in database_labels:
            print("image in database:%s" %image)
        else:
            unembed.append(img_path)
            shutil.copyfile(img_path,os.path.join(base_img_path,img_path.split("/")[-1]))
    elif os.path.isdir(img_path):

        test_labels = os.listdir(img_path)
        for label in test_labels:
            if label == '.DS_Store':
                continue
            if label in database_labels:
                continue
            # 保存文件
            unembed.append(os.path.join(img_path,label))
            shutil.copyfile(os.path.join(img_path,label), os.path.join(base_img_path, label))

    # 训练所有未转换的图片
    if len(unembed) == 0:
        print("all data in the database")
        return

    # 制作图片矩阵
    print("datas not in the database:")
    print(unembed)
    minsize = 20
    threshold = [0.6,0.7,0.7]
    factor = 0.709
    print("start emb...")
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            imgs = []
            labels = []
            for image in unembed:
                aligned = mtcnn(image,minsize,pnet,rnet,onet,threshold,factor)
                if aligned == None:
                    unembed.remove(image)
                    continue
                prewhitened = facenet.prewhiten(aligned)
                imgs.append(prewhitened)
                labels.append(image.split('/')[-1].split('.')[0])
            emb = embedPic(imgs)
            print("emb finish")
            print("start save...")
            np_sort = np.column_stack([np.array(labels),emb])
            current_emb = np.load("model/database.npy")

            current_emb = np.row_stack([current_emb,np_sort])
            np.save("model/database.npy", current_emb)
            print("save finish")










