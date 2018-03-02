#encoding=utf-8
from scipy import misc
from PIL import Image
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import os

model = "model/20170512-110547.pb"
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
        img = misc.imread(os.path.join(img_path, image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            print("can't detect face, use origin ", image) #当识别不到脸型的时候  直接把原图压缩处理
            bounding_boxes = np.array([[0, 0, img_size[0], img_size[1]]])
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')  # 这一步把脸部检测出来
        if use_to_save:
            result[image.split('.')[0]] = aligned
        else:
            prewhitened = facenet.prewhiten(aligned)  # 图片进行白化
            result[image.split('.')[0]] = prewhitened
    return result

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
        # result[str(os.path.join(save_path, image))] = prewhitened
        result[image.split('.')[0]] = prewhitened
    print("loaded data")
    return result

# def process_imgs(imgs):
#     list = {}
#     for image in imgs:
