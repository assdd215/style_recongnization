#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import os
import facenet
import align.detect_face
from PIL import Image

model = "model/20170512-110547.pb"
image_paths = {"imgs/2142867537.jpg","imgs/2287410834.jpg"}
image_size = 160
margin = 44
gpu_memory_fraction = 1.0
image_files = ["imgs/12.jpg","imgs/11.jpg"]

def main():

    images = load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            
            nrof_images = len(image_files)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, image_files[i]))
            print('')
            
            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    print('  %1.4f  ' % dist, end='')
                print('')
            
            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths = list(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, use origin ", image)
          bounding_boxes = np.array([[0,0,img_size[0],img_size[1]]])
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32) #bb分别对应裁切的图片的左、上、右、下坐标
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')  #这一步把脸部检测出来
        prewhitened = facenet.prewhiten(aligned)  #这里是对脸部进行一个预处理
        img_list.append(prewhitened)
        # img_list.append(aligned)
    images = np.stack(img_list)
    return images

def outputPic():
    images = load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction)
    img1 = images[1]
    new_img = Image.fromarray(img1.astype('uint8'))
    new_img.show()
if __name__ == '__main__':
    # outputPic()
    main()