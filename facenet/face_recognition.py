#encoding=utf-8
import tensorflow as tf
import numpy as np
import os
import facenet
import preProcess_mtcnn

model = "model/20170512-110547.pb"
pic1 = "/Users/aria/MyDocs/pics/resize_anchors/1154752421.jpg"
pic2 = "/Users/aria/MyDocs/pics/resize_anchors/1168973115.jpg"
database_file = "model/database.npy"


class SimpleData(object):
    key = 0.0
    value = 0.0
    target = 0.0
    def __init__(self):
        self.key = 0.0
        self.value = 0.0

    def __init__(self,key,value):
        self.key = key
        self.value = value

    def __init__(self,key,value,target):
        self.key = key
        self.value = value
        self.target = target


#测试用函数 检验两张图之间的距离
def compare():
    imgs_data = preProcess_mtcnn.get_processed_imgs()

    sample_data = np.array([imgs_data[pic1],imgs_data[pic2]])

    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0") # 将图片矩阵映射为128的特征向量
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            feed_dict = {images_placeholder:sample_data,phase_train_placeholder:False}
            emb = sess.run(embeddings,feed_dict=feed_dict)
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:] , emb[1,:]))))
            print("得到的欧式距离为：%f"% dist)

n = 100  #表示选择的推荐数量
def main():
    if os.path.exists(database_file) == False:
        print("当前没有数据文件，先进行制作")
        embedDatabaseAndSaveAsNpy()
    current_emb = np.load(database_file)
    #将用户的关注整理成列表,现在是这么理解的，数据库不变，用户的关注列表相当于是从数据库中抓取的图片来进行选择对比
    sample_labels = load_labels("imgs")
    all_result = []
    mDataDict = changeNpToDict(current_emb)
    #这个整理的代码感觉不是很优雅。 暂时先这样把
    for sample in sample_labels:
        sample_emb = mDataDict[sample].astype('float64')
        result = []
        for compare_index in mDataDict:
            if compare_index == sample:
                continue
            compare_data = mDataDict[compare_index].astype('float64')
            dist = np.sqrt(np.sum(np.square(np.subtract(sample_emb,compare_data))))
            data = SimpleData(float(compare_index),dist,float(sample))
            result.append(data)
            all_result.append(data)
    all_result.sort(cmp=cmp)  #对拿到的欧氏距离数组进行排序
    output = []
    count = 0
    for item in all_result:
        for input_label in sample_labels:
            if float(item.key) == float(input_label): #排除已存在用户的关注列表中的选项
                continue
        for exist_item in output:   #排除已存在已选择列表中的选项
            if exist_item.key == item.key:
                continue
        output.append(item)
        count = count + 1
        if count == n:
            break
    for item in output:
        print("key:%f,     value:%f,     target:%f"%(item.key,item.value,item.target))

def cmp(data1,data2):
    if data1.value == data2.value:
        return 0
    if data1.value < data2.value:
        return -1
    if data1.value > data2.value:
        return 1


def changeDictToArray(dict):

    img_index = []
    data_content = []

    for index in dict:
        img_index.append(index)
        data_content.append(dict[index])

    return np.array(img_index),np.array(data_content)


def changeNpToDict(emb):
    result = {}
    for col in emb:
        result[col[0]] = np.delete(col,0,axis=0)
    return result


def loadNpy():
    database = ''
    try:
        database = np.load(database_file)
    except Exception:
        print("load error!")
    return database


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


def load_labels(img_paths):
    labels = os.listdir(img_paths)
    output = []
    for label in labels:
        if label == '.DS_Store':
            continue
        output.append(label.split('.')[0])

    return np.array(output)


#这里用于更新数据库的npy便于后面的使用
def embedDatabaseAndSaveAsNpy():
    print("start load imgs...")
    imgs_dict = preProcess_mtcnn.load_imgs(use_to_save=False)  #先把图片保存为
    img_index,img_content = changeDictToArray(imgs_dict)
    print("img loaded")
    print("start embedding...")
    img_emb = embedPic(img_content)
    print("img emded")
    print("start saving...")
    np_soft = np.column_stack([img_index,img_emb])
    np.save("model/database.npy",np_soft)
    print("saved")


main()
