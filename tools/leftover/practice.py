import os
import sys
import math
import random
import time
import numpy as np
import time

from skimage import morphology, measure
from skimage.color import label2rgb

import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.platform import gfile



red = [255, 0, 0] # dent 
blue = [0, 0, 255] # dent
green = [0, 255, 0] # border
yellow = [255, 255, 0] # edge crack 
cyan = [0, 255, 255] # stain
pink = [255, 200, 200]
violet = [255, 0, 255]

back_ground, border_label, red_dent_label, blue_dent_label, stain_label, edge_crack_label, welding_hole_label, welding_line_label = 0, 1, 2, 3, 4, 5, 6, 7
dcolors = [(0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), [255, 200, 200], [255, 0, 255]]

label2name = {0:'bg', 1:'border', 2:'concave_dent', 3:'convex_dent', 4:'stain', 5:'edge_crack', 6:'welding_line', 7:'welding_hole'}

def print_label_info(labels):
    names = []
    for label in labels:
        if label > 0:
            names.append(label2name[label])
    print(names)


def img_debug(img, cmap=None):
    print(img.shape, img.dtype, img.min(), img.max())
    if cmap==None:
        plt.imshow(img)
    elif cmap=='gray':
        plt.imshow(img, cmap=cmap)
    plt.show()

def practicemodel():
    model_dir = os.path.join('model')
    f = gfile.FastGFile(os.path.join(model_dir, 'graph.pb'), 'rb')

    sess = tf.Session()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def)

    for i in tf.get_default_graph().get_operations():
        #print(i.name)
        pass


    graph = tf.get_default_graph()

    INPUT1 = graph.get_tensor_by_name("import/input_4:0")
    print(INPUT1)

    OUTPUT1 = graph.get_tensor_by_name("import/output_node0:0")
    print(OUTPUT1)

    in_val = np.array([
        [-0.4809832, -0.3770838, 0.1743573, 0.7720509, -0.4064746, 0.0116595, 0.0051413, 0.9135732, 0.7197526, -0.0400658, 0.1180671, -0.6829428],
        [-0.4810135, -0.3772099, 0.1745346, 0.7719303, -0.4066443, 0.0114614, 0.0051195, 0.9135003, 0.7196983, -0.0400035, 0.1178188, -0.6830465],
        [-0.4809143, -0.3773398, 0.1746384, 0.7719052, -0.4067171, 0.0111654, 0.0054433, 0.9134697, 0.7192584, -0.0399981, 0.1177435, -0.6835230],
        [-0.4808300, -0.3774327, 0.1748246, 0.7718700, -0.4070232, 0.0109549, 0.0059128, 0.9133330, 0.7188759, -0.0398740, 0.1181437, -0.6838635],
        [-0.4807833, -0.3775733, 0.1748378, 0.7718275, -0.4073670, 0.0107582, 0.0062978, 0.9131795, 0.7187147, -0.0394935, 0.1184392, -0.6840039],
    ])

    print(in_val.shape)
    in_val = np.expand_dims(in_val, axis=0)
    print(in_val.shape)

    kk = np.zeros((2,5,12))
    kk[0] = in_val
    kk[1] = in_val
    print(kk.shape)

    pred = sess.run( OUTPUT1, feed_dict={INPUT1: kk}) #input : ndarray, output : ndarray
    
    print(pred)
    print(type(pred))
    print('pred = ', pred.shape, pred.dtype)
    # C : -0.409784, -0.302862, 0.0152587, 0.690515
    # Python : -0.40978363, -0.30286163, 0.01525868, 0.6905151 
    # (?, 5, 12) -> (?, 4)
    # if (2,5,12) is input, then (2,4) is ouput
    # if (1,5,12) is input, then (1,4) is ouput
    # only applies in first question mark

def realmodel():
    model_dir = os.path.join('model')
    f = gfile.FastGFile(os.path.join(model_dir, 'mymodel.pb'), 'rb')

    sess = tf.Session()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def)

    for i in tf.get_default_graph().get_operations():
        #print(i.name)
        pass


    graph = tf.get_default_graph()

    INPUT1 = graph.get_tensor_by_name("import/input/Placeholder:0") #image
    print(INPUT1)

    INPUT2 = graph.get_tensor_by_name("import/input/Placeholder_2:0") #is_training
    print(INPUT2)

    OUTPUT1 = sess.graph.get_tensor_by_name('import/output/ArgMax:0') #pred_annotation
    print(OUTPUT1)

    
    
    test_dir = './images'
    test_list = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    print('test_list = ', test_list)

    test_file = test_list[0] #random.choice(test_list)
    test_path = os.path.join(test_dir, test_file)
    print(test_file)

    img = cv2.imread(test_path, cv2.IMREAD_COLOR)
    img = img[...,::-1] # bgr to rgb
    #img_debug(img)

    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.resize(img, None, fx=0.5, fy=0.5).astype('float32')
    img_debug(test_img, 'gray')

    test_img = np.expand_dims(test_img, axis=-1)
    test_img = np.expand_dims(test_img, axis=0)
    print('test = ', test_img.shape, test_img.dtype, test_img.min(), test_img.max())
    
    starttime = time.time()
    for i in range(1):
        pred = sess.run(OUTPUT1, feed_dict={INPUT1: test_img, INPUT2: False}) # https://www.tensorflow.org/api_docs/python/tf/Session#run 
    #The value returned by run() has the same shape as the fetches argument, where the leaves are replaced by the corresponding values returned by TensorFlow.
    endtime = time.time()
    print("DEEP TIME : ",endtime - starttime)


    print('pred = ', pred.shape, pred.dtype)
    pred_img = np.squeeze(pred)
    print('pred_label_type = ', np.unique(pred_img))
    print_label_info(np.unique(pred_img))

    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, figsize=(16,12), sharex=True,sharey=True)
    ax0.imshow(np.squeeze(test_img).astype('uint8'), cmap='gray')
    ax1.imshow(pred_img)
    plt.show()





if __name__=="__main__":
    realmodel()



