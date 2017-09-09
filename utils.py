import numpy as np
import h5py
import os
import sys
import time
from keras.utils import np_utils
from keras import backend as K


def gen_data(data_path):
    # 读取数据
    whole_5_classes = h5py.File(data_path, 'r')
    print whole_5_classes.keys()
    X = np.array(whole_5_classes['image_array'])
    print type(X)
    print X.shape
    y = np.array(whole_5_classes['label'])
    print type(y)
    print y.shape
    whole_5_classes.close()

    # keras要求格式为binary class matrices，转化一下，直接调用keras提供的这个函数
    # label是训练集的分类标签，一般为一个数组，可能为是否有病疵的两类
    label = np_utils.to_categorical(y, 5)
    # 读取X的三个维度长：(41820, 1050, 1680)
    a, b, c = X.shape
    # 更改形状为(48120, 1, 1050, 1680)，即把(1050, 1680)放在一组
    X = X.reshape(a, 1, b, c)

    print 'X.shape = ' + str(X.shape)
    print 'label.shape = ' + str(label.shape)

    return X, label


def before(output_path):
    print 'Program starting ...\n'

    K.clear_session()
    K.set_image_dim_ordering('th')

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.system('echo $CUDA_VISIBLE_DEVICES')

    original_stdout = sys.stdout
    sys.stdout = open(output_path, 'w')

    print 'Starting to write file at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    return original_stdout


def after(original_stdout):
    print '\nEnding to write file at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    sys.stdout.close()
    sys.stdout = original_stdout

    print '\nProgram ending ...'
