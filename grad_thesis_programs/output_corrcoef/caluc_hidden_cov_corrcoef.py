#-*- coding: utf-8 -*-
#中間層の共分散行列と相関行列を計算するプログラム
#データによって変更する行 75 80 251 mnist_load_initial.pyの 41
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from PIL import Image
import glob
import numpy as np
import csv
import matplotlib.pyplot as plt
import mnist_load_initial as myLoad
import math
import os

train, test = datasets.get_mnist(ndim=3)
corrcoef_matrix_list=[]

# Network definition
class CNN(chainer.Chain):

    def __init__(self, data1, data2, initialDropoutRatio,train=True):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 20, 5, initialW=myLoad.load_initial("conv1W",initialDropoutRatio))
            self.conv2 = L.Convolution2D(20, 50, 5, initialW=myLoad.load_initial("conv2W",initialDropoutRatio))
            self.fc1 = L.Linear(800,500, initialW=myLoad.load_initial("fc1W",initialDropoutRatio))
            self.fc2 = L.Linear(500,10, initialW=myLoad.load_initial("fc2W",initialDropoutRatio))

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.fc1(h))
        return self.fc2(h)

    def get_hidden_layer(self, x, out_layer):
        _conv1 = F.relu(self.conv1(x))
        _pool1 = F.max_pooling_2d(_conv1, 2)
        _conv2 = F.relu(self.conv2(_pool1))
        _pool2 = F.max_pooling_2d(_conv2, 2)
        _fc1 = F.relu(self.fc1(_pool2))
        _fc2 = F.relu(self.fc2(_fc1))

        if out_layer == 'conv1':
            return _conv1
        elif out_layer == 'conv2':
            return _conv2
        elif out_layer == 'fc1':
            return _fc1
        elif out_layer == 'fc2':
            return _fc2
        else:
            print("get_hidden_layer missed")

def main():

    datasets, img_list, label_list = load_image()

    send_out_layer = 'conv1'
    dropoutRatio_List = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    new_dir_path = '/Users/nagamuratoru/Desktop/4th4Q/mnistTL_Target/output_new_corrcoef_Source_v4/Source_v4_new_corrcoef_' + str(send_out_layer)
    if not os.path.isdir(new_dir_path):
        os.makedirs(new_dir_path)

    for dropoutRatio in dropoutRatio_List: #ノルムを計算する
        print("current dropoutRatio is ", dropoutRatio)
        out_cov_norm(img_list, dropoutRatio, send_out_layer, new_dir_path)


def out_cov_norm(img_list, dropoutRatio, send_out_layer, new_dir_path):
    out_vector_list = []
    for x in img_list:
        x = x[None, ...]
        ave_pic = get_ave_pic(x, dropoutRatio, send_out_layer)

        reshape_ave_pic=np.reshape(ave_pic, np.shape(ave_pic)[0] * np.shape(ave_pic)[0])

        out_vector_list.append(reshape_ave_pic)

    out_vector_list = np.array(out_vector_list)
    print("before transpose", np.shape(out_vector_list))
    out_vector_list = out_vector_list.transpose()
    print("after transpose", np.shape(out_vector_list))

    #相関の計算を開始
    list_size = np.shape(out_vector_list)[0]
    sqrt_list_size = int(np.sqrt(np.shape(out_vector_list)[0]))
    out_img_list=[]

    for i in range(list_size):
        new_corrcoef_matrix_list=[]
        print("cou:",i)
        for j in range(list_size):
            corrcoef_matrix_elem = np.corrcoef(out_vector_list[i,:], out_vector_list[j,:])
            new_corrcoef_matrix_list.append(corrcoef_matrix_elem[0,1])

        new_corrcoef_matrix_array = np.reshape(np.array(new_corrcoef_matrix_list), (sqrt_list_size, sqrt_list_size))
        print("new_corrcoef_matrix_list", np.shape(new_corrcoef_matrix_array))
        out_img_list.append(new_corrcoef_matrix_array)

    print("out_img_list", len(out_img_list))

    out_corrcoef_list(out_img_list, sqrt_list_size, send_out_layer, new_dir_path, dropoutRatio)


def get_ave_pic(x, tmp_dropoutRatio, out_layer):

    model = CNN(None,None,tmp_dropoutRatio)
    layer = model.get_hidden_layer(x, out_layer)
    minibatch, filter_num, h, w = layer.shape
    line_num = math.ceil(math.sqrt(filter_num))

    filter_shape = np.shape(layer.data[0,0])
    im_ave = np.zeros((filter_shape[0], filter_shape[0]))
    for i in range(0, filter_num):
        im_ave += layer.data[0, i] / filter_num

    return im_ave


def out_corrcoef_list(corrcoef_matrix_list, list_size, out_layer, save_dir_path, dropoutRatio):
    cou = 0
    fig, axes = plt.subplots(nrows=list_size, ncols=list_size, figsize=(20,20))
    for ax in axes.flat:
        print("cou",cou)
        print("in def corrcoef_matrix_list shape", corrcoef_matrix_list[cou].shape)
        im = ax.imshow(corrcoef_matrix_list[cou], cmap='bwr', vmin=-1.0, vmax=1.0)
        cou+=1

    plt.savefig(save_dir_path + "/all_corrcoef_pic_" + str(out_layer) + "drop" + str(dropoutRatio) + "_.png")


def load_image():
    pathsAndLabels = [] #リストを生成した？行列？

    for i in range(0,10):
        pathsAndLabels.append(np.asarray(['/Users/nagamuratoru/Desktop/4th4Q/mnist_dataSet/mnist_out_cov_source10/' +str(i)+ '/', i]))

    allData = []
    for pathAndLabel in pathsAndLabels:
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + '*')#画像の名前が入ってる
        for imgName in imagelist:
            allData.append([imgName, label])
    allData = np.random.permutation(allData)
    allData100 = allData[0:100]

    print("allData100:",np.shape(allData100))

    image_list = []
    label_list = []
    datasets = []
    for pathAndLabel in allData100:
        img = Image.open(pathAndLabel[0]).convert('L')  #Pillowで読み込み。'L'はグレースケールを意味する
        img = img.resize((28, 28)) # 28x28にリサイズ

        x = np.array(img, dtype=np.float32)
        x = x.reshape(1,28,28) # (チャネル、高さ、横幅)
        t = np.array(pathAndLabel[1], dtype=np.int32)

        image_list.append(x)
        label_list.append(t)
        datasets.append((x,t))

    return datasets, image_list, label_list


if __name__ == '__main__':
    main()
