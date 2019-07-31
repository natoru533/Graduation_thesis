# coding:utf-8
#cnn_1616の初期値をロードするプログラム
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import serializers
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt

# Network definition
class CNN(chainer.Chain):
    def __init__(self,data1,data2,train=True):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 20, 5)
            self.conv2 = L.Convolution2D(20, 50, 5)
            self.fc1 = L.Linear(800,500)
            self.fc2 = L.Linear(500,10)

def __call__(self, x):
    h = F.relu(self.conv1(x))
    h = F.max_pooling_2d(h, 2)
    h = F.relu(self.conv2(h))
    h = F.max_pooling_2d(h, 2)
    h = F.relu(self.fc1(h))
    
    return self.fc2(h)

def load_initial(layer_name,dropoutRatio):

    model = L.Classifier(CNN(None, None))
    serializers.load_npz('/Users/nagamuratoru/Desktop/4th4Q/mnistTL_Source/myMnistSourceModel_ver4/myMnistSourceModel_ver4_dropout'+ str(dropoutRatio) +'.npz', model) #v4 Source

    if layer_name=="conv1W":
        return model.predictor.conv1.W.data

    elif layer_name=="conv1B":
        return model.predictor.conv1.B.data

    elif layer_name=="conv2W":
        return model.predictor.conv2.W.data

    elif layer_name=="conv2B":
        return model.predictor.conv2.B.data

    elif layer_name=="fc1W":
        return model.predictor.fc1.W.data

    elif layer_name=="fc1B":
        return model.predictor.fc1.B.data

    elif layer_name=="fc2W":
        return model.predictor.fc2.W.data

    elif layer_name=="fc2B":
        return model.predictor.fc2.B.data
