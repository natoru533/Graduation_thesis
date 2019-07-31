#-*- coding: utf-8 -*-

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
import my_load_initial as myLoad

train, test = datasets.get_mnist(ndim=3)
train = train[0:20000] + train[30000:60000] #訓練データ

# Network definition
class CNN(chainer.Chain):
    dropoutRatio = 0
    count = 0
    print("dropoutRatio",dropoutRatio)

    def __init__(self, data1, data2, conv1_filter_size, conv2_filter_size, train=True):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 20, conv1_filter_size)
            self.conv2 = L.Convolution2D(20, 50, conv2_filter_size)
            self.fc1 = L.Linear(800,500)
            self.fc2 = L.Linear(500,10)

    def __call__(self, x):
        self.count += 1
        if self.count < 2:
            print("call dropoutRatio",self.dropoutRatio)

        h = F.relu(self.conv1(x))
        h = F.dropout(h, ratio = self.dropoutRatio)
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.dropout(h, ratio = self.dropoutRatio)
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


def my_cnn(dataset,
epoch_size,batch_size,instanceDropoutRatio,
conv1_filter_size,conv2_filter_size,
dirName,my_file_name):

    CNN.dropoutRatio = instanceDropoutRatio
    print("instanceDropoutRatio",instanceDropoutRatio)

    # Set up a neural network to train
    model = L.Classifier(CNN(None, None, conv1_filter_size, conv2_filter_size))

    print ("Setup an optimizer")
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    print("train size",len(train))
    train_iter = chainer.iterators.SerialIterator(train, int(batch_size))

    print("Set up a trainer")
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (int(epoch_size), 'epoch'), out='result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    print("Run the training")
    print("epochsize",epoch_size)
    print("batchsize",batch_size)
    trainer.run()

    serializers.save_npz('/Users/nagamuratoru/desktop/SPro_Kadai2/my_result/mymodel_1616_chIn1.npz', model)
    serializers.save_npz('./result/myMnistSourceModel_ver4_dropout'+ str(instanceDropoutRatio) +'.npz', model)

    #test part
    ok = 0
    acc = np.zeros((10,10))
    print("test start")
    for i in range(len(test)):
        x = test[i][0]
        t = test[i][1]
        y = model.predictor(x[None, ...]).data.argmax(axis=1)[0]

        if t == y:
            ok += 1
            acc[t][t] += 1

        else:
            acc[t][y] += 1 #配列accは例えば正しいラベルが2のものを3だと分類した場合はacc[2][3]+=1される。つまり行の合計が各クラスのテストデータの合計となっているはず

        if i % 1000 == 0:
            print("current" , i)

    correctRate = (ok * 1.0) / len(test) * 100
    print("correctRate",correctRate)

    # ファイルオープン
    f = open('correctRate_v4.csv', 'ab')
    writer = csv.writer(f, lineterminator='\n')
    correctRateList = []
    correctRateList.append(correctRate)
    writer.writerow(correctRateList)
    f.close()

    np.savetxt("./mnist_seatResult_v4/acc_v4_train50000_dropRate"+ str(instanceDropoutRatio) +".csv", acc, delimiter=",")


def main():

    dropoutRatio_List = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for sendDropoutRatio in dropoutRatio_List:
        my_cnn(1,
        '10','1000',sendDropoutRatio,
        5,5,
        'dirname','filename')

        # my_cnn(dataset,
        # layer1_copy,layer2_copy,layer3_copy,
        # layer1_FT,layer2_FT,layer3_FT,
        # epoch_size,batch_size,instanceDropoutRatio,
        # conv1_filter_size,conv2_filter_size,
        # dirName,my_file_name):
        # trueは１ falseは０ 


if __name__ == '__main__':
    main()
