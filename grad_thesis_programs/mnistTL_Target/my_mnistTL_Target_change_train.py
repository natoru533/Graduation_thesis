#-*- coding: utf-8 -*-
#5000枚作成したノイズ入りtrain画像のうち、毎回の訓練ごとに違った500枚(各クラス50枚)を取ってくる
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
import random

# Network definition
class CNN(chainer.Chain):
    dropoutRatio = 0
    count = 0
    print("dropoutRatio",dropoutRatio)

    def __init__(self, data1, data2,layer1_copy,layer2_copy,layer3_copy, layer4_copy, initialDropoutRatio,train=True):
        super(CNN, self).__init__()
        with self.init_scope():
            if layer1_copy == 1:
                self.conv1 = L.Convolution2D(1, 20, 5, initialW=myLoad.load_initial("conv1W",initialDropoutRatio))
                print("layer1 copy")
            else:
                self.conv1 = L.Convolution2D(1, 20, 5)

            if layer2_copy == 1:
                self.conv2 = L.Convolution2D(20, 50, 5, initialW=myLoad.load_initial("conv2W",initialDropoutRatio))
                print("layer2 copy")
            else:
                self.conv2 = L.Convolution2D(20, 50, 5)

            if layer3_copy == 1:
                self.fc1 = L.Linear(800,500, initialW=myLoad.load_initial("fc1W",initialDropoutRatio))
                print("layer3 copy")
            else:
                self.fc1 = L.Linear(800,500)

            if layer4_copy == 1:
                self.fc2 = L.Linear(500,10, initialW=myLoad.load_initial("fc2W",initialDropoutRatio))
                print("layer4 copy")
            else:
                self.fc2 = L.Linear(500,10)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.fc1(h))

        return self.fc2(h)

def my_cnn(dataset,
layer1_copy,layer2_copy,layer3_copy,layer4_copy,
layer1_Frozen,layer2_Frozen,layer3_Frozen,layer4_Frozen,
epoch_size,batch_size,instanceDropoutRatio,train,file_name):

    CNN.dropoutRatio = instanceDropoutRatio
    print("instanceDropoutRatio",instanceDropoutRatio)

    # Set up a neural network to train
    model = L.Classifier(CNN(None, None, layer1_copy, layer2_copy, layer3_copy,
    layer4_copy, instanceDropoutRatio))

    print ("Setup an optimizer")

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    if layer1_Frozen == 1: #Frozen or FTを決定する
        model.predictor.conv1.disable_update()
        print("layer1 is Frozen")
    else:
        print("layer1 is FT")

    if layer2_Frozen == 1:
        model.predictor.conv2.disable_update()
        print("layer2 is Frozen")
    else:
        print("layer2 is FT")

    if layer3_Frozen == 1:
        model.predictor.fc1.disable_update()
        print("layer3 is Frozen")
    else:
        print("layer3 is FT")

    if layer4_Frozen == 1:
        model.predictor.fc2.disable_update()
        print("layer4 is Frozen")
    else:
        print("layer4 is FT")

    ################################## train part###################################
    print("train length", len(train))
    print("test length", len(test))
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
    serializers.save_npz('./result_npzFile/'+ file_name +'target_model_dropout'+ str(instanceDropoutRatio)
    + "_e"+ str(epoch_size) + "b" + str(batch_size)
    + "trainSize" + str(len(train)) +'.npz', model)

    ################################## test part ###################################
    ok = 0
    acc = np.zeros((10,10))
    print("test start")
    for i in range(len(test)):
        x = test[i][0]
        t = test[i][1]
        y = model.predictor(x[None, ...]).data.argmax(axis=1)[0]

        y = int(y)
        t = int(t)

        if t == y:
            ok += 1
            acc[t][t] += 1

        else:
            acc[t][y] += 1 #配列accは例えば正しいラベルが2のものを3だと分類した場合はacc[2][3]+=1される。つまり行の合計が各クラスのテストデータの合計となっているはず

        if i % 1000 == 0:
            print("current" , i)

    #結果をファイルへ出力 correctRateというファイルが直下にできる
    correctRate = (ok * 1.0) / len(test) * 100
    print("correctRate",correctRate)

    # ファイルオープン
    f = open('./result_correctRate/' + file_name + '_correctRate'+ '_e'+ str(epoch_size)
                + 'b' + str(batch_size)+ "trainsize" + str(len(train)) +'.csv', 'ab')
    writer = csv.writer(f, lineterminator='\n')
    correctRateList = []
    correctRateList.append('dropoutRatio:')
    correctRateList.append(instanceDropoutRatio)
    correctRateList.append(correctRate)
    writer.writerow(correctRateList)
    f.close()

    np.savetxt("./result_table/" + file_name + "_contingency_table_dropRate"
    + str(instanceDropoutRatio) + "_e"+ str(epoch_size) + "b" + str(batch_size)
    + "trainSize" +str(len(train)) + ".csv", acc, delimiter=",")


def load_train_image(load_file_path, data_size):
    pathsAndLabels = [] #リストを生成した？行列？

    for i in range(0,10):
        pathsAndLabels.append(np.asarray([load_file_path + str(i) + '/', i]))

    # データを混ぜて、まばらになるように。
    allData = []
    for pathAndLabel in pathsAndLabels:
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + '*')#画像の名前が入ってる

        random.shuffle(imagelist) #クラスつける前にパスをシャッフル
        imagelist = imagelist[0:data_size / 10] #data_size / 10までのパスしか使わない(１クラス５０枚にするため)
        
        for imgName in imagelist:
            allData.append([imgName, label])
    allData = np.random.permutation(allData)

    # allData = allData[0:data_size]
    print("allData size",len(allData))
    datasets = []
    cou = 0
    for pathAndLabel in allData:

        if cou % 2000==0:
            print("current loading train data is :",cou)

        img = Image.open(pathAndLabel[0]).convert('L')  #Pillowで読み込み。'L'はグレースケールを意味する
        img = img.resize((28, 28)) # 28x28にリサイズ

        x = np.array(img, dtype=np.float32)
        x = x.reshape(1,28,28) # (チャネル、高さ、横幅)
        t = np.array(pathAndLabel[1], dtype=np.int32)

        datasets.append((x,t)) # xとtをタプルでリストに入れる
        cou+=1

    return datasets

def load_test_image(load_file_path, data_size):
    pathsAndLabels = [] #リストを生成した？行列？

    for i in range(0,10):
        pathsAndLabels.append(np.asarray([load_file_path + str(i) + '/', i]))

    # データを混ぜて、まばらになるように。
    allData = []
    for pathAndLabel in pathsAndLabels:
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + '*')#画像の名前が入ってる
        for imgName in imagelist:
            allData.append([imgName, label])
    allData = np.random.permutation(allData)

    allData = allData[0:data_size]
    print("allData size",len(allData))
    datasets = []
    cou = 0
    for pathAndLabel in allData:

        if cou % 2000==0:
            print("current loading train data is :",cou)

        img = Image.open(pathAndLabel[0]).convert('L')  #Pillowで読み込み。'L'はグレースケールを意味する
        img = img.resize((28, 28)) # 28x28にリサイズ

        x = np.array(img, dtype=np.float32)
        x = x.reshape(1,28,28) # (チャネル、高さ、横幅)
        t = np.array(pathAndLabel[1], dtype=np.int32)

        datasets.append((x,t)) # xとtをタプルでリストに入れる
        cou+=1

    return datasets


test = load_test_image('/Users/nagamuratoru/Desktop/4th4Q/mnist_dataSet/mnist_changed_target_test_data/impulse_data/', 10000)


def main():
    dropoutRatio_List = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for i in range(1,6): #何回評価を行って平均を取るかはここで決まる
        print("現在",i,"ループ目")
        train = load_train_image('/Users/nagamuratoru/Desktop/4th4Q/mnist_dataSet/mnist_changed_target_train_data_5000/impulse_data/',500)

        for sendDropoutRatio in dropoutRatio_List:
            my_cnn(1,
            1,1,1,0,
            1,1,1,0,
            '7','20',sendDropoutRatio,train,
            str(i) + '_impulse_frozen_l123TL_source_v4')
            #sendDropoutRatioでどのdropRatioのソースモデルを転移するか決める
            # my_cnn(dataset,
            # layer1_copy,layer2_copy,layer3_copy,layer4_copy,
            # layer1_Frozen,layer2_Frozen,layer3_Frozen,layer4_Frozen
            # epoch_size,batch_size,instanceDropoutRatio,train,
            # file_name):
            # 1:true, 0:false 

if __name__ == '__main__':
    main()
