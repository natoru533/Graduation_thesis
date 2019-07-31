# -*- coding: utf-8 -*-
#保存済みのmnist画像にノイズを加えるプログラム
import os
import numpy as np
from PIL import Image, ImageFilter

def main():
    data_dir_path_in = "/Users/nagamuratoru/Desktop/4th4Q/mnist_dataSet/mnist_otameshi/"
    missing_data_dir_path_out = "/Users/nagamuratoru/Desktop/4th4Q/mnist_dataSet/mnist_changed_target_train_data_50/missing_data/"
    impulse_data_dir_path_out = "/Users/nagamuratoru/Desktop/4th4Q/mnist_dataSet/mnist_otameshi/"
    gauss_data_dir_path_out = "/Users/nagamuratoru/Desktop/4th4Q/mnist_dataSet/mnist_changed_target_train_data_50/gauss_data/"
    # 各クラスの画像を保存しているディレクトリの名前が０〜９でない場合は以下のようにディレクトリの名前を取得する
    # inputDirlist = []
    # for f in os.listdir(data_dir_path_in):
    #     if os.path.isdir(os.path.join(data_dir_path_in, f)):
    #         inputDirlist.append(f)
    # print(inputDirlist)

    for i in range(0,10): #data_dir_path_inが指定するフォルダの中に０〜９という名のサブフォルダが入っている場合
        subDirName = str(i)
        #missing(data_dir_path_in, missing_data_dir_path_out, subDirName)
        impulseNoise(data_dir_path_in, impulse_data_dir_path_out, subDirName)
        #gaussNoise(data_dir_path_in, gauss_data_dir_path_out, subDirName)

def missing(tmp_data_dir_path_in, tmp_data_dir_path_out, tmpSubDirName): #真ん中を白抜き（欠損）

    file_list = os.listdir(tmp_data_dir_path_in + tmpSubDirName)

    for file_name in file_list: #file_listにはサブフォルダ０〜９（各クラス画像が入っている）の中身のファイルの名前が入ってる。
        root, ext = os.path.splitext(file_name)
        if ext == u'.png':
            img = Image.open(tmp_data_dir_path_in + '/' + tmpSubDirName + '/' + file_name).convert('L')  #Pillowで読み込み。'L'はグレースケールを意味する
            img = img.resize((28, 28)) # 28x28にリサイズ
            img = np.array(img) #numpyに変換

            img[11:17, 11:17] = 255 #真ん中の6x6部分を白にする

            saveImg = Image.fromarray(np.uint8(img))
            saveImg.save(tmp_data_dir_path_out + '/' + tmpSubDirName + '/' + root +'_missing.png')

def impulseNoise(tmp_data_dir_path_in, tmp_data_dir_path_out, tmpSubDirName): #真ん中を白抜き（欠損）

    file_list = os.listdir(tmp_data_dir_path_in + tmpSubDirName)

    for file_name in file_list: #file_listにはサブフォルダ０〜９（各クラス画像が入っている）の中身のファイルの名前が入ってる。
        root, ext = os.path.splitext(file_name)
        if ext == u'.png':
            img = Image.open(tmp_data_dir_path_in + '/' + tmpSubDirName + '/' + file_name).convert('L')  #Pillowで読み込み。'L'はグレースケールを意味する
            img = img.resize((28, 28)) # 28x28にリサイズ
            img = np.array(img) #numpyに変換

            # 白
            pts_x = np.random.randint(0, 28-1 , 100) #0から(col-1)までの乱数を千個作る
            pts_y = np.random.randint(0, 28-1 , 100)
            img[pts_x, pts_y] = 255 #指定された要素を白にする

            # 黒
            pts_x = np.random.randint(0, 28-1 , 100) #0から(col-1)までの乱数を千個作る
            pts_y = np.random.randint(0, 28-1 , 100)
            img[pts_x, pts_y] = 0 #指定された要素を黒にする

            saveImg = Image.fromarray(np.uint8(img))
            saveImg.save(tmp_data_dir_path_out + '/' + tmpSubDirName + '/' + root +'_impulse.png')

def gaussNoise(tmp_data_dir_path_in, tmp_data_dir_path_out, tmpSubDirName): #真ん中を白抜き（欠損）

    file_list = os.listdir(tmp_data_dir_path_in + tmpSubDirName) #file_listにはサブフォルダ０〜９（各クラス画像が入っている）の中身のファイルの名前が入ってる。

    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        mean = 0
        sigma = 15.0
        if ext == u'.png':
            img = Image.open(tmp_data_dir_path_in + '/' + tmpSubDirName + '/' + file_name).convert('L')  #Pillowで読み込み。'L'はグレースケールを意味する
            img = img.resize((28, 28)) # 28x28にリサイズ
            img = np.array(img) #numpyに変換

            gauss = np.random.normal(mean,sigma,(28,28)) #正規分布に従う乱数を生成し画像に足し合わせることで輝度値にノイズを加える。
            gauss = gauss.reshape(28,28)
            img = img + gauss

            saveImg = Image.fromarray(np.uint8(img))
            saveImg.save(tmp_data_dir_path_out + '/' + tmpSubDirName + '/' + root +'_gauss.png')

if __name__ == '__main__':
    main()
