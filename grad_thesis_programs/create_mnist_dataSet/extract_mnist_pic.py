# coding:utf-8
# chainer mnistからダウンロードしたデータを.png画像として保存する
# 0~9の数字画像が、各クラスのフォルダにextract_num枚だけ入る
import os
from PIL import Image
import chainer

save_dir_name = '/Users/nagamuratoru/Desktop/4th4Q/mnist_dataSet/mnist_changing_target_train_data_5000/'

def save(data, index, num):
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(data[i+j*28]*256)
    filename = str(num) + "/target" + "{0:04d}".format(index) + ".png" #ラベルと同名のフォルダに対応する画像を保存
    img.save(save_dir_name + filename)
    print filename

def main():
    test, _ = chainer.datasets.get_mnist()
    test = test[20001:40000]
    extract_num = 500
    cou0=cou1=cou2=cou3=cou4=cou5=cou6=cou7=cou8=cou9=1

    for i in range(10): #名前が0~9のディレクトリを作成
        dirname = str(i)
        if os.path.isdir(save_dir_name + dirname) is False:
            os.mkdir(save_dir_name + dirname)
    for i in range(len(test)): #mnistのtest dataに対して、ラベルを参照し対応するフォルダへpng形式で保存

        if cou0==cou1==cou2==cou3==cou4==cou5==cou6==cou7==cou8==cou9==extract_num:
            break

        if test[i][1]==0 and cou0<=extract_num:
            save(test[i][0], i, test[i][1])
            cou0+=1
        elif test[i][1]==1 and cou1<=extract_num:
            save(test[i][0], i, test[i][1])
            cou1+=1
        elif test[i][1]==2 and cou2<=extract_num:
            save(test[i][0], i, test[i][1])
            cou2+=1
        elif test[i][1]==3 and cou3<=extract_num:
            save(test[i][0], i, test[i][1])
            cou3+=1
        elif test[i][1]==4 and cou4<=extract_num:
            save(test[i][0], i, test[i][1])
            cou4+=1
        elif test[i][1]==5 and cou5<=extract_num:
            save(test[i][0], i, test[i][1])
            cou5+=1
        elif test[i][1]==6 and cou6<=extract_num:
            save(test[i][0], i, test[i][1])
            cou6+=1
        elif test[i][1]==7 and cou7<=extract_num:
            save(test[i][0], i, test[i][1])
            cou7+=1
        elif test[i][1]==8 and cou8<=extract_num:
            save(test[i][0], i, test[i][1])
            cou8+=1
        elif test[i][1]==9 and cou9<=extract_num:
            save(test[i][0], i, test[i][1])
            cou9+=1

if __name__ == '__main__':
    main()
