# 卒業研究で使用したプログラム
卒業研究で用いた、転移学習と共適応とドロップアウトの３者の関係性を調査するためのプログラムです。
2系のPythonで、機械学習系ライブラリのChainerを使っています。

### コード概要
1. create_mnist_dataフォルダ
 * extract_mnist_pic.py : chainer mnistからmnistの画像をダウンロードし、クラスごとに保存する
 * change_mnist.py : mnist画像にノイズを加える
 
2. mnistTL_Sourceフォルダ
 * my_mnistTL_Source.py : 転移学習のベースとなるネットワークの作成

3. mnistTL_Targetフォルダ
 * my_mnistTL_Target.py : ベースのネットワークの情報を使った転移学習により、ネットワークを作成する
 * mnist_load_initial.py : ベースのネットワークから情報を受け取るために、my_mnistTL_Target.pyで使用するメソッドのコード
 
4. output_corrcoefフォルダ
 * caluc_hidden_cov_corrcoef.py : 学習済みネットワークの中間層の共分散行列と相関行列を可視化する
 * caluc_hidden_cov_eigv.py : 学習済みネットワークの共分散行列の固有値を算出する
