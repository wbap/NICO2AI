# 第10回：リカレントニューラルネットワーク (2)

## 到達目標
* 勾配爆発・消失問題を理解する
* LSTMの目的と働きを数式を追いながら理解する
* Chainerを用いてLSTMを用いたモデルを記述できる
* RNN特有の学習テクニックを理解し、正しく使用できる
* RNNの用途とより高度の使い方を知る

## 講師へのお願い
* RNNの工学的問題を知ってもらうために、勾配爆発・消失問題に時間を割いてください
* LSTMのアイデアを直感的かつ数式的に理解できるようにしてください
* 講義後半は、RNNの現在の応用先と、単なる時系列モデルを超えた拡張も含めて、具体例を多数紹介してください

## キーワード
* Long Short-Term Memory (LSTM)
* 長期依存/短期依存
* 勾配爆発 (消失) 問題
* Gradient Clipping
* Neural Turing Machine (NTM)
* Differential Neural Computers (DNC)

## タイムスケジュール
### 前回の復習 (5分)
### 講義・基礎演習 (85分)
#### Part 1. LSTMの基礎と学習 (50分)
* 系列データ予測の問題設定 (復習)
* RNNの順伝播・逆伝播 (復習)
* 勾配爆発 (消失) 問題
* Constant Error Carousel (CEC)
* LSTMの順伝播
* LSTMの逆伝播
* Truncated Backprop (unchain\_backward)
* GRU vs LSTM (Pascanu, 2012)
* ChainerにおけるLSTM (F.lstm, F.n\_step\_lstm/gru)
* Gradient Clipping (chainer.optimizer.GradientClipping)
* RNNの学習のコツ (Gradient clipping, ミニバッチ学習の時は長さを揃える (F.pad\_sequence), Truncated backprop, RMSPropまたはAdamを使用、学習率の減衰)
* LSTMの学習の可視化 (隠れ層の発火の可視化)

#### 小課題1: 人工データを用いたLSTMの学習 (15分)
* 第9回と同じデータのLSTM版を実行し、結果を比較
* 簡単に比較できるコードがあると良い

#### Part 2. RNNの活用と発展 (20分、詳細には立ち入らない)
* RNNの工学的応用 (音声認識、文章生成、形態素解析、動作認識)
* Encoder-Decoder (seq2seq) model (機械翻訳、画像キャプショニングへの応用)
* Neural Turing Machine (NTM)
* Differential Neural Computers (DNC)
* Sequential Prediction (画像生成)

### 実践演習 (85分)
#### 課題1: 獲得した特徴表現の可視化 (15分)
* LSTMVisの体験
* Live Demoを通じて、生徒に特徴的なセルを探してもらう
* LSTMの獲得特徴の体験的理解を意図

#### 課題2: 手書き文字のストローク予測 (55分)
* (Graves, 2013) Section 4の実装をIAM On-Line Handwriting Databaseを用いて行う
* Mixture Density Networks (MDN) (Bishop, 1994) の簡単な解説
* 学習済みモデルを用意 (セルとしてRNN, LSTM, GRUを用いた場合)
* 学習コード、可視化コードを用意、モデル定義のMDNを除いた部分を書いてもらう
* 予測結果の確率分布の可視化 (予め用意するか、もし可能なら一部を課題にしても良い) を通じてMDNの役割と学習結果を理解

#### 解説・コードレビュー (15分)
### フィードバック・次回予告 (5分)

## 参考文献
* わかるLSTM ～ 最近の動向と共に
http://qiita.com/t\_Signull/items/21b82be280b46f467d1b

* Deep Learning Lecture 12: Recurrent Neural Nets and LSTMs
https://www.youtube.com/watch?v=56TYLaQN4N8

* Deep Learning Lecture 13: Alex Graves on Hallucination with RNNs
https://www.youtube.com/watch?v=-yX1SYeDHbg

* (Karpathy, 2016) Visualizing and Understanding Recurrent Networks
https://pdfs.semanticscholar.org/8390/c96f0b2ff3b36b232f7f9918401e51632f4e.pdf

* (Bengio, 1994) Learning long-term dependencies with gradient descent is difficult
http://www.dsi.unifi.it/~paolo/ps/tnn-94-gradient.pdf

* (Hochreiter, 1997) Long Short-Term Memory
http://www.bioinf.jku.at/publications/older/2604.pdf

* (Pascanu, 2012) On the difficulty of training recurrent neural networks
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.421.8930&rep=rep1&type=pdf

* (Bishop, 1994) Mixture Density Networks
https://publications.aston.ac.uk/373/1/NCRG\_94\_004.pdf

* (Graves, 2013) Generating Sequences With Recurrent Neural Networks
https://arxiv.org/abs/1308.0850

* IAM On-Line Handwriting Database (IAM-OnDB)
http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database

* Handwriting Generation Demo in TensorFlow (Blog)
http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/

* Generative Handwriting Demo using TensorFlow (GitHub)
https://github.com/hardmaru/write-rnn-tensorflow

* LSTMVis: Visual Analysis for Recurrent Neural Networks
http://lstm.seas.harvard.edu/
