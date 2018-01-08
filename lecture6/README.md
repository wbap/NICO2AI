# 第６回：Chainer入門

## 到達目標
* Chainerを使って全結合のニューラルネットを記述し、実行することができる
* ニューラルネットとその他の手法との関係性とその特性の差を理解する
* ニューラルネットの基本的な学習テクニックを学ぶ

## キーワード
* Chainer
* Define-by-Run
* Link, Chain, Optimizer

## タイムスケジュール
### 前回の復習 (5分)
### 講義・基礎演習 (85分)
#### Part 1. Chainer入門 (50分)
##### 講義 (10分)
* Chainerとは
* "Define-and-Run"と"Define-by-Run"
* Chainerの特長と他フレームワークとの比較
* 計算グラフの記述
* (GPUへの対応)

##### 基礎演習 (40分)
* Variable
* 自動微分
* Link
* Chain
* L.Linear, F.relu, F.softmax\_cross\_entropy
* 多層パーセプトロンのChainerによる記述
* Optimizer
* (Trainerを用いない)ニューラルネットの学習
* モデルの保存と読み込み (Serializer)
* (Trainer/Updater)
* (datasets/iterators)
* (Extension(Evaluator, LogReport, PrintReport, ProgressBar, snapshot))
* (Trainerを用いた)ニューラルネットの学習
* (GPU対応コードの実装)
* 課題:Chainerを用いたロジスティック回帰の実装

#### Part 2. ニューラルネットの学習テクニック (15分)
* NNの最適化手法(Momentum法(MomentumSGD)/AdaGrad/Adam)
* NNの正則化手法(Dropout)
* 勾配消失問題
* Heの初期化
* Batch Normalization

#### Part 3. 他手法との比較 (20分)
* 汎化誤差・交差検証 (復習)
* 機械学習モデルの性能を決める要素 (特徴選択・前処理・線形性)
* SVMとNNの比較
* モデル選択

## 参考文献 (講師の方も随時追加お願いします)
