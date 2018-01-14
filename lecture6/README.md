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

##### 基礎演習 (40分)
* 計算グラフの記述
* Link
* Chain
* L.Linear, F.relu, F.softmax\_cross\_entropy
* 多層パーセプトロンのChainerによる記述
* Optimizer
* Variable
* Trainerを用いないニューラルネットの学習
* Trainer/Updater
* datasets/iterators
* Extension(Evaluator, LogReport, PrintReport, ProgressBar, snapshot)
* Trainerを用いたニューラルネットの学習
* GPU対応コードの実装
* モデルの保存と読み込み (Serializer)
* 課題:Chainerを用いたロジスティック回帰の実装

#### Part 2. ニューラルネットの学習テクニック (15分)
* NNの最適化手法(Momentum法(MomentumSGD)/AdaGrad/Adam)
* NNの正則化手法(Dropout)
* 勾配消失問題とHeの初期化
* Batch Normalization

#### Part 3. 他手法との比較 (20分)
* 汎化誤差・交差検証 (復習)
* 機械学習モデルの性能を決める要素 (特徴選択・前処理・線形性)
* SVMとNNの比較
* モデル選択

### 実践演習 (85分)
#### 課題1: Chainerを用いた多層パーセプトロンの実装 (30分)
* 第5回と同様、MNISTデータセットに対して今度はChainerで多層パーセプトロンを実装

#### 課題2: 線形・ガウシアンSVMを用いたMNIST分類 (20分)
* Scikit-learnを使いつつ、線形・ガウシアンSVMを用いた分類を行い、NNと比較
(適切なデータセットがあればMNIST以外のデータセットでの比較も考えられる)

#### 課題3: DropoutとBN (20分)
* Dropout及びBNを課題1のコードに追加し、バリデーション誤差や正解率に対する影響を見る
* 各層の中間層の値を取り出して、その傾向を見る

#### 解説・コードレビュー (15分)

## 参考文献 (講師の方も随時追加お願いします)
* 岡谷貴之『機械学習プロフェッショナルシリーズ 深層学習』(講談社、2015)
