# 第9回：リカレントニューラルネットワーク (1)

## 到達目標
* 系列データ予測の問題設定と応用先を理解する
* シンプルなRNNの順伝播・逆伝播アルゴリズムの原理を理解する
* Chainerを用いてRNNを実装できる
* 誤差逆伝播法よりBiologically-Plausibleなアルゴリズムとして、ESN及びLSMの原理を学ぶ
* Numpyを用いてESNのモデルを記述できる

## 講師へのお願い
* RNNはニューロンとの関係が深いため、神経科学との接続をとった解説をしてください
* BPTTのアイデアは数式レベルで理解してもらってください
* 課題分量が大きすぎると感じた場合、実践演習の課題を1つけずって、その分人工データの演習を拡張するなどの工夫はして構いません

## キーワード
* Recurrent Neural Network (RNN)
* Elman Network
* Back Propagation Throught Time (BPTT)
* Echo State Network (ESN)

## タイムスケジュール
### 前回の復習 (5分)
### 講義・基礎演習 (85分)
#### Part 1. RNNの基礎と学習 (40分)
* 動機:ニューロンの相互結合のモデル化
* Hopfield Network
* Elman Network (Elman, 1990)
* 系列データ予測の問題設定
* Recurrent Neural Network (RNN)
* シンプルなRNNの順伝播
* シンプルなRNNの逆伝播
* ChainerにおけるLSTM (F.lstm, F.n\_step\_lstm/gru)
* Back Propagation Thorough Time (BPTT)
* 言語予測への適用 (RNNLM (Mikolov, 2010), Character-Level RNN)

#### 小課題1：人工データを用いたRNNの学習 (20分)
* 適当な決定性オートマトンに従うシンボル列の学習
* 0,1,...1,0 (1が指定回続く) 系列を用いた記憶容量の検証
* 生徒は、Chainerを用いて1からモデルを記述する (関数名、引数などの外枠は与えられる)

#### Part 2. 誤差逆伝播法を使わないRNN: Echo State Network (25分)
* Echo State Network (ESN)
* Resoervoir Computing
* RNNと記憶機能 (前の状態を明示的に持たなくとも、ネットワークがそのダイナミクスを保持していることの解説)
* Liquid State Machine (LSM) (IAF, HHなどのスパイク発火モデルで利用されていることに言及)
* ESNの学習 (最小二乗法の復習)
* (Predictive Codingとの関係性)

### 実践演習 (85分)
#### 課題1: (仮) 神経活動データを用いたESNの学習 (30分)
* 適当な神経活動データ (平均発火率の時系列で表されたもの) を用意し、その入出力関係を学習
* データは例えば、トリガーが引かれるとその後の入力にかかわらず周期的に発火するようなデータが望ましい (水谷さんに相談)
* 簡単なので、必要な変数とその初期化のコードだけ用意して、学習部分は全て書いてもらう
* 必要な解説は数式レベルでしっかり追うこと
* 課題1は適当な信号同士の積などが計算できることを確認できれば良く、原理を重視

#### 課題2: シェイクスピアをCharacter-Level RNNで学習する (40分)
* Karpathyのページ (http://karpathy.github.io/2015/05/21/rnn-effectiveness/) のコンテンツの紹介
* Shakesphereデータセット (http://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt) の解説
* One-hot vectorの解説
* F.EmbedID
* 2~3層のRNNを記述してもらう (実行重視)
* 事前検証で学習が上手くいかない場合は、LSTM及びGradient Clippingを先行導入することを検討
* 学習ずみモデルを用意し、生徒がすぐに生成を試せるようにする (料理番組的な) 結果重視

#### 解説・コードレビュー (15分)
### フィードバック・次回予告 (5分)

## 参考文献
* (Elman, 1990) Finding Structure in Time
http://machine-learning.martinsewell.com/ann/Elma90.pdf

* (Mikolov, 2010) Recurrent neural network based language model
http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov\_interspeech2010\_IS100722.pdf

* The Unreasonable Effectiveness of Recurrent Neural Networks
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

* Sample Echo State Network source codes
http://minds.jacobs-university.de/mantas/code
