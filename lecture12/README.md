# 第12回：強化学習 (2)・脳型アーキテクチャから作るANN
(脳型アーキテクチャから作るANNはAIラボ担当のため、未記載)

## 到達目標
* 方策勾配法の問題設定と用途を知る
* 深層強化学習の仕組みとその意義を知る
* DQNの学習アルゴリズムを理解し、実行できる
* ChainerRLを用いてNNベースのエージェントを記述できる

## 講師へのお願い
* 方策勾配は基礎として必須ですが、問題意識を理解してもらうだけで十分です
* Deep Q-Networkの学習は困難なため、学習済みモデルの実行及びトイタスクによる学習で、短い時間で体験できる工夫をしてください
* 本回後半の脳型アーキテクチャに接続するトピックを研究動向では扱ってください

## キーワード
* 方策勾配法
* 方策勾配定理
* Actor-Critic
* 深層強化学習
* Deep Q-Network (DQN)
* Experience Replay
* A3C, TRPO

## タイムスケジュール
### 講義 (45分)
#### Part 1. 方策勾配に基づく学習 (20分)
* 連続状態・行動に対する強化学習とその課題
* 方策勾配の考え方と基本的な学習アルゴリズム
* 方策の評価計算 (収益の期待値)
* 方策勾配定理
* 価値反復との違い
* REINFORCE
* Actor-Critic

#### Part 2. 深層強化学習・最新の研究動向 (25分)
* Arcade Learning Environment (ALE)
* ピクセルから価値へ：Deep Q-Network (DQN)
* NNによるQ関数の近似：DQNの学習アルゴリズムの解説
* Experience Replay/Target Freezing/Reward Clipping/Frame Skipping
* Experience Replayとエピソード記憶の関係
* DQNの派生モデル (Double DQN/Dueling DQN/Prioritized Experience Replay)
* 最近の研究動向 (特に、汎用エージェントに向けた研究があれば)
* 現代の方策勾配法：A3C、TRPO

### 演習 (45分)
* ChainerRLの紹介
* ChainerRLの提供アルゴリズムとその使い方
* DQNのコード解説及び学習済みモデルの実行
* DQNの学習結果の可視化 (学習済みモデルのフィルタなど)
* 簡単なタスク (Cartpoleなど) を選定し、ChainerRLを用いて実行してもらう NNベースのQ関数の定義を書かせる

## 参考文献
* 牧野他『これからの強化学習』(森北出版、2016)

* Tutorial: Deep Reinforcement Learning
http://icml.cc/2016/tutorials/deep\_rl\_tutorial.pdf

* 深層強化学習の動向 / survey of deep reinforcement learning
https://speakerdeck.com/takuseno/survey-of-deep-reinforcement-learning

* ChainerRL Quickstart Guide
https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb
