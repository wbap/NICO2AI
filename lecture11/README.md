# 第11回：強化学習 (1)

## 到達目標
* 強化学習の問題設定と、教師あり学習との違いについて説明できる
* 強化学習問題の用語を覚える
* 価値反復に基づく学習アルゴリズムを理解し、実装できる
* OpenAI Gymの基本的な使い方を覚える
* Q-learningを用いて簡単な強化学習問題 (Cartpole) が解ける

## 講師へのお願い
* 用語が多いため、生徒の理解状況を見ながら丁寧に言葉を導入してください
* アルゴリズム・実装・問題設計の3つをバランスよく解説してください
* 口頭での解説に加え、生徒に考えさせる時間をこまめに設けてください

## キーワード
* 強化学習 (Reinforcemnt Learning)
* 状態・行動・報酬・方策
* マルコフ決定過程 (MDP)
* 収益・割引率・価値関数・行動価値関数
* Bellman方程式
* TD-learning・Q-learning・Actor-Critic
* 方策オン型・オフ型
* OpenAI Gym
* greedy法、ε-greedy法
* Cartpole
* 多腕バンディット問題
* 大脳基底核
* 線条体

## タイムスケジュール
### 前回の復習 (5分)
### 講義・基礎演習 (85分)
#### Part 1. 強化学習の問題設定 (25分)
* 強化学習とは
* 強化学習の問題設定
* 環境とエージェント
* 探索-利用トレードオフ
* 基本的な用語1 (状態・行動・報酬・方策)
* マルコフ性
* マルコフ決定過程 (MDP)
* 基本的な用語2 (収益・割引率・価値関数・行動価値関数)
* Bellman方程式及び価値関数の漸化式記述
* 小課題：強化学習問題を設計して、状態、行動、報酬、(ヒューリスティックな)方策を考える

#### Part 2. 価値反復に基づく学習、探索 (30分)
* 強化学習アルゴリズムの分類 (モデルベース/フリー、方策オン/オフ、価値反復/方策勾配)
* TD-learning
* 簡単なMDPにおける、TD-learningのアルゴリズム
* Q-learning
* 方策の定め方
* 探索戦略の種類 (素朴な探索、不確かなときは楽観的に、etc.)
* greedy方策, ε-greedy方策 UCB1
* 小課題：簡単なMDPを用いた、価値関数の伝播計算 (手計算)

#### Part 3. 基礎演習:OpenAI Gym入門 (15分)
* OpenAI Gymの機能
* Cartpole-v0の紹介 (タスク及び状態・報酬・行動の数値的定義)
* observation, state, reward, action, done
* 小課題：ルールベース記述 (講師の指示にある程度従いつつ、actionをルールで書き下して動かす)

#### Part 4. 生物における強化学習 (15分)
* 脳は強化学習しているか
* (連合学習：Rescorla-Wagner Model (Rescorla, 1972))
* 大脳基底核 (線条体、淡青球、黒質、視床下核)
* 大脳基底核の各部位の機能
* サル脳のドーパミン神経細胞と報酬予測誤差との対応性 (Schultz, 1997)
* 線条体における行動価値の学習 (Samejima, 2005)
* 強化学習モデルとしての大脳基底核 (Doya, 2007)

### 実践演習 (85分)
#### 課題1: Bandit問題 (10分)
* 多腕バンディット問題の解説
* 探索アルゴリズムとの競争 (Can you beat the bandit?) に挑戦 (腕の数が10程度だと面白い)
* 講師は予め各探索アルゴリズムの平均的な性能を計測して、紹介する

#### 課題2: OpenAI GymとCartpole-v0 (60分)
* ルールベースに基づく解 (OpenAI Gymより引っ張ってきてそのアイデアを検証、動かしてみる)
* 状態を離散化したテーブルベースQ-Learningに基づく強化学習アルゴリズムを実装してもらう
* コードの骨子及び設計方針は予め用意し、一部を穴埋めしてもらう形式
* 学習曲線の可視化を行いつつ、ランダム探索や使用する状態を変化させるなどして検証 (重要)

#### 解説・コードレビュー (15分)
### フィードバック・次回予告 (5分)

## 参考文献
* 牧野他『これからの強化学習』(森北出版、2016)

* Lecture 9: Exploration and Exploitation
http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching\_files/XX.pdf

* (Rescorla, 1972) A Theory of Pavovian Conditioning: Variations in the Effectivenes of Reinforcement and Nonreinforcement
https://sites.ualberta.ca/~egray/teaching/Rescorla%20&%20Wagner%201972.pdf

* (Schultz, 1997) A Neural Substrate of Prediction and Reward
http://www.gatsby.ucl.ac.uk/~dayan/papers/sdm97.pdf

* (Samejima, 2005) Representation of action-specific reward values in the striatum
https://www.ncbi.nlm.nih.gov/pubmed/16311337

* (Doya, 2007) Reinforcement learning: Computational theory and biological mechanisms
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2645553/pdf/HJFOA5-000001-000030\_1.pdf

* Can you beat the bandit?
http://iosband.github.io/2015/07/28/Beat-the-bandit.html

* The Neuroscience of Reinforcement Learning (ICML2009 Tutorial)
http://www.princeton.edu/~yael/ICMLTutorial.pdf

* Cart-Pole Balancing with Q-Learning
https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
