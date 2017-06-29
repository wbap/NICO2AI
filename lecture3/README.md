# 第3回：線形回帰

## 到達目標
* Pythonの利便性を上げるクラスなどの構文が書けるようになる
* 二乗誤差最小化に基づく線形回帰アルゴリズムを数式で記述し、その実装ができる
* 実データを用いて線形回帰を行い、その結果をMatplotlibを用いて可視化できる

## キーワード
* 最小二乗回帰
* map, reduce, filter
* Advanced indexing
* Matplotlib

## タイムスケジュール
### 前回の復習 (5分)
### 講義(30分)
#### 統計学の復習
* 確率変数
* 期待値、分散、標準偏差
* 二乗和誤差
* 最小二乗回帰の解析解

###基礎演習(60分)
#### Part 1: Python・numpyのさらなる活用 (45分)
* map, reduce, filter
* (イテレータ (iterator) / ジェネレータ (generator) の紹介と使用例) -> 第6回
* (無限数列の生成など) -> 第6回
* (yield文を使った例があるとよい) -> 第6回
* クラスとその構成要素 (__init__, self, super, Python流のクラスの書き方)
* Advanced indexingの解説と使い方
* Advanced indexingの使用例
* 変数のマスク
* 次元による配列アクセス速度の比較、viewとadvanced indexingの違い、Row/Column major-order (http://kaisk.hatenadiary.com/entry/2015/02/19/224531を参考に)
* numpy配列の分割と結合
* np.stack, np.vstack, np.hstack, np.concatenate
* np.eye
* 方程式を解く (np.eye, np.linalg.solve)

#### Part 2: Matplotlib入門 (15分)
* Matplotlibの解説
* 折れ線グラフ (plt.plot)
* 散布図 (plt.scatter)
* ヒストグラム (plt.hist)
* 画像の表示 (plt.imshow)
* グラフの分割(subplot, axis)

### 実践演習(90分)
#### 課題1: Advanced indexing、変数のマスクを使いこなす (20分)
* 行列の一定割合をランダムにマスク
* 他、適当に

#### 課題2: 二乗誤差最小化に基づく線形回帰の実装 (50分)
* 2-1: 関数近似
* 2-2: 実データで何か -> Allen

#### 解説・コードレビュー (15分)
* できた人のコードを読んで、良い点・改善点があれば指摘
* そのうえで答えの解説

#### フィードバック・次回予告 (5分)

## 参考文献 (講師の方も随時追加お願いします)

