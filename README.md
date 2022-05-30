### AHC011

15 パズル　木の章

---

#### 第一投

スコア関数を rust から移植して簡単なビームサーチを書く  
全域木を作ることは考えず、とりあえず最大の木のサイズを評価値にする  
幅は 5000

提出するとスコアは 17432944

だいたい 7 割くらい使った木が作れている

---

15 puzzle は置換云々を検索すると「適当な 2 つの駒をつまみ上げて交換する動作を偶数回行った局面にした移行できない」ことがわかる ([参考](https://manabitimes.jp/math/979))

所与のピースを用いて全域木を構築する方法で、偶数回の swap によって達成できるものを列挙して、手数の短いものを採用すればよさそう？（全域木の構築がまず難しそうではあるが…）

揃えるフェーズは MM117 RotatingNumbers とかを参考にすれば何とかなりそう

順位表を見ると maspy さんが 31.7M を出していて、25M を超えているので全域木を作っていることがわかる

---

16 種類のピースで N^2-1 (N>=6) マスの盤面を埋めるとき、鳩の巣原理よりどれかのピースは複数枚使うことになる

このとき、初期盤面と目標盤面の転倒数は同じ種類のピースを交換することで自由に調節できるので、偶置換・奇置換を考慮しなくてよくなる（はず）

---

input で与えられたピースを用いて木を作る方法を実装した

まず入力生成と同様の方法でランダムに全域木を作る

input で与えられたピースの種類ごとの個数を配列にした target_ctr[16]
ランダム全域木についてピースの種類ごとの個数を配列にした tree_ctr[16]
を用意する

使われていない辺を on にして、出来た閉路（木なので必ず閉路ができる）の辺のうち一つを off にする遷移によって、木を保ったまま変形することができる

上述の遷移によって、sum_{i=0}_{15} abs(target_ctr[i]-tree_ctr[i]) を最小化する山登りをすると直ぐにコスト 0 の解を見つけることができる