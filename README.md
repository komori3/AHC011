### AHC011

15 パズル　木の章

---

#### 第 1 投 (2022-05-28 14:32:39)

スコア関数を rust から移植して簡単なビームサーチを書く  
全域木を作ることは考えず、とりあえず最大の木のサイズを評価値にする  
幅は 5000  

提出するとスコアは <b>17432944</b>

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

---

#### 第 2 投 (2022-05-30 06:34:09)

山登りを多点スタートして所与のピースで作れる候補の木を大量に生成して目標盤面とする

input と目標盤面のピースを最大重みマッチング(Primal Dual)で対応付ける　コストはマンハッタン距離の総和

対応ができたらあとはルールベースで N^2-1 puzzle を解けばよい（解けない盤面を弾く必要があることに注意）

https://igorgarbuz.github.io/n-puzzle/ でたくさんプレイして雰囲気をつかむ

[MM117 RotatingNumbers](https://togetter.com/li/1501039) っぽく N * N 正方形の 2 辺を揃えて (N-1) * (N-1) の正方形に…とどんどん小さくしていく感じ

かなり発狂しながらルールベースを書いた　改善の余地は大いにあり

提出するとスコアは <b>32889877</b>

---

#### 第 3 投 (2022-05-30 08:11:33)

山登りを焼きなましに変更してより沢山の候補を生成するようにした

提出するとスコアは <b>34230648</b>

---

#### 第 4 投 (2022-05-31 01:53:25)

ルールベースのみでは上位の点に追いつくのは辛い気がする

正方形領域がある程度小さくなれば A* やビームサーチなどが使えそう

A* は RotatingNumbers で大量 TLE を出した苦い思い出があるので、まずはビームサーチで実験する

マンハッタン距離の総和をコストとしてビームを撃つと、かなり時間は掛かるが幅 10000 で seed=0 が 800000 点台に乗る

カリカリに高速化したら N=6,7 あたりは何とか解が出せそうだったので、ルールベースのものと合わせて一番よいものを採用する

提出するとスコアは <b>36174376</b>、順位は 8 位で 3 日目に差し掛かって一桁台は今までにない好調

---

今の実装はビームサーチとルールベースを独立に走らせている

N > 7 などのビームサーチが間に合わないケースでは、途中までルールベースで解いて 6x6 以下になったらビームサーチすればよさそう（やれば 38M とか乗るんじゃない？と思っている）

あとは盤面の評価値を n 手先読みしたときのベストスコアにするやつも試したい

ルールベースもコーナー処理等の無駄を削ぎ落とすことができそう

15puzzle 関係の論文を漁っていて出てきた linear conflict heuristics を理解したが、3sec で使うには少々重そう  
一応考慮には入れておく

---

#### 第 5 投 (2022-05-31 12:43:03)

7x7 までルールベースで解いて、残りはビームサーチに切り替える

提出するとスコアは <b>37247312</b>

---

思ったより伸びなかった

ルールベース解法のコーナー処理とか、移動パスを選ぶ際に manhattan distance を小さくするようなものを選ぶとかすればさらに良くなるはず

それでも 1 位の 41M はわけがわからない、10x10 も激ヤバヒューリスティクスでビーム飛ばしてそう…

試す: Chokudai search, 領域を絞ったビームサーチ, 先読みビームサーチ

N=6 の時だけビーム幅増やしてもいいかも

---

#### 第 6~8 投 (~2022-06-02)

ルールベースの精緻化（地獄実装）

空マスを数字の近くまで持ってく際にマンハッタン距離が増えたらコスト 2、そうでなければ 0 の 01-BFS をする
* 根拠はない　ダイクストラで係数いじれるようにした方がいいかも

数字を現在地 S から目的地 T まで移動させる際に幅少ないビームサーチを入れる
* 細かいこと色々やってるけど後で書く

スコアは <b>38949583</b> まで伸びた　上位の 42M は何？

---

chokudai search, 先読みビームサーチ試したがあまりよくない

Primal Dual を Hungarian に変えてみる
* N=10 で 7~8 倍くらい速い　でかした

N 大のときにビーム -> ルールベース -> ビームとしてはどうか？

linear conflict を試す

mincost flow に空マスを含めてしまっていた　なしで試してみる -> 違いわからず

最初に揃える上辺や左辺に近いほどコストに重みを掛けてみる？

8x8 over は領域を 4 分割して解く？ <- これが正解…？　なんかそういう気がしてきた　もうそういう気しかしない　デカ盤面でビームサーチしてまともに答え出すとか不可能そうだし

