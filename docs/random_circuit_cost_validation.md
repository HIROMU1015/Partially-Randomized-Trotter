# ランダム回路compiled-cost加法モデルの検証

最終更新：2026-08-23 JST

## 1. 検証したい近似

最終総コストを計算する前に、ランダム回路の期待compiled costを、より小さい回路単位の
コスト和で近似できるかを調べる。今回のpilotでは、同じRTE trajectoryに対して次の二つを
比較した。

### RTE event単位の加法モデル

RTE occurrenceに含まれるevent列を一つの回路としてtranspileしたコストを
$C_{\mathrm{RTE,full}}$、各eventを別々にtranspileして足したコストを
$C_{\mathrm{RTE,event\text{-}sum}}$とする。

$$
C_{\mathrm{RTE,event\text{-}sum}}
=\sum_{j=1}^{r}C_{\mathrm{compile}}(E_j)
$$

この比較は、event境界をまたぐゲート相殺やtranspiler最適化を無視できるかを調べる。

### partial-$S_2$の3部分加法モデル

partial-$S_2$全体を一つの回路としてtranspileしたコストを$C_{S_2,\mathrm{full}}$とし、
決定論forward half、RTE occurrence全体、決定論reverse halfを別々にtranspileした和を
$C_{S_2,\mathrm{3part}}$とする。

$$
C_{S_2,\mathrm{3part}}
=C_{D,\mathrm{forward}}+C_{\mathrm{RTE,full}}+C_{D,\mathrm{reverse}}
$$

各metricについて、加法近似の符号付き相対誤差を

$$
\eta_C=\frac{C_{\mathrm{add}}-C_{\mathrm{full}}}{C_{\mathrm{full}}}
$$

と定義した。正の値は、加法モデルが一体compileよりコストを過大評価することを表す。
全比較は同一trajectory上で対にして行い、Monte Carloでは差$C_{\mathrm{full}}-C_{\mathrm{add}}$
自体の標準誤差も保存した。

## 2. pilot条件

| 項目 | 条件 |
|---|---|
| model | H4 chain、原子間距離1.0 Å |
| basis / DF | STO-3G、DF rank 12 |
| split | $L_D=3$ |
| partial-$S_2$ step | $\delta=0.1$、$q=1$ |
| finite RTE | $K=0$、$r=1$または2 |
| compiler | Qiskit 1.3.0、basis gates `rz,sx,x,cx`、optimization level 1、結合制約なし、seed 17 |
| 古典MC | seed 20260822。同じseedを使い、標本数を増やしたrunは同じ乱数列のprefixを再利用 |
| cost metrics | RZ count/depth、CX count/depth、total depth、circuit size |

これはcost近似のpilotであり、実backend、controlled Hadamard interrogation、量子shot、
ノイズ、全round RPEまたは最終総コストの評価ではない。加法モデルの受理閾値もまだ
決めていない。

H4のDF分解は別processで再構築すると`preparation_hash`が変わり、回路表現と絶対compiled
costも変化することを確認した。このため、主比較の$r=1,2$はHamiltonianを一度だけ構築する
batchで生成し、同じ`preparation_hash`を持つことを確認した。別processの300標本runは
独立なDF回路表現でのreplicateとして扱い、100標本から300標本への収束比較には用いない。

## 3. 結果

### 3.1 $r=1$：partial-$S_2$の完全列挙基準

$K=0$の1 short stepには218種類のeventがあり、全trajectoryを完全列挙した。
$C_{S_2,\mathrm{3part}}$は全metricで一体compileを過大評価した。

| metric | $C_{S_2,\mathrm{full}}$ | $C_{S_2,\mathrm{3part}}$ | $\eta_C$ |
|---|---:|---:|---:|
| RZ count | 3089.984 | 3100.764 | 0.349% |
| RZ depth | 607.710 | 613.709 | 0.987% |
| CX count | 1156.138 | 1156.317 | 0.015% |
| CX depth | 425.745 | 425.745 | $2.5\times10^{-7}$% |
| total depth | 1391.305 | 1401.378 | 0.724% |
| circuit size | 6160.593 | 6176.969 | 0.266% |

100標本MCによる一体compile期待値は、完全列挙値との差が最大0.213%で、
標準誤差換算の最大値は約2.41だった。標本数30から100で観測誤差が全metricについて
単調減少したわけではないため、これだけでMC収束則を確定したとは扱わない。一方、
加法モデルの相対誤差は100標本でも完全列挙値から最大0.021 percentage point以内だった。

### 3.2 $r=2$：event単位の加法モデル

trajectory空間は$218^2=47{,}524$なので完全列挙せず、$r=1$と同じDF回路表現上の
100標本古典MCで比較した。

| metric | $C_{\mathrm{RTE,full}}$ | $C_{\mathrm{RTE,event\text{-}sum}}$ | $\eta_C$ | 差の絶対z-score |
|---|---:|---:|---:|---:|
| RZ count | 417.990 | 625.340 | 49.61% | 16.53 |
| RZ depth | 81.340 | 128.080 | 57.46% | 17.79 |
| CX count | 120.900 | 179.300 | 48.30% | 15.86 |
| CX depth | 43.600 | 67.580 | 55.00% | 16.80 |
| total depth | 182.650 | 287.820 | 57.58% | 17.64 |
| circuit size | 829.530 | 1241.960 | 49.72% | 16.29 |

event別compileの和は、全metricで一体compileを48--58%過大評価した。差は16--18標準誤差
なので、少なくともこの条件では統計揺らぎでは説明できない。別processで得た異なる
`preparation_hash`の300標本replicateでも53--61%の過大評価、30--32標準誤差となり、
event境界効果の定性的結論は維持された。ただし、これはMC標本数だけを変えた収束比較ではない。

同じ$r=2$の100標本では、RTE occurrenceを一つの部分として用いる
$C_{S_2,\mathrm{3part}}$の過大評価は0--0.959%だった。したがって、大きなずれは
主にRTE event間のcompile境界を人工的に切ったことから生じ、RTE occurrenceを一体compile
すればpartial-$S_2$外側の3部分加法近似は比較的小さいずれに留まった。

## 4. pilotから採用する当面の方針

1. 個々のRTE eventコストを足すだけのモデルは、compiled costの主モデルに採用しない。
2. 補正なしの加法proxyでは、最小較正単位を少なくともshort-step列を含むRTE occurrence
   全体とする。event単位を使う場合は、独立加算せず境界補正を明示的に含める。
3. partial-$S_2$では「決定論forward half + RTE occurrence + reverse half」の和を安価な
   初期近似として残し、一体compileとの差を補正項または不確かさとして較正する。
4. この判断はH4の1点に対するpilotであり、$L_D,\delta,r,K,q$およびcompiler条件を変えた
   holdout検証を通るまで一般化しない。

その後、固定short-step分布の1--3 event列で局所境界補正を較正し、未使用の4・6 event列で
検証した。event単純和の最大絶対相対誤差157.51%に対し、pair補正は8.73%、triple補正は
4.06%だった。詳細は[RTE境界補正検証](rte_boundary_cost_validation.md)を参照する。

## 5. 実装と高速化

`random_circuit_cost_validation_v1`は、完全回路と加法回路を同一trajectory上で評価し、
versioned JSON、SHA-256 fingerprint、生成command、source hash、compiler条件、実行時間を
保存する。完全列挙と複数のMC標本数の間ではtranspile cacheを共有する。

H4・$r=1$の完全列挙部分は420秒、artifact全体は696秒を要した。$r=2$のpartial-$S_2$を
含む30/100標本runは299秒だった。一方、RTE occurrence比較だけを選択するscopeを追加し、
300標本を85秒で実行した。また、複数の$L_D,\delta,r,K$を一つのprocessで走査し、
同じHamiltonianを再利用するbatch実行を追加した。以後の広いparameter gridではこの
軽量scopeとbatch実行を先に使い、
代表点だけpartial-$S_2$全体または完全列挙へ進む。

## 6. 次に必要な検証

1. H4で$L_D=1,3,6$、$\delta=0.05,0.1,0.2$、$r=2,4$を先に
   RTE-occurrence-only scopeで走査する。
2. $K=2$を加え、event長の変化で非加法性がどう変わるか確認する。
3. 代表点で300、1000標本を比較し、差の信頼区間とMC停止条件を決める。
4. 代表点で$q=2,4$のpartial-$S_2$反復を一体compileし、step間境界の非加法性を測る。
5. optimization level、結合制約、controlled回路を変え、今回の結論がcompiler固有でないか
   holdout検証する。
6. H4の固定short-step pilotで得たpair/triple境界補正を、上記の未使用parameter条件へ
   外挿して検証し、RTE-occurrence-level proxyとの使い分けと受理閾値を決める。

## 7. 成果物と再実行

- 完全列挙基準：
  `artifacts/random_circuit_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_r1_k0_v1.json`
- $r=2$、partial-$S_2$を含む30/100標本：
  `artifacts/random_circuit_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_r2_k0_v1.json`
- $r=2$、RTE occurrence 300標本：
  `artifacts/random_circuit_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_r2_k0_mc300_v1.json`
- 生成script：`scripts/run_random_circuit_cost_validation.py`
- schema・検証実装：`src/trotterlib/random_circuit_cost_validation.py`
- 専用test：`tests/test_random_circuit_cost_validation.py`

成果物はdirty local worktreeで生成したlocal evidenceであり、immutable CI evidenceではない。
