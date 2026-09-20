# RPE短ラウンドの仮想Hadamard測定・失敗確率検証

## 目的

既存のfinite-RTE信号から求めた測定回数が、暫定的に割り当てた座標誤差、統計位相誤差、
失敗確率と整合するかを検証した。対象は短い3 roundの測定統計だけであり、全roundの
RPE位相復元、エネルギー推定、実backend、noise、状態準備、最終総コストは含まない。

## 固定条件

- H4鎖、原子間距離1.0 Å、STO-3G、8 qubit。
- 保存した同一DF Hamiltonian snapshot、DF rank 12、$L_D=3$。
- $delta=0.1$、RTE short-step数$r=4$、有限Taylor cutoff $K=2$。
- $q_m=1,2,4$。
- $(\beta_{\mathrm{PF}},\beta_{\mathrm{RTE}},\beta_{\mathrm{stat}})
  =(0.08,0.08,0.24)$ rad、$\beta_{\mathrm{RPE}}=0.40$ rad。
- $\alpha_{\mathrm{tot}}=0.05$を3 round・cosine/sineの6軸へ均等配分し、
  $\alpha_{m,b}=0.05/6$とした。
- finite-RTEの物理full-$H$基底状態に対するattenuated event-mean signalを測定の真値とした。
- 期待信号から直接生成する仮想実験は10万回。別に、各軸で必要shot数だけfresh IIDな
  RTE trajectoryを明示抽出する1 batchを実行した。

## 二段階の検証

### 1. 周辺Bernoulli分布

cosine軸の$\pm1$結果の平均を$\operatorname{Re}Z_m$、sine軸を
$\operatorname{Im}Z_m$とした。RTE trajectoryを各shotで独立に生成する場合、trajectoryを
周辺化した各測定結果も、この平均を持つIID Bernoulli変数になる。この性質を使い、

- 座標誤差が$\epsilon_{\mathrm{coord},m}$以上となる厳密二項確率、
- cosine・sineから復元した位相誤差が$\beta_{\mathrm{stat}}$を超える厳密確率、
- 10万回のMonte Carlo反復と厳密確率の一致、

を評価した。Monte Carloの点推定だけで合否を決めず、Clopper--Pearson区間も保存した。

### 2. fresh IID trajectoryの明示検査

各Hadamard shotに固有のseedを割り当て、sector内で$r q_m$個のRTEイベントと決定論halfを
状態へ作用させた。そのtrajectory固有の複素期待値を条件付き平均として$\pm1$結果を生成した。
これはイベントsampler、符号規約、fresh-IID実装を確認するためのproduction-size 1 batchであり、
低確率事象の発生率推定には使っていない。

## 結果

### 厳密二項確率

| $q_m$ | 各軸shot | $\epsilon_{\mathrm{coord},m}$ | cosine座標失敗率 | sine座標失敗率 | 位相失敗率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 389 | 0.1679383 | $1.70\times10^{-22}$ | $7.19\times10^{-4}$ | $1.28\times10^{-6}$ |
| 2 | 390 | 0.1677955 | $2.07\times10^{-11}$ | $3.00\times10^{-4}$ | $2.02\times10^{-7}$ |
| 4 | 391 | 0.1675104 | $1.62\times10^{-5}$ | $1.46\times10^{-6}$ | $9.86\times10^{-10}$ |

6軸のいずれかで座標誤差が許容量以上となる厳密確率は
$1.0363\times10^{-3}$、3 roundのいずれかで統計位相誤差が0.24 radを超える確率は
$1.4775\times10^{-6}$だった。いずれも暫定$\alpha_{\mathrm{tot}}=0.05$以内である。

10万回の仮想実験では、6軸合成の座標失敗は111回、失敗率0.00111、片側95%上限
0.001299だった。位相失敗は0回で、片側95%上限は$2.996\times10^{-5}$だった。
厳密確率は各軸・各roundの事前指定99% Monte Carlo区間に入り、座標条件を満たしながら
位相予算を超える例は厳密な離散gridでもMonte Carlo標本でも0だった。

### fresh IID trajectory

3 round・2軸で合計2340 trajectoryを生成し、seedはすべて異なった。trajectory信号平均と
解析的finite-RTE平均の差は最大2.063標準誤差だった。6軸すべての条件付き測定結果は、
解析的な周辺二項分布の99.9%中央区間内に入った。

## 判断と限界

H4の上記短round条件では、389、390、391という各軸shot数と暫定失敗確率配分を受理する。
Hoeffding式はこの条件の実失敗率に対して保守的だった。ただし、これは経験的な短round実装検証であり、
PF係数が経験値であること、基準信号半径下界が一般には外部仮定であることは変わらない。

また、10万回反復は解析的な周辺信号を使った古典シミュレーションであり、実量子backendの
10万実験ではない。全roundのbranch selectionと最終位相復元を実行していないため、
「RPE全体の成功率を検証した」とは扱わない。次は、holdout検証済みの$q>4$ Hadamard
1 shot cost proxyをresource accountingへ接続する。

## 成果物

- 結果：
  `artifacts/rpe_hadamard_failure_validation/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_marginal100000_fresh_v1.json`
- 実装：`src/trotterlib/rpe_hadamard_failure_validation.py`
- 生成：`scripts/run_rpe_hadamard_failure_validation.py`
- test：`tests/test_rpe_hadamard_failure_validation.py`

成果物はlocal dirty-worktree evidenceであり、immutable CIまたは外部再現結果ではない。
