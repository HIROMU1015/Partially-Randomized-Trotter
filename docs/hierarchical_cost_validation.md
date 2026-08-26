# 階層compiled-cost modelの拡張holdout検証

最終更新：2026-08-24 JST

## 1. 目的

H4の先行検証では、RTE eventを個別compileした単純和を棄却し、短いevent列から
pair/triple境界補正を較正する候補を得た。本検証では、次の未確認点を分離して調べた。

1. $K=0$で、$C_2,C_3$から求めた補正が未使用の長さ$L=8$にも移送できるか。
2. 有限Taylor cutoffを$K=2$へ上げても、RTE event列の局所cluster modelが機能するか。
3. controlled partial-$S_2$反復回路のcostを、$q=1,2$から未使用$q=4$へ外挿できるか。

これは最終総コスト評価ではない。大規模系で完成回路を全trajectoryについてcompileせずに
期待compiled costを推定するための、局所近似モデルの検証である。

## 2. 比較したモデル

固定short-step分布の長さ$L$のRTE occurrenceを一体compileした期待costを$C_L$とする。
pairおよびtriple予測は

$$
\widehat C_L^{(2)}=(2-L)C_1+(L-1)C_2,
\qquad
\widehat C_L^{(3)}=(3-L)C_2+(L-2)C_3
$$

とした。$C_1,C_2,C_3$のcalibrationと$C_L$のholdoutには独立seedを使い、各平均の
標準誤差を予測差へ伝播した。

controlled partial-$S_2$の$q$反復一体compile期待costを$D_q$とし、$q=1,2$で較正する
affine model

$$
\widehat D_4=-2D_1+3D_2
$$

を、独立seedの$q=4$で評価した。対象はordinary controlled time-evolution subcircuitであり、
ancillaのHadamard、軸変更、測定、状態準備および量子shot実行は含めない。

暫定判定はRZ countを主指標とする相対5%とした。点推定と、calibration/holdout双方の
標準誤差を加えた点ごとの正規近似95%上側診断を分けて記録した。これは比率の厳密な信頼上界や
全指標同時被覆ではない。この5%は最終resource accountingの受理閾値
として確定したものではない。

## 3. 共通条件

| 項目 | 条件 |
|---|---|
| model | H4 chain、原子間距離1.0 Å |
| basis / DF | STO-3G、DF rank 12 |
| split | $L_D=3$、旧3-artifactではRTE component 324個。共有再検証batchでも各artifact内の表現を固定 |
| time | $\delta=0.1$、$r=4$、short-step時間0.025 |
| compiler | Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、結合制約なし、seed 17 |
| sampling | fresh IID trajectory、calibration/holdoutで独立seed |
| metrics | RZ/CX count・depth、total depth、circuit size |

一つの親Hamiltonianを3 workerへ渡し、全artifactで
`preparation_hash = fbed0c65...11c3`が一致することを確認した。成果物はdirty worktreeで生成した
local evidenceであり、immutable CI evidenceではない。

## 4. 結果

### 4.1 $K=0$：未使用$L=8$

$C_1$は324 eventの厳密列挙、$C_2,C_3$は各2000標本、$C_8$は別seedの2000標本とした。

| 指標 | pair絶対相対誤差 | triple絶対相対誤差 | triple absolute z-score |
|---|---:|---:|---:|
| RZ count | 8.794% | 1.744% | 0.521 |
| CX count | 7.697% | 1.460% | 0.442 |
| RZ depth | 0.368% | 0.250% | 0.067 |
| CX depth | 2.320% | 0.479% | 0.133 |
| total depth | 0.823% | 0.117% | 0.031 |
| circuit size | 8.851% | 1.482% | 0.446 |

triple補正の全指標最大点誤差は1.744%、最大absolute z-scoreは0.521だった。RZ countの
点ごとの正規近似95%上側診断は8.307%で、暫定5%を統計的に保証する幅には達していない。一方、pair-onlyの
count/size残差約8%が未使用$L=8$でも再現し、tripleで約1.5--1.7%へ下がった。

したがって、この$K=0$条件ではcount/sizeにtriple、depthにpairを用いるmetric別候補を
支持する。この条件・精度では4-event以上のcluster係数が必要だという証拠は得られなかった。

### 4.2 $K=2$：Taylor次数を強制した再検証

旧IID pilotでは1 short stepのorder-2確率が
$p_2=1.062925\times10^{-4}$だった。seedとsampling conventionを再現して監査すると、
$C_1,C_2,C_3,C_4,C_6$の全8000 event位置でorder-2は1回だけだった。したがって旧表の
RZ count最大4.113%という値は、ほぼ$K=0$の列を評価したpilotであり、$K=2$ event内部の
妥当性を示す根拠には使わない。

再検証では各Taylor次数列を条件付きで生成し、解析的な次数確率で再結合した。all-order-0列は
各長さ500標本、order-2がちょうど1回の各配置は100標本、2回以上は各2標本とした。
独立な$C_1,C_2,C_3$較正と$L=4,6$ holdoutの結果は次の通りだった。

| holdout | 条件 | RZ count点誤差 | absolute z-score | 点ごとの正規近似95%診断 |
|---:|---|---:|---:|---:|
| $L=4$ | all-order-0 | 3.926% | 1.048 | 11.267% |
| $L=4$ | order-2が1回 | 7.701% | 1.548 | 17.449% |
| $L=6$ | all-order-0 | 9.119% | 1.650 | 19.948% |
| $L=6$ | order-2が1回 | 3.882% | 0.728 | 14.336% |

全指標最大点誤差は9.119%、最大z-scoreは1.651だった。これは5%判定を通らないが、差は
約1.7標準偏差以内であり、独立に推定した大きな$C_2,C_3,C_L$のMonte Carlo分散と
cluster打切り誤差を分離できない。

そこで同一trajectoryから1・2・3-event局所窓と全回路をcompileし、
$\widehat C_L^{(3)}-C_L$をtrajectoryごとに直接取る対応あり残差を追加した。all-order-0は
各長さ100標本、order-2が1回の各配置は25標本とした。

| holdout | 条件 | RZ count点誤差 | 対応ありz-score | 点ごとの正規近似95%診断 |
|---:|---|---:|---:|---:|
| $L=4$ | all-order-0 | 0.457% | 3.014 | 0.754% |
| $L=4$ | order-2が1回 | 0.321% | 3.611 | 0.496% |
| $L=6$ | all-order-0 | 1.373% | 6.366 | 1.796% |
| $L=6$ | order-2が1回 | 1.097% | 7.535 | 1.382% |

全六指標の最大点誤差も1.373%で、RZ countの点推定と95%診断は暫定5%を通過した。
z-scoreが大きいので4-event以上に由来する小さい系統残差は存在するが、この条件では
相対5%以内である。したがって、$K=2$の3-event局所式はこの代表条件の候補として残す。
一方、独立係数を各500標本で求める運用推定器は精度不足であり、解析的次数重み、対応ありの
connected-cluster係数、標準誤差停止条件を用いる必要がある。

### 4.3 controlled partial-$S_2$：未使用$q=4$

$D_1,D_2$を各300標本で較正し、別seedの$D_4$を300標本でholdout評価した。

| 指標 | q=4絶対相対誤差 | absolute z-score | 点ごとの正規近似95%診断 |
|---|---:|---:|---:|
| RZ count | 0.293% | 0.922 | 0.917% |
| CX count | 0.159% | 0.925 | 0.498% |
| RZ depth | 0.131% | 0.886 | 0.420% |
| CX depth | 0.067% | 0.893 | 0.214% |
| total depth | 0.137% | 0.896 | 0.435% |
| circuit size | 0.307% | 0.923 | 0.958% |

全指標で点推定と点ごとの正規近似95%診断の両方が暫定5%を通過した。この代表条件では、controlled化後の
決定論部分、RTE occurrenceおよびstep間のtranspiler最適化を含む完成回路costが、$q=1,2$
からのaffine modelで$q=4$へ移送できた。

## 5. 判断と次の検証

1. $K=0$の同一条件では、count/sizeにtriple、depthにpairを用いる候補を採用し、cluster次数
   の検証は一旦終了する。
2. controlled $q$方向はaffine modelを次の候補とする。ただし$q>4$、別$L_D$、結合制約、
   optimization levelおよびHadamard wrapper全体へ一般化しない。
3. $K=2$の3-event局所式はTaylor次数条件付き・対応あり残差で暫定5%を通過した。ただし
   500標本の独立係数推定器は通過していないため、運用時はconnected-cluster係数を対応ありで
   推定し、解析次数確率で再構成する。
4. 次の優先検証は、別$L_D$またはshort-step条件への係数移送と、compiler/coupling contextの
   holdoutである。今回と同一条件の単純なフルMC増量は優先しない。
5. これらの局所結果を量子shot数、誤差・失敗確率配分または全round総コストへまだ接続しない。

## 6. 成果物と再実行

- 実装：`src/trotterlib/hierarchical_cost_validation.py`、
  `src/trotterlib/rte_order_stratified_cost_validation.py`
- 生成script：`scripts/run_hierarchical_cost_validation.py`、
  `scripts/run_rte_order_stratified_cost_validation.py`
- 専用test：`tests/test_hierarchical_cost_validation.py`、
  `tests/test_rte_order_stratified_cost_validation.py`
- artifact：`artifacts/hierarchical_cost_validation/`の3 JSON、
  `artifacts/rte_order_stratified_cost_validation/`の共有batch 2 JSONを主証拠とする。
  先行する別process 2 JSONはDF表現hashが異なるため比較診断に限定する

再実行：

```bash
.venv311/bin/python scripts/run_hierarchical_cost_validation.py
.venv311/bin/python scripts/run_rte_order_stratified_cost_validation.py
.venv311/bin/python scripts/run_rte_order_stratified_cost_validation.py --shared-both
```

実行時間は$K=2$が1155秒、$K=0,L=8$が3832秒、controlledが4275秒だった。3 workerを
並列実行した。Taylor次数条件付きの独立推定は433秒、対応あり残差は504秒だった。
各JSONはversioned schema、SHA-256 fingerprint、source hash、生成command、
compiler条件、標本数、独立seed、event/trajectory digest、標準誤差を保存する。
