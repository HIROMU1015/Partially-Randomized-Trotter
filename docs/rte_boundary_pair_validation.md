# RTE境界補正のfragment層別・高統計検証

最終更新：2026-08-23 JST

## 1. 目的

H4の300標本pilotでは、event単純和の過大評価をpair境界補正で大きく減らせたが、
pair補正後にも最大8.73%の観測差が残り、最大absolute z-scoreは1.50だった。この差が
Monte Carlo揺らぎか系統的残差かを判定し、pair補正をDF fragment境界の確率から構成できるか
調べた。

検証は次の二つからなる。

1. 同一Hamiltonian表現上の独立2 seedで、1--3 event clusterのcalibrationと4・6 event
   holdoutを各1000標本へ増やす。
2. pair境界をsame-fragmentとdifferent-fragmentに層別化し、1500標本のcalibrationから
   求めた条件付き補正を、別seedの1500標本pair holdoutへ予測する。

これはcompiled-cost近似のlocal validationであり、最終総コスト評価ではない。

## 2. 条件

| 項目 | 条件 |
|---|---|
| model | H4 chain、原子間距離1.0 Å、STO-3G |
| DF / split | DF rank 12、$L_D=3$、RTE component 218個 |
| finite RTE | $K=0$、short-step時間0.025、fresh IID trajectory |
| compiler | Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、結合制約なし、seed 17 |
| full cluster | seed 20260823、20261823。各seedで$C_2,C_3,C_4,C_6$を各1000標本、$C_1$を厳密列挙 |
| pair層別 | calibration seed 20263823を1500標本、holdout seed 20263824を1500標本 |

親processでHamiltonianを一度だけ生成して各workerへ渡した。三つのartifactは同じ
`preparation_hash = 400b18b...98d44`を持つ。以前の300標本artifactは別processで生成され、
`preparation_hash`が異なるため、1000標本runのprefix収束比較には使用しない。

## 3. 1000標本・2 seedのcluster holdout

| seed | event単純和：最大誤差 / z | pair：最大誤差 / z | triple：最大誤差 / z |
|---:|---:|---:|---:|
| 20260823 | 165.15% / 104.70 | 8.07% / 2.96 | 2.33% / 0.79 |
| 20261823 | 165.54% / 106.99 | 8.18% / 3.02 | 3.73% / 0.97 |

pair-onlyの約8%残差は両seedで再現し、最大z-scoreは約3だった。したがって、少なくとも
count/size指標ではMC揺らぎだけでは説明できない。一方、triple補正後の全比較は両seedで
absolute z-score 1未満だった。

### metric別の最大holdout誤差

表は2 seed、$L=4,6$の最大値である。

| metric | pair絶対相対誤差 | pair最大z | triple絶対相対誤差 | triple最大z |
|---|---:|---:|---:|---:|
| RZ count | 8.18% | 3.01 | 3.73% | 0.97 |
| CX count | 7.39% | 2.74 | 3.37% | 0.91 |
| circuit size | 8.17% | 3.02 | 3.73% | 0.97 |
| RZ depth | 2.57% | 0.97 | 2.15% | 0.72 |
| CX depth | 1.75% | 0.67 | 2.33% | 0.79 |
| total depth | 2.31% | 0.88 | 2.06% | 0.70 |

pair-onlyの統計的な不適合はcount/sizeで生じ、depthでは確認されなかった。したがって、
最終的に選ぶcost metricによって必要なcluster次数が異なる可能性がある。

triple係数$\mu_3$自体のabsolute z-scoreは、全metric・両seedを通じて最大1.40だった。
すなわち、triple予測はholdoutを覆ったが、$\mu_3\ne0$を2標準誤差以上で確定したわけではない。
count/sizeではtripleを次の候補とするが、$C_3$ calibrationの追加精度確認が必要である。

## 4. fragment境界の層別化

隣接eventが同じfragmentとなる解析確率は

$$
p_{\mathrm{same}}=\sum_f p_f^2=0.7310604,
\qquad
p_{\mathrm{different}}=0.2689396
$$

である。calibrationとholdoutはいずれも1500標本中same 1104、different 396だった。

各metricの境界補正を

$$
\mu_2
=p_{\mathrm{same}}\,\mathbb E[B\mid\mathrm{same}]
+p_{\mathrm{different}}\,\mathbb E[B\mid\mathrm{different}]
$$

として構成した。比較のため、different境界の補正をゼロとするsame-onlyモデルも評価した。

| metric | same条件付き補正 | different条件付き補正 | same-only誤差 / z | 二分類誤差 / z |
|---|---:|---:|---:|---:|
| RZ count | -321.79 | -25.62 | 3.65% / 2.51 | 0.82% / 0.57 |
| CX count | -90.90 | -5.21 | 2.77% / 1.85 | 0.72% / 0.48 |
| circuit size | -636.87 | -44.49 | 3.33% / 2.26 | 0.85% / 0.58 |
| RZ depth | -62.30 | -4.92 | 3.62% / 2.59 | 0.82% / 0.59 |
| CX depth | -32.90 | -1.53 | 2.17% / 1.48 | 0.50% / 0.34 |
| total depth | -140.05 | -9.98 | 3.29% / 2.33 | 0.75% / 0.53 |

same-fragment境界が主補正であることは確認できたが、different-fragment境界にも全metricで
負の補正がある。different補正をゼロとすると最大z=2.59の系統差が残り、二分類モデルでは
最大誤差0.849%、最大z=0.587へ低下した。

最大確率の `fragment-3 → fragment-3` は全境界の解析確率72.344%を占め、RZ countの
条件付き補正は約-323だった。主要な異種pairである `3→4`、`3→5`、`5→3`、`4→3`、
`3→6`、`6→3`にも、RZ countで約-9から-39の補正があった。

## 5. 判断

1. event単純和は引き続き不採用とする。
2. pair補正は境界効果の主要項だが、count/sizeの長いevent列にはpair-onlyでは不十分である。
3. pair係数を確率から作る場合、same-fragmentだけでなくdifferent-fragmentの条件付き補正も
   必要である。このH4条件では二分類でpair holdoutを1%未満に予測した。
4. depthを目的metricとする場合、現状ではpair-onlyの統計的不適合は確認されていない。
5. count/sizeではtripleモデルを次の候補とする。ただし$\mu_3$自体は2標準誤差で非ゼロと
   確定していないため、追加の$C_3$ calibrationまたは未使用$L=8$ holdoutが必要である。
6. この結論はH4、$L_D=3$、$K=0$、一つのcompiler条件に限定する。

## 6. 成果物

- 1000標本 seed 20260823：
  `artifacts/rte_boundary_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k0_n1000_seed20260823_v1.json`
- 1000標本 seed 20261823：
  `artifacts/rte_boundary_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k0_n1000_seed20261823_v1.json`
- fragment層別1500/1500標本：
  `artifacts/rte_boundary_pair_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k0_ncal1500_nhold1500_v1.json`
- 層別実装：`src/trotterlib/rte_boundary_pair_validation.py`
- 並列生成script：`scripts/run_rte_boundary_cost_replication.py`
- 専用test：`tests/test_rte_boundary_pair_validation.py`

三artifactのfingerprint、記録したsource hashおよび共通preparation hashは検査済みである。
dirty local worktree evidenceであり、immutable CI evidenceではない。
