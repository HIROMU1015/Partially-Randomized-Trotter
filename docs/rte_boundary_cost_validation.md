# RTE境界補正cost modelのpilot検証

最終更新：2026-08-23 JST

## 1. 目的

RTE eventを別々にcompileしたコストの単純和は、H4の先行pilotでRTE occurrenceの
一体compile期待値を約48--61%過大評価した。そこで、隣接event間のゲート相殺を
期待値レベルの局所的な境界補正として取り込み、短いevent列で較正した補正が、較正に
使っていない長いevent列を予測できるか検証した。

これは最終総コスト評価ではない。random circuitのcompiled-cost proxyを構成する前段階の
近似検証である。

## 2. 検証したモデル

同じshort-step時間と同じIID event分布からなる、長さ $L$ のRTE event列の
一体compile期待値を $C_L$ とする。1、2、3 eventの期待値から

$$
\mu_1=C_1,\qquad
\mu_2=C_2-2\mu_1,\qquad
\mu_3=C_3-3\mu_1-2\mu_2
$$

を定義した。$\mu_2$ は平均的な1境界の補正、$\mu_3$ は2つの隣接境界を同時に
含む3-event clusterの残差である。比較した予測は次の三つである。

$$
\begin{aligned}
\widehat C_L^{(1)}&=L\mu_1,\\
\widehat C_L^{(2)}&=L\mu_1+(L-1)\mu_2,\\
\widehat C_L^{(3)}&=L\mu_1+(L-1)\mu_2+(L-2)\mu_3.
\end{aligned}
$$

$C_1$は全218 eventの厳密列挙で求め、$C_2,C_3$は各300標本で較正した。
較正に使っていない長さ $L=4,6$ を、それぞれ独立seedの300標本でholdout評価した。
各長さでshort-step時間を0.025に固定し、eventの角度と抽選分布を変えずに境界数だけを
増やした。予測値とholdout平均の標準誤差を合成し、absolute z-scoreも保存した。

## 3. pilot条件

| 項目 | 条件 |
|---|---|
| model | H4 chain、原子間距離1.0 Å |
| basis / DF | STO-3G、DF rank 12 |
| split | $L_D=3$、RTE component 218個 |
| short step | 0.025。基準として $\delta=0.1,r=4$ |
| finite RTE | $K=0$、fresh IID trajectory |
| compiler | Qiskit 1.3.0、basis gates `rz,sx,x,cx`、optimization level 1、結合制約なし、seed 17 |
| calibration | $C_1$厳密列挙、$C_2,C_3$各300標本 |
| holdout | $C_4,C_6$各300標本、calibrationと別seed |
| metrics | RZ count/depth、CX count/depth、total depth、circuit size |

異なるprocessでDF回路表現が非canonicalになる既知の問題を避けるため、Hamiltonian生成から
全calibration/holdoutまでを一つのprocessで実行した。

## 4. 結果

### 4.1 holdout予測誤差

各行の範囲は六つのcost metricにおける最小値から最大値である。

| holdout長 | モデル | 絶対相対誤差 | 最大absolute z-score |
|---:|---|---:|---:|
| 4 | event単純和 | 108.45--123.47% | 40.01 |
| 4 | pair補正 | 0.22--5.55% | 1.17 |
| 4 | triple補正 | 2.05--2.94% | 0.50 |
| 6 | event単純和 | 140.09--157.51% | 48.21 |
| 6 | pair補正 | 0.38--8.73% | 1.50 |
| 6 | triple補正 | 2.61--4.06% | 0.52 |

二つのholdoutをまとめた最大絶対相対誤差は、単純和157.51%、pair補正8.73%、
triple補正4.06%だった。最大absolute z-scoreはそれぞれ48.21、1.50、0.52だった。
単純和のずれは古典Monte Carloの標本不足では説明できず、局所境界補正によって大部分を
説明できた。

最長の $L=6$ について、代表的な指標の期待値は次のとおりである。

| metric | holdout一体compile | event単純和 | pair補正 | triple補正 |
|---|---:|---:|---:|---:|
| RZ count | 705.47 | 1735.44 | 767.04 | 734.13 |
| CX count | 212.39 | 509.92 | 228.95 | 220.42 |
| RZ depth | 132.52 | 341.27 | 133.67 | 136.54 |
| total depth | 295.25 | 755.82 | 296.37 | 302.96 |
| circuit size | 1382.72 | 3382.61 | 1501.26 | 1431.61 |

### 4.2 補正の内容

fragment確率から計算した、隣接する二eventが同じDF fragmentに属する確率は

$$
p_{\mathrm{same}}=\sum_f p_f^2=0.73106
$$

だった。最大の `df-fragment-3` だけで確率0.85055を占める。したがってこの条件では、
同じ基底変換が隣接してtranspilerにより回収される境界が高確率で生じる。

pair補正 $\mu_2$ は全metricで負で、$\mu_1$の約66.1--73.0%に相当する大きさだった。
一方、triple残差 $\mu_3$ は $\mu_1$の約$-3.09$--$+2.93$%で、全metricにおいて
推定標準誤差より小さかった。triple補正はcount/size指標の最大相対誤差を改善したが、
depth指標ではpair補正より一様に良くはならなかった。

## 5. 判断

1. event単純和は、event列が長くなるほど過大評価が拡大するため不採用を維持する。
2. expectation-level pair境界補正は、H4のこのholdoutでは全metricをabsolute z-score 1.50以内で
   覆い、次の最小cost model候補として残す。
3. 3-event clusterを超える効果がこの条件で必要だという統計的証拠は得られなかった。
   triple補正は診断用に保持するが、pair補正より常に優れるとは扱わない。
4. 受理閾値は未決定であり、$L_D,\delta,r,K$、compiler、結合制約およびcontrolled回路を
   変えた未使用条件を通るまでは、一般的なproxyとして採用しない。

### 5.1 1000標本・複数seedによる更新

同一Hamiltonian表現上の独立2 seedで、calibrationとholdoutを各1000標本へ増やした。
pair補正の最大絶対相対誤差は8.07%、8.18%、最大absolute z-scoreは2.96、3.02となり、
count/size指標の約8%残差が系統的であることを確認した。triple補正では最大誤差2.33%、
3.73%、最大z-score 0.79、0.97だった。

ただし、pair-onlyの不適合はcount/sizeに限られ、depth指標はpair補正でも最大z-score 1未満
だった。また、$\mu_3$自体のabsolute z-scoreは最大1.40である。したがって現判断を、
「全metricでpairを採用」から次へ更新する。

- depth：pair-onlyを引き続き候補とする。
- count/size：tripleを次の候補とするが、$C_3$の追加精度確認が必要。
- pair係数の構成：same/different-fragmentの二分類を用いる。

詳細は[fragment層別・高統計検証](rte_boundary_pair_validation.md)を参照する。

今回のcluster係数は、同一fragment境界だけを手で数えた値ではなく、全IID trajectoryを
一体compileした期待値から求めた。そのため、同一基底の相殺に加え、異なるfragment間の
transpiler最適化も平均として含む。

## 6. 成果物と再実行

- artifact：
  `artifacts/rte_boundary_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k0_v1.json`
- schema・検証実装：`src/trotterlib/rte_boundary_cost_validation.py`
- 生成script：`scripts/run_rte_boundary_cost_validation.py`
- 専用test：`tests/test_rte_boundary_cost_validation.py`

再実行：

```bash
.venv311/bin/python scripts/run_rte_boundary_cost_validation.py
```

artifactはversioned schema、SHA-256 fingerprint、生成command、source hash、compiler条件、
calibration/holdout seed、標準誤差および実行時間557秒を保存する。dirty local worktreeで
生成したlocal evidenceであり、immutable CI evidenceではない。
