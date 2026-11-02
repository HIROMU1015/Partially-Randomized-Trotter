# WP06-b sequence-aware basis policyとproxy再較正

## 目的と事前規則

WP06-aではsupport限定basisが単発Z/ZZを大きく削減した一方、同一basisの長さ3列ではfull basis共有より
RZが15.0%増えた。そこでWP06-bでは、元のDF basisが連続する最大runを選択単位とし、run長が固定閾値
以下のときだけsupport限定basisを使うpolicyを比較した。production RTE builderには明示basis planを
受け取る経路を追加したが、既定のfull-basis policyは変更していない。

policyは独立training列だけで固定した。選択規則は、各training trajectoryのRZがfull basisより5%を超えて
悪化しない候補に限定し、pooled RZ、CX、total depthの順で辞書式に最小化するものとした。holdoutを見た
policy変更、trajectoryごとの事後oracle選択、最終総costへの直接採用は行わない。

## 条件

| 項目 | 条件 |
|---|---|
| 物理系 | H4直鎖、1.0 Å、STO-3G、8 qubit |
| DF・分割 | rank 12、$L_D=3$ |
| RTE | $\delta=0.02$、$K=2$ |
| compiler | Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなし |
| training | event列長1, 2, 4、各8本、master seed 2026092201 |
| holdout | 未使用列長3, 6、各12本、master seed 2026092202 |
| schedule transfer | $r=1,2,4,8,16,32$、各8本、master seed 2026092203 |
| 同値性 | sampled列長1, 3と、強制$K=2$非零phase event |

eventは固定snapshotの有限RTE物理分布から生成した。training、holdout、schedule transfer、同値性probeの
seedは相互に分離している。support completionは必要なorbital列を保存して構成し、324個の
basis-support組を登録した。保存列の最大残差は浮動小数点精度内だった。

## policy選択

比較したのはfull共有、run長1以下、2以下、3以下でsupport限定、ならびに全runをsupport限定とする
5候補である。

| policy | training RZ合計 | CX合計 | total depth合計 | trajectory単位5% guard |
|---|---:|---:|---:|---|
| full basis共有 | 11,944 | 3,862 | 5,749 | pass |
| support if run $\leq1$ | 9,590 | 3,254 | 5,550 | pass |
| support if run $\leq2$ | 9,114 | 3,220 | 6,617 | fail、最大+7.90% RZ |
| support if run $\leq3$ | 9,065 | 3,216 | 6,811 | fail、最大+7.90% RZ |
| support for all runs | 10,504 | 3,716 | 8,235 | fail、最大+99.78% RZ |

従って、**同一元basisのsingleton runだけsupport限定へ置換し、run長2以上はfull basisを共有する**
`support_run_le_1`を固定した。これは各trajectoryをcompile後に選び直すoracleではなく、列を見る前から
適用できるrun規則である。

## 独立holdout

未使用の列長3, 6をまとめると、固定policyはfull共有に対して次の変化だった。

| metric | full平均 | policy平均 | 相対変化 |
|---|---:|---:|---:|
| RZ count | 609.04 | 544.08 | -10.67% |
| CX count | 211.08 | 193.92 | -8.13% |
| RZ depth | 145.46 | 142.42 | -2.09% |
| CX depth | 107.46 | 105.88 | -1.47% |
| total depth | 320.88 | 313.67 | -2.25% |
| circuit size | 1,221.29 | 1,091.42 | -10.63% |

列長3ではsingleton runが現れず差は0、列長6ではRZ -16.50%、CX -12.46%、total depth -3.40%だった。
固定policyのtrajectory別oracleに対するpooled RZ regretはfull RZ基準で0.390%であり、事前の5%基準内だった。

controlled演算子は、物理分布からの列長1, 3に加え、通常samplingでは極めて出にくい$K=2$の
非零Taylor phase eventを強制して比較した。relative ancilla phaseを含む最大operator残差は
$1.34\times10^{-15}$で、判定値$10^{-10}$を通過した。強制eventではfull、選択policy、support-allの
relative phaseがすべて$3\pi$で一致した。

## schedule移送とadditive proxy bridge

独立streamで既存scheduleの$r=1,2,4,8,16,32$へ移送した。中央controlled RTE occurrenceのpooled変化は
RZ -20.33%、CX -15.20%、total depth -6.30%だった。ただし既存Hadamard proxyには決定論blockとwrapperも
含まれるため、中央RTEの差を$q$ slopeへ加えるだけのadditive bridgeとした。

| $r$ | 旧RZ $q$ slope | 中央RTE差 | bridge後slope | 相対変化 |
|---:|---:|---:|---:|---:|
| 1 | 5,641.0 | -144.38 | 5,496.63 | -2.56% |
| 2 | 5,542.63 | -70.75 | 5,471.88 | -1.28% |
| 4 | 6,045.25 | -63.00 | 5,982.25 | -1.04% |
| 8 | 6,730.50 | -236.88 | 6,493.62 | -3.52% |
| 16 | 7,309.25 | -420.88 | 6,888.38 | -5.76% |
| 32 | 9,739.0 | -779.00 | 8,960.0 | -8.00% |

最大変化は8.00%で、WP06-aの5% triggerを超えた。proxyのaffine式と$r$ domainは維持するが、係数は
WP05でfull wrapperを再transpileして更新する必要がある。

## $L_D=3/12$順位への限定的な影響

WP04のround、shot、$\alpha$配分、wrapper interceptを固定し、中央RTE slopeだけを差し替えたbridgeでは、
$L_D=3$のRZ点推定は$1.7848\times10^{12}$から$1.6503\times10^{12}$へ7.54%低下した。
$L_D=12$は$1.6963\times10^{12}$のままで、点順位は$L_D=12$優勢から$L_D=3$優勢へ反転した。

一方、補正のsampling不確かさを加えた$L_D=3$のlocal区間
$[1.3137,1.9868]\times10^{12}$は、$L_D=12$の
$[1.6115,1.7811]\times10^{12}$と重なる。さらにfull wrapperのtranspile、shot数、$\alpha$配分を再最適化
していないため、これは順位感度を示すbridgeであり、科学的優位性またはdecision-grade総costではない。

## 判断

`support_run_le_1`は独立holdoutとcontrolled同値性を通過したため、WP05の明示入力として採用する。
production builderの既定値はfull basisのままとし、旧proxyを確定値として上書きしない。次はWP05で、
選択policyを含むcomplete controlled partial-$S_2$／Hadamard interrogationを$q=1,2$以上で直接transpileし、
additive bridge、wrapper境界効果、RZ順位反転を検証する。

## 後続WP05-aの結果

WP05-aではこのpolicyをcomplete controlled partial-$S_2$と測定付きHadamard wrapperへ明示伝播した。
$r=1,2,4,8,16,32$の$q=1,2$較正から、独立$q=4$のRZを最大2.29%、全6 metricを最大2.43%で
予測した。中央RTE additive bridgeのfull-wrapper RZ残差も最大2.63%で5%基準を通過した。
固定WP04条件の点順位は$L_D=3$となったが区間は重なる。詳細と制限は
[WP05-a full-scope接続](research_direction_full_scope.md)を参照する。次は$q=8$と$\delta=0.01$の
WP05-bだった。その後WP05-b/RとWP01-D/C07まで完了し、$L_D=3$の点推定は13.92%低くなったが、
25%移送区間は重なった。頑健な方向判断は未確定であり、次はM08/G08で支配的な後半roundの
$q>8$ proxy精度を定量化する。production既定値は引き続きfull basisである。詳細は
[WP05-b/R](research_direction_full_scope_extension.md)と
[WP01-D/C07](research_direction_decision_cost.md)を参照する。

## 成果物と再生成

- artifact：`artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json`
- runner：`scripts/run_research_direction_sequence_policy.py`
- test：`tests/test_research_direction_sequence_policy.py`

```bash
.venv311/bin/python scripts/run_research_direction_sequence_policy.py
.venv311/bin/python -m pytest -q tests/test_research_direction_sequence_policy.py
```

専用testは`5 passed`、変更後のlocal全suiteは`520 passed, 4 warnings`だった。warningは既存の
grouped-UWC test由来である。

artifactはsnapshot、WP06-a、WP04、6個の旧proxy、生成sourceのSHA-256を記録するlocal dirty-worktree
evidenceであり、immutable CIまたは外部再現結果ではない。
