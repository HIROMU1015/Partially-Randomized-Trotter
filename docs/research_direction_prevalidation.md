# 研究方向screening：WP00・WP02・WP01-S

最終更新：2026-09-21 JST

## 1. 目的と位置付け

144項目の事前検証候補を一括実行する前に、研究方向を変え得る最初の3段階を
`WP00 -> WP02 -> WP01-S`の順で実施した。対象はH4 linear chain、1.0 Å、STO-3G、
DF rank 12の固定Hamiltonian snapshotである。

WP01-Sは`model_conditional_screening`であり、最終総コスト評価でも、部分ランダム化の
最終的な優位性判定でもない。状態準備、noise、実backend実行および位相復元を含めず、
長い$q$の1 shot costには短$q$からの外挿を使う。

## 2. WP00：比較契約

比較条件を次のように固定した。

| 項目 | 固定値 |
|---|---|
| model | H4 linear chain、1.0 Å、STO-3G |
| DF | rank 12、8 qubit、4電子sector、次元70 |
| 候補 | $L_D=0,3,12$ |
| 主精度 | `CA/10`。CAと`CA/100`は感度scenario |
| $delta$候補 | 0.01、0.0125、0.02 |
| phase配分 | $(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.02,0.02,0.36)$ |
| failure配分 | $\alpha_{\rm total}=0.05$、候補内の全round・両軸へ一様配分 |
| 回路scope | 状態準備なしの1回のHadamard interrogation |
| control | ordinary controlled $\operatorname{diag}(I,U)$ |
| compiler | Qiskit 1.3.0、`rz,sx,x,cx`、optimization 1、seed 17、結合制約なし |
| 主指標 | 期待RZ count |

固定snapshotのHamiltonian hashは
`56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`である。
既存PF artifactのhashは`68a219c8...f60b6`でcost snapshotと一致しなかったため、
`L_D=0,3,12`のPF検証を同一snapshotから再生成した。$L_D=3$のD6係数は再生成前後で
相対$4.15\times10^{-9}$しか変わらないが、fingerprintで結合できないartifactを暗黙に
混ぜないための措置である。

`L_D=0`のPF artifact全体は、旧$C_D$ surrogateがnoise floorとなる判定を含むため
`overall_pass=false`である。一方、本screeningで使用するsame-snapshotの論文D6係数検証は
通過している。ここではD6係数を経験的入力として使い、厳密上界とは扱わない。

## 3. WP02：精度とround horizon

$\beta_{\rm RPE}=0.4$とし、

$$
\frac{\beta_{\rm RPE}}{2^M\delta}\leq\epsilon_E
$$

を満たす最小$M$を計算した。PF欄はsame-snapshotの$L_D=3$ D6係数
$C=0.0133991364$を用いる経験的$qC\delta^3$診断である。

| 精度 | $\delta$ | $M$ | $q_{\max}$ | PF phase proxy | 0.02 rad予算 |
|---|---:|---:|---:|---:|:---:|
| CA | 0.01 | 15 | 32,768 | $4.39\times10^{-4}$ | pass |
| CA | 0.0125 | 15 | 32,768 | $8.58\times10^{-4}$ | pass |
| CA | 0.02 | 14 | 16,384 | $1.76\times10^{-3}$ | pass |
| CA/10 | 0.01 | 18 | 262,144 | $3.51\times10^{-3}$ | pass |
| CA/10 | 0.0125 | 18 | 262,144 | $6.86\times10^{-3}$ | pass |
| CA/10 | 0.02 | 17 | 131,072 | $1.41\times10^{-2}$ | pass |
| CA/100 | 0.01 | 22 | 4,194,304 | $5.62\times10^{-2}$ | fail |
| CA/100 | 0.0125 | 21 | 2,097,152 | $5.49\times10^{-2}$ | fail |
| CA/100 | 0.02 | 21 | 2,097,152 | $2.25\times10^{-1}$ | fail |

CA/10では既存の3 scheduleと全56点のsector行列検査を再利用した。CAでは同じscheduleを
そのまま総コストへ使わず、短いhorizonに合わせたfailure配分・shot数の再計算が必要である。
CA/100は3つの既存$\delta$がすべて経験的PF予算を超えるため、長回路cost compileより先に
小さい$\delta$と新しいscheduleを作る必要がある。

## 4. WP01-S：小系の条件付き比較

各候補に3つの$\delta$を与え、roundごとの$(r_m,K_m)$を既存の成分作用数proxyで選んだ。
`L_D=3`では現れる7種類の$(r,K)$について、$\delta=0.02$、$q=1,2$の状態準備なし
Hadamard wrapperを各8 trajectoryで直接compileした。両軸・各metricにaffine modelをfitし、
全roundへ外挿した。$L_D=12$はtailなしの専用経路で$q=1,2$を厳密評価した。

### 4.1 $L_D=0$のkill-switch

$L_D=0$は$\lambda_R=13.7237$と大きいため、この候補だけ$r=1,\ldots,65536$、
$K=0,2,\ldots,16$へ探索域を広げた。成分作用数proxyの最良候補は$\delta=0.02$で、
最小半径下界0.5473、全roundの両軸shot合計25,400、shot重み付きランダム成分作用数は
$4.4839\times10^{12}$だった。選択された最大$r=16384$、最大$K=2$は探索境界に当たっていない。

同じ目的関数で見た$L_D=3$の$\delta=0.02$は$7.9980\times10^9$であり、$L_D=0$は
560.6倍となる。この固定taskの解析的screenでは逆転可能性が低いと判断し、compiled RZ costの
精密化前に除外した。ただし、これはcomponent-application proxyの比であってcompiled RZ cost比
ではなく、別のRTE構成、配分、精度に対する一般的・理論的な棄却でもない。

### 4.2 $L_D=3$対決定論endpoint

両候補とも$\delta=0.02$が点推定の最小となった。

| 候補 | 最良$\delta$ | no-prep総RZ点推定 | 較正SEの保守的95%半幅 | 5% scenario＋較正 | 25% scenario＋較正 |
|---|---:|---:|---:|---:|---:|
| $L_D=3$ | 0.02 | $3.0814\times10^{12}$ | $3.0997\times10^{11}$ | $[2.6174,3.5455]\times10^{12}$ | $[2.0011,4.1617]\times10^{12}$ |
| $L_D=12$ | 0.02 | $2.4329\times10^{12}$ | 0（厳密短$q$較正） | $[2.3112,2.5545]\times10^{12}$ | $[1.8246,3.0411]\times10^{12}$ |

点推定では$L_D=3/L_D=12=1.2666$で、決定論endpointが$L_D=3$より21.0%低い。
5% scenarioでは区間が分離するが、25% scenarioでは重なる。後者は未使用holdoutを持たない
schedule別$q=1,2$外挿、$\delta=0.02$から他$\delta$への移送、長$q$適用の不確かさを
意図的に広く見た感度幅であり、統計的信頼区間ではない。

したがってWP01-Sの判定は次の通りである。

- $L_D=0$はこの固定taskの拡張解析的component-application screenで強く不利。
- 点推定と狭いscenarioは決定論endpointを支持する。
- 保守scenarioでは$L_D=3$と$L_D=12$を分離できず、研究方向としては`undetermined`。
- 部分ランダム化の優位性は主張できない。次は予定通りWP04で配分・schedule寄与を分け、
  WP03でPF係数選択感度を調べる。

## 5. 成果物と再実行

主成果物は次の通りである。

- `artifacts/research_direction_prevalidation/2026-09-21/wp00_comparison_contract_v1.json`
- `artifacts/research_direction_prevalidation/2026-09-21/wp02_round_horizon_coverage_v1.json`
- `artifacts/research_direction_prevalidation/2026-09-21/wp01s_model_conditional_screening_v1.json`
- `artifacts/research_direction_prevalidation/2026-09-21/pf_delta_same_snapshot/`
- `artifacts/research_direction_prevalidation/2026-09-21/wp01s_calibrations/`

再実行は次を用いる。

```bash
.venv311/bin/python scripts/run_pf_delta_validation.py \
  --snapshot artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz \
  --n-electrons 4 --ld-values 0,3,12 \
  --output-directory artifacts/research_direction_prevalidation/2026-09-21/pf_delta_same_snapshot
.venv311/bin/python scripts/run_research_direction_prevalidation.py --stage all --samples 8
```

これらはdirty worktree上のlocal evidenceであり、immutable CIまたは外部再現結果ではない。
WP01-Sの長$q$値を最終総コストとして引用せず、WP05後のWP01-Dでdecision-gradeに再評価する。
変更後のlocal全test suiteは`499 passed, 4 warnings`だった。warningは既存grouped-UWC test由来である。
