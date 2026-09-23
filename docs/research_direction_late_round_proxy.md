# G08 / M08 後半round proxy精度検証

## 目的

WP01-D/C07のlocal 5%区間は僅かに分離したが、対称model discrepancyの分離限界は
5.0484%で、採用した5%との差は0.0484 percentage pointしかない。G08で全roundのcost・
proxy不確かさ・PF/RTE riskを分解し、M08で判断を変え得る最小限の$q>8$ holdoutを実行する。

## G08結果

既存のWP01-D/C07 schema-v2 computeと判断統合artifactだけを再集計した。新しい回路compileは
行っていない。

| 候補 | 最後3 roundのcost割合 | 最後3 roundの較正不確かさ割合 | 最大cost round | 最大RTE-risk round |
|---|---:|---:|---:|---:|
| $L_D=3$ | 90.94% | 87.70% | 17 | 7 |
| $L_D=12$ | 83.03% | 0% | 17 | 該当なし |

$L_D=3$の最後3 roundは全て$r=32$で、$q=32768,65536,131072$である。最大cost/PF-riskは
最終roundだが、最大finite-RTE riskはround 7であり、costとfailure-riskの最大点は一致しない。
従ってM08は後半$r=32$ proxyの精度に集中する一方、RTE feasibilityは既存round 7検査を維持する。

G08 artifactは
`artifacts/research_direction_round_dominance/2026-09-22/g08_round_cost_risk_proxy_dominance_v1.json`、
fingerprintは`e696ced27b06e871368f3afa164f507c240d4aa6693223f9f6abe7990b30d064`である。

## M08結果

実scheduleの$q$をそのまま直接transpileせず、まず既存$q=1,2$ affine式を変更せずに未使用
$q=16,32$を直接holdoutする。

| 項目 | 条件 |
|---|---|
| 物理系 | H4、1.0 Å、STO-3G、8 qubit、DF rank 12、$L_D=3$ |
| 回路 | `support_run_le_1`とfull basisのcomplete controlled Hadamard wrapper |
| 条件 | $\delta=0.02,r=32,K=2,q=16,32$ |
| 標本 | 各$q$ 8 fresh trajectory、cosine/sine両軸 |
| 判定 | selected RZ・全metric、full-basis RZが5%以内、selected RZが5.0484%分離限界以内 |
| 精度診断 | direct RZ relative SEが2%以内 |
| compiler | Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなし |

全64本のmeasurement-bearing wrapperを直接transpileした。M08 artifactの全checkは通過した。

| 指標 | 最大相対誤差 | 判定 |
|---|---:|---|
| selected policy RZ | 2.466% | 5%および5.0484%分離限界以内 |
| selected policy 全metric | 2.569% | 5%以内 |
| full basis RZ | 3.286% | 5%以内 |
| direct RZ relative SE | 1.353% | 2%以内 |

事前規則による$q=64$ follow-upは発火しなかった。M08 artifactは
`artifacts/research_direction_proxy_precision/2026-09-22/m08_late_round_q16_q32_proxy_precision_v1.json`、
fingerprintは`e010a63bfa5aecd7de01ba074f56300615460de9e6f51a40dca352954992aaf8`である。

## M08測定値によるWP01-D/C07再集計

点推定、最適化されたschedule、shot数、較正95% half-widthは変更せず、M08測定値を両候補に
共通の対称model-discrepancy幅として置く反実仮想区間を追加した。これは$L_D=12$をM08で直接
測定したという意味ではなく、同じ許容幅で候補比較を再集計するscenarioである。

| scenario | $L_D=3$区間（RZ） | $L_D=12$区間（RZ） | 結果 |
|---|---:|---:|---|
| M08 selected RZ 2.466% | $[1.34344,1.56815]\times10^{12}$ | $[1.64943,1.73282]\times10^{12}$ | 分離 |
| M08観測RZ最大 3.286% | $[1.33149,1.58010]\times10^{12}$ | $[1.63555,1.74670]\times10^{12}$ | 分離 |
| 従来local 5% | $[1.30654,1.60504]\times10^{12}$ | $[1.60657,1.77568]\times10^{12}$ | 僅かに分離 |
| transfer 25% | $[1.01538,1.89620]\times10^{12}$ | $[1.26834,2.11390]\times10^{12}$ | 重なる |

M08実測domain内では$L_D=3$区間が低いというlocal判断が補強された。ただし直接検証domainは
$q\leq32$であり、実scheduleは$q=131072$まで達する。従って2.466%または3.286%を長$q$へ
無条件に移送せず、25%移送感度を残す。頑健な方向判断は
`undetermined_under_transfer_sensitivity`のままで、部分ランダム化の科学的優位性や最終総costを
示さない。次はlocal $q$精度の追加ではなく、主張範囲の見直しまたは外部条件での移送検証である。

再集計artifactは
`artifacts/research_direction_proxy_precision/2026-09-22/wp01d_c07_m08_measured_discrepancy_reaggregation_v1.json`、
fingerprintは`6aa69bc756a97aa025e26994f596f9ee3df999be2e64be870ce938dca5180b3e`である。

## 再生成

- G08 runner：`scripts/run_research_direction_round_dominance.py`
- M08 runner：`scripts/run_research_direction_proxy_precision.py`
- 再集計runner：`scripts/run_research_direction_m08_reaggregation.py`
- tests：`tests/test_research_direction_round_dominance.py`、
  `tests/test_research_direction_proxy_precision.py`、
  `tests/test_research_direction_m08_reaggregation.py`

成果物はlocal dirty-worktree evidenceであり、immutable CIまたは外部再現ではない。変更後の
local全suiteは`535 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。
