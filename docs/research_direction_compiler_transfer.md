# M06/L08 compiler-transfer解析・再集計

## 1. 目的

WP01-D/C07で得た候補差がQiskit transpilerのoptimization level 1に固有かを確認するため、
同一trajectoryをoptimization level 2で再compileした。検証対象はM06のcompiler間proxy移送と
L08のcompiler条件感度である。

これは最終総costの再最適化ではない。optimization level 2の直接計算がある範囲だけを置換する
focused感度分析と、未測定roundへ比率を一様移送する反実仮想を分離して記録する。

## 2. 固定条件と変更因子

- H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12
- 候補：`L_D=3`の`support_run_le_1`とtail-free `L_D=12`
- `delta_time=0.02`、`L_D=3`は`r=32,K=2`
- cosine/sine Hadamard wrapper、状態準備なし、RZ countを主指標
- basis gates `rz,sx,x,cx`、coupling mapなし、transpiler seed 17、Qiskit 1.3.0
- 変更因子はoptimization level 1から2だけ
- `L_D=3`のq=1,2はWP05-bRと同じ32 trajectory、q=16,32はM08と同じ8 trajectory
- `L_D=12`はq=1,2,16,32の決定論回路

## 3. proxy検証

optimization level 2のq=1,2 affine式を固定し、q=16,32を直接holdoutした。

| 診断 | 最大値 | 判定 |
|---|---:|---|
| selected policy RZ相対誤差 | 2.340% | 5%以内 |
| selected policy全metric相対誤差 | 2.470% | 5%以内 |
| full-basis RZ相対誤差 | 3.420% | 5%以内 |
| full-basis全metric相対誤差 | 3.594% | 5%以内 |
| direct RZ relative SE（両policy最大） | 1.387% | 2%以内 |
| tail-free `L_D=12`全metric誤差 | 0% | affine exact |

従って、`r=32,q<=32`の範囲ではoptimization level 2でもproxy精度基準を通過した。
これはq>32または`L_D=3,r<32`の直接検証を意味しない。

## 4. 同一trajectoryでのcompiler効果

selected policyのRZ countはoptimization level 1比で18.20--18.28%減り、同じ4個のqと両軸を
平均したlevel-2/level-1比は0.817588だった。full basisのRZ減少は17.92--18.19%である。
一方、全metricをまとめるとCX等には最大+0.857%の変化もあり、単一の削減率を全指標へ
一般化してはならない。

tail-free `L_D=12`の固定plan再集計では、RZ点推定が
`1.691123354272e12`から`1.327822299806e12`へ21.48%減った。compiler最適化の影響は両候補で
同じではない。

## 5. focused固定plan再集計

WP01-D/C07で選択したround schedule、shot数、alpha、betaを固定した。`L_D=3`は直接証拠のある
`r=32`の最後3 roundだけをoptimization level 2へ置換し、`r<32`はlevel 1の値を保持した。
`L_D=12`は決定論的affine provider全体をlevel 2へ置換した。

| 項目 | `L_D=3` | `L_D=12` |
|---|---:|---:|
| 元の点推定 | 1.455792e12 | 1.691123e12 |
| focused再集計点推定 | 1.215990e12 | 1.327822e12 |
| 元値からの変化 | -16.47% | -21.48% |
| calibration 95% half-width | 6.398093e10 | 0 |

点推定では`L_D=3`が8.42%低い。しかし、対称discrepancyを置いて区間分離できる上限は
1.881%であり、optimization level 2で実測したselected RZ discrepancy 2.340%を下回った。

| 対称discrepancy scenario | `L_D=3`区間 | `L_D=12`区間 | 判定 |
|---|---:|---:|---|
| selected実測 2.340% | [1.123560e12, 1.308421e12] | [1.296757e12, 1.358888e12] | 重なる |
| 観測RZ最大 3.420% | [1.110418e12, 1.321562e12] | [1.282406e12, 1.373239e12] | 重なる |
| local 5% | [1.091210e12, 1.340771e12] | [1.261431e12, 1.394213e12] | 重なる |
| transfer 25% | [0.848012e12, 1.583969e12] | [0.995867e12, 1.659778e12] | 重なる |

したがって、WP01-D/C07で見えたlocal区間分離はcompilerを変更しても保たれるとは確認できない。
focused結果は`L_D=3,r<32`をlevel 1のまま残すため、完全なoptimization-level-2候補比較でもない。

## 6. 一様比率移送の反実仮想

selected RZの平均比0.817588を`L_D=3`の全roundへ一様に移すと、点推定差は10.36%、分離上限は
2.981%となる。この仮定ではselected実測2.340%の区間だけは分離するが、観測RZ最大3.420%、
5%、25%では重なる。未測定`r<32`への比率移送とdiscrepancyの選び方で判定が変わるため、
このケースは直接証拠でなく感度用の反実仮想である。

## 7. 判断

- optimization level 2でも`r=32,q<=32`のaffine proxy adequacyは通過した。
- 固定plan点推定では`L_D=3`を選ぶが、compilerをまたいだlocal区間分離は確立しない。
- 頑健な方向判断は`undetermined_under_compiler_and_transfer_sensitivity`とする。
- 次は大きな外部instance計算ではなく、N07/P03相当の主張範囲とbreak-even条件を整理する。
- full optimization-level-2固有の順位が必要になった場合だけ、`r<32`の再較正と候補別の
  schedule・shot・alpha・beta再最適化を別検証として行う。

状態準備、実backend、noise、coupling map、別snapshot・系サイズ、q>32の直接compile、最終総cost、
科学的優位性は本検証に含まない。

## 8. 証拠

- raw compute：
  `artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_same_trajectory_compute_v1.json`
  （fingerprint `87f66c8944dedfaa9fe5f0edf864d2a2a07cb82e23245af4f5944f90bda83ad6`）
- 解析・再集計：
  `artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_focused_analysis_reaggregation_v1.json`
  （fingerprint `7ebe8815a633a182b4afa4608b969df5c53d1b6d1efe6eeaf2ce4e653fcbbd95`）
- 実装：`src/trotterlib/research_direction_compiler_transfer_compute.py`、
  `src/trotterlib/research_direction_compiler_transfer_analysis.py`
- runner：`scripts/run_research_direction_compiler_transfer_compute.py`、
  `scripts/run_research_direction_compiler_transfer_analysis.py`
- 専用test：`tests/test_research_direction_compiler_transfer_compute.py`、
  `tests/test_research_direction_compiler_transfer_analysis.py`

専用testは3件、変更後のlocal全suiteは538 passed, 4 warnings（既存grouped-UWC由来）である。
証拠statusはlocal dirty-worktreeであり、immutable CIまたは外部再現ではない。
