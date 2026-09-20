# RPE短ラウンドの信号・shot・回路cost接続検証

## 目的

finite-RTE信号検証、RPE資源計上、直接compiled-cost評価を同じ条件で接続し、

1. 信号半径とfinite-RTE上界から得るshot数が一致するか、
2. 1 shot当たりcostとshot数の積がround costへ正しく入るか、
3. 古典Monte Carlo標本数を量子shot数へ誤って掛けていないか、
4. controlled time-evolution部分とHadamard interrogation全体のscopeを混同していないか、

を確認した。これは短ラウンドの接続検証であり、全RPE roundの総cost評価ではない。

## 固定条件

- H4鎖、原子間距離1.0 Å、STO-3G、8 qubit。
- 保存した同一DF Hamiltonian snapshot、DF rank 12、$L_D=3$。
- partial-$S_2$時間幅$delta=0.1$、RTE short-step数$r=4$、有限Taylor cutoff $K=2$。
- $q_m=1,2,4$（$m=0,1,2$）。
- 暫定位相配分
  $(\beta_{\mathrm{PF}},\beta_{\mathrm{RTE}},\beta_{\mathrm{stat}})
  =(0.08,0.08,0.24)$ rad、$\beta_{\mathrm{RPE}}=0.40$ rad。
- $\alpha_{\mathrm{tot}}=0.05$を3 round・2軸へ一様配分し、
  $\alpha_{m,b}=0.05/6$。
- PF係数はH4の既存Eq. (D6)経験値$C=0.01342567$を固定入力とした。
  本検証でこの係数を再評価したわけではない。
- Qiskit 1.3.0、basis gates `rz,sx,x,cx`、optimization level 1、seed 17、
  coupling mapなし。
- compiled-costの古典Monte Carloは各$q_m$で8 trajectory。これは接続検査用であり、
  cost期待値を高精度に再評価する標本数ではない。

## 比較した二つの回路scope

| scope | 含むもの | 含まないもの |
|---|---|---|
| `compiled_time_evolution_subcircuit` | ordinary controlled repeated partial-$S_2$ | Hadamard軸回転、測定、状態準備、backend実行 |
| `single_hadamard_interrogation_without_state_preparation` | 上記時間発展、ancilla Hadamard、軸変更、測定 | 状態準備、backend実行、複数shot実行、全roundの位相復元 |

両scopeは同じRTE seed・古典標本数で評価した。scopeを分けた結果は、Hadamard側の
1 shot当たりcostとして後者を資源計上へ使う必要があることを確認するためのものである。

## 結果

### 信号半径とshot数

| $q_m$ | 物理full-$H$基底状態でのPF信号半径 | 単位半径仮定の各軸shot | 実半径を用いた各軸shot |
|---:|---:|---:|---:|
| 1 | 0.9999999992805422 | 389 | 389 |
| 2 | 0.9999999971442701 | 390 | 390 |
| 4 | 0.9999999889235008 | 391 | 391 |

信号半径の1からのずれは最大$1.11\times10^{-8}$で、この3 roundでは
$\rho_{\star,m}=1$というspecializationを実半径へ置き換えても整数shot数とround costは変わらなかった。
これはH4の明示条件での結果であり、一般に$\rho_{\star,m}=1$を保証するものではない。

### 回路scope差とRZ cost

| $q_m$ | time-evolutionのみ | cosine Hadamard | sine Hadamard | 各軸の増分 |
|---:|---:|---:|---:|---:|
| 1 | 6261.875 | 6263.875 | 6263.875 | +2 RZ |
| 2 | 12281.25 | 12283.25 | 12283.25 | +2 RZ |
| 4 | 24007.5 | 24009.5 | 24009.5 | +2 RZ |

回路sizeは全$q_m$で+5、CX count/depth、RZ depth、total depthはこのcompiler条件で
同じだった。Hadamard wrapperの増分は小さいが、scopeは明示的に異なるため、理論上も実装上も
同一costとは扱わない。

### 接続整合性

全$q_m$で次を通過した。

- finite-RTEのoperator・signal・radius・phase上界検査。
- 信号検証側の単位半径shot数とresource-accounting側のshot数の一致。
- time-evolution providerとHadamard providerの$\epsilon_Z$、attenuation、shot数の一致。
- `round_cost = N_c g_c + N_s g_s`の直接再計算との一致。
- 古典Monte Carlo標本数8を上式へ追加で乗算していないこと。
- 二つの回路scopeが期待した識別子で保存されていること。

この条件で得たHadamard interrogationのRZ round costは$q=1,2,4$でそれぞれ
4,873,294.75、9,580,935、18,775,429だった。ただし8 trajectoryの点推定であり、
候補間の精密なcost順位または全RPE総costとしては用いない。

## 結論と次段階

短ラウンドでは、既存のfinite-RTE近似からshot数を求め、状態準備を除くHadamard
interrogationの1 shot当たりcompiled costを掛ける接続経路が整合した。H4のこの条件では
単位信号半径specializationの影響は整数shot数に現れなかった。

未検証なのは次である。

- $q_m>4$のHadamard cost proxyをresource accountingへ接続すること。
- 全roundを合計し、誤差・失敗確率配分を変えて比較すること。
- 全roundのbranch selectionと最終位相復元を含むend-to-end成功率。
- 状態準備、実backend実行、ノイズ、最終的な部分ランダム化優位性。

同じ短ラウンド条件の仮想Hadamard測定は、その後の
[失敗確率検証](rpe_hadamard_failure_validation.md)で実施した。厳密二項確率、10万回の
周辺Bernoulli反復、shotごとのfresh IID trajectory検査が暫定$\alpha$配分と整合した。
次はlong-$q$ proxy接続と$\beta,\alpha$配分比較へ進む。

## 成果物

- 結果：
  `artifacts/rpe_round_cost_connection_validation/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_mc8_v1.json`
- 実装：`src/trotterlib/rpe_round_cost_connection_validation.py`
- 生成：`scripts/run_rpe_round_cost_connection_validation.py`
- test：`tests/test_rpe_round_cost_connection_validation.py`

生成コマンド：

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv311/bin/python \
  scripts/run_rpe_round_cost_connection_validation.py
```

成果物はlocal worktree上の検証結果であり、immutable CIまたは外部再現結果ではない。
