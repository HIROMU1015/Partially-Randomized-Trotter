# 現行Full-H R448固定構成の確認

対象はC公開 `367dc98d832ecb4d769f2a3da3d174bc697331a5` の `FULL_H_K2_QDYADIC_R448`（configuration `cfa6aee3260de664de9acd444a628faae388c459a2eaf68a786c707573029d1c`）。保存PLAN/round/REPORT/SCOPEとSOURCEを読取確認した。数値は保存値の転記、またはq・R・λ・Tのmetadata算術だけで、signal・価格の再解析はしていない。

## 物理構成

| 項目 | 内容 | SOURCE |
|---|---|---|
| ランダム化範囲 | one-body（one-body固有基底のZ 8成分）＋DF二体10 fragment全部の対角成分。非identity成分368、決定論block0、head0 | `9b83b96b:scripts/h46_full_h_pure_rte.py:prepare_full_h` |
| one-bodyの扱い | `H_one=(Σε)/2·I−Σ(ε_j/2)UZ_jU†`。非scalar Zを保存DF tail末尾へ追加。fragment順・成分順・閾値0は不変 | 同上 |
| identity抽出 | 抽出scalar=tail identity＋Σε/2 = −2.6245790607368162、Hamiltonian定数2.29310124732、合計−0.33147781341681615 | 9b83 `input_and_validation_summary.json` |
| scalar適用 | 総Tで一回。signalは `exp(−iT(c+s))`、nativeはancillaの `p(−scalar·T)` | `baseline_signal_numerics.evaluate`、`h46_long_time_native.compose` |
| λ | λ_R=21.02794414280327（one-body分7.30424549435506を含む全368成分のL1）。確率 p_j=|a_j|/λ_R | 9b83 summary `H4.lambda_R` |
| K | finite K2：偶数次数{0,2}、paired numerator次数K+1=3、重み `τ^k/k!·√(1+(τ/(k+1))²)` | `71ba49bb:src/trotterlib/rte.py:finite_rte_distribution` |
| grouping | 各round q=2^m（RPEのq_RPEと同じ値）、group内R=448。**qはgroupingでありqPFではない**（B側q_PF=null） | B config rounds、D `layout_mapping` |
| 総event | s=qR=448·2^m、event時間dt=(T/q)/R、τ=λ_R·dt。T=0.1·2^mなのでτ=λ_R·0.1/448≈4.69e−3が全roundで一定 | D `physical_schedule` |
| 正規化 | 1 eventのb(τ)=Σ_k w_k、logB=qR·log b。Raw=corrected·e^{−logB}をcorrected/rawの独立伝播で計算（closure 1e−30台） | `evaluate`（PURE_FULL_H_GROUPED_SPECTRAL_POWER） |
| law | `ideal_finite_RTE_law`、`PAIRED_FINITE_TAYLOR_COMPONENT_ORDER_LAW`。PCG64/CDF samplerとの距離は未資格 | B law binding |
| profile | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、compiler hash `1239304c…`、control `symmetric_directional`、測定付きfull Hadamard X/Y、P0 | spec.profile |
| RTE lowering | 旧Full-H価格はprovider c31ef424の `RZZGate.control(1)`。R448自体のcompiled価格は**未取得** | D S1 `3d39f8c0` |

PR LD0やB3とは異なる。B3はone-bodyを決定論側に残し（H4 B3保存最良のλ_R=13.72）、PR prefix0もPRのsplit規約に従う。Full-Hのpreparationは `prepare_full_h` 由来に限り、B controllerとD adapterはone-body非randomizeやB3 aliasを拒否する（`B.generate`、`D.check_full_h_preparation`）。

## R448の採択経緯

1. T=2.0最適化 `7a8bc7f1`：H4 T2.0固定でq∈{1..64}×R∈{4..512}＋局所R、K∈{0,2,4,6}を探索。旧高確率型shot会計（u floor1e−9、α_axis.025）・旧lowering・n9価格で、ε≤.005の取得域最良がq8/R448/K2（s=3584）。R512とのn33独立確認でG比0.987。
2. c546のbaseline計画 `c5466946`：Full-HをPR/Full PF/B0Iと同じ13roundへ接続する「単純な接続seed」として、**R=448/groupだけを流用しq=2^m**とした。REPORTはS2・p7・B0I・R448を最適又は必要十分としないと明記している。
3. その結果、group時間はT2.0の0.25から0.1に変わり、単位時間あたりevent密度は1792から4480に上がった。T2.0最適点のκ_eff≈2.03を各roundで保つ構成ではない。

## 保存結果（`367dc98d` SIGNALS.csv転記）

|m|T|s=qR|κ_eff=s/(λT)²|(λT)²/s|保存logB|保存B|保存attenuation|
|---:|---:|---:|---:|---:|---:|---:|---:|
|0|0.1|448|101.3|0.00987|0.00987|1.0099|0.990|
|3|0.8|3,584|12.66|0.0790|0.0790|1.082|0.924|
|5|3.2|14,336|3.166|0.316|0.316|1.371|0.729|
|6|6.4|28,672|1.583|0.632|0.632|1.881|0.532|
|7|12.8|57,344|0.792|1.263|1.263|3.54|0.283|
|9|51.2|229,376|0.198|5.053|5.053|156.5|6.39e−3|
|10|102.4|458,752|0.0989|10.11|10.11|2.45e4|4.08e−5|
|12|409.6|1,835,008|0.0247|40.43|40.43|3.61e17|2.77e−18|

κ_effと(λT)²/sは本調査のmetadata算術（[KAPPA_CANDIDATES.json](KAPPA_CANDIDATES.json) `R448_schedule`）。他roundは同ファイルとSIGNALS.csvを参照。

## 問題点と有効範囲

- **固定R・q∝Tなのでs∝T、logB∝T。** κ_effはm0の101からm12の0.025へ単調に下がり、m5とm6の間でκ=2を横切る。m0〜m5は必要以上のeventで回路を長くし、m7以降は減衰が指数的に拡大する。
- 13 meanはこのR448構成の**ideal finite-K law平均として数値gateを通過した有効な値**である。R448構成のbaseline、grouped spectral power経路の動作確認、「固定R scheduleは長時間roundで不適」という観察には使える。
- m9以降のRawは減衰6.4e−3以下、m12は2.8e−18。数値計算として得られていても、B²に比例するshot数のもとではRPE推定に実用的でない。RMSE/N/価格の資格はなく、**Full-H pure RTE自体の最適性能や劣位を示す結果ではない**。
- κ型の各候補とphysical IDは共有しない（[KAPPA_CANDIDATES.json](KAPPA_CANDIDATES.json) の `shared_physical_ids_with_R448` は全て空）。R448のsignal・logB・将来の価格をκ候補へ流用できない。
