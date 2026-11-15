# B0I p7/S2：同一shot配分での要求幅0.1と16h/7のQ3精度比較

2026-11-15 JST。要求幅0.1を**新規1 call・retryなし**で評価し、比較先16h/7は保存結果を再利用した。両幅とも採択済みQ3モデル閾値に適合し、幅0.1ではMSEと閾値までの余裕が改善した。これは固定配分の精度比較であり、幅0.1の費用比較ではない。

## MSEと閾値までの余裕

X/Y各軸、m0〜m12の固定整数配分：

~~~text
[59,55,51,47,43,39,35,31,27,23,19,13,4]
~~~

exact閾値は τ = (0.00159360010199040 Ha)² = **0.0000025395612850638132820416921600 Ha²**。原APIのexact有理数で判定・区間演算し、以下の小数端点は表示用に丸めた。原値は[EXACT_COMPARISON.json](EXACT_COMPARISON.json)、表示要約は[SUMMARY.json](SUMMARY.json)。

| 要求幅 | 直接モデルMSE区間（Ha²、表示） | Q3判定 | τ−MSEの区間（Ha²、表示） |
|---|---|---|---|
| **0.1＝h** | [1.113188114132502061033934E−6, 1.113188114132502061033945E−6] | 適合 | [1.426373170931311221007747E−6, 1.426373170931311221007758E−6] |
| **16h/7** | [1.506660616251764083585358E−6, 1.506660616251764083585362E−6] | 適合（保存結果） | [1.032900668812049198456330E−6, 1.032900668812049198456334E−6] |

**MSE差（幅h − 幅16h/7）**の区間は
**[−3.934725021192620225514282E−7, −3.934725021192620225514126E−7] Ha²**。
区間全体が負なので、同じ配分では要求幅hの方がMSEが小さい。閾値までの余裕は同じ大きさの正の区間だけ増えた。これは今回の2幅・1配分についての結果であり、幅全体の単調性や最適性は推定しない。

| 診断（Ha²、区間の表示概数） | 幅h | 幅16h/7（保存） |
|---|---:|---:|
| bias² | 5.760665952744145E−9 | 3.743142061942309E−7 |
| variance | 1.107427448179758E−6 | 1.132346410057533E−6 |

bias²・Varは同じtotal推定量の診断であり、新たな誤差資格ではない。AのδEやδE²は加算していない。

## SOURCE・物理条件と既存処理の再利用

- 実行前にcaller・[INPUTS](INPUTS.json)・人工検査を **650017fa1c6d3f408f34e5eb391d038e3b3c1475** にcommit/pushして固定。callerは `scripts/run_p7_s2_d01_same_shots_q3_fixed.py`、既存PF intake/total binding・Q3 kernel・高速DP・worker保存経路を再利用した。
- 幅hの元configurationは **e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc**、total viewは **b8ceb77562c4648f2ae51d7beebdc4069edb0f273d316dfcaee500d1d33015fb**。保存packet/PLANは **5fb2c32e39e63267ea1748861269c362a93a1b86** の `docs/research/truncated_fullh_signal_results_20261114/run01/B0I_P7_S2_D01/`。元signal producerは5a172629db4d32a4230f8bb10a871b42807b1cd5、numerical SOURCEは5cc7c07e112ce97205948465c94441cba0042416。
- 比較先は **b4296b110b129d226dbef2743c14f725a27c3bf6** の `docs/research/pf6_cost_budget_q3_results_20261115/run01/B0I_P7_S2_16h_over_7__B_PRICE/API_RETURN.json`。その実行SOURCEは06f4210e1bdfd5b07f42b941fa22d17e228533fc、元configurationはa6e557770a2e4d96585ad5ca5903c4626ae47dda9775e9a9babc756149c38d84。
- 元configuration.specは要求幅delta_pf以外が完全一致。H/state/snapshot、p7/S2、RPE h=0.1/M12、prior/decoder、target/lawを照合。幅hの固有schedule qPF=[1,2,…,4096]・各round実幅0.1を使い、比較先scheduleへ付け替えていない。13roundの元physical IDとallocationは[C_RESULT_JOIN](C_RESULT_JOIN.json)で追える。
- Z1＋T1 total decoder、保存DF10 solver energy **−2.1663874486347634 Ha（条件付きproxy）**、全事象平方loss、保存deterministic raw・ideal IID二項・EXACT_AFFINE_OF_BINARY64_RAWを維持。raw/normalization/scalar/targetの元packetは不変。旧Minkowskiは別参照。
- §10.5照合で同等の幅h・同vectorの保存total Q3/進行中jobを確認範囲で未同定。幅16h/7や他方式の結果は今回の代替にせず、比較先だけ保存再利用した。範囲と検査は[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)。

## 数値gate・保存・負荷

- 新count格子の有限binary64 good閉包を**2,555,322隣接組**で確認。全13roundの質量・first-unsafe分割・残余会計、原API/RESULT整合、MSE上下界・moments・exact閾値を検査した。宣言scopeはρ=1、outcomes4,225、verification55,043,173、非終端frontier8,192、transitions141,559,036を維持。
- 実追跡は**86,533,919遷移**。非終端frontier最大8,192、終端11,039状態（終端はfrontier制限対象外）。残余mass **2.908829677505633E−31**、残余loss上界 **1.133191632207093E−29 Ha²**を捨てずに会計した。
- any-zero確率は約**2.252044702736076E−4**。m12の追跡zero-holdは同値、tie-argminは約**6.897961101680257E−6**。total decoderの規則で全事象lossに含め、未解決として除外していない。追跡外のmassは上記残余として残る。
- 数値判定は `MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`、保存状態は別に `WORKER_ARTIFACTS_SAVED`。原[API_RETURN](run01/B0I_P7_S2_D01/API_RETURN.json)、[RESULT](run01/B0I_P7_S2_D01/RESULT.json)、coarse/lemma/total-DP partials、[RESOURCE_FINAL](run01/B0I_P7_S2_D01/RESOURCE_FINAL.json)を保存。親caller・記録済み子process100件は全て終了code0、monitor errorなし。
- worker wall **143.316秒**、worker CPU **135.160秒**、child CPU **155.647秒**、process-tree CPU **290.807秒**。worker peak RSS **600,200 KiB**、完了child単体peak最大 **435,148 KiB**、親monitorのsampled subtree peak **3,886,860 KiB**。単体peakの合計をRAM peakと扱わず、観測範囲を原記録に保存した。1科学job＋既存8 DP worker経路。
- 新人工/metadata検査**3件PASS**。異なるconfiguration拒否、幅固有schedule・raw/target/law保全、区間差/余裕の符号を確認した。[TESTS](TESTS.txt)に科学実行前の技術修正と最終PASSを保存。

遷移allowance 1E−12 radは既存のbinary64 engineering allowanceであり、厳密数値certificateではない。保存solverのground資格、state/target包含・実sampler lawへの移送も未認証。`sufficient_N`、`global_RMSE_bound`、`G`、`G_qualified`、winnerはnullを維持する。

## 価格・終了

**435,481,090 RZは比較先の幅16h/7だけの条件付き保存費用**。幅hの全価格は未取得のため、今回の幅h費用はnullであり、費用の優劣・最良幅を結論しない。新信号、MC、価格予測、build/compile、別配分、追加Q3、retryは実施していない。旧結果・SOURCE・STOPを保全し、公開照合後 **mandatory STOP**。
