# 研究C M4-Native128：固定32校正・条件付き総資源比較

2026-10-28 JST。**332/332 logical units COMPLETE、correctness FAIL0、retry/resume0。固定32 trajectoryのraw/compiled full512、actual adjoint、complex X/Yを完了した。** 以下は保存M4の2仮定scenarioにおける有限cohort価格校正に基づく条件付き点比較である。母平均優位・有意性・actual CA・厳密enclosure・保証shots/totalは認定しない。

## 固定対象・SOURCEと記述revision

保存直鎖H4・隣接1.00Å・STO-3G、通常number Proposed CDF8（保存StageA γ=.1）、Nα=Nβ2/sector36、system8＋last ancilla/full512。元順frame0..7/LD2、元g補正κ/nuclear scalar一度/tail identity抽出、partial ordered symmetric S2 δ=h=.1/s1/r128/K0/m0..12、2^m coherent kernels、own-DF ground oracle/状態準備除外。CA=.00159Ha/DFreserveCA/8等の資格・shotsは保存M4をそのまま用いた。

- 起点commit：`4887d6af81d4b26a31c2f85cb0430670b620f4e6`。
- 科学SOURCE commit：`dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad`（実行開始後無変更）。
- [実行契約revision2](research_c_pr_aware_df_M4_native128_contract_revision2_20261028.json) SHA256：`0a5d20032656dcb456a20d84ee8fdfc6d7b44ed36d4c88201a7046ecb9e12d1f`。
- [SOURCE88](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/technical/SOURCE.json) SHA256：`3c0fc544b31b4aad9dd776c83eeb6392d59ee03c5b8a3059433cd1b65d56e12f`。
- [準備報告](research_c_pr_aware_df_M4_native128_revision2_preparation_20261028.md)／[66人工・回帰PASS](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/technical/tests.log)／[preflight](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/technical/preflight.json)。
- 元input SHA256：`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、Proposed J/U：`f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`。元P-H4来歴：`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。

旧契約の3説明に残った64 eventsを128へ修正する独立revisionだけを作成した。旧契約/manifest/SOURCE87を保全し、実装は一切変更していない。新manifestは元87＋metadata回帰test1。科学条件・32標本・call332・compiler・数値allowanceは不変。歴史的unapproved metadataは承認機構ではなく、今回のチャット許可で一度実行した。承認証明/原文保存/新grantなし。

既存M4-native adapter/runner、B2/M3 sampler・reference_context/trajectory_reference・full512 gates/calibration、StrongBuilder/PrecisionBuilder、P0、SharedQiskitCallsを再利用した。新sampler/CAR/compiler/runnerは作らない。旧M3/M4科学runnerの再起動0。

## Sampling・native policy・正しさ

固定32非paired PCG64 substreams。Master2026102805/cohort`M4_r128_native_fixed32_v1`/axis`M4_R128_NATIVE_ONLY`、SOURCE88 digestと保存JU identityに結合したseedを各sampleに保存。全32件が128 signed K0 events、旧cohortとpoolなし、追加標本/適応停止なし。32はclassical価格校正標本であり、RPEの量子shots実行ではない。Seedの分離はPCG64 joint streamやideal-law IID性の証明ではない。

Head内部relative Gaussian融合、tail隣接同一basis共有、cross-head/tail一般relative融合なし。Qiskit1.3.0/P0/opt1/seed17/rz,sx,x,cx/approximation_degree1/qubits_initially_zero=false、MPFR160 declared phase、phase fittingなしを維持。Scalar/identity相対位相、control0 identity/control1 reference、全512入力出力を含む比較であり有限probeによる代用ではない。

Full256 CAR9因子・sector対応/和、216 signed atom law、Gaussian9/relative4、保存head/X/Y shell3組がPASS。32 raw、32 compiled、64 actual inverse、32 raw+compiled complex X/YがPASS。Matrix component allowance1e-10、complex signal2e-10、rtol0。単一trajectoryのsignalにideal attenuationを掛けていない。

| 観測最大残差（32件） | 値 |
|---|---|
| raw_reference | 2.242727e-14 |
| raw_unitarity | 4.485301e-14 |
| compiled_reference | 2.554363e-12 |
| compiled_raw | 2.553138e-12 |
| compiled_unitarity | 1.227907e-13 |
| raw_adjoint_reference | 2.087369e-14 |
| compiled_adjoint_reference | 2.560195e-12 |
| raw_adjoint_return | 4.107825e-14 |
| compiled_adjoint_return | 1.190172e-13 |
| raw_XY | 3.557768e-14 |
| compiled_XY | 9.267111e-14 |
| raw_XY_norm | 3.241851e-14 |
| compiled_XY_norm | 9.581225e-14 |

[全32の正しさ](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/correctness32.csv)、[raw証拠](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01/)へ個別結果を保存。QPY128件（raw/compiledと各actual adjoint）、独立CAR reference32件、seed/event列32件。有限32回路のbinary64 numerical pilotであり、未観測trajectory/global all-law synthesis/validated CAR certificateへ拡張しない。

## Kernel価格校正

| 指標 | mean | sample variance | descriptive SE | observed min | observed max |
|---|---|---|---|---|---|
| rz_count | 14,076.437500 | 2,083,079.286290 | 255.139624 | 11192 | 16567 |
| rz_depth | 4,342.281250 | 145,573.176411 | 67.447474 | 3642 | 5041 |
| cx_count | 5,579.437500 | 198,374.770161 | 78.735072 | 4674 | 6340 |
| cx_depth | 3,424.875000 | 40,705.145161 | 35.665611 | 3046 | 3778 |
| total_depth | 9,574.531250 | 686,118.902218 | 146.428193 | 8042 | 11080 |
| circuit_size | 28,014.437500 | 8,459,508.060484 | 514.159145 | 22204 | 32993 |

[個別metrics32](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/native_metrics32.csv)／[校正6指標](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/calibration6.csv)。観測min/maxは全law supportではない。Kernel平均RZは **14,076.437500**、M3 r64の9,105.593750から4,970.843750増、1.545911倍（54.591%増）。同一標本のpaired比較やevent一個当たりの因果費用ではない。

## 保存shotsと条件付き総資源

M4のPF13/RTE13/attenuation/energy/Hoeffding shots/W/S/全36閾値を再評価しない。新fit/optimizer/rank/積分/ground eigensolve/PF/RTE/shots/CA再判定はすべて0。主scenario shotsは保存16056対M3保存20734、Wは16631888対32417320で、coherent-kernel重みは48.694%少ない。減衰改善による重み削減とnative価格増をC=Wμ+Sで接続する。

`ideal_exact_native`はexact synthesis/ideal finite-RTE IIDを仮定した反実条件、`assumed_raw_signal_budget`は全law synthesis/law raw-signal誤差がそれぞれ.001A以内という未検証感度条件。今回の有限full512 PASSでこれらの仮定を証明したことにはならない。保存M4のevidence laneは未資格/nullのまま。有限cohort平均をideal-law期待価格の記述的proxyとして用いる限定会計である。

同一32校正平均を全roundへ使用するため **SEtotal=W·SEkernel**。Round独立としてsqrt13で縮小しない。Shellは保存Sをinterrogationにつき一度、cross-segment未観測融合なし。将来の各coherent occurrenceは独立trajectoryのモデルであり、校正sampleを繰り返す実行モデルではない。

| scenario | 指標 | 条件付きtotal | descriptive total SE |
|---|---|---|---|
| ideal_exact_native | rz_count | 234,117,788,135.000000 | 4,243,453,649.482139 |
| ideal_exact_native | rz_depth | 72,220,391,610.500000 | 1,121,778,835.161280 |
| ideal_exact_native | cx_count | 92,796,579,603.000000 | 1,309,512,899.671833 |
| ideal_exact_native | cx_depth | 56,962,137,414.000000 | 593,186,442.540874 |
| ideal_exact_native | total_depth | 159,242,635,766.500000 | 2,435,377,305.397034 |
| ideal_exact_native | circuit_size | 465,933,091,247.000000 | 8,551,437,321.951055 |
| assumed_raw_signal_budget | rz_count | 235,155,559,637.250000 | 4,262,263,563.116383 |
| assumed_raw_signal_budget | rz_depth | 72,540,522,177.375000 | 1,126,751,332.742038 |
| assumed_raw_signal_budget | cx_count | 93,207,918,053.250000 | 1,315,317,564.122159 |
| assumed_raw_signal_budget | cx_depth | 57,214,632,898.500000 | 595,815,854.023797 |
| assumed_raw_signal_budget | total_depth | 159,948,508,924.375000 | 2,446,172,577.495011 |
| assumed_raw_signal_budget | circuit_size | 467,998,428,053.250000 | 8,589,343,190.793096 |

[全12 totals](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/conditional_totals12.csv)。比較対象はM3 r64、M2同J/U全frame決定論S2 s1、旧PR s4/r16。各targetの保存scenario総費用を使い、新しいtarget評価は行わない。

| scenario | target | RZ total比（r128/target） | 差 | 有限cohort点判定 |
|---|---|---|---|---|
| ideal_exact_native | M3_r64 | 0.793138 | -61,061,230,817.750000 | LOWER_CONDITIONAL_POINT |
| ideal_exact_native | M2_deterministic | 2.299196 | 132,291,858,680.000000 | HIGHER_CONDITIONAL_POINT |
| ideal_exact_native | old_PR_s4 | 0.433319 | -306,172,371,974.000000 | LOWER_CONDITIONAL_POINT |
| assumed_raw_signal_budget | M3_r64 | 0.792717 | -61,489,642,350.125000 | LOWER_CONDITIONAL_POINT |
| assumed_raw_signal_budget | M2_deterministic | 2.309388 | 133,329,630,182.250000 | HIGHER_CONDITIONAL_POINT |
| assumed_raw_signal_budget | old_PR_s4 | 0.433251 | -307,613,847,067.750000 | LOWER_CONDITIONAL_POINT |

副指標の条件付き比も次に示す。1未満が小さい点値で、統計的有意差を意味しない。

| scenario | 指標 | 対M3r64 | 対M2det | 対旧PRs4 |
|---|---|---|---|---|
| ideal_exact_native | rz_count | 0.793138 | 2.299196 | 0.433319 |
| ideal_exact_native | rz_depth | 0.729099 | 1.404494 | 0.329093 |
| ideal_exact_native | cx_count | 0.731692 | 1.435469 | 0.331013 |
| ideal_exact_native | cx_depth | 0.667952 | 0.982795 | 0.257420 |
| ideal_exact_native | total_depth | 0.731897 | 1.425297 | 0.332543 |
| ideal_exact_native | circuit_size | 0.799163 | 2.411805 | 0.445465 |
| assumed_raw_signal_budget | rz_count | 0.792717 | 2.309388 | 0.433251 |
| assumed_raw_signal_budget | rz_depth | 0.728712 | 1.410720 | 0.329042 |
| assumed_raw_signal_budget | cx_count | 0.731303 | 1.441832 | 0.330962 |
| assumed_raw_signal_budget | cx_depth | 0.667597 | 0.987152 | 0.257380 |
| assumed_raw_signal_budget | total_depth | 0.731508 | 1.431614 | 0.332491 |
| assumed_raw_signal_budget | circuit_size | 0.798738 | 2.422496 | 0.445396 |

## 全36損益分岐判定

保存元のbinary64閾値と直接比較し、表示丸め値は使わない。μ<thresholdで小さい点値、等値はTIE。下表は各6指標×3target×2scenario、[元精度CSV36](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/comparisons36.csv)へmean/threshold/差/比/SEを保存する。

| scenario | 指標 | M3r64 閾値／判定 | M2det 閾値／判定 | 旧PRs4 閾値／判定 |
|---|---|---|---|---|
| ideal_exact_native | rz_count | 17,747.772397 / LOWER_CONDITIONAL_POINT | 6,122.327980 / HIGHER_CONDITIONAL_POINT | 32,485.193738 / LOWER_CONDITIONAL_POINT |
| ideal_exact_native | rz_depth | 5,955.678439 / LOWER_CONDITIONAL_POINT | 3,091.703191 / HIGHER_CONDITIONAL_POINT | 13,194.687447 / LOWER_CONDITIONAL_POINT |
| ideal_exact_native | cx_count | 7,625.391668 / LOWER_CONDITIONAL_POINT | 3,886.840147 / HIGHER_CONDITIONAL_POINT | 16,855.629162 / LOWER_CONDITIONAL_POINT |
| ideal_exact_native | cx_depth | 5,127.429037 / LOWER_CONDITIONAL_POINT | 3,484.829823 / LOWER_CONDITIONAL_POINT | 13,304.598053 / LOWER_CONDITIONAL_POINT |
| ideal_exact_native | total_depth | 13,081.795154 / LOWER_CONDITIONAL_POINT | 6,717.569344 / HIGHER_CONDITIONAL_POINT | 28,791.874931 / LOWER_CONDITIONAL_POINT |
| ideal_exact_native | circuit_size | 35,054.742050 / LOWER_CONDITIONAL_POINT | 11,615.543192 / HIGHER_CONDITIONAL_POINT | 62,888.035974 / LOWER_CONDITIONAL_POINT |
| assumed_raw_signal_budget | rz_count | 17,757.215094 / LOWER_CONDITIONAL_POINT | 6,095.309351 / HIGHER_CONDITIONAL_POINT | 32,490.240422 / LOWER_CONDITIONAL_POINT |
| assumed_raw_signal_budget | rz_depth | 5,958.847157 / LOWER_CONDITIONAL_POINT | 3,078.059096 / HIGHER_CONDITIONAL_POINT | 13,196.737287 / LOWER_CONDITIONAL_POINT |
| assumed_raw_signal_budget | cx_count | 7,629.448754 / LOWER_CONDITIONAL_POINT | 3,869.687025 / HIGHER_CONDITIONAL_POINT | 16,858.247741 / LOWER_CONDITIONAL_POINT |
| assumed_raw_signal_budget | cx_depth | 5,130.157083 / LOWER_CONDITIONAL_POINT | 3,469.450824 / LOWER_CONDITIONAL_POINT | 13,306.664967 / LOWER_CONDITIONAL_POINT |
| assumed_raw_signal_budget | total_depth | 13,088.755321 / LOWER_CONDITIONAL_POINT | 6,687.923827 / HIGHER_CONDITIONAL_POINT | 28,796.347847 / LOWER_CONDITIONAL_POINT |
| assumed_raw_signal_budget | circuit_size | 35,073.392916 / LOWER_CONDITIONAL_POINT | 11,564.282291 / HIGHER_CONDITIONAL_POINT | 62,897.805843 / LOWER_CONDITIONAL_POINT |

## 研究上の問いと限界

1. Q1/Q2：r128平均RZ 14,076.437500、r64比1.545911倍。128-event complete P0 kernelの有限32校正値を取得した。
2. Q3/Q4：主scenarioのr128/M3条件付き総RZ比は0.793138、変化-20.686%。Shots総数だけでなく、高mの2^m重みを反映したW削減との交換条件で判断する。
3. Q5：同J/U決定論比はM3の約2.898859から2.299196へ変化した。PR内部の改善と強決定論への競争力を分ける。新しい同精度無条件優位の認定ではない。
4. Q6：対r64と対旧PRでは両scenarioの全6指標が低い点値となった。対決定論ではCX depthだけが低く（主scenario比0.982795）、RZ count/depth・CX count・total depth・sizeの5指標は高い。小さいCX depth差も有意性の認定ではない。副指標を混ぜた未登録加重scoreは作らない。
5. Q7：保存M2の同J/U決定論kernelは11026 RZで、今回14076.4375は約1.277倍。一方、条件付き総RZ差は2.299倍である。Kernel価格だけでなく、attenuation/補正RTE誤差を反映した高m shotsのcoherent重みも残る負担である。保存総shotsも16056対決定論13610。128 random eventsを含む価格とshots側の両方を改善する必要がある。Headは同一構成を再利用するが、head/tail別の新しい費用ablationは行っていない。Head・events・fusionの厳密寄与や因果帰属は未取得であり、headが主因とは断定しない。
6. Q8（提案のみ）：次の一つはγ最適化を通じたtail λと実complete-kernel費用の交換条件を検証する最小の追加γ点を推奨する。rをさらに増やす前に、必要random microsteps自体を減らせるかを調べる価値がある。新fitやγ候補の採択・実行は今回行わない。Native融合改善も代替案だが、今回データだけではhead/tail別の削減余地が分からず、保証できる改善量はない。

正式GO/NO-GOはユーザー／研究Cへ戻す。母平均・有意性・PCG64/ideal-law間の誤差・全law synthesis・cost support・actual CA・厳密保証は未資格。条件付き総資源を無条件の総資源優位と呼ばない。状態準備費用も未計上である。

## Calls・資源・保全・mandatory STOP

332=attachment7＋native準備35＋固定trajectory処理288（32×9）＋calibration1＋保存M4会計1。全attempt332/COMPLETE332/FAIL0、correctness UNRESOLVED0。All-law未資格は別問題としてnullを維持する。Retry/resume0、追加候補/標本0。

既存plannerの32worker/BLAS1、immutable reference/Gaussian cacheを再利用。Native build/P0 compile/actual inverse作成は同一process mutex、CAR/reference/Operator decode/行列比較は並列。Affinity32、開始時空きRAM52.133GiB、wall 919.830秒、processCPU 12820.017秒、peakRSS 2998.645MiB。CPU/RAM/worker/wall人工capなし。Task-local venv/cache/process/output、依存read-only、共有設定・他job操作0。比較実測なしの高速化率は主張しない。

[call journal](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01/science_calls.jsonl)／[resources](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01/resources.json)／[STOP](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01/STOP.json)／[postflight](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/execution_postflight.json)。新raw 364files/164349050bytes、QPY128/reference32。SOURCE88/input63/旧保護642件のhash不変。M4/M3run01/run02/M2/M1/B1/B2/C1/F2-K1/F3-K1の旧SOURCE/契約/raw/grant/ledger/marker/STOPを上書きしない。

Branch `codex/research-c-M4-native128-20261028`。結果・SOURCE・証拠をcommit/pushしremote SHA/treeとcommitted bytesを照合する。**公開照合後mandatory STOP。** 別r/s/K/γ/追加sampling/分子/fit/全law保証/K1 certificate/詳細GPT科学レビューは開始しない。次はユーザー／研究Cが最小追加γ検証を採用するか判断し、Codexは別途具体的な許可があった場合だけ準備・実行する。
