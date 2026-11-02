# CDF7 native32：full512校正と条件付き総資源比較

2026-10-30 JST。**332/332 COMPLETE、correctness FAIL0、retry/resume0。Raw／compiled／actual adjoint／X/Yは各32/32 PASS、QPY128件を保存した。主指標RZの条件付き点費用では、今回のCDF7は登録した両比較先を下回らなかった。** Kernel平均RZは17754.9375±169.1912（descriptive SE）、ideal仮定の条件付き総RZは638880821593.5。保存通常CDF8γ0比1.0188545、M4比2.7288863である。母平均の優劣や実CA保証を判定したものではない。

## 対象・SOURCE・実行範囲

保存直鎖H4・1.00 Å・STO-3G、通常number CDF7、Nα=Nβ=2／sector36、元順LD2／tail F2..6、partial ordered symmetric S2、δ=h=.1／s1／r128／K0。各DFのground-state oracle、状態準備費用除外、共通CA=.00159 Haの既存条件を維持した。今回の量子回路検証はsystem8＋ancilla1のfull512である。

ユーザーの一括指示に基づき、[技術準備](research_c_cdf7_native32_preparation_20261030.md)・人工検証・科学SOURCE公開とremote照合後に一度実行した。中断は科学開始前だったため、既存科学runの再起動・resumeはない。今回の許可を証明するための記録・新grantは作っていない。

| 項目 | 固定値 |
|---|---|
| Branch | `codex/research-c-cdf7-native32-20261030` |
| 科学SOURCE commit | `173ba2fd83172f41f08900fcba7e2a977565f57f` |
| Scope | `C_H4_CDF7_NATIVE32_CONDITIONAL_COST_V1` |
| [契約](research_c_cdf7_native32_contract_20261030.json) SHA256 | `8922f3ef93f5a226d0ea6527e147c4eedeb4c3fe112549d0715acb872bb6a8ee` |
| [SOURCE107 manifest](../../artifacts/research_c_cdf7_native32/2026-10-30/technical/SOURCE.json) SHA256 | `e90e04f4ebc9a957cd6089eb224fd70e01a34e851e83d61fe7a1c58ada5d7a13` |
| 保存fit SOURCE／export SHA256 | `c13d6f82a22d497b3334eaaa40d8430a5010f70f`／`4e4939d1380386ebb98c3b39a13954cbfc3148e35fb8e2cfd3cf536bd0b57ff4` |
| 保存PF/RTE・RPE SOURCE | `3e75b16901f0f4a9966d2bd5b9ff6bde8e75f472` |
| 元P-H4 input SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| Fresh output | `artifacts/research_c_cdf7_native32/2026-10-30/run01` |

旧native本体323 unitsに、独立entryの保存入力読込・再接続9 unitsを明示計上し、最大332とした。候補や32 trajectoryを増やす変更ではない。新fit・optimizer・rank探索・DF誤差・PF/RTE・ground eigensolve・RPE/shotsの再計算は行わない。

独立worktree・task専用venv/cacheを使用し、SOURCE107・input33・旧保護2292件と依存版を照合。人工新9＋既存17＝26 distinct tests PASSを実行前に確認した。既存B1/B2/CAR、precision Gaussian、P0、Operator decode、actual adjoint、X/Y、metrics、calibrationを再利用し、科学SOURCEは実行中に変更しない。

P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1／qubits_initially_zero=false。Headの相対Gaussian融合、tailの隣接exact-key共有、head-tail融合なしを両比較先と同じ条件で使用。未観測の追加最適化効果を仮定しない。

## 正しさ・有限cohortの費用

8因子のfull/sector対応、full-CARのsigned law＋identity、8 Gaussian、4 head-relative transition、保存3 shellがすべて通過した。32本は全512入力・出力を含むoperator行列で比較し、個別Gaussianのsector射影や有限state probeで代用していない。

| 観測量（32本中の最大） | 誤差 | 固定allowance |
|---|---:|---:|
| Raw／独立controlled CAR | 2.1541284966324435e-14 | 1e-10 |
| Compiled／独立controlled CAR | 3.701760038206692e-12 | 1e-10 |
| Compiled／raw | 3.700291150202197e-12 | 1e-10 |
| Raw／compiled unitarity | 4.3076653355456074e-14／2.2426505097428162e-13 | 各1e-10 |
| Raw／compiled actual-adjoint reference | 2.2205769744839184e-14／3.687471793771653e-12 | 各1e-10 |
| Raw／compiled adjoint往復identity | 4.363176487925092e-14／2.1517355464660708e-13 | 各1e-10 |
| Raw／compiled complex X/Y | 2.7350553208580254e-14／1.948331193323566e-13 | 各2e-10 |

[各32本の残差](../../artifacts/research_c_cdf7_native32/2026-10-30/presentation/gates32.csv)、[metrics](../../artifacts/research_c_cdf7_native32/2026-10-30/presentation/metrics32.csv)、[原本とQPY](../../artifacts/research_c_cdf7_native32/2026-10-30/run01/)を保存した。上表は既存gate出力の観測最大で、strict boundや全law上界ではない。

| 一h-stepのcompiled kernel指標 | Sample mean | Descriptive SE | 観測min–max |
|---|---:|---:|---:|
| RZ count | **17754.9375** | **169.19122685576718** | 15966–19617 |
| RZ depth | 5334.90625 | 47.978854536252605 | 4810–5850 |
| CX count | 6685.3125 | 52.317933872020376 | 6136–7242 |
| CX depth | 3889.34375 | 24.993935126241286 | 3607–4160 |
| Total depth | 11698.78125 | 105.01002001321304 | 10531–12850 |
| Circuit size | 35374.5 | 340.72037359669434 | 31742–39061 |

[6指標のcalibration](../../artifacts/research_c_cdf7_native32/2026-10-30/presentation/calibration6.csv)。保存通常CDF8γ0のRZ mean±SEは17454.1875±195.9084、保存M4は14076.4375±255.1396。これらは固定inputのJSONを引用したもので、旧QPYや旧metricsを再評価していない。非pairedな小さいcohort同士であり、CDF7と通常CDF8の小差を統計的に確定したrank効果とは呼ばない。

## 保存shotsを使った条件付き比較

[前段119-call結果](research_c_cdf7_PR_execution_20261030.md)のshots／W／Sをそのまま再利用した。Idealはshots21768、W35983272、S_RZ76188。各`.001*A_m`誤差budget仮定ではshots21862、W36152228、S_RZ76517。新しいshots式・PF/RTE・energyの計算は行っていない。

| 仮定lane | CDF7条件付き総RZ | 保存CDF8γ0比 | 保存M4比 |
|---|---:|---:|---:|
| ideal_exact_native | **638880821593.5** | **1.0188544963837933** | **2.7288862870389856** |
| assumed_raw_signal_budget | **641880625142.75** | **1.0189625266680766** | **2.72960004064081** |

Idealの保存比較先はCDF8γ0が627057959562.5 RZ、M4が234117788135 RZ。新cohortの総RZ校正SEはideal6088053935.964775／budget6116639808.889419。このSEは新cohortだけをW倍した値で、比較先の校正不確かさを加えたjoint intervalではない。

| 指標 | Ideal条件付き合計 | 保存CDF8γ0比 | 保存M4比 |
|---|---:|---:|---:|
| RZ count | 638880821593.5 | 1.0188545 | 2.7288863 |
| RZ depth | 191967458876.25 | .9977447 | 2.6580783 |
| CX count | 240559418092.5 | 1.0124337 | 2.5923307 |
| CX depth | 139951314057.75 | .9867934 | 2.4569182 |
| Total depth | 420960569279.25 | .9934914 | 2.6435167 |
| Circuit size | 1272890396856 | 1.0157468 | 2.7319167 |

[2 lane×2対象×6指標の全比較](../../artifacts/research_c_cdf7_native32/2026-10-30/presentation/conditional_comparisons24.csv)。Depth類は同じserial accountingで足した指標であり、全roundを一体compileした回路depthや実機並列実行時間ではない。Coherent occurrence間の未測定の追加融合を仮定していない。

事前のideal RZ kernel損益分岐17426.37199236634（通常CDF8γ0）／6506.293033801929（M4）に対し、新mean17754.9375は両方を上回った。登録主指標の記述的結論は**`NO_RZ_POINT_IMPROVEMENT_VS_EITHER_REGISTERED_TARGET`**。通常CDF8に対してdepth類がわずかに小さい指標もあるが、主指標のRZ削減は観測しておらず、保証付きの優劣は未資格である。

## 資格と解釈

Master2026103007、cohort `CDF7_s1_r128_native32_v1`、axis `CDF7_NATIVE_KERNEL`、SOURCE digest／JU hash／sample0..31で固定したPCG64有限bit実装の32本。旧cohortと非paired、poolなし、価格による選別・追加samplingなし。各本は128 event、K0。費用モデルは各coherent occurrenceがfresh independent trajectoryを使うもので、単一sampleを全roundへ繰り返す量子実行ではない。

32本のfull512 raw／compiled／actual adjoint／complex X/Yが数値gateを通ることと、全lawの合成・implemented-law誤差上界は区別する。Max-entry／unitarity絶対1e-10、complex signal2e-10、normalization1e-12、mass相対1e-12／絶対1e-14、rtol0、global-phase fittingなし。これはbinary64数値pilotであり、厳密rounding enclosureではない。

校正値は有限32本のsample mean／variance／descriptive SE。観測min-maxをsupport boundにせず、費用support・ideal-law母平均・保証付き優位は未資格のまま。保存W/Sを使う`C_metric=W*mean_kernel_metric+S_metric`は仮定laneの条件付き点評価である。校正SEは`W*SE(mean)`で伝播し、13 roundの独立した校正として小さくしない。比較先は固定された保存finite32点費用で、両cohortを合成したconfidence判定は行わない。

`evidence` laneの合成／law誤差、actual CA、保証付き総資源はnull。Ideal仮定はexact native・ideal finite-RTE IID law、もう一つは合成・lawのraw-signal誤差が各`.001*A_m`以下という未検証の仮定である。条件付き点費用の増減を、実CA達成、状態準備込み費用、CDF一般の優位、rankだけの因果効果へ昇格させない。新CDF7と保存CDF8は初期化・最適化来歴も異なる。

## Call・資源・保全・STOP

保存入力接続9＋native本体323＝**332 attempts／332 COMPLETE／FAIL0／inflight0**。[Kind別calls](../../artifacts/research_c_cdf7_native32/2026-10-30/presentation/calls.csv)、[journal](../../artifacts/research_c_cdf7_native32/2026-10-30/run01/science_calls.jsonl)、[STOP](../../artifacts/research_c_cdf7_native32/2026-10-30/run01/STOP.json)。32 sample、32 build／compile、actual adjoint64回路を含むQPY128件、full-CAR reference32件を保存した。Retry/resume0、新fit／optimizer／rank探索／旧PF-RTE・RPE/shots再計算／量子shots実行0。

起動前affinity32 CPU、空きRAM56,256,086,016 bytes、load約1.22、CPU/memory pressure avg10=0。既存resource plannerで32 worker、BLAS1を選んだ。Build／P0／actual-adjoint buildの既存process-local mutexを維持し、それ以外のreference／Operator decode／行列比較／X/Yを並列化した。Sourceを途中で高速化版へ差し替えず、人工CPU/RAM/worker/wall上限、共有設定・共有venv・他jobへの変更はない。[Preflight](../../artifacts/research_c_cdf7_native32/2026-10-30/execution_preflight.json)、[監視snapshot](../../artifacts/research_c_cdf7_native32/2026-10-30/monitor_resources.jsonl)。

[GNU time](../../artifacts/research_c_cdf7_native32/2026-10-30/process_time.txt)：wall **20分04.12秒**、user14270.23秒、system295.03秒、peak RSS3407732 KiB、swap0、exit0。Runner wall1203.7128743秒、CPU14564.856104421秒。Qiskit lock wait5815.3556秒は各workerの累積で、単一の経過時間ではない。測定なしのspeedup倍率は主張しない。

[Summary](../../artifacts/research_c_cdf7_native32/2026-10-30/execution_summary.json)／[postflight](../../artifacts/research_c_cdf7_native32/2026-10-30/execution_postflight.json)でSOURCE107・input33・旧保護2292件の不変、raw369件／196959283 bytes、QPY128、call実績、未知値nullを照合した。CSVは登録済み出力の転記と観測残差の集約で、新しい物理条件や数値operator評価を加えていない。旧H2/H4/H6、SOURCE・raw・grant・ledger・marker・STOPを保全した。

結果・原本・SOURCE・契約・QPY・STOPをcommit/pushし、remote SHA／SOURCE／input／成果物を照合して**mandatory STOP**する。担当Codexによる今回のnative32実行と条件付き比較は完了。次はユーザー／研究Cが、低rank化のみの候補を継続するか、basis切替・head価格等を扱う構造設計へ戻るかを判断する。新たなrank／fit／seed探索や、未資格項目を埋める追加科学計算へは進んでいない。
