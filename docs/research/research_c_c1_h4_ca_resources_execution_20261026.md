# H4 CDF8／保存順XDF8：共通CA・RPE shots・条件付き総資源の限定実行

2026-10-26 JST。独立branch `codex/research-c-c1-h4-ca-resources-execution-20261026`。

**固定SOURCEを変更せずfresh run01で一度実行し、119/119 logical unitsがCOMPLETE。correctness FAIL・inflight・retry・resumeは0。4候補とも13/13 roundを通過し、`NUMERICAL_PILOT_CA_PASS`となった。同sの条件付き総RZはCDF8がXDF8よりs1で11.85385%、s4で11.99082%小さかった。これは保存level B誤差・ideal IID Born・own-DF ground oracle・状態準備除外・serial native policyに条件付けた数値pilotであり、厳密CAや無条件の総資源優位ではない。公開照合後mandatory STOP。**

- [SOURCE・全raw19件・call journal・STOP](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/run01/)
- [機械可読要約](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/execution_summary.json)／[preflight](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/execution_preflight.json)／[postflight・raw hash一覧](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/execution_postflight.json)
- [52 round accuracy](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/round_accuracy.csv)／[104軸shots](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/shots_axes.csv)／[104軸serial費用](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/serial_axis_costs.csv)
- [候補別総費用4行](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/candidate_totals.csv)／[同s比較2行](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/same_s_comparisons.csv)
- [技術準備・再利用・数式・制約](research_c_c1_h4_ca_resources_preparation_20261026.md)／[固定119-call契約](research_c_c1_h4_ca_resources_contract_unapproved_20261026.json)

## 固定対象と来歴

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、sector36。通常number CDF rank8と、保存OpenFermion1.6.1 eigh由来の元順λ/G prefix8 XDF。元g補正κ・核scalar一回・spin/bit/frame順を変更せず、全frame ordered symmetric S2、s1／4、δ=.1、m0..12、PR splitなし。s1が主対象、s4が対照であり、cross-s探索・事後選択を行っていない。

共通CA=.00159 Ha、DF reserveCA/8=.00019875 Ha、β_total=π/6、β_stat=π/12、deterministic phase room=π/12。各DF自身のground-state oracleと保存alias条件を仮定し、状態準備費用・実機雑音を除外した。failure配分は候補ごとに.01、round/axisごとに.01/26で、4候補同時・事後選択した勝者の成功保証ではない。

| 固定対象 | Identity |
|---|---|
| 今回の科学SOURCE commit | `18045333a1438740160d7b9e44e78139ad59f2b5` |
| SOURCE tree | `dd53c67efa51027ed1084f88134fdd55e6dba5ea` |
| SOURCE manifest SHA256 | `fc03dae7a2db264d4aa4a92abf0ff0e6c812dd6b25879afd8cddfd56d299d521` |
| 固定119-call契約 SHA256 | `7540af39daec4e1584f2805a8338c4ab91d2042923d83830dcaef183f5fb4edb` |
| native/full512科学SOURCE／結果公開 | `862a97d4213121749448ca716cde820a44b99772`／`f4de9f62c2ff1cbaaa75b6cdaa07217d00dfd957` |
| 元P-H4入力 SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| CDF元fit SOURCE／export SHA256 | `da2edb465d5ae6432439fbd1df8e57512ee59c11`／`c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| CDF PF／XDF8 PF SOURCE | `6ea5c6a9c10cc4b5411039a0d61a9f19acc4f34f`／`328f31007ee8069d1f7ba24f91029a3748b59109` |

保存full512合成52 profile・保存DF norm＋元guard・PF/ground来歴・P0 metricsを読んだ。新たなNPZ/J/U/ground/Operator/QPY decode、DF/PF/合成誤差の再生成、fit、回路build/compileはない。両arm共通のH6 Gaussian精度修正・MPFR160 P0、Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=false、および既存完全同一basis共有・fsum角度結合を維持した。

チャット上の明示許可に基づく実行。歴史的な`unapproved`ファイル名・metadataは変更せず、承認証明書・承認原文保存・新grantを作成していない。

## Gateと数値CA判定

| Gate | 結果 |
|---|---|
| SOURCE24・保存input12・契約・依存9版 | preflight PASS、実行後protected36件不変 |
| 前段native305-call完了・SOURCE・full512 coverage・P0/precision | PASS、旧runの再起動なし |
| 保存DF reserve・PF/合成round対応・ground/alias来歴 | 4候補PASS、旧誤差を再計算せず再利用 |
| 13 round×4のsignal/phase・新shots・固定N margin | **52/52 `CONDITIONAL_ROUND_PASS`** |
| horizon12・candidate failure・energy | **4/4 `NUMERICAL_PILOT_CA_PASS`** |
| 104軸serial費用coverage・数値容量・totals | 4候補PASS、unknown/clip/別集計器なし |
| 同s比較 | s1／s4とも`CDF_LOWER_CONDITIONAL_INTERROGATION_COST` |

PF operator誤差と合成complex signal上界pilotを一回ずつ既存`RoundEvidence`へ入力し、返されたcombined値と保存値の同一性を確認した。決定論モデルのattenuation1／RTE補正0／law discrepancy0を維持した。未知のreference・合成丸めを0とは認定していない。

DF誤差はsignalへ足さず、既存`assess_energy`で`DF norm＋guard + β_total/(δ2^12)`としてHaで一度だけ接続した。下表はこの**数値十分条件によるenergy誤差上界pilot**であり、実際に測定したenergy誤差ではない。

| 候補 | 保存DF norm＋guard [Ha] | energy誤差上界pilot [Ha] | CA=.00159 Ha | X/Y合計shots |
|---|---:|---:|---|---:|
| CDF8 s1 | 7.6662159163e-5 | .0013549794824010532 | 数値PASS | 13,606 |
| CDF8 s4 | 7.6662159163e-5 | .0013549794824010532 | 数値PASS | 13,314 |
| XDF8 s1 | 9.5339014551e-5 | .001373656337788953 | 数値PASS | 13,600 |
| XDF8 s4 | 9.5339014551e-5 | .001373656337788953 | 数値PASS | 13,314 |

両armのDF近似誤差は完全一致していないが、同じDF予算と最終CA目標を満たす数値条件で比較した。新shotsは保存合成誤差を反映した既存Hoeffding十分数であり、旧CDF shotsや旧XDF10 shotsを流用していない。実際の量子測定・RPE outcome samplingは0。

各roundのX/Yは対称Nで、m0は全候補511 shots/axis、m12はCDF s1 590、XDF s1 589、両s4 516。既存固定N検査の最小statistical marginはそれぞれ2.16696e-6、2.18514e-6、2.63014e-6、2.67918e-6で、すべて非負だった。全104値とmarginはCSVに保存した。candidateのstatistical_failure_upperはすべて.01。

## 条件付きserial総資源

各round/axisで`n=s2^m`、`cost=n×saved_kernel+saved_head+saved_axis_tail`を既存関数で構成し、新shotsを掛けた。測定を含むreadout size/depthは二重加算していない。state preparationは含まず、測定数＝executions＝上表の合計shots。

| 候補 | 総RZ count | serial RZ depth | 総CX count | serial CX depth | serial total depth | 総circuit size |
|---|---:|---:|---:|---:|---:|---:|
| CDF8 s1 | 113,219,727,909 | 54,183,488,163 | 68,160,609,352 | 59,414,499,760 | 118,146,374,763 | 217,204,369,161 |
| CDF8 s4 | 413,834,761,607 | 198,048,553,871 | 249,137,131,232 | 217,168,804,160 | 431,842,276,925 | 793,913,930,693 |
| XDF8 s1 | 128,445,454,592 | 55,315,213,190 | 71,444,533,608 | 59,580,098,424 | 120,790,723,934 | 246,316,079,720 |
| XDF8 s4 | 470,217,718,215 | 202,499,839,919 | 261,546,777,184 | 218,113,016,352 | 442,194,889,173 | 901,722,729,901 |

| 同s比較 | CDF/XDF総RZ比 | CDFの削減率 |
|---|---:|---:|
| s1（主対象） | .8814615376514202 | **11.853846%** |
| s4（対照） | .8800918076374575 | **11.990819%** |

比・百分率は保存済みfinal totalsの表示用算術であり、別条件のcost再集計や候補再探索ではない。s1ではCDFのshotsがXDFより6多く、単一kernel RZ約11.99%減が、そのまま総RZの百分率にはならなかった。s4は合計shotsが同じである。旧CDF8／XDF10の約27%結果は別baseline・条件の歴史的結果として保持し、今回のCDF8／XDF8結果へ置換しない。

deterministicな固定kernel/shellの明示serial modelなので費用intervalはsingleton（lower=mean=upper）。この算術的supportは精度の丸めenclosureではない。depthはall-wire fenceによる`scheduled_serial_sum`で、長round全体を新たにcompileした最小depthや実機時間ではない。literal/shared未観測の追加削減・一般相対Gaussian融合は仮定していない。

## 科学call・資源・保全

| 科学call | 実績／上限 |
|---|---:|
| 保存bundle読込・状態/来歴/horizon確認 | 1/1 |
| 候補attachment | 4/4 |
| round accuracy・shots・固定N確認 | 52/52 |
| energy判定 | 4/4 |
| X/Y一組のround serial費用 | 52/52 |
| candidate totals/classification | 4/4 |
| 同s比較 | 2/2 |
| **合計** | **119/119 COMPLETE** |

journal attempt=COMPLETE=固定上限、FAIL/inflight0。fresh run01は消費済みでretry/resume0。raw19件／571,700 bytes、52 accuracy行、104軸shots・104軸cost、4候補totals・2比較を保存した。SOURCE24＋input12のprotected36件、さらに前段protected62件・raw109件のhash不変を確認した。旧H2/H4/H6のSOURCE・結果・grant・ledger・marker・STOPは変更していない。

実行直前の人工21 testsが21 PASS／1.91秒。前段の新21＋既存67＝88 unique人工・回帰PASSも保全した。科学SOURCEとrunnerは無変更。read-only既存Python3.11.1／NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0ほか固定9版・旧lockを再利用し、別環境の書換えなし。

affinity32 CPU／空きRAM約52.7 GiBから、既存resource_planが独立4候補のworker4を選択した。重い行列処理はなく、scalar経路へBLAS上限は設定していない。runner wall1.48467秒／process CPU9.77566秒／peak RSS349844 KiB。cache/tmpはtask内、bytecode出力無効。共有環境・他ユーザーjobを変更せず、CPU/RAM/worker/wallの恣意的capを追加していない。

新fit／optimizer／rank探索／DF・PF・合成誤差再計算／state-action／build/compile／量子shots抽選はすべて0。保存出力のCSV・要約整形でoperator・norm・shots solverを追加呼出ししていない。これは公開されたローカル実行証拠であり、外部再現・CI再現ではない。

## 資格の限界・次の担当・mandatory STOP

今回得られたのは、保存level B誤差を仮定した共通CAの数値十分条件、新Hoeffding shots、明示serial native modelの条件付き費用比較である。CAR/expm・Operator・norm・PF基準の厳密な丸めenclosure、実機noise、状態準備込み資源、4候補同時／選択後の成功保証、無条件の総資源優位は未資格。`CA_achieved`／`rigorous_enclosure`／`guaranteed_actual_shots`はnull、`guaranteed_total_resource_superiority=false`を維持した。

**ユーザー／研究C**が今回のCDF8／XDF8比較と既存H2/H4/H6のscope差を確認し、結果統合・詳細科学レビューや追加検証の要否を判断する。**Codex**は具体的な次の依頼に応じて必要な記録更新・技術準備を行い、追加科学計算と詳細科学レビューはそれぞれ必要な別明示許可後のみ実施する。今回の契約内で未完了の技術blockerはない。

**commit・push・remote SHA／SOURCE／成果物照合後mandatory STOP。次の科学計算・再最適化・分子拡張・レビューへ自動進行しない。**
