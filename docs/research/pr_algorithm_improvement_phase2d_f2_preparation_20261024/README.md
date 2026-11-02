# Phase 2D-F2：finite-target資源比較の実行前準備

**TECHNICAL_PREPARATION_COMPLETE_STATISTICAL_ADOPTION_AND_QUALIFICATION_PENDING_MANDATORY_STOP**。
F2専用SOURCE、保存入力接続、case/call計画、集計、実行入口とpreflightを実装し、mock／管理検査を通過した。
未実装の実行接続をblockerとして残していない。ただし科学処理による動作確認は未実施で、
**主比較を実行するには統計条件の正式採択と未取得資格の解決が必要**。
今回は科学実行・研究回路build・QPY生成・compile・数値再解析・新grant発行をすべて0件に保った。

## 採択範囲と固定条件

今回ユーザーが採択した主比較は **finite-target比較**。
physical signal／energy推定全体の厳密な同精度改善は主比較に含めない。
この準備をF2本体の実行承認に読み替えない。

F1と同じ4 mode・2粒子・occupied(0,1)・L_D=1・残差density-square 2fragmentの合成fixture。
分子geometry／AO basis／分子DF rankは非該当、full16／controlled32、JW little-endian・ancilla-last。
Hamiltonian、state、sector、scalar、D/R split、faithful_identity_in_tailを維持する。
F2条件は **T=1/5、q=1、delta=1/5、r=2、h=1/10、K=0**。
F1のK=2のlaw・mean・価格・信号・compile結果は流用しない。

比較対象はSS／SP／PS／PP。各axisのfull wrapperにprep X0X1、ancilla H、
actual controlled trajectory、X/Y readout、ancilla terminal measureを含める。
全arm共通のQiskit1.3.0／opt1／seed17／rz,sx,x,cx／topology-free／approximation_degree1／
qubits_initially_zero=Falseを用いる。P0のrtol=0・phase fitting禁止と数値閾値を維持する。

起点はF1正常結果 `b0dad8cc2f7a4cb821f53185c4a65f3a42e8f381`。
[F1結果](../pr_algorithm_improvement_phase2d_f1_rerun_20261024/EXECUTION_REPORT.md)、
[既存F2草案](../pr_algorithm_improvement_phase2d_preparation_20261023/F2_CONDITIONAL_CONTRACT_DRAFT.md)、
P0 `b8d54708f90ccef275eda3757143100b2d0c677e`、I `bfaab3831042f64a7112296437c0001e34920209`、
B結果 `32c76bc6392ba277ce9fdd8d4bc54ffb5339b767`、B科学SOURCE `3de42d3a3efe9ebc74dde8a1d0e1f1bdbadd607c`、
F1-PRE `31e8391a56fbb1a0e64f477f78e242655329de99` を参照した。
正式共通指示2026-10-23の§8.4／§10／§10.3を適用した。原文の編集・公開はしない。

## SOURCEと入力の再利用

| 処理 | 再利用／今回の差分 |
|---|---|
| exact spec／context／typed atom | Iのfixture.py、execution._atom、prototype.pyを再利用。F2 contextは既存対応済み。 |
| F2 law／event | make_tail、finite_rte_distribution、ordered_eventを承認後に使用。K=0を明示し、新lawを4armで各一回構築する。 |
| 回路・state/readout | 既存build_evolution／hadamard_wrapper／BasisProviderをそのまま使用。 |
| 独立reference | CAR、exterior-minor、event_operator、finite_target／physical_target、wrapper_operatorと90/128digitsを再利用。 |
| compile／cache | 既存compiled_wrapperの固定compilerを維持。完全IR key生成をgates.compile_cache_keyへ抽出し、重複判定と同じ実装を使う。 |
| 実費用metric | rte_compiled_costの_gate_counts／_filtered_depthを再利用し、measurement込み／除外depthを区別。 |
| interval／ceil | 既存scalar_context／rational_interval／endpoints／shot_count_from_enclosureを再利用。資格は未取得のまま。 |
| selector／既知四候補 | 既存select_dictionary／exhaustive_fourを、資格条件を満たすactual N加重RZに限り使用。literal scoreは使わない。 |
| grant消費／STOP | 既存authorization.claim_packetを再利用。F1専用verifyはF2を拒否するため、F2用の薄いpreflightを追加。 |

追加SOURCEはf2_plan.py、f2_preflight.py、f2_statistics.py、f2_execution.pyと
[scripts/run_pr_dictionary_f2_saved.py](../../../scripts/run_pr_dictionary_f2_saved.py)。
F1 runner／SavedF1はF1 context・K2 reference・F1 countsに固定されるため、そのまま呼べない。
F2では全trajectory・共通target・actual費用と必要Nを結ぶ専用接続だけを追加した。
旧gatesの数値処理を再実装せず、cache keyの抽出を除いて既存科学primitiveを変更していない。
旧固定SOURCEや既存grantの内容は変更していない。

保存Bのphysical.json、4arm JSON、input-only f1_bundle.json、requirements.lockの **7file** を
hash照合してprivateの新しい準備領域へコピーした。名前f1_bundle.jsonは過去の保存pathであり、
ここからspec・typed atom・rational frameだけを読む。catalog、F1 laws、literal prices、score、F1結果はF2入力にしない。
F1 bundleのprice／score欄が空であることもpreflightで確認する。
辞書生成・collectは再実行せず、その古典費を0やF2実測とせず `NOT_REEXECUTED_SAVED_B_INPUT_REUSED` と記録する。

## Case・callと処理構成

Bの保存Lと既存descriptorのQ=13,674を整数計算だけで照合した。

| arm | L | ordered trajectory L² | X/Y full wrapper 2L² |
|---|---:|---:|---:|
| PP | 51 | 2,601 | 5,202 |
| PS | 26 | 676 | 1,352 |
| SP | 58 | 3,364 | 6,728 |
| SS | 14 | 196 | 392 |

物理trajectory合計6,837、X/Y要求Q=13,674。
共通prep/readout-only診断X/Yの **2件を追加** し、logical full-wrapper build／compile requestは各 **13,676**。
診断はphysical evolutionを含まない参考値で、compiler fusionが非加法的なため主費用から差し引かない。
両axisで同じcontrolled evolutionを再利用し、evolution-onlyの追加compileはしない。

| 登録単位 | 件数 |
|---|---:|
| compile_requests | 13,676 |
| compiled_operator_checks | 13,676 |
| compiled_signal_checks | 13,676 |
| conditional_selector | 1 |
| full_wrapper_build_requests | 13,676 |
| known_four_search | 1 |
| law_bindings | 4 |
| mean_comparisons | 4 |
| ordered_trajectory_pairs | 6,837 |
| physical_bias_diagnostics | 1 |
| probability_mass_checks | 4 |
| quantum_measurement_draws | 0 |
| raw_operator_checks | 13,676 |
| raw_signal_checks | 13,676 |
| reference_precision_evaluations | 27,356 |
| resource_aggregation | 1 |
| shot_qualifications | 4 |

reference precision evaluationsは `2×(13,676 wrapper + finite target1 + exact physical1)=27,356`。
wrapper各件で90/128のoperator・signalを照合し、共通targetも二精度で照合する。
precision違い、operator check、event、build、compile requestを科学call総数へ単純合算しない。
4armのactual corrected meanは、検査したraw full wrapperからprep/readoutを外して得るcontrol1 blockの全列挙平均から構成する。
追加component probeやevolution-only build／compileは行わない。
physical bias診断1件は数値診断であり、主比較のphysical bias budget判定には使わない。

承認後の実行は、入力復元・F2 law → 並列trajectory build／reference／raw check → 完全IR重複照合 →
unique full-wrapper compile → logical alias全件のoperator／signal再検査 → mean／統計資格／actual集計、の順。
U_F2は未確定（`U_F2<=13,676`）。未承認のIR生成で事前決定せず、承認後のIR bindingで解決する。
角度・global phase・register/mapping・compilerとSOURCE/input/env identityをすべて含むkeyを使い、
各unique missを一回compileする。cache hitでもlogical検査は省かず、旧runのcacheは使わない。
保存QPYのbytesはhashで照合し、読戻し後の再serializationのbitwise同一は前提にしない。
読戻した回路の科学的同一性は固定operator／signal gateで確認する。
全ordered列挙を維持し、少数Monte Carloや試行結果による条件探索へ変更しない。

## 統計条件と未取得資格

**eta_stat=0.025、alpha=0.05と既存N式は候補であり、未採択。実際のNは未計算。**
主比較のfinite targetを採択したことは、これらの数値や保証を採択したことを意味しない。

既存候補は複素signalのaxis許容差eta/sqrt(2)、axis failure alpha/2、
`N_X=N_Y=ceil(4 Btot²/eta² log(4/alpha))`。
各shotでfresh IIDのevent列と独立bounded outcome±1、固定state/time/axis、
補正後range[-Btot,Btot]、trajectoryの恣意的選別なし、という条件を必要とする。
今回も将来F2もquantum measurement shotの抽選は0件で、Nは採択条件に基づく必要数として扱う。
全列挙やmockだけで実samplerのIIDを実証したとは表示しない。

| 必要な判断・資格 | 現在の状態とコード上の扱い |
|---|---|
| eta／alpha／N式の正式採択 | 未採択。実行値はnull、候補値を別欄に保存。 |
| alphaの範囲 | per-arm complex signalか、4arm同時保証かをユーザー/GPTが判断。後者はalphaを4armへ配分する接続を用意した。未選択。 |
| IID／range | 既存bounded-IID規則の条件を整理。外部から採択された条件・根拠を必要とし、フラグだけで証明済みとしない。 |
| finite-bit allowance gamma | 未決定・未保証。確率丸め／正規化・角度・corrector・実装／compile誤差の根拠が必要。0を自動設定しない。 |
| etaへの実装差の配分 | 接続案はeta_sampling=eta_stat-gamma>0。数値と妥当なgammaの根拠を正式採択してからNへ使用する。 |
| interval provider／log／端点／ceil | 未資格。SOURCE／adapter／環境lockと資格根拠をbindingし、整数境界を跨ぐceilはSHOT_COUNT_UNRESOLVEDでSTOP。 |
| physical bias保証 | 主比較には不要。二精度の小さい差を厳密enclosureやphysical certificateへ読み替えない。 |

F2のK0での既存関係 `Btot=B²=1+(lambda*h)²` を補正rangeとして記述した。
実行時のlawはbinary64 event確率を一度exact有理数として正規化し、全trajectoryの重みをその積で保持する。
理想law、実際に使うfinite-bit law、actual circuit mean、理想finite targetを区別し、差を記録する。
これらの入力依存の数値やB／lambdaを今回は評価していない。

P0の1e-12やoperatorの1e-10等は数値受理gateであり、それだけから厳密なfinite-bit allowanceを認定しない。
新しい保証や科学的数値確認を要する部分は未解決のまま返す。
[f2_statistics.py](../../../src/trottertracks/pr_dictionary_validation/f2_statistics.py)は既存interval primitiveとceilを接続するだけで資格を発行しない。
既存score_enclosureはliteral price・候補eta/alphaに固定されているためF2 actual比較へ直接転用しない。
将来の資格根拠はpacketのqualifications/以下へ保存し、contractのqualification_evidence_filesでhashをbindingする。
SOURCEは根拠の内容を数学的に認証する機構ではなく、別途採択された根拠と実行条件のidentityを照合する。

未採択・未資格では、F2本体のpreflightを拒否する。
Nや認証score、最適辞書をfallback SSや0で埋めず、未適格な主比較をcost-onlyやphysical比較へ自動変更しない。
資格が承認後の登録検査で成立しなければ、取得済み実費用・失敗理由を保全してSTOPする。
準備のmock PASSはSOURCEの科学的正しさや統計資格のPASSではない。

## 資源集計と古典費

各arm／axisで、同じ全列挙lawの重み付きactual RZ/CX、RZ/CX/total depth、circuit size、measurement countを集計する。
total depthはterminal measurement込み、quantum-only depthも別に保持する。
RZ/CX depthのfilter定義は既存helperで全arm共通。
期待費用は保存integer metricと有理数weightの和として保持し、trajectory MC標準誤差は付けない。
`C_RZ=N_X E[RZ_X]+N_Y E[RZ_Y]`、CXも別座標で同じ式を使う。
N加重actual RZを主座標とし、literal期待価格や1-shot費用だけで資源優位を認定しない。
RZ／CX／depth／秒／bytesを恣意的に換算しない。

辞書生成は保存Bの再利用、入力復元／F2 law、build／reference、unique compile／alias検査、
mean、統計／resource集計、selector、既知四候補探索の費用を分けて記録する。
各actual attemptのwall・CPU・peak RSSと、unique compile時間・cache aliasを保存する。
古典費の反復benchmarkや償却回数を今回追加せず、未測定の辞書生成費をF2の実測に混ぜない。

実行時のCPU affinity・MemAvailableとF1観測worker RSS595,704 KiBをもとにworker数を選ぶ。
spawnで独立trajectory／unique compileを並列化し、BLAS/OpenMP/Numba/Qiskit/Rayonの内側threadは各1に調整する。
CPU・RAM・wall timeの恣意的な上限は設けず、共有venv／OS／quota／scheduler／他jobを変更しない。
reference primitive／basisをworker内で再利用し、matrixは小系16／32に限定、結果を専用outputへ順次保存する。
QPYサイズとUに依存する正確な保存bytesは承認後に実測する。今回はIRを生成して見積りを埋めない。

## 技術確認・preflight・次の担当

新しい管理／mock test6件、既存metadata proxy test2件がPASS。
ID計画・重複IRのgroup化・両axisの人工weighted集計・診断非減算・worker mockのattempt/STOP・
F1 grant拒否・統計未採択／未資格拒否を確認した。
既存metadata testではQiskitのmetadata descriptorをproxyに使い、QPYとtranspileをmockした。
実際のQuantumCircuit／Operator／Statevector／高精度ctx／interval ctxは生成していない。
旧科学testは追加実行せず、F1やBの数値結果を再解析していない。

[contract.draft.json](contract.draft.json)がSOURCE103file、保存B入力7file、件数、数値条件、候補と未資格事項の正本。
private準備packetに対するCLI `--check-only`はSOURCE/input/plan照合を通過し、
統計未採択・interval未資格・最終契約未採択・公開SOURCE commit未bindingの4項目を返した。
科学libraryをimportせず、grant消費・output生成も0件であることを確認した。

```bash
"$F2_PYTHON" -I -B "$F2_SOURCE/scripts/run_pr_dictionary_f2_saved.py" \
  --check-only --packet "$F2_PACKET" --source-root "$F2_SOURCE"
```

上記は技術照合のみ。科学実行では新しい明示承認原文とF2専用one-shot grant、
正式採択済みcontract、公開済み最終SOURCE commit、入力・環境・資格根拠を新packetへbindingする。
専用outputのparentをgrant消費前に準備し、fresh namespaceと未使用markerをpreflightで確認する。
`source_commit=null`とdraft statusは未承認資料の意図した状態であり、公開後の採択packetで埋める。
使用済みF1/B grantを流用せず、失敗・成功ともSTOPし、retry／resumeや追加研究を自動開始しない。

**次の担当はユーザー／GPT**：候補eta／alpha／失敗確率範囲、IID/range条件、
finite-bit allowanceの根拠と予算配分、interval／ceil資格の取得方針、最終F2契約と科学実行承認を判断する。
必要な保証・資格検証を新たに行う場合は、その科学的範囲を先に明示して承認する必要がある。
その後のCodex作業は、採択内容のbindingとpreflight、別途承認されたF2実行に限る。

既存B/F1のSOURCE、raw、grant、ledger、marker、STOPは保全する。
今回はF2科学実行0件、新grant0件。**mandatory STOP。physical同精度比較・追加実験・重要科学レビューへ自動移行しない。**
