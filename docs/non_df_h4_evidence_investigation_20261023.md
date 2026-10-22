# H4非DF/DF pilot：N/G未確定と監視終了証拠の原因調査

2026-10-23 JST。既存SOURCE・保存値・監査記録の静的調査である。新規数値解析・shot再計算・G再会計・科学実行は行っていない。**20行のN/Gはnull、40軸はUNDETERMINEDのまま維持する。元case監視のnative PASSは証明できず、消失原因はUNKNOWN。** 調査完了後mandatory STOP。

## 1. 対象、来歴、調査方法

対象はlinear H4、隣接1.00 Å、STO-3G、共通未閾値MO積分。Pauli-JWは非identity184項・PR保持prefix27、DF-nativeはtol-only1e-8/rank10・PR保持prefix1＋one-body。8 spin qubit、共通の非DF Nα=Nβ=2 sector36最低状態、各表現自身のfull-H reference。S2、T=.8/q8/δ=.1、PR r4/K4・各256 paired trajectories。ε_sig=.05/.01/.005/.001/.0005、full ordinary-controlled cosine/sine Hadamard wrapperのRZ、準備費用P=0。Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/topologyなし。別系列H2のS2/Yoshida4順位逆転や旧H4 DF12 cohortと混合しない。

GitHubから対象branch、commit metadata、主要8資料と科学source4ファイルを実際に取得した。対象branchは取得時に[`6892e95f`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/6892e95ffdfc78e75425e7432e9e4ccdb9958df1)、その親は科学SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`である。GitHub内容とローカル固定Git object・保存ファイルを照合した。CSVのCRLF/LF表現差は内容照合時だけ正規化し、元ファイルのGit blobとbytesは変更していない。取得結果は[github_retrieval.json](../artifacts/non_df_h4_evidence_investigation/2026-10-23/github_retrieval.json)、照合結果は[verification.json](../artifacts/non_df_h4_evidence_investigation/2026-10-23/verification.json)。branch/refの後続移動を証拠として用いず、上記commitへ固定する。

報告書の専用branchは`codex/non-df-h4-evidence-investigation-20261023`、指定結果commit `6892e95f`から直接分岐した。作業領域は`non-df-h4-saved-diagnostics-20261023/`。対象旧branch/worktreeは読取専用である。準備中だった別の保存値解析コードは今回の制限を受けて退避し、実H4データに対して実行していない。

構造検査はstdlibのみで、科学moduleをimport/実行しない。保存値の転記、stored headroomの符号とstatusの検査、sourceの制御フロー照合、SHA/ログ/CSV照合だけを行った。新しいheadroom、u/headroom比、N、G、allowance、normalization、event費用統計は計算していない。

資源規約は、repositoryに収録された[2026-10-17統合正式版の写し](progress/track_a/figures/pr_pf_analytic_vs_circuit_CNG_20261022/sources/common_instructions.md)の§8.4を確認した。新しい恣意的なCPU/RAM/worker上限を設けず、共有設定・他ユーザーのprocess/files・既存契約は変更していない。

## 2. 調査A：nullになった実際のコード経路

### 2.1 呼出しと保存順

科学SOURCEの[`h4_pilot_science.py:122`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_pilot_science.py#L122)は、H4 adapter/stateを既存`numerical_evidence`に渡し、`evidence_kind='EMPIRICAL'`で評価する。共通関数のファイル名が`h2_non_df_pilot.py`でも、ここで使われる入力とcontractは今回のH4であり、H2結果の転写ではない。

`evaluate_audited_signal`はreference/raw/corrected/signed_bias、`log_B`、`log_B_upper`を保存する。その後`precision_accounting`を5精度について適用し、**第1精度についても上書き後のshots/total_RZをprimary resultへ戻してから保存する**（`h4_pilot_science.py:139–141`）。一次計算時の仮のshot値が最終受理値になる実装ではない。

### 2.2 直接の棄却条件

[`fixed_signal_accounting.py:26`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/fixed_signal_accounting.py#L26)の既存条件は、各軸について

`h_a = ε_a − |bias_a| − u_a`、`N_a = ceil(2 B_upper² log(2/α_a) / h_a²)`。

これはsourceの式の引用であり、今回Nを評価していない。`h>0`なら基礎会計のstatusはELIGIBLE。`h<=0`でも、`max(0,bias−u)>=ε_a`が成立しない場合はUNDETERMINEDであり、科学的な不適格を断定しない。

その後、[`h2_non_df_pilot.py:269`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h2_non_df_pilot.py#L269)の`precision_accounting`が、`shots is not None`かつ`allowance/headroom > maximum_u_over_headroom`の軸を**UNDETERMINED・shots=nullへ変更**する。今回の固定閾値は0.01。いずれかの軸のshotsがnullなら`usable=false`、`N_total=null`、`total_RZ=null`となる。

保存値とsourceから直接特定できる経路は次の二つである。

- **R（比率guard）**：保存headroomが正、`log_shot_bound`が存在し、最終statusがUNDETERMINED。この状態へ書き換える上記追加guardに到達したことが分かる。元の一時shot数や比率値は再計算しない。
- **H（allowanceと精度の重なり）**：保存headroomが非正、`log_shot_bound=null`、statusはUNDETERMINED。誤差余裕を確保できず、同時に基礎会計のINELIGIBLE判定にも至らなかった。

40軸の構造的な内訳はR=22軸、H=18軸、20行ではR/R=11行、H/H=9行。4方式共通の会計経路であり、方式ごとのu・bias・normalizationの保存値が、どの精度でR/Hとなるかを変えている。**nullは0でも、ELIGIBLEでも、INELIGIBLEでもない。** 全軸の保存statusはUNDETERMINEDである。

### 2.3 4方式×5精度の保存状態

各セルは「cosineの経路 / sineの経路」。**全セルでN/G=null、両軸UNDETERMINED。** 精度や閾値は変更していない。

| case | ε=.05 | .01 | .005 | .001 | .0005 |
|---|---|---|---|---|---|
| pauli_jw_PF | R/R | H/H | H/H | H/H | H/H |
| pauli_jw_PR | R/R | R/R | R/R | H/H | H/H |
| df_native_PF | R/R | R/R | R/R | H/H | H/H |
| df_native_PR | R/R | R/R | R/R | R/R | H/H |

40軸のepsilon/alpha/bias/allowance/headroom/status/log_shot_boundは[precision_state_inventory.json](../artifacts/non_df_h4_evidence_investigation/2026-10-23/precision_state_inventory.json)へそのまま転記した。nullの元値は同ファイルの`stored_N_total`/`stored_G`に保持した。

### 2.4 保存済み数値証拠の意味

以下は元結果からの転記であり、新評価値ではない。uは両軸で同じ。

| case | B | log_B | log_B_upper | u |
|---|---:|---:|---:|---:|
| pauli_jw_PF | 1.0 | 0.0 | .0074791908265582525 | .0074791908265582525 |
| pauli_jw_PR | 1.2873709223899619 | .25260209407554185 | .25409221019507633 | .0014901161195344985 |
| df_native_PF | 1.0 | 0.0 | .0008797645570268785 | .0008797645570268785 |
| df_native_PR | 1.1775851413163583 | .1634658511448318 | .16400229294796304 | .0005364418031312327 |

confidence allocationはcosine/sine各α=.025、union=.05、各軸ε=ε_sig/√2で保存されている。bias・allowance・log_B_upper・allocationの**値の欄が欠測していること**が今回の直接原因ではない。

[`h2_non_df_pilot.py:251`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h2_non_df_pilot.py#L251)のuは、`safety_factor * max(floor, rounding_proxy, agreement)`という経験的見積り。保存されたrounding_proxyがfloor/独立計算法のagreementより大きく、これが選ばれる。`work`は状態空間次元の3乗等を使う宣言済み操作量proxyで、実際の丸め誤差の証明や計測runtimeではない。PFにも共通式中のr/K項が含まれる事実は確認できるが、その項を削れば妥当な上界になるとは今回判断しない。

## 3. 調査B：certificateと数値証拠の不足

### 3.1 falseフラグと実際の判定を分ける

`fixed_signal_shots`の返却値は`certificate_verified=False`を固定している。`evaluate_audited_signal`も`normalization_certificate_verified=False`を固定する。これらは証明器が実行されていないことを示すが、**そのboolean自体を参照してNをnullにする分岐はない**。

`EMPIRICAL`は今回の明示的入力である。`CERTIFIED`も会計関数のenumとして受け付けるが、そのラベルから証明を検査・生成する実装ではない。ラベルを変更するだけで厳密な証拠が得られるわけではなく、今回変更していない。厳密certificateがなくても条件付きshot数を出せる関数設計と、今回固定guardにより全行を保留した結果は区別する。

| 項目 | 本来必要な条件 | 保存されている情報 | 不足・回収可能性 | N/Gとの関係 |
|---|---|---|---|---|
| PF normalization | 決定論経路のB=1と測定範囲の根拠 | law=deterministic、B=1、log_B=0、adapterの非randomized分岐 | 確率的normalizationは不要という構造はsourceから確認可。共通falseフラグをcertificate発行記録には読み替えない | PFの直接棄却もu/headroom。PR固有問題ではない |
| PR normalization | 固定finite lawの正規化と、浮動小数点評価誤差を含む検証済みの包含 | q/r/K、lambda_r、finite order weights/probabilities、B、log_B、log_B_margin、log_B_upper、分布hash | 数式と入力は回収可。directed roundingや誤差伝播の証明記録は未保存 | log_B_upperは計算欄が存在しても厳密な上界証明とは限らない |
| log_B上界 | 保存されたモデル・lawに対応した上界とその適用範囲 | PRは`q*r*log(B_K)`、PFは0、上位で経験的marginを加算 | `exact_finite_distribution`は有限和の法則名で、格納はfloat。`paper_upper_bound`/truncation boundも保存されるが、丸め誤差込みcertificateへ結合した記録ではない | 名前にupper/boundがあるだけでは厳密shot保証を支えない |
| numerical allowance | reference・law mean・normalization・演算誤差を覆う妥当なu | dense/sparseの差、law_mean/raw_normalization gates、rounding_proxy、work、floor/safety factor、u | 経験的式の来歴は回収可。uが全誤差を覆う証明や、より鋭い妥当なuは未取得 | 直接のR/H棄却原因。guard閾値を緩めることで解決しない |
| signed bias | 固定Hamiltonian/state/referenceでのbiasと、その数値誤差の包含 | real/imagのsigned bias、raw/corrected/reference、入力snapshot/state hash | 値は回収可。厳密bias区間・完全なforward/backward error証明は未保存 | 保存点biasだけで適格・不適格を断定できない |
| confidence/estimator | 事前固定のε/α配分、独立な新しい測定とbounded outcomeの条件 | α=.025/.025、union=.05、ε分割、`B_times_fresh_independent_plus_minus_one_mean` | 配分欠測はない。量子shot実行やその実測証拠を取得したrunではない | cost用256 trajectoryは量子shot数Nとは別物 |
| Gの適用範囲 | 有効なNと適用可能な回路単価。母平均保証なら別の費用上界/統計的根拠 | full-wrapper価格・sample mean/MC SE・event/価格join | 保存価格は存在。PRの未観測tail/母平均保証は保存SEからは得られない | Nが未確定なので条件付きGもnull。Nの解消だけで母平均優位性まで証明されない |

Pauli-JW/DF-nativeはHamiltonianの構成、tailとprefix、u/work、bias、Bの値が異なるが、証拠schemaと会計guardは共通である。DF rank10のgate PASSやPauli/DF演算子一致は、この固定入力での技術的一致検査であり、数値誤差全体の厳密certificateとは区別する。

## 4. 調査C：元監視の構成、残った証拠、回収限界

### 4.1 構成と時系列

- 科学supervisor PID71878。入力worker PID71882のstage完了後、4独立sessionのcase worker PID71952/71953/71954/71955を起動・releaseした。各PR case内の256 trajectoryはこの旧SOURCEで逐次処理され、後に作った複数worker版へ途中移行していない。
- 別の読取進捗monitorが`run01/progress_observations.jsonl`へ追記。元のsaved-only後処理PID74215は、supervisor identityを確認しながら元STOPを待つ構成だった。
- **入力stage**は`input_run/input_generation/process.json`と`input_run/STOP.json`にnative exit0、reaped=true、all_children_reaped=true、COMPLETE_MANDATORY_STOPを保存済み。これは回収可能な正常終了証拠である。
- case stageの元ledgerはBATCH_CONSUMED、4 LAUNCHED、RELEASEDで終わる。MANDATORY_STOPはない。進捗monitorの最終151番目の記録は**04:27:49 JST**（2026-10-22 19:27:49 UTC）、DF PRは161 joins/162 event records。他3ケースは保存statusがCOMPLETE_FIXED_CASE。
- 元finalizer保存logの最終行は**04:27:53 JST**、DF PR161 joinsを待機中。このlogはsupervisorのnative終了logではない。sourceでは各待機反復の前にsupervisor identityを確認するが、最終行から正確な消失時刻までは特定できない。
- recoveryの初期記録**05:31:39 JST**では元supervisor/元finalizerは不在。DF PR PID71955は同じstart_ticks=5479158、PPID=1、R状態で239 joins/240 event records。元workerが継続した経緯を保存している。
- recoveryは新しい科学workerを起動せず、既存workerの保存を観測した。**05:45:40 JST**に256/256とworker不在を記録。4caseのstatus/result/validation/精度表が保存されている。
- 結果commitのGitHub時刻は**05:45:50 JST**。外部`/tmp/h4-recovery-859ce9f6.receipt.json`にも同commitとSUPERVISION_GAPを記録している。このreceiptは後処理commitの記録で、元caseのexit/reapを証明しない。

主要記録は[monitor_inventory.json](../artifacts/non_df_h4_evidence_investigation/2026-10-23/monitor_inventory.json)へ転記した。時刻は既存timestampのJST表記であり、消失時刻を補間していない。

### 4.2 存在する証拠と欠けた証拠

4caseともlaunch/ready/worker_consumed/marker、result/status/validation、event/price join、actual call accountingがある。4caseの`stdout.log`/`stderr.log`は存在するが0 bytes。FAILED.jsonはない。ただし、空logやFAILEDがないことをnative exit0の証明には使わない。

一方、全4caseの`process.json`、case stageの`run/STOP.json`、最上位の`run01/STOP.json`、元pipelineの`run01/summary.json`は欠ける。`STOP.request`もない。元finalizerの成功receiptも確認できない。

[`isolated_qiskit_supervisor.py:255`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/isolated_qiskit_supervisor.py#L255)は実行中にPopenのpoll結果をメモリへ保持し、**process.jsonを永続化するのはbatch全体のfinally（319–326行）**、STOP永続化は355–362行である。先に終了した3caseについても、caseの科学result保存とnative process receipt保存は別工程になっている。従って、3caseが保存完了していてもnative終了recordが全件欠けることは、この保存設計と整合する。実際のpoll値を事後復元できたという意味ではない。

元supervisorを失った後に別processから、過去の親Popen/waitpid結果やsubreaper回収の履歴を再作成することはできない。PIDの現在不在、marker、COMPLETE_FIXED_CASE、recoveryの新receiptはその代用にならない。既に存在する外部scheduler/launcherのexit監査が別途提供されれば照合対象になり得るが、今回の既知の保存集合には該当する原記録は見つからない。新しいSTOP/process receiptは作成していない。

### 4.3 原因はUNKNOWN

元case/launcher logに、終了signal、OOM、launcher return code、監視例外やcleanup実行結果はない。契約のrun_timeout/heartbeat_timeout/ready_timeoutはnullである。一時PSI/CPU競合だけで強制終了する設定も確認されない。最終資源観測だけで後のOOMや外部停止を否定することもできない。

外部sessionの終了、catch不能な停止、host/保存系の異常などは可能性として残るが、どれかを示す直接証拠はなく、**原因はUNKNOWN**。finallyの記録がないことからSIGKILL等を断定しない。

## 5. 保存データの整合性と実行終了監査を分離する

元の`postprocessing_recovery01/verification.json`は`SAVED_DATA_VALIDATED_SUPERVISION_INCOMPLETE`。SOURCE738 files、runtime3428 files、旧保全2154 files、入力gates、4case/20行、514 trajectory・16384 production events・1028 compile requests、event/価格hash等を確認した**既存記録**である。本調査ではこれらの物理数値検証や誤差最大値を再評価せず、その固定JSONとsourceを確認した。

今回新たに確認したのは、GitHubの固定内容とローカルbytes、科学SOURCE738 files、対象結果・claim1147 filesのhash、40軸status/保存値、元log/receiptの構造である。入力snapshotはhash照合だけで、配列の読込や状態・信号計算はしていない。

この保存整合性の確認は、欠けた元native exit/reap/STOPを回復しない。反対に、元native終了証拠が欠けることだけを根拠に、保存値が破損したと判断もしない。元verificationの`original_native_completion_verified=false`、`all_children_reaped=null`、N/G=nullは変更しない。

## 6. 調査D：問題別の解決可能性

区分A＝既存記録の確認、B＝新しい保存値の数値再解析、C＝追加科学計算/診断、D＝原証拠欠落のため事後証明不能、E＝原因/解決方法未確定。

| 問題 | 区分 | 固定根拠・判断 | 今回の到達点 |
|---|---|---|---|
| 20行のnullの直接経路 | A | precision_rows.json各軸＋SOURCEの基礎/追加guard | R/Hを特定。nullの意味を保留として確定 |
| certificate falseの意味 | A | 会計/evaluatorの固定False、EMPIRICAL呼出し | false自体によるnullではないと確認 |
| normalization/bias/α/ε/uの欄の回収 | A | result.json、numerical_evidence.json、contract、event distribution | 保存値を転記。証明の充足とは区別 |
| 入力stageのnative終了 | A | input_run/STOP.json、process.json | 既存exit0/reap証拠を回収 |
| 全保存結果・SOURCEのbytes保全 | A | GitHub/commit blob、SOURCE manifest、grant seal | hash一致を確認。数値再実行なし |
| uの必要な改善量・各εでの数値感度 | B | signed_bias/allowance/headroomは存在 | 新しい比率・許容u・shot/G計算は未実施。別承認が必要 |
| 妥当なnumerical error/bias包含を追加取得 | C | snapshot/state/referenceと既存codeはあるが厳密包含なし | 新しい信号/行列/高精度・区間等の診断が必要になり得る。成功を保証しない |
| PR normalizationの数値certificate | C | finite lawは保存、検証済み丸め込み包含なし | 誤差伝播/証明手法の設計・検証が必要。通常float値の再計算だけでは証明にならない |
| 元4caseのnative exit/reap/STOP | D | 全process.json、元2段階STOP欠落。入力stageのものは別 | 既知の保存集合から回収不可。新runでも過去の証拠を補完できない |
| 元監視消失の具体的原因 | E | 最終進捗log、recovery時の不在、空stderr | UNKNOWN。外部原監査の存在は未確認 |
| 証拠追加後に固定5精度でN/Gが確定するか | E | 現在は全行UNDETERMINED | 未実行の結果を予告しない。閾値/候補を変えて成功扱いにしない |

## 7. 最小の次段階案（未承認・未実行）

機械可読案は[next_execution_contract_DRAFT.json](../artifacts/non_df_h4_evidence_investigation/2026-10-23/next_execution_contract_DRAFT.json)。execution_authorized/grant_valid/dispatch_enabledはfalse、SOURCE/namespace/実行commandは未発行である。

1. **B：保存値の数値診断だけを独立承認単位にする。** 固定4case/5精度と現行guardのまま、保存u/bias/log_Bの必要精度と感度を調べる。元N/Gを上書きせず、新しい結果領域へ限定する。今回の調査では比率や新shot/Gを計算していない。
2. **C：必要なら、保存入力を使う数値証拠生成を別契約にする。** 共通snapshot/stateを再利用し、固定4caseのreference/law mean/biasとPR normalizationに対して、誤差伝播を明示した高精度/区間等の方法をまず技術検証する。単なる独立法の一致をCERTIFIEDへ格上げしない。具体的アルゴリズム・証明対象・call数は実装案を確定してからSOURCEへsealし、一括承認対象に含める。
3. この不足を調べる最小範囲では、新SCF/FCI/DF/状態最適化・追加sampling・回路compileは不要と考えられる。既存価格とeventは保持する。N/Gの再会計が必要になれば別の保存会計段階として明示承認する。PR母平均費用の保証や候補探索まで自動拡張しない。
4. 元の監視終了証拠は独立のD問題であり、B/Cを成功させても解消しない。将来、新しい実行終了証拠を得るための新runを行う場合は、永続化設計・新SOURCE・新namespace・新grant・call予算を別途承認する。旧grantのreuse/retry/resumeや旧STOPの補充にはしない。

## 8. 保全、commit、担当、STOP

対象旧worktreeのSOURCE/raw/input/grant/ledger/marker/STOP/claimを変更していない。調査前後のhashは`preservation_before.json`/`preservation_after.json`に保存する。共通の研究数値・4次PF資料・別系列H2の結果には統合していない。調査資料と必要な専用入口だけを新branchへlocal commitする。今回の新規pushは0。取得時点で元commitがGitHubに存在することとは区別する。

次の担当は**ユーザー**：Bの保存値再解析、Cの数値証拠生成の設計/実装/実行、または外部原監査の提示のどこまで進めるか判断する。追加作業を承認する場合、対象・出力・方法・科学call範囲・SOURCEと新namespaceを明示する。**Codexは承認前に新規数値解析・科学計算を開始しない。** 現在の依頼は原因調査のlocal commitまでで終了しmandatory STOP。
