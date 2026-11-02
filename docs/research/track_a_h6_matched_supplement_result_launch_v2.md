# Track A H6精度一致・測定込み資源比較：補完v3 launch_v2 結果・GPT引渡し

状態：**H6_MATCHED_SUPPLEMENT_STOP / mandatory STOP**。原reason：`RuntimeError:WORKER_EXIT:-6`、runner exit `1`。
これはCodexによる原記録・来歴・欠測の整理で、GPT独立科学レビューや科学的GOの代行ではない。

## 実行identityと一次証拠

Repository `HIROMU1015/Partially-Randomized-Trotter`、branch `track-a-ax2b-h4-post-review-20261010`。
起動準備公開commit `990ef294a36c1dc0bc760a06d35a8f9281a72cd4`、科学source `5a903aea2492c2d211c1e15e7737df0d50694c12`、metadata source `48b489ca3465ef7f4bb202f61c8ca45e9455271e`。
[元科学契約](track_a_h6_matched_supplement_execution_contract_v3.md)、[起動準備v4](track_a_h6_matched_launch_preparation_contract_v4.md)、[sealed manifest](../../artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11/sealed_manifest_launch_v4.json)。
manifest digest `d350e7dcca318eb729e5cfe43fad0d313cb49ec604f79560aeb3e74f28ce2865`、SHA256 `c7e4dcc354515dd9cdd1bf656aa6cea0cf323fefd204585fc801a1e3f5adb617`、reuse digest `c82aa9223c3f67f52085cf821547b92c37b2e110c78e98c2bb256f7c521ead70`。
新[authorization-only grant](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/authorization_launch_v2.json)はcommit `dc651bb83de44fa26d9d268cd0ab94976de4954b`、SHA256 `3564a78601474d18aceaa269d292bb8650b291d42042353fb5dd901331b50a57`。
[独立remote事前照合](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/remote_prelaunch_verification_launch_v2.json)、[同じforeground session内の起動直前検査](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/final_prelaunch_receipt_launch_v2.json)、[一回起動](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/launch_once_launch_v2.json)、[process](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/process_start_observation_launch_v2.json)、[exit](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/runner_exit_launch_v2.json)、[原stdout/stderr](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/runner_stdout_launch_v2.log)。
runner v3を一回だけ呼び、retry/resume・追加科学計算・source救済を行っていない。起動親はwrite probeで確認、最終leafはrunnerのexclusive mkdirでclaimした。
[原terminal](../../artifacts/resource_applicability/track_a_h6_matched_supplement_v3/2026-10-11/launch_v2/execution_terminal.json)、[原inventory](../../artifacts/resource_applicability/track_a_h6_matched_supplement_v3/2026-10-11/launch_v2/execution_inventory.json)、[原raw directory](../../artifacts/resource_applicability/track_a_h6_matched_supplement_v3/2026-10-11/launch_v2)、[外部観察](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/execution_observation_launch_v2.json)。

### Phase Aの技術的STOP証拠

[worker原log](../../artifacts/resource_applicability/track_a_h6_matched_supplement_v3/2026-10-11/launch_v2/cost_exploration_0447/worker.log)は`memory allocation of 1200 bytes failed`。最後の[ResourceTrace](../../artifacts/resource_applicability/track_a_h6_matched_supplement_v3/2026-10-11/launch_v2/cost_exploration_0447/diagnostic_0019.json)はstage `transpile`、axis `cosine` の`begin`、elapsed `524.126911923755` 秒。
これは開始記録で、transpile完了やmetric取得ではない。workerはexit−6、supervisorが`RuntimeError:WORKER_EXIT:-6`を保存。worker terminalとfailure_diagnosticsは未保存で、後から例外stackを補作しない。
[技術診断要約](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/phase_A_diagnostic_summary_launch_v2.json)には原log、最後のstage/memory、compile_attempt、validation、batch_finallyの対応を保存する。
logの1200 bytesは失敗したallocation要求量であり、必要な追加メモリ量や回路全体のメモリ量ではない。AS8 GiBは変えず、last VmSize/RSSとaggregate batch peakを別記する。
原inventoryの成功batch peak0・source_input_after_verified=falseは保全し、外部の事後static identity PASSと区別する。
新しい完了費用groupやwrapperは得られていない。Phase B/Cは未開始。再試行、memory増加、compiler変更、B1 S4 q8除外、proxy補完は行っていない。

## 科学条件とcoverage

linear H6、1.00 Å、STO-3G、12 modes、alpha3/beta3 sector400、保存state、T=0.8。
DF tol-only1e-8/rank19/cutoff0、signed generation orderと採用weighted Hermitian projectionを維持。
B0/B2 prefix5/10/15、B1 full19 S2/global Yoshida S4、q1/2/4/8、B2 r1/2/4・R=qr・K2/4、delta0.8/0.4/0.2/0.1。
92候補、epsilon_sig0.05/0.01/0.005/0.001、normalization・経験的数値allowance・Hoeffding shot会計・選抜・seedを変更していない。
quantum shotは解析会計で、測定shot samplingではない。finite-time signal精度を化学的energy精度へ換算しない。

| 項目 | 保存旧結果再利用 | 新規取得・受理 | 残り |
|---|---:|---:|---:|
| signal | 92 | 0（再計算なし） | 0 |
| B3 normalization row | 84 | 0（再計算なし） | 0 |
| 原探索group | 449/592 | 0 | 143 |
| wrapper | 902 | 保存record 0 / 完了group受理 0 | 欠測をproxyで補わない |
| 独立確認group | 0 | 0 | 登録None、未選抜なら未登録 |

[全592 groupのsource別来歴・未完了一覧](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/missing_coverage_launch_v2.json)。TASK_COMPLETE/両軸/event/seed/compilerを満たすgroupだけを受理し、partial wrapperを完了groupへ昇格しない。
旧source `54986a9f54515c8c2289e38fa16fd81e2b6008c2`、旧結果 `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb`、[旧原結果](track_a_h6_matched_accuracy_result_v2.md)のMemoryError STOPは変更していない。
[旧v3起動時STOP](track_a_h6_matched_supplement_launch_stop_v3.md)と旧grantも保全。旧launch_v1は未使用のまま。
Phase Aは原index447 `B1_p19_q8_4th`を単独取得。成功時だけ残142、全592受理後だけ原B2上位2/精度・union最大8/独立32/X/Yへ進む条件を維持。
探索8と確認32のpool、候補変更、追加sample、H8は行わない。

登録されたmap・選抜・統計・order診断の保存状況：

- `integrated_exploration.json`：欠測。
- `exploration_resource_map.json`：欠測。
- `confirmation_tasks.json`：欠測。
- `confirmation_resource_map.json`：欠測。
- `cost_order_diagnostics.json`：欠測。
- `aggregate_calls.json`：欠測。
- `cost_coverage.json`：欠測。

## 資源・実行監査と限界

coordinator AS soft2/hard8 GiB、cost worker8/8、CPU [0,2,5,6]、cost最大2 process/threads1、aggregateRSS18 GiB。
Phase Aは単独。phase/total wall cap=null、GPUなし、既存output/log/call/instruction guard・新wrapper最大798・元上限1708を維持。
compiler basis rz/sx/x/cx、optimization1、seed17、backend/coupling/layout/routing指定なし。full measured Hadamard wrapper、state preparation除外。
sector/prepared cache、disposable process、streaming fingerprint、single-owner wrapperは固定sourceのまま。実測比較のない高速化倍率は主張しない。
[静的audit](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/static_identity_audit_launch_v2.json) exit `0`。STOP時PASSは保存bytes/lineage/limitsの整合性で、全探索完了や数値certificateではない。
原inventoryのpeakは成功して戻ったbatchだけの場合があり、失敗batch peakは原progressのbatch_finallyを参照する。snapshotから失敗allocationのpeakを推定しない。
compile_attemptはfingerprint前の登録で、実transpile開始と同義ではない。旧中断call総数はnull、不明を0へ変更しない。
数値allowanceは経験的、SD/SEは有限標本の記述量。母平均winner・PR優位性・ground-state/化学的精度は未認定で`UNDETERMINED`を維持。
[保全](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/preservation_launch_v2.json)、[証拠inventory](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/execution_evidence_inventory_launch_v2.json)、[GitHub再取得照合](../../artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/remote_result_verification_launch_v2.json)。
科学source224/validation5・metadata2/test1、旧source216/旧raw5220、保存入力33を照合。既存dirty/untracked・旧STOP・grant・freezeを保全。

## GPTへ戻す判断と停止

原結果とResourceTraceに基づく技術的失敗の解釈、完全なH6 mapが得られた場合の比較結果・有限標本限界、追加検証の必要性はGPT独立レビューへ戻す。
Codexは科学的GO/STOP・PR優位・H8可否・論文化の最終判断を代行しない。
今回のgrantは一回起動で消費済み、retry/resumeなし。**H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / next_stage_authorized=false / H8_authorized=false / mandatory STOP**。
