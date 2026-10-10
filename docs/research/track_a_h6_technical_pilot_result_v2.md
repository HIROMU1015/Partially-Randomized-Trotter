# Track A H6技術pilot v2：一回実行の結果・監査・GPT引渡し

2026-10-11 JST。2026-10-10 JSTにユーザーが明示認可した固定H6技術pilot v2を一回実行し、原status **`H6_TECHNICAL_PILOT_STOP`** を保存した。
原reason `PHASE_WALL_CAP:wrapper_cost` でSTOP。correctness 7/7、wrapper 32/36、欠測7件を保存。
この資料はCodexによる実行記録と証拠索引であり、独立科学レビューやH6本検証の認可ではない。
一回認可は消費済み、science retry/resume0、救済変更0。**mandatory STOP、GPT独立レビュー待ち。**

## 固定条件・source・入力・認可

linear H6/1.00 Å/STO-3G、12 modes、α3/β3、sector400、全19 signed DF fragments生成順。
DF tol-only1e-8、cutoff0、final_rank/fallback/fragment削除なし。保存済みweighted projection政策を維持。
T0.8、B0/B2 prefix10、B1 full19、B3 prefix0（one-body deterministic）。二次PF q1/q2、四次PF q1/q2、
B2 R4/r2/K2、B3 R4/r2/K6。epsilon_signal0.001は診断用ラベルで、同一要求精度の適格性を認定する閾値ではない。
delta=T/qはq1で0.8、q2で0.4。この7cellのみを実行、grid拡張・最適化・refitを行っていない。

- 実行science source `0b04886869efb9d08b07d6517300da2bc0123f4a`、準備公開 `6414184d0039d2007cc9f52fef7b0dcd9ad6ec66`、実行認可seal `fc97d46db9dc84d255eaf317d1fea75fd1d1c015`。
- [準備](track_a_h6_technical_pilot_preparation_v2.md)、[実行seal](track_a_h6_technical_pilot_execution_seal_v2.md)、[source freeze](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/source_freeze_v2.json)。
- [固定manifest](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/frozen_pilot.json) digest `2b8bc391d4e7f41d323ed3fe70812da3111a06223a1bc21fa4119532cc153be8`、[専用grant](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/authorization_v2.json) SHA `8799bbac78a0728dbbb18742c2288e6716a4ea3e995b42b0af4159a56d38da0a`。
- [prelaunch remote照合](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/prelaunch_remote_verification_v2.json)、[launch call](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/launch_call_v2.json)、[session receipt](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/session_launch_v2.json)、[foreground terminal](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/foreground_launcher_terminal_v2.json)。
- [親入力完成結果](track_a_h6_saved_df_completion_parallel_result_v2.md)。snapshot SHA `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`、Hamiltonian hash `223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`、state hash `664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3`。

saved amplitudesを再正規化せず使用。新SCF・DF再分解・state solver0。過去のHermitization STOPを成功runで置換しない。
旧[H6 pilot v1 STOP](track_a_h6_technical_pilot_stop_v1.md)・旧source・旧grant・旧rawを保持。
v2はregistered_validation_times_v2を含むcoverage schema接続とgate前保存だけを修正した別source/別execution identity。
数値schedule、probe、gate、rank、policy、seed、caps、cost/signal pathは凍結計画どおり。

## 一次結果と完了範囲

- worker terminalはwatchdog停止時に未保存（欠測）。最後の保存boundary以後の未完了処理量は不明。[watchdog terminal](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/terminal_status.json)、[worker claim](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/worker_claim.json)、[log](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/worker.log)。
- [保存bytes監査](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/saved_run_audit_v2.json)、[実行後source/input/environment監査](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/execution_audit_v2.json)、[結果要約](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/result_summary_v2.json)、[全raw hash・欠測inventory](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/execution_evidence_inventory_v2.json)。
- [保存済みpartial task/event metadata監査](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/saved_partial_bytes_audit_v2.json)、[監査helper](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/audit_saved_partial_bytes_v2.py)、[全raw directory](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/)。helperはrun後に追加したstdlib保存値監査用で、凍結worker sourceの一部ではない。
- [actual preparation/bounds](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/actual_prepared_representation.json)
- [actual instruction bounds](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/actual_bounds_v2.json)
- [actual coverage](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/actual_coverage.json)
- [coverage比較](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/coverage_comparison_v3.json)
- [参照計算](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/input_reference.json)
- [全primitive probe](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/primitive_validation.json)
- [parallel receipt](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/parallel_resource_receipt.json)

- [H6_B0_S2_q2 correctness](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/H6_B0_S2_q2_correctness.json)
- [H6_B1_S2_q1 correctness](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/H6_B1_S2_q1_correctness.json)
- [H6_B1_S2_q2 correctness](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/H6_B1_S2_q2_correctness.json)
- [H6_B1_S4_q1 correctness](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/H6_B1_S4_q1_correctness.json)
- [H6_B1_S4_q2 correctness](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/H6_B1_S4_q2_correctness.json)
- [H6_B2_K2_q2_R4 correctness](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/H6_B2_K2_q2_R4_correctness.json)
- [H6_B3_K6_q2_R4 correctness](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/H6_B3_K6_q2_R4_correctness.json)

|cell / method / PF / prefix / q|saved total signal discrepancy|native–oracle maximum signal discrepancy|log B|
|---|---:|---:|---:|
|H6_B0_S2_q2 / B0 / 2nd / 10 / 2|0.0264142608896|5.73137515187e-15|0|
|H6_B1_S2_q1 / B1 / 2nd / 19 / 1|0.0112836805341|8.80512554375e-15|0|
|H6_B1_S2_q2 / B1 / 2nd / 19 / 2|0.00269773709837|1.93382877443e-14|0|
|H6_B1_S4_q1 / B1 / 4th / 19 / 1|0.00343597838414|1.47492616956e-14|0|
|H6_B1_S4_q2 / B1 / 4th / 19 / 2|0.000236742718454|3.48786772035e-14|0|
|H6_B2_K2_q2_R4 / B2 / 2nd / 10 / 2|0.00269773715901|3.67884423497e-15|0.000205719741307|
|H6_B3_K6_q2_R4 / B3 / 2nd / 0 / 2|0.00220807406831|2.85437454388e-15|20.2133626291|


表は保存されたtotal_absoluteとoracle discrepancyの転記・要約。oracle一致はbinary64の技術的整合性で、
PF/RTE truncation errorやrepresentation errorの認定ではない。B0のdiscard/PF、B2/B3のouter-PF/finite-RTEは
各correctness artifactにsigned complex差を分離保存し、絶対値の加法性は主張しない。raw×Bとcorrectedの照合も保存。
参照Hamiltonianは採用済みDF表現。元integralsとのrepresentation誤差をこのpilotで新たに認定したとはしない。

primaryはsymmetric_directional、ordinaryは同一event trajectoryのpaired sensitivity、cosine/sineを別wrapperとして保存。
36 wrapper予定は9 replica groups×2 control×2 axis。B2/B3は各n2、計4 cost trajectories/8 occurrencesの計画。
実際には32 wrapper/8 complete paired replica groupsを保存、B2 n2・B3 n1、計3 random cost trajectories/6 occurrencesを最後のboundaryに保存。
欠測7件はcost_summary、worker_terminal、B3 replica1 trajectory、wrapper32–35。B3 n2や36 wrapper完了とは主張しない。
これは測定shotではない。compiler・設定・seed・metrics・event_digestは各wrapper/trajectory rawに記録。
random n2でも個別費用のengineering統計に限られ、期待費用・方式winner・精度一致の優位性を確立しない。このSTOPでは最終cost_summary自体が未保存で、後から再集計して原runの出力とはしない。
state preparation費用を含めない。N/Gnull、shot見積りや最終total-cost評価を行っていない。
STOPの場合、未到達の予定を完了事項とはせず、inventoryのmissing_recordsを参照する。

## 古典計算資源・実行範囲・保全

CPU IDs [0,2,5,6]、science worker1、Numba4/OMP4/BLAS1、chunk1、GPUなし。
phase上限1800/1800/3600秒、total7200秒、AS8GiB/output512MiB/log64KiB。
watchdog wall `3913.9031868656166` 秒、最大保存worker peak RSS `7092899840` bytes。
固定wrapper_cost phase3600秒の上限に達してwatchdogがworkerをSIGKILL（exit -9）した。保存されたSTOP理由は時間上限で、科学gate拒否は記録されていない。最終worker確認と全36wrapper完了には到達していない。
これはpilot評価の古典計算資源。量子回路のdepth/CX等とは区別する。RSSは保存boundary/cell記録の最大値で、追加測定はしていない。
最後の保存実行counter `{"compile": 32, "control_probe": 192, "occurrence": 6, "primitive": 735, "reference_matvec": 400, "trajectory": 3}`。source 206件・親入力 33件・環境pinを実行後にも再照合した。
raw 142件/31592752bytes。新しいtest結果を科学的証拠として追加していない。
旧source/freezes/results/contracts/STOP、既存dirty/untracked、root独立レビュー原本、Track Bを保全する。

2026-10-11の[ユーザー指定](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/future_wall_time_policy_v1.json)により、今後新規に準備・個別認可する計算ではphase/total wall上限を設定しない。
このrunの起動時上限・凍結source/manifestは変更していない。将来の新runner/契約へ反映し、メモリ/出力/log/call/instruction上限、科学gate、別実行認可、mandatory STOPは維持。

## GPTが判断すべき論点とSTOP

完了した7cell/保存32wrapper（予定36、B3 n1）の技術的整合性と対称controlledの検証範囲が、H6本検証計画をレビューする根拠として十分か。
保存signed誤差の分離、有限RTE normalization、native sector leakage・独立oracle一致、random n2の限定性、
representation/u/ground-state未認定を踏まえ、必要な追加検証と主張可能範囲を独立に判断する。
Codexは科学GO/STOPや表現・Hermitization政策の変更を決定しない。
`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、N/Gnull、u/ground-state未認定、`UNDETERMINED`、next_stage_authorized=false。
mandatory STOPを維持。H6本検証・H8・追加pilotを自動開始しない。

## GitHub再取得・公開bytes確認

結果保存commit `735475b3cb11a8214eefcf14826bacd90ed528fa` を独立bare repositoryへGitHubから再取得し、[remote照合記録](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/remote_verification_v2.json)を保存した。
science source206・validation source3・親入力33・raw142件のGit blob SHA256/ローカルbytesが一致。旧v1 STOP raw14件も旧結果commitと同じbytes。
相対リンク37件をremote treeで確認。保全対象4266ファイル・root独立レビュー原本43件、既存dirty/untrackedのstatusを保全。
この照合は公開bytesの確認で、数値再評価・sampling・build/compile・科学再実行・科学GOではない。原STOP/欠測を維持しmandatory STOP。
