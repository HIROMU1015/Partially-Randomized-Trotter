# Track A H6技術pilot v1：coverage schema STOP・一次証拠

2026-10-10 JST。固定pilotを一回実行し、**原status `H6_TECHNICAL_PILOT_STOP`、worker reason `ValueError:ACTUAL_PRIMITIVE_COVERAGE`**で停止した。
参照matvec・primitive/control probe・trajectory/occurrence・compileは全counter0、correctness0/7・wrapper0/36。
これはinput-reference段階の実装interface STOPであり、H6のPF/RTE/回路正しさ・精度・費用・PR優位性の結果は得ていない。
原source・manifest・grant・STOP・欠測を保全。新sourceへの修正、救済変更、科学再実行を行っていない。

## 条件・来歴・一次artifact

linear H6/1.00 Å/STO-3G、12 modes、α3/β3、sector400、全19 signed fragments生成順、tol-only1e-8/cutoff0。
weighted projection政策維持、T0.8、mid-prefix10/full19/0、q1/q2、partial R4/r2/K2またはK6。
CPU IDs [0,2,5,6]・worker1/Numba4/OMP4/BLAS1を固定。phase1800/1800/3600秒、total7200秒、AS8GiB/output512MiB/log64KiB。
今回はmatrix-free参照に到達しておらず、実Numba thread receiptも欠測。並列性能やH6本計算の所要時間を実測したとはしない。

- 実行source `a643220cde1e24b7e3d637f4bc5d1b0342ce2e86`、実行前seal/認可commit `9bba761da7dd157bc02ef1f63b4023168e87bbdd`。
- [準備](track_a_h6_technical_pilot_preparation_v1.md)、[一回認可・seal](track_a_h6_technical_pilot_execution_seal_v1.md)、[source freeze](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/source_freeze_v1.json)。
- [専用grant](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/authorization_v1.json) SHA `64d3322bb6e383e411189580229da6b12a07dd75ba86377d58b740aff3d24228`、[prelaunch remote proof](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/prelaunch_remote_verification_v1.json)。
- [実行時manifest](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v1/2026-10-10/launch_v1/frozen_pilot.json) digest `f05a6c26ae2d1a19a15808a0dd45b416848cadf872f84b9396c526b77bb8aba5`、[worker claim](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v1/2026-10-10/launch_v1/worker_claim.json)、[exact認可copy](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v1/2026-10-10/launch_v1/authorization_source.json)。
- [worker terminal](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v1/2026-10-10/launch_v1/worker_terminal.json)、[watchdog terminal](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v1/2026-10-10/launch_v1/terminal_status.json)、[最後の進捗](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v1/2026-10-10/launch_v1/progress_0003.json)、[worker log](../../artifacts/resource_applicability/track_a_h6_technical_pilot_v1/2026-10-10/launch_v1/worker.log)。
- [保存bytes監査](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/saved_run_audit_v1.json)、[scope/実行後identity監査](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/execution_audit_v1.json)、[結果要約](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/result_summary_v1.json)、[全raw hashと欠測inventory](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/execution_evidence_inventory_v1.json)。

入力snapshot SHA `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`、Hamiltonian hash `223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`、saved state hash `664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3`。
[親入力結果](track_a_h6_saved_df_completion_parallel_result_v2.md)と同じbytesを使用。実行後にもsource202・親入力33・環境pinを再照合した。
旧Hamiltonian/state・Hermitization policy・rank・seed・gate・caps・H4/Track Bの科学結果は変更しない。

## 完了範囲と欠測

watchdog wall `2.618358921725303` 秒。最後の保存progressのworker peak RSS `359473152` bytesはnative準備前の値で、final peakは未保存。
原raw 14件/287068 bytesを保存し、byte auditorは`SAVED_PILOT_STOP_RECORDS`。この監査はSTOP記録のbytes/認可binding確認でありH6数値PASSではない。
source制御フローと例外位置から、policy-bound snapshot load・sector/basis bridge・7 native preparation・actual_bounds/check_boundsまでの到達を推定できる。
これは保存されたcall counterやprepared receiptで実測した件数ではない。prepared representation hash/actual boundsは保存前にSTOPした。
実sourceのsource-parent-environment再照合はPASS。新SCF/DF再分解/state solver0。signal/probe/sampling/full evolution・Hadamard wrapper build/transpile/compile0。
軌道basisのprepared representation作成は`U_to_qiskit_ops_jw`→`BogoliubovTransform`を内部で使い、basis変換回路を構築する。これはsource制御フローからの推定で件数receiptはない。すべてのQuantumCircuit構築が0だったとは主張しない。

欠測は[保存監査の一覧](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/saved_run_audit_v1.json)に全59件を記録：actual prepared representation/bounds、input reference、parallel receipt、
primitive validation、7 correctness、9 trajectory records、36 wrapper、cost summary、correctness/wrapper phase等。
actual bounds・735 probesのruntime countやinstruction上限の値を後から再計算して旧runの実測として代用しない。
事前に固定した245 keys/735 probesは計画値であり、このrunで実測・保存されたcoverage countではない。

## 静的診断：科学条件を変えずに確認できたこと

[静的coverage診断](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/static_coverage_diagnosis_v1.json)はstdlibの登録scheduleと凍結sourceを照合した資料。real H6再prepare・数値計算・bounds再取得は行っていない。

- [旧actual_bounds](../../src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py)はscheduleに`registered_validation_times_v2`を追加して返す。
- [新fixed coverage](../../src/trottertracks/resource_applicability/ax2b_h6_pilot_contract_v1.py)はその追加fieldを含まない。
- [H6 portのgate](../../src/trottertracks/resource_applicability/ax2b_h6_pilot_port_v1.py)はschedule dict全体のdigestを比較するため、数値が一致しても追加fieldだけで拒否する。

このschema不一致はSTOPの十分な静的原因。runのactual_boundsが未保存なので、他の差異まで存在しないとは結論しない。
準備時の129 local synthetic testsはこのactual_bounds→fixed coverageのinterfaceを直接検査しておらず、不一致を検出できなかった。
新しい科学的比較条件・DF表現政策・信号値への判断は行わない。

## 起動infra記録と一回認可

最初のdetached起動要求はプロセスが維持されず、output directory/worker claimが未作成、ホスト上にも対象runnerがなかった。
[最初の起動要求](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/launch_call_v1.json)、[PID記録](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/launcher_pid_v1.json)、[空stdout](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/launcher_stdout_v1.log)、[infra未起動記録](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/launch_infrastructure_stop_v1.json)を保全した。
source順序ではoutput/bindingを作る前に科学workerをspawn/importしないため、この要求では科学worker0。
続いて継続可能なexec sessionで同じsource/manifest/grantを起動し、worker claim一件と原STOPを保存した。
[session起動記録](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/session_launch_call_v1.json)、[foreground terminal](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/foreground_launcher_terminal_v1.json)。
startup invocationは2、科学worker実行は1、科学retry/resumeは0。一回pilot認可は消費済み。旧grant/outputを科学再実行へ流用しない。

## 引渡しと次の候補（未認可）

現時点でGPTが科学評価できる新しいH6 correctness/cost証拠はない。ここからPRの適用範囲やH6本検証へのGOは判断できない。
[独立レビュー](track_a_h6_df_hermitization_independent_review_2026-10-10.md) §15は方針を変えない通常実装修正をCodexへ委ねているため、
次の候補は旧v1/STOPを保全した新versionのcoverage schema接続・同じ全時間/probe範囲の固定・actual/differenceのgate前保存・synthetic interface統合検証・新sealの準備。
fieldを無条件に捨ててgateを緩和せず、登録validation timesそのものと全scopeの一致を検査する。
数値schedule/probe/gate/primary target/coefficient/独立性を変える必要が出た場合はGPTへ戻す。修正後の科学実行には別の明示認可が必要。
今回は修正・新seal・再実行を行わず、原証拠公開とremote照合まででmandatory STOP。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u/ground-state未認定・UNDETERMINED、next_stage_authorized=false。
