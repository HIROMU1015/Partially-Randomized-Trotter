## 2026-10-26 B3 canonical fingerprintの高速化

`h46_b3_fast_fingerprint.py`は、同じcanonical v2 byteを生成する処理で、標準gateの定義とbit位置の反復処理を削減する。毎回、現在のparameter・phase・custom definitionを確認し、callをまたぐobject ID/digestの再利用は行わない。compiler・回路生成・標本数・seedは変更しない。

`h46_b3_accelerated_run.py`で保存済み代表入力を旧実装と比較した。linear1Å/STO-3G/DF10・19、B3 L_D=0、T=.8/q8/S2、H4 R4/K2・H6 R8/K6。両軸を毎回同じcompilerで新規compileし、価格6指標・event byte・canonical SHAと全作用検査が一致した。H4 23.79→12.36秒、H6 103.55→63.34秒。この2入力の実装比較を統計標本へ加えず、全研究の短縮率とは区別する。公開記録は`artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/acceleration_validation.json`。対応する人工テストは`test_h46_b3_fast_fingerprint.py`。

高速版は、現行run03をそのまま完了させた後の復旧・独立確認用として起動する。未着手分を途中で引き継ぐための現行controller制御は、自動承認審査が既存の非中断条件と再開整合性へのリスクを理由に拒否したため実施していない。

## 2026-10-26 B3 run03のSIGBUS診断と未完了分の復旧

`h46_b3_trace_recovery.py`は、run03の正常workerを自然終了まで保ち、技術的SIGBUSの未完了分を固定seedで1回だけ復旧する。小さい2件の診断は別private directoryへ保存し、元controller終了後にPASSとevent再現・result SHAを確認して同じcohort/indexへ転用する。完成済みtrialは再取得しない。SOURCE・入力・compiler・候補・n・seedは維持し、科学STOPやSIGBUS以外の異常は自動で再開しない。

`strace`のsignal記録で再発時のsi_code等を取得する。人工SIGBUSの実動作、tracee PID/RSSの参照、終了signal伝達を含む人工7件と既存B3/復旧6件の計13件PASS。SIGBUSの原因は現時点では未確定で、fresh spawnだけで解消したとはしない。自分専用service `prt-h46-b3-20261026-run04.service`はrun03の自然終了を待ってから既存scheduler・会計・独立確認・図表作成を継続する。対応する`complete-run04.service`が最終確認とlocal commitを行う。新たな失敗を再帰的にretryせず記録して停止する。

## 2026-10-26 実行中B3探索の並列化補助

`h46_b3_prefetch.py`は、未着手の固定B3探索seedに対応するfull-wrapper compiled cacheを空きCPUで先行作成する。本体`prt-h46-b3-20261026-run03.service`の実行SOURCE・scheduler・進行中trialは変更しない。同じsampler/native builder/compilerとfingerprint排他lockを使用し、追加の統計標本やtrial resultは作らない。本体で必要なnative/compiled作用チェックと受理を行う。

補助serviceは`prt-h46-b3-20261026-prefetch-v2.service`。CPUの実測空きと両処理の将来メモリ増加を考慮する。今回のfresh代表trialが全チェックを終えてPASSし、event数が比較可能な場合に限り、ピークRSS実測に25%の余裕を付けて補助側の予測を更新する。小さな実測から2倍を超えるevent数には外挿しない。現時点の低いRSSだけで予測を下げない。異常時は補助の追加投入をやめ、正常なworkerと本体計算は中断しない。

進捗は`artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/parallel_prefetch.json`、固定seed・fingerprint・実測ピークは専用private `parallel_prefetch/`。最初の1 trial相当の両軸compile完了を確認した。新しいメモリ投入条件の人工7件と既存B3/復旧6件の計13件PASS。処理全体の短縮率は未確定。終了処理`h46_b3_finalize.py`が先行compileを別記し、受理済み標本へ二重計上しない。

## 2026-10-26 H4/H6 長時間PF・PR比較

[REPORT](../REPORT.md)と[成果物](../artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/)。科学取得は終了済み。以下の取得driverを再実行せず、保存済み結果を参照する。

- `h46_long_time_signal.py`：保存入力・signal/shot評価。`h46_long_time_extension.py`：S2限定q32追加。
- `h46_long_time_native.py`：既存native builder・compiler・paired統計を再利用した単価取得。
- `h46_long_time_report.py`：固定独立確認設計・会計・図表。`h46_long_time_fixed_delta.py`：固定δ=.2不足候補だけ取得。
- `h46_long_time_pipeline.py`：取得後の段階処理。報告だけ出す場合は`report()`を使い、取得pipeline全体を起動し直さない。
- `h46_long_time_recover.py`：今回の実行セッション消失後に未完了だけを再開した処理。中断履歴と同じseed/eventを保全し、完成試行を再取得しない。
- `h46_long_time_extension_fill.py`：未着手q32試行の並列処理。`h46_long_time_saved_metrics.py`：旧価格指標の回収cache。`h46_long_time_k6_probe.py`：条件付き正しさ確認（費用標本と分離）。
- 人工テスト：`tests/test_h46_long_time_{signal,native,accounting,recover}.py`。科学的性能や母平均保証の証拠とは別。

## 2026-10-22 H246共通決定論PF：588候補の固定比較完了

[全結果・契約・監査](../docs/research/track_a_h246_pf_common_final_20261022.md)。588候補の評価が全て完了。H2/H4は再計算せず参照結合。H6不足63metadata・新102signal・1380 endpoint・固有新価格209候補418軸を取得し終了監査PASS。旧187価格は保存値受入監査、新22価格は新grant取得。中断予約を含むcompile実呼出438軸、旧parentのexit/reapは不明のまま保全。linear1Å/STO-3G/tol-only rank3/10/19、固定L_D/q≤8・δ=.8/.4/.2/.1/T=.8/P=0。ε=.0005の全公式PF対保存PR削減率：H2 -25.14%、H4 +37.37%、H6 +40.96%。経験的allowance・有限探索条件付きで母平均保証なし。旧SOURCE/raw/grant/STOP不変、local commit後mandatory STOP・push0。新経路 `h6_pf_metadata_completion.py`、監査 `audit_h6_pf_metadata.py` / `audit_h6_pf_common_run.py`、固定参照統合 `finalize_h6_pf_common_results.py`、数値別照合 `check_h246_pf_common_final.py`。stage driverと消費済みgrantの再使用は禁止。以下は履歴。

## 2026-10-22 H6共通PF：新grantで残22候補の取得開始

[承認・SOURCE・予算・起動証跡](../docs/research/track_a_h6_pf_recovery_execution_start_20261022.md)。明示再開指示を直前の追加内部予算案に結合し、新SOURCE bdf6772aa・新pack/新grantで残22候補44軸を起動。旧187完成価格とSIGNAL/QUALIFYは参照し再計算0、旧394 compile予約を非返却維持。22 worker、正常終了後に別ACCOUNT grant・監査・H2/H4参照結合・local commit/mandatory STOPへ進む。完了/順位は未確定、旧STOP不変、push0。新経路 `h6_pf_recovery_execution.py` / `run_h6_pf_recovery_pipeline.py`、終了処理 `finish_h6_pf_recovery_results.py`、人工 `tests/test_h6_pf_recovery_execution.py` / `tests/test_h6_pf_recovery_exports.py`。以下は履歴。

## 2026-10-22 H6回復制御の修正・科学実行なし

[修正・人工検証・未承認仕様](../docs/research/track_a_h6_pf_recovery_control_repair_20261022.md)。起動処理へ異常終了STOP・SIGTERM/SIGINT時の自分の子process回収・終了heartbeat・PID再使用検査を追加。人工13件と自分専用user service実動作PASS。残22候補の元入力/完成価格/control計6504照合項目を実行無効仕様へ封印。追加予算は未承認、科学call/新grant/push0、H6全域比較UNRESOLVED。旧科学SOURCE・raw・grant・STOP不変。制御 `run_detached_local_job.py`、非実行準備 `prepare_h6_pf_cost_recovery.py`、人工 `tests/test_h6_pf_recovery_preparation.py`。旧full-run driverのresumeは不可。以下は履歴。

## 2026-10-22 H6 metadata完了・COST外部中断

[取得結果・中断監査・予算差分](../docs/research/track_a_h6_pf_cost_interruption_20261022.md)。不足63 metadata・新102 signal・1380 endpointはPASS。COSTは187 pair完成後にprocess不在、残22 pair・H6全域比較はUNRESOLVED。完成1122 archiveを別経路で保存値監査し、394 compile予約を非返却計上。元手法のfresh22候補案は内部piece上限を1229超えるため未実行。H2/H4再計算・新PR sampling・push0。linear1Å/STO-3G/tol-only rank19、B0/B0I L_D5..15・Full19、q1/2/4/8、T=.8/P=0、δ=.8/.4/.2/.1。旧SOURCE/grant/ledger/marker/STOP保全。中断原因・元exit/reapは不明。ローカルcommit後mandatory STOP。監査 `audit_h6_pf_interruption.py`、自分専用一時service起動 `run_detached_local_job.py`、人工 `tests/test_h6_pf_interruption.py`。launcherは科学実行承認を追加しない。以下は履歴。

## 2026-10-22 H6 loaderの修正と実行前検査

[修正・人工検証・保全](../docs/research/track_a_h6_snapshot_loader_repair_20261022.md)。H6 loader修正：scalar shape規約を維持しgrant前にNPY header/dtype/SHA検査。人工26件・保存8配列byte照合PASS。科学再実行・新分子grant・push0、旧STOPとH2/H4結果不変。 checker `pf_common_snapshot_receipt.py`、人工 `tests/test_pf_common_snapshot_receipt.py`、接続先 `pf_common_run_native.py` / `pf_common_run_contract.py` / `h6_pf_metadata_completion.py`。以下は履歴。

## 2026-10-22 H6 metadata取得：snapshot読込後のloader shape検査でSTOP・修正を保存

[実行・STOP・修正・監査](../docs/research/track_a_h6_pf_metadata_STOP_20261022.md)。不足63件を新grantで起動したが、snapshot_restore1後に0次元scalar shapeの比較不具合でexit1。入力SHAは原値一致。metadata/signal/compile/後続stage0、失敗時STOP条項に従いretry0。別SOURCE修正・人工5PASS、旧H2/H4結果75file不変。588登録中312完了/276未解決を維持。local commit後mandatory STOP、push0。実装 `h6_pf_metadata_completion.py`、人工 `tests/test_h6_pf_metadata_completion.py`、数値依存 `pf_common_run_native.py`。消費済みMETADATA grantを起動し直さない。以下は履歴。

## 2026-10-22 H2/H4共通PF比較完了・H6原metadata STOP

[条件・全候補・費用・監査](../docs/research/track_a_pf_common_domain_results_20261022.md)。588登録中H2 84/H4 228を評価完了、H6 276は63原metadata不足でUNRESOLVED。新signal84・compile169候補338軸・1560 endpoint、同native保存22 pair再利用。全公式PF対PRではH2は5精度ともPF低費用、H4はε=.001/.0005でPRが15.01/37.37%低費用、緩い3精度はPF低費用。linear1Å/STO-3G/tol-only rank3/10/19、T=.8/P=0、固定prefix/q≤8域・経験的allowance・保存探索cohort条件付き。両系4 stage/終了監査PASS、旧PR不変、新sampling0。全energy/RPE最適化・母平均保証ではない。local commit後mandatory STOP、push0。runner: `prepare_pf_common_run.py`（pack封印）、`run_pf_common_run.py`（4 stage一回実行）、`pf_common_run_contract.py`、`pf_common_run_native.py`、`audit_pf_common_run.py`（読み取り監査）。人工 `tests/test_pf_common_run.py`。消費済みgrantを再実行しない。以下は履歴。

## 2026-10-22 S2/Yoshida4/m5決定論PF：共通探索域の準備完了・科学実行未承認

[候補照合・固定追加予算・未解決条件](../docs/research/track_a_pf_common_domain_preparation_20261022.md)。H4 DF10 p1..9/H6 DF19 p5..15、q1/2/4/8、Full/B0/B0Iを3公式で統一し504候補。既存354候補（raw326/固定endpointのみ28）、不足150signal＝75基礎＋75overlay、独立forward75、compile最大373候補746軸。217候補の旧費用接続成功時は最大156候補312軸。H6原metadata28件と旧profile/runtime接続が未解決。新23＋既存30人工PASS、実分子call/新費用順位/有効grant/push0。既存PR結果不変、専用branchローカル保存、mandatory STOP。以下は履歴。

H246固定評価の別経路照合：`audit_h246_reconstructed_results.py`。全3stage終了後の保存整数moment・paired covariance・台帳とraw目録の監査。

H246再構築確認：`h246_reconstructed_contract.py / run_h246_reconstructed.py`。新packで元native数値・固定118/236取得と固定評価を接続する。`tests/test_h246_reconstructed.py`で境界と人工3公式を検証。

H246 reference復元：`scripts/run_h246_reconstruction_forward.py`。固定3 reference＋欠落H2 signal2を別枠で封印する。

## 2026-10-22 H246移行再構築

[run_h246_migration.py：元生成式と新規grantを分離する実行入口](../docs/research/track_a_h246_migration_execution_20261022.md)。旧STOP保持。人工34PASS、科学入力生成は別枠one-shot、確認は固定118trajectory/236軸。

## 2026-10-21 H246高次PFの保存値統計・総費用

[実行結果・範囲・限界](../docs/research/track_a_h246_saved_cost_results_20261021.md)。純粋な保存値計算`h246_saved_cost_evaluation.py`、専用preparer/one-shot runner、人工23件を追加。新sampling/compileは0。集約JSON・全1000行CSV・30比較・60費用分解・36単価統計・144層を保存し、SOURCE/認可/終了監査を結合した。母平均保証・独立確認P・AD03採択は行わない。

## 2026-10-21 H246保存値評価の実行前gate

[政策未採択STOP・根拠・採択案](../docs/research/track_a_h246_higher_pf_saved_evaluation_preflight_STOP_20261021.md)。`scripts/audit_h246_saved_evaluation_preflight.py`はmetadata/SHAだけを照合し、科学moduleを呼ばない。人工検査は`tests/test_h246_saved_evaluation_preflight.py`。統計・shot・総費用の実行は停止中。

H4/H6高次PF PR本探索の実行契約・軽量高速化：[実行範囲](../docs/research/track_a_h46_pr_exploration_execution_20261021.md)。取得後mandatory STOP、統計・会計・AD03は含まない。

## 2026-10-20 H4/H6高次PF：本探索標本数設計・実行準備（未採択）

[設計・予算・契約・次の承認](../docs/research/track_a_h46_pr_exploration_readiness_20261020.md)。H4/H6各12群の保存pilot統計・216名目nを再利用し、ρ5/2/1%×分散倍率1/2/4×群別/4q共通×名目のみ/下限33の36未採択案を用意。第一候補ρ2%・f4・群別・下限33はH4 793/H6 607本・計2800軸。EXPLORATION別2無効契約、SOURCE96259cbe、seed15879個/既知209156個衝突0。新16/回帰88・CLI6拒否/改変10拒否PASS、旧17867file不変。新実分子sampling/compile/signal/shot/会計/順位0、n_E/shot/AD03保留。ユーザーの設計採択・系別取得承認待ち、mandatory STOP。以下は履歴。

## 2026-10-20 H2/H4/H6高次PF：既存再利用とpilot評価完了

[結果・標本設計案・残る限界](../docs/research/track_a_h246_higher_pf_pilot_evaluation_results_20261020.md)。H2の5精度統合比較はcommit2d415da2から転記のみ。H4/H6各12群×33本を初評価、paired共分散/MCSE・層診断・216名目n_E案を保存。相対MCSEはH4約0.094〜2.776%、H6約0.103〜2.244%、S4全24群未観測。2 process正常exit/reap・新12/回帰6/拒否8PASS。新sampling/compile/shot/総費用順位0、n_E/AD03未採択、mandatory STOP。次はユーザーのρ/f・tail扱い・n_E判断。以下は履歴。

## 2026-10-20 H6 PR pilot：全396本の取得範囲が揃い終了監査PASS

[完了・SOURCE・資源・残る承認](../docs/research/track_a_h6_pr_pilot_continuation_run01_results_20261020.md)。新SOURCE 0ff0f704で322/322本・644軸、30 worker全exit0/reap、約58分6秒。1932 archiveのnative/原SHA復元・終了監査PASS。保存74本の再監査と参照結合により396登録/792軸を被覆、旧STOP保持。統計/MCSE/n_E/shot/会計/AD03未実施、mandatory STOP。次はユーザーのpilot数値評価の別承認。以下は履歴。

## 2026-10-20 H6 PR pilot：74本受入・残り322本を並列起動

[継続契約・起動gate・進捗](../docs/research/track_a_h6_pr_pilot_continuation_run01_start_20261020.md)。SOURCE 0ff0f704、保存74本を独立受入し元固定322登録/最大644軸を一回起動。元seed/33本×12群・旧STOP保持。fresh gate/人工8＋回帰16/改変拒否11PASS、context共有後30worker。完了/native exit/reapは未確定。統計/n_E/shot/会計/AD03は保留、終了時mandatory STOP。以下は履歴。

## 2026-10-20 H6 PR pilot：cutoff修正・軽量高速化（再実行未承認）

[修正・人工検証・保全・再開提案](../docs/research/track_a_h6_pr_pilot_cutoff_repair_20261020.md)。SOURCE e7595e8f。cutoffの演算順序差を保存metadataで特定し、factoryと同じτ計算へ修正。H6専用opt-inのcontext共有・実空きCPU/RAMによる並列数・SHA検査付きJSON cacheを追加。新16＋回帰161件PASS。74完了pair/444 archive/4404片を読取照合、旧STOP保持。新実H6計算・有効grantは0。残り322本/最大644軸は別受入・継続契約と実行承認待ち。実H6の高速化率は未測定、AD03/統計/会計は保留、mandatory STOP。以下は履歴。

## 2026-10-20 H4/H6 DESIGN_PILOT：33本/群の取得を承認

[新保存契約・実行範囲](../docs/research/track_a_h46_pr_design_pilot_execution_20261020.md)。新規QPY/NPZだけをbyte可逆level1圧縮し、manifest永続化・原SHA復元確認後に専用raw一時fileを解放する。H4/H6各396 trajectory/792軸、独立one-shot。実行状況・終了PASSは別の取得receiptを正とし、この契約記述から認定しない。n_E/統計評価/shot/会計/AD03は保留、取得終了後mandatory STOP。以下は履歴。

## 2026-10-20 H4/H6 m5：決定論的費用92軸完了・終了監査PASS

[単価・終了監査・次の担当](../docs/research/track_a_h46_m5_det_cost_run01_results_20261020.md)。H4 38/38・H6 54/54軸、全46pairを正常終了、両native exit0・全worker回収。科学SOURCE635ba563…不変、metadata監査c1b419cd…/人工5件、終了940項目PASS・保存368gate・台帳1012/1430一致。旧3137file不変。新92＋保存4＝適格m5決定論48候補96軸を直接実測区分で転記。PR24群n_E・shot政策・AD03は保留、会計/順位/fit/自動retry0。両契約mandatory STOP、次はユーザーのPR標本設計と後続会計の個別判断。以下は各時点の履歴。

## 2026-10-20 H4/H6 m5：決定論的費用run01を独立2契約で起動

[承認・起動gate・進捗snapshot](../docs/research/track_a_h46_m5_det_cost_run01_start_20261020.md)。H4 38軸/H6 54軸を独立承認・grant・ledger・markerで起動。科学SOURCE635ba563…は不変、起動補助9e44ec88…/人工5件PASS。SOURCE/input/compiler・3137file保全・fresh RAM/scratch gateを通過、各3 workerで進行。compiled CT01 1e-7未満/他1e-9未満、異常時は当該stageのみSTOP・retryなし。全PASS/最終費用/exit/reapは未確定。PR/shot/会計/rankingは未承認、AD03保留。次はCodexの終了監査・結果報告後mandatory STOP。以下は各時点の履歴。

## 2026-10-20 H4/H6 m5：回路費用取得・標本設計準備

[取得対象・再利用・未採択事項](../docs/research/track_a_h46_m5_cost_sampling_readiness_20261020.md)。新SOURCE635ba563…、保存88候補/880軸ラベルを同精度でjoin。m5決定論48候補中保存4軸を再利用し、追加H4 38/H6 54軸案。12保存軸は同m5系列内でidentity照合、8軸は今回の適格領域外。PR24群のn_E・新seed・総call数は未確定。native adapter/並列pool/4独立無効契約を実装、人工18＋回帰18・metadata84PASS、3137file不変。実分子signal/compile/sampling/shot/会計/ranking・有効grant0、AD03保留。次はユーザーの対象/標本設計/資源政策と独立実行承認の判断。公開資料のみcommit・push後mandatory STOP。以下は各時点の履歴。

## 2026-10-20 H4/H6 m5：修正版signal run02正常終了

[signal結果・経験的適格性・終了監査](../docs/research/track_a_h46_m5_signal_run02_results_20261020.md)。修正版SOURCEbec018c6で固定88候補（82新signal＋6raw再利用）を正常終了。H4 40/H6 48、5精度両軸880行、2系process・各30候補thread、wall57.68秒/外部61.41秒、全exit0/reap。47終了監査/2065seal、旧2899file＋run01 STOP18file不変。reference条件付き経験的政策、shot/costはnull、AD03保留。新compile/sampling/reference/会計/ranking/push0。次はユーザーの費用取得計画準備の判断、local commit後mandatory STOP。以下は各時点の履歴。

## 2026-10-20 H4/H6 m5：signal run01初期化STOP・端点修正

[実行STOP・原因・修正・次の判断](../docs/research/track_a_h46_m5_signal_run01_stop_and_fix_20261020.md)。承認済み固定88条件のrun01を起動したが、両系がfull-prefixの打切り専用helper呼出しで初期化STOP。候補signal/ledger0、外部exit1・全回収。旧SOURCE480ca450/run01を保全し、別worktreeのSOURCEbec018c6で空tail分岐と元SOURCE path検証を修正。人工18・metadata37PASS、2899旧file＋18run file不変、新2064seal・無効grant。新科学retry/compile/sampling/会計/push0、AD03保留。次はユーザーの修正版一回実行判断、local commit後mandatory STOP。以下は各時点の履歴。

## 2026-10-20 H4/H6 m5：基本88条件のsignal実行準備

[実装・契約・検証・次の担当](../docs/research/track_a_h46_m5_signal_readiness_20261020.md)。基本88条件のsignal専用adapter・prefix別context・2系process/候補thread並列・one-shot契約を準備。SOURCE480ca450…、82新signal上限＋6raw再利用・保存reference2件。人工14＋既存回帰16、metadata17・改変拒否7PASS、2046seal/2899file保全。allowanceはreference条件付き経験的提案、未採択。実分子signal/compile/sampling/会計/有効grant/push0、AD03保留。次はユーザーの限定signal実行・allowance使用の判断。local commit後mandatory STOP。以下は各時点の履歴。

## 2026-10-19 H4/H6 m5：最小比較範囲の新DRAFT

[範囲・再利用・並列計画](../docs/research/track_a_h46_m5_minimal_comparison_plan_20261019.md)。旧ordinary/DF12のq比較と、新tol-only directional pilotの条件差を明示。基本88 signal条件（3公式PR24＋m5対照64）、outer32/K4/q1/2/4/8を提案。局所20は別枠、12軸を再利用候補へ。n/allowance/科学SOURCEは未seal、AD03保留、実signal/compile/再会計0。次はユーザーの範囲判断とCodexのsignal-only契約準備、local commit後mandatory STOP・push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：CT01 sine一軸正常終了・両軸receipt取得

[実測・終了監査・次の担当](../docs/research/track_a_h6_m5_ct01_sine_run01_results_20261019.md)。SOURCE885bbd71…、元compilerでsine新compile/作用各1、raw1.494e-9でCT01の1e-7未満PASS。保存cosine1.504e-9も別受入、両軸831662 RZ／軸。wall約31分20秒/native exit0/全回収、63終了監査/2025seal/2816file保全。旧58軸・STOP不変、AD03/会計保留、retry/次Phase/push0。次はユーザーの後続範囲判断、local commit後mandatory STOP。以下は各時点の履歴。

## 2026-10-19 H6 m5：CT01で未実施sine一軸を一回起動

[承認・fresh gate・進行snapshot](../docs/research/track_a_h6_m5_ct01_sine_run01_start_20261019.md)。SOURCE885bbd71…、元compiler/保存cosine831662 RZと先行PASSを再利用、sine新compile/作用各最大1。1e-7未満/他1e-9を保持、2025seal/2816file保全、一回grant消費。進行中・最終raw/単価/native exit/reapは未監査。次はCodexの終了監査・結果報告後mandatory STOP。旧STOP/58軸/AD03保留、会計/retry/次Phase/push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：CT01保存cosine受入・sine一軸準備完了

[保存受入・新契約・検証・次の担当](../docs/research/track_a_h6_m5_ct01_sine_readiness_20261019.md)。SOURCE885bbd71…、元compilerのcosine raw1.504e-9をCT01の1e-7未満で別受入、831662 RZ/軸。旧STOP/58軸は保持。未実施sineだけの1compile/1作用を準備、terminal H Sdg Hを人工照合、28test/metadata16/改変拒否10PASS。2025seal/2791file保全、新実分子compile・作用・有効grant・会計0。次はユーザーの一回実行判断、local commit後mandatory STOP・push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：丸め0の代数的縮減の実検証・SINGLE_STATE_PASS

[結果・実行証拠・採用範囲](../docs/research/track_a_h6_m5_algebraic_zero_run01_results_20261019.md)。SOURCE0d47f59c…、保存PASS再利用・cosine新compile1/作用1。追加丸め0、Euler/小角度閾値/wrapなし、全raw検査は保存結果参照。診断gate数と受理費用を区別し、旧STOP/58軸/AD03保留。40項目終了監査、2016seal/2758file保全、会計/追加軸/retry0。次はユーザーの残る検証・政策採否判断。local commit後mandatory STOP・push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：丸め0の代数的1qubit縮減・実行準備完了

[方式・人工比較・未承認一軸契約](../docs/research/track_a_m5_algebraic_oneq_readiness_20261019.md)。SOURCE0d47f59c…（prototype29f5d5f0…）、正確なRZ和とSX/X恒等式でEuler/閾値/wrap/phase加算を避ける。人工30件/14条件PASS、丸め0の人工RZ24→1と小角度全保持を確認。H6での縮減・raw精度・時間は未検証。2016seal、metadata20/改変拒否11、2720file保全、実H6 compile/作用/有効grant/会計0。旧PASS/STOP/58軸/AD03保留。次はユーザーの新SOURCE・固定cosine一軸の実行判断、承認後Codexが新grant/fresh gateで一回実行。local commit後mandatory STOP・push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：1qubit保持対照の実検証完了

[結果・旧STOPとの区別・費用上の制限](../docs/research/track_a_h6_m5_preserve_oneq_run01_results_20261019.md)。SOURCE68ee03a7…、H6でraw誤差の改善を確認し、登録した3検査を全て通過した。 保存PASS再利用・新compile1/作用1、1qubit再合成全体を外す対照なので局所簡約だけの原因認定・旧費用互換性は未確立。旧STOP/58軸/AD03保留・sine/会計/retry0。全exit/reapと保全を監査、次はユーザーの技術方針判断。local commit後mandatory STOP・push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：1qubit再合成を保持する対照を一回起動

[scope・承認・fresh gate・起動snapshot](../docs/research/track_a_h6_m5_preserve_oneq_run01_start_20261019.md)。SOURCE68ee03a7…、小角度を含む1qubit runを保持し他opt1処理を維持する新policy。保存PASS再利用・新cosine1compile/1作用、raw3検査各1e-9未満。人工12/preflight22/改変拒否12件PASS、2011seal・2658保全。単独原因/旧費用互換性を認定せず、旧STOP/58軸/AD03保留。完了/exit/reapは未監査、次はCodexの終了監査と報告後mandatory STOP。local commit・push0。以下は各時点の履歴。

## 2026-10-19 m5：Euler位相生成・角度簡約の人工切分け完了

[局所結果・限界・次の担当](../docs/research/track_a_m5_local_resynthesis_diagnostic_20261019.md)。SOURCE057d954b…、45入力×3生成/Euler45/積9/局所反復4を4processで完了。既定の小角RZ省略で約9.1e-13差、正確な外側phase batchでも反復で蓄積する人工例を再現。簡約無効の対照は約2.9e-16。H6での発生・原因・修正十分性は未確定。新人工12＋回帰13件PASS、2598file保全、実分子QPY/NPZ/compile/作用/会計0、旧STOP/58軸/AD03保留。次はユーザーの保存H6局所run限定照合の判断。local commit後mandatory STOP・push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：exact phase batchの再計算もraw基準超過でSTOP

[結果・実行証拠・終了監査](../docs/research/track_a_h6_m5_phase_batch_run01_results_20261019.md)。SOURCE 96cd7cd4…、保存lowering PASS再利用で新compile1/作用1、約7分20秒。88pass・502636位相増分の正確な集計を完了したが、compiled vs保存変換後vectorは1.432e-9で固定1e-9未満FAIL、第3gate未実施null。worker exit0/launcher exit1・全回収、終了監査48項目PASS。2570file/2011seal・旧STOP/58軸/AD03保留、費用受理/retry/sine/会計0。修正未解決、次はユーザーの内部phase生成・matrix/Euler・局所簡約の切分け判断。local commit後mandatory STOP・push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：exact phase batchで問題cosine一軸を再計算起動

[実装・承認・保存PASS再利用・fresh gate](../docs/research/track_a_h6_m5_phase_batch_run01_start_20261019.md)。SOURCE 96cd7cd4…、元Rust再合成器を保持した隔離passでphaseを正確蓄積・境界で一度丸める。保存lowering QPY/vector/PASSを新契約で受け入れ、重複を避け新compile1/作用1、raw3検査は各1e-9未満（初回reuse）。人工13件・起動前22項目・12改変拒否PASS、2011seal/2570保全file。旧STOP/58軸/AD03保留、費用受理/sine/会計/retry0。全PASS/native exit/reapは未監査。次はCodexの終了監査・結果報告後mandatory STOP。local commit・push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：保存callbackから1qubit位相蓄積を優先候補へ

[限定調査・SOURCE経路・未実装修正案](../docs/research/track_a_h6_m5_postrepair_phase_static_20261019.md)。調査器SOURCE c3b43e9e…、保存626callbackのphase bit変更65件は全てOptimize1qGatesDecomposition（88呼出し）。各置換scalarの逐次binary64加算/wrapへ追跡、人工の空DAGで情報落ちを確認。ただし最初の誤差pass・H6残差の原因・修正完了は未確定。人工12件PASS、2500file/1964seal・旧STOP/58軸/AD03保留。新compile/再生/QPY・NPZ復元/実vectorのphase再解析/有効grant/会計0。次はユーザーの位相蓄積案の隔離実装・人工検証判断。local commit後mandatory STOP・push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：明示basis修正の一回再計算もraw不一致でSTOP

[結果・限界・終了監査](../docs/research/track_a_h6_m5_phase_repair_run01_results_20261019.md)。SOURCE02cb8b12…、cosine1compile/2作用・wall約15分38秒。修正後vs保存wrapperは2.326e-11でPASS、compiled vs修正後は1.472e-9で固定1e-9未満を満たさずSTOP、第3gate未実施null。worker exit0/reapとlauncher exit1/reapを区別、63項目の証拠監査PASS。2422file/1964seal・旧STOP/58軸保全、費用受理/自動retry/sine/会計0。修正完了・原因passは未確定、旧費用非互換・AD03保留。次はユーザーの限定静的調査継続判断。local commit後mandatory STOP・push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：明示basis修正・cosine一軸再計算を起動

[承認・固定scope・fresh gate・起動snapshot](../docs/research/track_a_h6_m5_phase_repair_run01_start_20261019.md)。SOURCE02cb8b12…/plan9c70e90b…、利用者の実H6再計算承認に基づき新grantを一回消費。最大cosine1compile/2作用、各3raw差1e-9未満を維持。人工11件・起動前22項目PASS、1964seal/2422保全file。1 productive worker＋独立監視、transient PSIでkillせず無関係worktree非凍結。新入力政策は旧費用非互換、旧STOP/58軸/AD03保留。全PASS/native exit/reapは未監査、次はCodexの終了監査・結果報告後mandatory STOP。local commit・push0。以下は各時点の履歴。

## 2026-10-19 H6 m5：compiler位相欠落の人工再現・明示basis修正候補

[調査・人工検証・一軸DRAFT計画](../docs/research/track_a_h6_m5_phase_repair_design_20261019.md)。SOURCE9fe4981a…、固定BasisTranslatorでparameter付き等価則の数値定数phase欠落を人工最小例で再現。P-onlyは単独修正として不採用、明示native→basis候補は人工raw演算比較PASS、18 test method。H6旧微小差との因果接続/実H6改善は未検証、実行adapter/closure未完成・無効grant。実分子QPY/NPZ/compile/再生/有効grant0、2369file/1932seal保全、旧STOP/58軸/AD03保留。local commit後mandatory STOP・push0。次はCodexの固定一軸実行準備、その後ユーザーの別実行承認。以下は各時点の履歴。

## 2026-10-19 H6 full m5：scalar位相台帳照合完了・旧STOP維持

[結果・解釈・終了監査](../docs/research/track_a_h6_m5_phase_ledger_run01_results_20261019.md)。SOURCEf0022cec…/plan b24b9fea…、保存2QPYを2 workerでscalar抽出し約39秒でnative exit0・全回収。正確なbinary64有理数と数学的πの80/120桁でδ_det約1.48572348e-9 rad、保存拡張common phaseとの差約−1.69657e-11 rad。内部安定性PASSだが発生pass/全状態等価性は未確定。人工13件・起動前22項目・終了35項目PASS、2,324file保全/1,932seal。新compile/回路再生/費用受理/補正0、旧STOP/raw基準/58軸/AD03保留を維持。local commit後mandatory STOP・push0。次はユーザーの追加調査・修正計画判断。以下は各時点の履歴。

## 2026-10-19 H6 full m5：位相帳簿の静的調査完了・原因未確定

[保存構造・source経路・次の最小案](../docs/research/track_a_h6_m5_phase_ledger_static_20261019.md)。保存2QPYを2processでmetadata照合し、ancilla scalar P1回/H2回・gate順序hashを確認。Qiskit公式1.3.0の11 source/選択Python5file byte一致、binary64再合成/位相加算・2π wrapへ追跡したが原因未確定。新再生/compile/実角度総和0、人工構造5＋記号3の8件PASS、2223file/1831seal保全。再compile不要のdeterminant scalar台帳案は未実装・未承認。旧STOP/58軸/raw基準/AD03保留を維持。local commit後mandatory STOP・push0、次はユーザーの台帳準備/実行判断。以下は各時点の履歴。

## 2026-10-19 H6 full m5：保存精度診断完了・旧raw FAIL維持

[結果・解釈・終了監査](../docs/research/track_a_h6_m5_saved_phase_run01_results_20261019.md)。SOURCE99e8723d…、保存4再生が15:17:11 JSTにnative exit0・全worker回収、wall約13分04秒で終了。通常2本は保存vectorと差0。拡張でもraw1.5039e-9/位相差が残り、旧1e-9未満基準FAIL・旧STOP維持。26終了監査PASS、旧684＋準備10file・1831seal不変。新compile/費用受理/会計/retry0、AD03/Core保留、local commit後mandatory STOP・push0。次はユーザーの静的調査継続判断。以下は各時点の履歴。

## 2026-10-19 H6 full m5：保存4再生の一回解析起動・完了未確認

[承認・fresh gate・進行snapshot](../docs/research/track_a_h6_m5_saved_phase_run01_start_20261019.md)。SOURCE99e8723d…/plan b2b2a818…に明示承認、新grant一回消費・15:04:07 JST起動。1831file seal/旧684file保全・fresh gate・保存診断/精度/2QPY構造PASS、初回通常再生が進行。最大4jobだがidle CPUから今回は1worker順次。新compile/sampling/会計0、旧STOP/58軸保全、AD03保留。最終結果・exit/reap未監査、次はCodexの終了監査とmandatory STOP。push0。以下は各時点の履歴。

## 2026-10-19 H6 full m5：保存回路の精度診断・実行準備完了／実解析未承認

[実装・固定契約・検証・次の承認](../docs/research/track_a_h6_m5_saved_phase_readiness_20261019.md)。SOURCE 99e8723d…、保存2回路×通常/拡張の4再生runnerを実装。人工24件/metadata拒否12件PASS、1831file seal・旧684file不変。実H6 QPY/NPZ復元/解析/compile/有効grant0、原raw基準/旧STOP/58軸保全。通常再現atol1e-12は新契約案・未承認。最大4独立workerとvector化/cache、host PSIでkillせず他worktreeを凍結しない。SOURCE/plan/無効grantをlocal保存しmandatory STOP、push0。次はユーザーの別解析実行承認判断。以下は各時点の履歴。

## 2026-10-19 H6 full m5：保存phase追加診断計画・解析未実行

[固定入力・4再生DAG・解釈・実装残件](../docs/research/track_a_h6_m5_saved_phase_diagnostic_plan_20261019.md)。起点282cba7d…、保存cosine2回路/初期状態1つを通常・拡張精度で各2作用する未承認案。新compile/sampling/格子/受入変更0、QPY/NPZ未復元、runner未実装・SOURCE未seal・無効grant。12 roleのbyte SHAを結合、拡張精度の実mantissa・gate whitelist・通常再現をfail closedにする計画。旧STOP/58軸保全、684保護file、AD03/Core保留。次はCodexのrunner/人工検証/契約準備、その後ユーザーの別実行承認、local commit後mandatory STOP。以下は各時点の履歴。

## 2026-10-19 H6 full m5：固定1点再計算もraw不一致を再現・診断保存STOP

[結果・phase診断・保全](../docs/research/track_a_h6_m5_full_recovery_run01_results_20261019.md)。SOURCE f743a882…、14:22:24 JST終了・約27分08秒。cosine raw差1.504e-9（元と一致）でSTOP、sine0・新受理費用0、全worker/launcher回収。保存共通位相1.503e-9 rad、phase-aligned差6.068e-11は診断専用で基準変更なし。QPY/NPZ保存とSHA照合に成功。旧58軸/629file・新330 seal保全、累積予約60、retry/再会計0、AD03/Core保留、mandatory STOP。次はCodexの保存証拠追加診断計画とユーザーの解析承認判断。以下は各時点の履歴。

## 2026-10-19 H6 full m5：承認済み1構成の再計算起動・完了未確認

[診断保存adapter・人工検証・新契約・起動](../docs/research/track_a_h6_m5_full_recovery_run01_start_20261019.md)。SOURCE f743a882…、固定H6 full m5/p19/q2・最大2軸、cosine FAILならsine抑止。元raw基準1e-9未満・science ASTを維持し、QPY/比較ベクトル/未検証費用を判定前保存。最終15人工PASS・metadata8種拒否、closure330/QPY runtime10、旧629file保全。13:55:17 JST一回起動・fresh gate PASS、1 productive workerでnative_build。旧STOP/58軸保全、AD03/Core保留、終了後mandatory STOP、次はCodexの終了監査。以下は各時点の履歴。

## 2026-10-19 H6 full m5：compile不一致の静的調査・根本原因未確定

[調査根拠・未検証証拠保存の修正案](../docs/research/track_a_h46_m5_compile_stop_static_analysis_20261019.md)。SOURCE c7d53f04…のraw差1.504e-9/固定1e-9未満を保持。global phase適用/complex精度/opt1経路を確認したが、失敗回路・比較ベクトル未保存のため原因未確定。選択Qiskit source8件RECORD照合。固定1点・最大2軸の診断取得は未承認DRAFT（旧59予約保全、累積最大61案）、adapter実装/新数値検査/compile0、旧STOP不変、mandatory STOP。次はCodexの診断adapter/人工test準備、その後ユーザーの別実行承認判断。以下は各時点の履歴。

## 2026-10-19 H4/H6 m5：tol-only pilot終了・29/30 PASS、H6 full m5でSTOP

[結果・数値不一致・終了監査](../docs/research/track_a_h46_m5_tol_pilot_run01_results_20261019.md)。SOURCE c7d53f04…、13:30:50 JST終了、約27分25秒。H4全lane/H6 PR完了、H6 full m5/q2 cosineのcompiled vector差1.504e-9が固定1e-9未満基準を超過。58受理軸・59予約・sine1未投入、全24 worker exit0/reap・launcher exit1/reap。元SOURCE/input/grant/STOPを保全しretry/追加compile/会計0、AD03/後続Core保留、mandatory STOP。次はCodexの静的原因調査案と、ユーザーの追加検証判断。以下は各時点の履歴。

## 2026-10-19 H4/H6 m5：tol-only technical pilot一回起動・完了未確認

[起動・fresh gate・承認範囲・終了監査](../docs/research/track_a_h46_m5_tol_pilot_run01_start_20261019.md)。SOURCE c7d53f04…、最大60軸/PR12本/reference2、最大30 worker。13:07 JST snapshotでreference両PASS・21/30 task完了、STOPなし。新4grantを一回消費し旧DRAFT/INVALIDとSOURCEを保全。m5採用AD03/後続Core/会計/fitは未承認。バックグラウンド継続、終了後自動mandatory STOP。Codexの終了監査・結果報告は未実施。以下は各時点の履歴。

m5 tol-only：`h46_m5_tol_contract.py` / `h46_m5_tol_native.py` / `h46_m5_tol_saved_input.py`、`prepare_h46_m5_tol_pilot.py`、`launch_h46_m5_tol_pilot.py`。既存code再利用・新series/seal、既定はscope表示、科学実行は明示承認が必要。

[check_h46_m5_protocol_alignment.py](check_h46_m5_protocol_alignment.py)：m5計画metadataと承認/Phase境界の構造検査のみ。科学moduleはimportしない。

## 2026-10-19 H4/H6：m5_best native pilot準備完了・実計算未承認

[実装・4契約・並列化・STOP](../docs/research/track_a_h46_m5_best_native_readiness_20261019.md)。SOURCE7a4d126a…、3公式/full・B0・B0+I・PRの4laneを接続。24条件/PR12本/60軸/共有full-H reference2件。人工回帰120件PASS、metadata改変20件拒否、暫定seed120/既知4243衝突0。SOURCE/input/runtime closure各303/305 file封印、4無効grant。旧main profileは参照のみ、新native費用系列として識別。実分子signal/sampling/compile/会計/fit0、mandatory STOP。以下は各時点の履歴。

## 2026-10-19 H4/H6：m5_bestの並列検証計画・科学未実行

[計画・再利用・未完了範囲](../docs/research/track_a_h46_m5_best_parallel_plan_20261019.md)。既存4th(m5_best)・11段S2を再利用するH4/H6計画。主controlはsymmetric_directional。4lane独立並列pilot案24条件・PR12本・60軸、後続は依存DAG/各Phase STOP。新12人工PASS、計画SOURCE834b738a…。実分子計算0、native adapter/launcher未seal・無効grant。係数非ZEROのみで棄却せず、主Track Aへ接続する差分を準備。以下は各時点の履歴。

## 2026-10-19 H6：高次PF独立確認の固定2点順位を再現・正常終了

[結果・終了監査・限界](../docs/research/track_a_h6_higher_pf_confirmation_run01_results_20261019.md)。SOURCE e3ac6fd8…、最小3候補13/45/45本・新103 trajectory/206軸・control536件PASS。2026-10-18 23:28:31 JST、native exit0/全56 worker回収、約42分12秒。固定2点順位は両方REPRODUCED、近似区間は両方0を含まず。点費用はY4/q2がS2/q8比3.748%低下、Y4/q4がq2比1.751%増。s4/s2_multi未観測、母平均保証なし、探索n8混合/追加sampling/compile/retry0。mandatory STOP。以下は各時点の履歴。

## 2026-10-18 H6：最小3候補の独立確認native準備完了・実計算未承認

[実行契約・検証・限界](../docs/research/track_a_h6_higher_pf_confirmation_native_readiness_20261018.md)。H6高次PFの最小3候補S2/q8・Yoshida4/q2/q4について専用native adapter・one-shot launcher・paired集計を実装。SOURCE e3ac6fd8…、13/45/45本＝103本206軸は名目90%の未採択案。306 closure・暫定seed914/旧seed3329との衝突0・人工回帰92件PASS。無効grantで実CLIの事前拒否を確認。目的Aの点順位再現に限定し、目的Bの母平均保証・A_COSTを分離。新分子sampling/compile/signal/費用再解析0、有効grant0、mandatory STOP。以下は各時点の履歴。

## 2026-10-18 H4/H6：独立確認の3/7候補計画、実計算未承認

[標本数・希少層・資源・契約案](../docs/research/track_a_h46_higher_pf_confirmation_readiness_20261018.md)。固定1fc58aa3…のn8を使用。最小3候補の名目80%案10/35/35（80本160軸）、90%案13/45/45（103本206軸）、分散4倍感度案39/137/137。拡張7候補・family7も別案。14設計・47人工PASS、metadata71file。全案未採用、native C adapter/SOURCE/n_C未確定、新sampling/compile/signal0、mandatory STOP。以下は各時点の履歴。

## 2026-10-18 H4/H6：高次PFの統合比較完了・独立確認は未実行

[結果・選択・確認案](../docs/research/track_a_h46_higher_pf_integrated_run01_results_20261018.md)。SOURCE a65e4f6e…、18適格費用群n8/24条件・旧shot24行/費用6群再現、全6層の点選択完了。両系S2 q8/Yoshida4 q2/new_2元8桁 q1。H6 Y4 q2/q4は約0.974%差、S2対Y4は約3.294%差。native exit0・wall0.580秒、42人工PASS。MCSE案は未採用n=null、C sampling/新compile/signal0。母平均・厳密biasを認定せずmandatory STOP。以下は各時点の履歴。

## H4/H6 new_2数値診断（2026-10-18）

[runner](run_h46_new2_validation.py)と[core](h46_new2_core.py)は、保存入力からの高次PF・有限RTE期待signal専用。固定SOURCE/provider/input/planと利用者認可を検査する。新trajectory・native compile・分子入力生成を行わない。[結果と限界](../docs/research/track_a_h46_new2_diagnostic_20261018.md)。

## 2026-10-18 H8：追加6精度の保存値検証完了

[結果・実測/model区分・来歴](../docs/research/track_a_h8_epsilon_boundary_run01_results_20261018.md)。SOURCE3156b036…、利用者の準備後実行承認に基づくC専用one-shot。保存anchor1,520行を再現後、新2,280行/12比較、native exit0・最終gate PASS、wall0.832秒。通常ε=.00075で+12.77%、.000625で−0.50%（有限標本の点比較）。人工164件PASS、新compile/sampling/signal/fit0。他のA/B/D/E/Fは未実行、条件付き実行承認は保持し、各単位のSTOPを維持。以下は各時点の履歴。

## 2026-10-18 Targeted parallel verification plan

[prepare_track_a_targeted_plan.py](prepare_track_a_targeted_plan.py) は未承認scope・無効grant・依存DAGをmetadataだけで出力する。science dispatchなし、`--execute`拒否。[範囲・並列化・未完成adapter](../docs/research/track_a_targeted_parallel_validation_plan_20261018.md)。

## 2026-10-18 H8：希少層境界・選択効果の保存値解析完了

[結果・来歴・限界](../docs/research/track_a_h8_rare_sensitivity_selection_run01_results_20261018.md)。RARE 40条件（32境界/8不適格）とSELECTION元4精度の探索的bootstrap2000を別grantで各一回完了、両native exit0・mandatory STOP。人工135件PASS。未知費用はnull、母平均/厳密bias/selection-adjusted coverageなし。PR sampling/compile/fit/shot再計算0、旧結果/ledger/STOPは保持。 以下は各時点の履歴。

## 2026-10-18 H8 rare-tail / selection readiness

`run_h8_rare_tail_selection_readiness.py`はbuilt-in syntheticだけ。science/config/grantは入力を開く前にexit2拒否。`audit_h8_rare_tail_selection_evidence.py`は36固定Git fileとpilot closure9のmetadata/literal監査だけ。新確率/SE/費用/順位を評価しない。[設計](../docs/research/track_a_h8_rare_tail_selection_readiness_20261018.md)。

## 2026-10-18 Track A H4〜H8：横断証拠監査（対象結果2026-10-17）

[証拠整理・metadata検査](audit_track_a_cross_system_evidence_v1.py)。19系列・222保存値/状態・32固定ファイル、12比較・13不足を別表に記録。実測/model/有限標本を分離し、新科学計算・再会計・ランキング・統計解析は0。H8希少次数pilotは別チャットの未完了作業として収録せず、既存SOURCE/grant/ledger/他worktreeを変更しない。新しい科学的採否・新規性判断なし。以下の記録は各時点の履歴。

このscriptはstdlib/Git blobによる文字列・schema・SHA検査のみ。既存science runnerやaccounting checkerは実行しない。

## 2026-10-17 H8：現runを保持した準備工程の並列化

[実装・検証・契約](../docs/research/track_a_h8_key_compile_parallel_preparation_20261017.md)。準備を3構成並行・最大6 workerへ拡張し、wrapper生成も軸workerへ移した。人工32件PASS、SOURCE 7f8d74660b2914a1223d0ac7990dde8d4645032e。compileは固定順直列・各軸2%以上でSTOP。新方式の科学起動0・無効grant・未seal。旧runの25 immutable filesはSHA一致し、稼働継続。速度/peak RSSは未測定。以下は各時点の履歴。

## 2026-10-17 H8：主要3構成・6軸の直接compile準備（未承認）

[runner](run_h8_key_compile.py) / [契約](h8_key_compile_contract.py) / [adapter](h8_key_compile_adapter.py) / [metadata準備](prepare_h8_key_compile.py)。SOURCE `352581103b7dbd4fa58e28af7a6ca65570d36e8e`。保存入力11件のbyte照合、人工22件PASS。準備2並列・compile直列/各軸2%未満、own-process cleanupを実装。実H8 build/control/compile/費用再会計0、無効grant・CPU未指定・未seal。実compileと後続再会計は別承認。以下は各時点の履歴。

## 2026-10-17 H8：非相殺COST_COMPARISON準備・未承認

[run_h8_noncancellation_cost.py](run_h8_noncancellation_cost.py) / [cost契約](h8_noncancellation_cost_contract.py) / [metadata準備](prepare_h8_noncancellation_cost.py)。default planのみ、別承認で保存shot再利用cost一回。旧shot/通常会計を再実行しない。 以下は各時点の履歴。

## 2026-10-17 H8 Phase C：B2完了結果の統合・実行準備（未承認）

[資料入口](../docs/research/track_a_h8_phase_c_integration_readiness_20261017.md)。既存Phase C snapshotを統合し、B2最終4軸契約（PASS_REGISTERED_FOUR_AXES、保存3＋新規1、軸別SOURCE）へ再結合。B0 p15/q1 S4は実測168,353 RZ優先、未測定S4はmodel推定と明示、非相殺行は拒否、旧32/184は別判定。資源上限とPSI-STOP observerを廃止しwait4 launcherへ。人工37件PASS（実launcher→driver人工e2eを含む）。SOURCE 29ed76544441d534ad99e5c708a2ae17eec78c18、plan 9e42fc76…、unsealed・未承認・無効grant。実H8費用再会計・ランキング・削減率・新signal/compile/sampling/fit0、B2/M1-S/既存worktree不変。以下は各時点の履歴。

## 2026-10-17 H8 B2：登録S4全4軸の実測PASS・正常終了

[結果・結合来歴・終了監査](../docs/research/track_a_h8_b2_completed_results_20261017.md)。H8/隣接1Å/STO-3G/DF28/T=.8、B0 p15q1・B0+I p21q8 S4の両軸を実測。B0各168,353 RZ/誤差0.2153638062%、B0+I各1,852,640 RZ/誤差0%、登録4軸PASS。2026-10-17T17:46:48.422662+09:00に最後のsineが終了、SOURCE f58da959…/native driver・launcher exit0/全回収。保存3軸を別受入契約で結合し、元runと過去のSTOP・未settle予約は保持。最新run wall1632.031秒/peak26.252GiB、33source/265provider/34273runtime/31input・702保護/135frozen/286過去immutableを照合。新fit/費用再会計/Phase C/非相殺stage0、全格子・厳密bias・PR母平均の保証は未認定。mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 B2：PSI契約修正・sine再開snapshot

[実測・技術STOP・再開と保全](../docs/research/track_a_h8_b2_sine_psi_running_20261017.md)。本人session full PSI0.36%による17:01のsine中断を保全。現行資源方針と技術修正/再開承認に基づき、新SOURCE f58da959…ではPSIを診断記録へ変更し、OOM・実効headroom・観測異常STOPを維持。26人工tests PASS。2026-10-17T17:22:24.738031+09:00時点、2回fresh gate PASS後、保存sine回路の読込/照合を157.1秒継続確認。保存cosine実測1,852,640 RZ/誤差0%を再利用し、新候補/control再計算/fit/Phase Cは0。33source/265provider/34,273runtime/31input、過去286 immutable・702保護/135 frozen worktreeと明示1並行観測。全4軸PASSは未確定、B2終了時mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 B2：cosine実測PASS・sine限定再開snapshot

[実測・技術STOP・再開と保全](../docs/research/track_a_h8_b2_sine_recovery_running_20261017.md)。保存cosine実測1,852,640 RZ/誤差0%/native exit0を確認。元attempt02はsine読込中に上位cgroup full PSI0.18%でSTOP、旧記録/未settle保持。利用者の技術回復承認で、検証済みQPYからsine1軸だけを新SOURCE e3ecc03a…で再開。2026-10-17T16:48:38.918552+09:00時点、2回fresh gate PASS・136.0秒の同一worker CPU増加確認、最終RZ/判定は未保存。21人工tests PASS、33source/265provider/34,273runtime/31inputを固定。699 immutable/133 frozen worktreeを厳密照合し、明示1並行worktreeは読取専用の更新観測。Phase C未承認、B2終了時mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 B2：CPU起動修正・新試行の稼働snapshot

[修正・承認・STOP保全・稼働確認](../docs/research/track_a_h8_b2_cpu_handshake_running_20261017.md)。起動時CPU固定・witness blocking wait・同窓process CPU証拠を修正。SOURCE2e604e4a…、40人工tests PASS。利用者の技術修正/再開承認に基づくattempt01は科学前CPU再確認でSTOPし全回収・原記録保持、別grant/未使用域のattempt02は2回fresh gate PASS。2026-10-17T14:40:26.878362+09:00時点、S4回路構築を194.8秒の複数観測で継続確認、新compile結果なし。31source/265provider/34,273runtime/16input、旧692保護/127worktree・117過去immutable不変。準備2 worker/compile1の固定範囲。Phase C未承認、完了時mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 B2 recovery run01：初回CPU gateで科学起動前STOP

[失敗counter・native exit・保全](../docs/research/track_a_h8_b2_recovery_run01_cpu_stop_20261017.md)。H8 B2残り2軸の新scopeを利用者承認で一回dispatchしたが、初回fresh CPU gateでcore6のraw busy82%>5%を検出し、科学driver前STOP。driver/worker/control/QPY/compile/fit/再会計0、output/control/lock/fresh receiptなし。SOURCE77ce12d8…、31source/265provider/34,273runtime/16input照合。独立witnessが実native launcher exit1をwait完了・保存、dispatcher exit0と区別。旧688保護/125worktree・元B2原45files不変。CPU再割当/retry0、旧STOP/未settle1/C未承認を保持し、公開後mandatory STOP。以下は各時点の履歴。

[固定contract](../scripts/h8_phase_b2_recovery_contract.py)／[dispatcher/witness](../scripts/h8_phase_b2_recovery_witness.py)／[保存STOP checker](../artifacts/resource_applicability/track_a_h8_b2_recovery_run01_stop/2026-10-17/verify_saved_stop.py)。

## 2026-10-17 H8 B2：停止記録修正・2 worker並列準備、科学再実行は未承認

[修正・並列化・受入契約・資源](../docs/research/track_a_h8_phase_b2_recovery_parallel_readiness_20261017.md)。B2 run01 MEMORY_PRESSURE/STOPを保持し、残るB0I p21q8 S4の2軸用に修正・並列準備を実装。left/rightとcosine/sine検証・fingerprint/QPY保存を各2 worker、compileは各軸2%判定を挟む直列1。検証済み回路をQPYへ保存し、親の大回路を解放して1軸ずつ読込。PSI raw/対象groupをcleanup前保存、独立witnessでCLI終了後もnative waitを記録、marker集計をaxis_*へ修正。31 source/265 provider/34,273 runtime/16 inputsをseal、34人工testsと2 qubitの4人工compileで指標・fingerprint一致。5物理core/112.3125 GiB案、無効grantのみ。旧687保護/124worktree・B2原45files保持、H8新compile/fit/有効grant0。旧B0/旧B2 STOP・V2 NO_GO、C未承認を維持、公開後mandatory STOP。以下は各時点の履歴。

[contract](../scripts/h8_phase_b2_recovery_contract.py)／[並列control・QPY](../scripts/h8_phase_b2_recovery_parallel.py)／[adapter](../scripts/h8_phase_b2_recovery_adapter.py)／[dispatcher/witness](../scripts/h8_phase_b2_recovery_witness.py)／[launcher](../scripts/launch_h8_phase_b2_recovery.py)／[人工tests](../tests/test_h8_phase_b2_recovery.py)／[provider fixture](../tests/check_h8_phase_b2_recovery_synthetic_provider.py)／[bundle](../artifacts/resource_applicability/track_a_h8_phase_b2_recovery_readiness/2026-10-17/README.md)。

## 2026-10-17 H8 Phase B2：契約STOP・自動再実行なし

[実測・判定・終了監査](../docs/research/track_a_h8_phase_b2_run01_results_20261017.md)。H8/1Å/STO-3G/DF28/T=.8、S4の固定4軸を直列・一回実行。実compile entry3、保存判定2軸（PASS2）。SOURCE244c45e7…、model8420b9f2…を固定。STOP理由は結果報告と原terminalに保存（MEMORY_PRESSURE）。B0I cosineは中断、sine未起動、未settle予約1を保持し、後続compile・retry/refit0。driver native exit−9、launcher報告exit1（外側native wait欠測）、全owned process回収。旧684保護files/123worktree・B0原53/B1原34 rawを保持。新fit/signal/再会計0、元B0 STOP/V2 NO_GOを維持。C未承認、公開後mandatory STOP。以下は各時点の履歴。

[固定contract](../scripts/h8_phase_b2_contract.py)／[native adapter](../scripts/h8_phase_b2_adapter.py)／[serial launcher](../scripts/launch_h8_phase_b2.py)／[人工tests](../tests/test_h8_phase_b2.py)／[保存結果checker](../artifacts/resource_applicability/track_a_h8_phase_b2_run01/2026-10-17/verify_saved_result.py)。

## 2026-10-17 H8 Phase B2：S4の固定4軸・直列検証を実装、実行未承認

[実装・固定条件・資源・承認範囲](../docs/research/track_a_h8_phase_b2_execution_readiness_20261017.md)。B1正本83a3c4a6…/model8420b9f2…を固定し、B0 p15q1・B0I p21q8 S4の両軸を順番に検証するadapter/launcherを準備。各軸2%以上で次軸を起動せずSTOP。SOURCE244c45e7…、21 source/265 provider/34,273 runtime/15 inputをseal、新19＋既存100＝119人工testsと2 qubit S4接続11項目PASS（人工4compile、実H8 0）。3物理core・80.25 GiB、無効grant/CPU[]/command null、fresh gate未発行。旧683保護filesと別チャットの許可済み1worktree更新を区別、更新後122worktree不変。B1・元B0 STOP・V2 NO_GOを保持、新fit/再会計/有効grant0、公開後mandatory STOP。以下は各時点の履歴。

[contract](../scripts/h8_phase_b2_contract.py)／[runtime](../scripts/h8_phase_b2_runtime.py)／[adapter](../scripts/h8_phase_b2_adapter.py)／[driver](../scripts/run_h8_phase_b2.py)／[launcher](../scripts/launch_h8_phase_b2.py)／[prepare](../scripts/prepare_h8_phase_b2.py)／[tests](../tests/test_h8_phase_b2.py)／[人工provider接続](../tests/check_h8_phase_b2_synthetic_provider.py)／[bundle](../artifacts/resource_applicability/track_a_h8_phase_b2_readiness/2026-10-17/README.md)。

## 2026-10-17 H8 Phase B1：全4層PASS・費用model固定・mandatory STOP

[結果・model・検証・保全](../docs/research/track_a_h8_phase_b1_run01_results_20261017.md)。H8/1Å/STO-3G/DF28/T=.8。新B0保存8軸と既存B0I16軸で4独立S2 fit。B0のp21/q8独立holdoutは両軸0%、B0Iは各8 training anchorの残差0%。係数は各層(a,b,d)=(1592,1314,25271/7)。SOURCE6fb8a0a7…、model8420b9f2…、計算本体wall.092秒/native driver・launcher exit0/全回収、SOURCE19/runtime34,273/input82照合。35人工tests PASS、旧681保護/119worktree不変。元B0 run02 STOPは保持、新compile/signal/再会計0、S4は未実測仮説、B2/C別承認。以下は各時点の履歴。

[fit/STOP](../src/trotterlib/h8_phase_b1_fit.py)／[実行入口](../scripts/h8_phase_b1_execution.py)／[人工tests](../tests/test_h8_phase_b1_execution.py)／[evidence](../artifacts/resource_applicability/track_a_h8_phase_b1_run01/2026-10-17/README.md)。

## 2026-10-17 H8 B0：終了監査修正・保存8軸の読取専用audit完了

[技術判定・修正・証拠](../docs/research/track_a_h8_b0_saved_anchor_audit_20261017.md)。H8/1Å/STO-3G/DF28/T=.8、p14/21・q1/8のB0 S2保存8軸を照合し、欠測・hash/lineage不一致なし。再compileを要求する技術的欠損はなく、B1入力候補として構造的に再利用可。元run02のPROCESS_THREADS/契約STOP・過去の違反PID未記録を保持。管理Gitはthread1/optional lock無効、監視は違反identity/exe/thread数をcleanup前保存。新14＋既存83の97人工testsを固定PythonでもPASS。旧681保護/118worktree不変、科学compile/fit/新grant0、B1別承認・公開後mandatory STOP。以下は各時点の履歴。

[audit runner](../scripts/audit_h8_b0_saved_anchors.py)／[metadata/Git helper](../src/trotterlib/h8_b0_saved_anchor_audit.py)／[人工tests](../tests/test_h8_b0_saved_anchor_audit.py)／[evidence](../artifacts/resource_applicability/track_a_h8_b0_saved_anchor_audit/2026-10-17/README.md)。

## 2026-10-16 H8 B0 run02：8軸を保存、終了時のthread監視でSTOP

[保存測定・停止箇所・保全](../docs/research/track_a_h8_b0_anchor_run02_stop_20261016.md)。H8/1Å/STO-3G/DF28/T=.8、B0 S2のp14q1/p21q1/p14q8/p21q8各cosine/sineを4 worker・2 waveで一回取得。8 worker全exit0/reap、TRAIN6/HOLDOUT2保存。約80.76分後、終了時のworktree確認中にPROCESS_THREADSでSTOPし、driver exit−9・launcher exit1、全回収。8軸の保存と取得契約の正常完了を区別し、fit/残差/再会計/自動retry0、旧B1 STOP・V2 NO_GOを維持。以下は各時点の履歴。

## 2026-10-16 H8 B0：CPU選択・起動前STOP記録を修正、再起動なし

[修正・診断・条件維持・人工検証](../docs/research/track_a_h8_b0_cpu_gate_fix_20261016.md)。H8 B0 run01のCPU gateを修正。新SOURCE db559a51…、承認前の3固定window候補probe、全選択coreのcounter/負荷率/自身のCPU使用、科学起動前STOP receipt、試行域の再利用拒否を追加。raw5%・6物理core/176.25 GiB・4構成8軸/4 worker/2 waveは不変。新11＋既存27 tests PASS、source15/provider265/runtime34,273/input8照合。run01原因は遡及断定せず、旧677保護/116worktreeを維持。新compile/fit/有効grant/取得再起動0、公開後mandatory STOP。以下は各時点の履歴。

`probe_h8_b0_anchor_cpus.py --output NEW_FILE`はgrant前の読取専用候補表示。grant変更・affinity変更・科学起動機能はない。

## 2026-10-16 H8 B0 run01：fresh CPU gateで科学起動前STOP

[停止点・証拠・未実測・保全](../docs/research/track_a_h8_b0_anchor_run01_stop_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。SOURCE572a337d…の4構成8軸取得を利用者承認で一回起動したが、fresh CPU idle checkがPHYSICAL_CORE_BUSYで拒否。driver/worker・fresh receipt・output/control/lock・科学予約・compileすべて0。自動retry/CPU再割当0。失敗coreの詳細sampleとlauncher exit codeは未捕捉のまま保持。旧676保護/115worktree不変、SOURCE14/provider265/runtime34,273/input8照合PASS。旧B1 STOP/V2 NO_GO・B1/B2/C別承認を維持し、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8 B0：8軸取得の実装・seal完了、実行未承認

[実装・gate・人工検証・承認範囲](../docs/research/track_a_h8_b0_anchor_execution_readiness_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。固定4構成8軸・4 worker/2 waveのadapter/launcher、SOURCE/input/profile・資源監視/回収を実装。SOURCE 572a337d…、14 own/265 provider/34,273 runtime/8 inputのbytes照合PASS。17新unittest＋旧計画10、2 qubit fixture8項目PASS（人工4 compile・実H8 0）。6物理core/176.25 GiBとfresh gate必須、無効grant・CPU[]。旧B1 STOP/V2 NO_GO・旧673保護/114worktreeを維持。科学起動/fit/残差/再会計0、公開後mandatory STOP。以下は各時点の履歴。

defaultの`run_h8_b0_anchors.py` / `launch_h8_b0_anchors.py` はplan表示だけ。新scopeの有効grant・SOURCE seal・fresh gateなしに科学処理を行わない。

## 2026-10-16 H8 B0：新anchor最小取得計画・未承認

[候補・来歴契約・並列資源・STOP](../docs/research/track_a_h8_b0_anchor_acquisition_plan_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。新B0 S2はTRAIN p14q1/p21q1/p14q8とHOLDOUT p21q8の各cosine/sine、最小4構成8軸を提案。設計座標のみのdet−49、実fit/compile/再会計0。4 worker/thread1・2 wave、後続B1で各軸holdout誤差2%以上STOP。265 provider/7入力照合・10人工tests PASS。旧B1 STOP・B0I16軸再利用・V2 NO_GOを維持。実adapter/launcher未seal、無効grant案のみ。旧673保護/113worktreeを保全し公開後mandatory STOP。以下は各時点の履歴。

`python3 scripts/check_h8_b0_anchor_plan.py` は未承認計画だけを検査する。科学実行入口なし。

## 2026-10-16 H8 Phase B1：anchor照合完了・来歴不足でSTOP

[照合・契約・資源・STOP判定](../docs/research/track_a_h8_phase_b1_anchor_readiness_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。保存32実測軸を照合し、B0I16軸はprofile/preparationへ結合、旧B0の16軸はruntime全体・cost用basis未結合で全B1 STOP。H6係数移送・実rank/fit/compile/再会計0。82既存証拠、8人工tests PASS、無効契約/grant案とB2事前選択4軸（予測null）を公開。旧673保護/112worktreeを保全、V2 NO_GO、公開後mandatory STOP。以下は各時点の履歴。

`python3 scripts/check_h8_phase_b1_readiness.py` は保存台帳のSTOP整合性検査だけを行う。数値fit/compile入口なし。

## 2026-10-16 H8 Phase A：実装・人工検証完了、科学実行は未承認

[H8 adapter・保存操作列復元・並列制御・契約](../docs/research/track_a_h8_phase_a_implementation_readiness_20261016.md)。追加152候補は不変。22 primitive tasks/1,650 probesと13 prefix tasks、最大16 worker。28人工tests＋2 qubit fixture8項目PASS、provider265/runtime34,273/input11を結合。実H8復元・signal・数値再解析・compile・新有効grant0、別途明示承認とfresh gateが必要。V2 NO_GO・旧grant/ledger・V1/H5/H7を保持し、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8対照強化の準備・V2現行残差設計のNO_GO終了

[H8準備状況・152追加候補案・Phase別資源/STOP](../docs/research/track_a_h8_strengthened_deterministic_readiness_20261016.md)。旧32候補・適格B0I16実測軸・保存full preparation exportを再利用。S2 p8〜20（14再利用）とS4 p15〜21、B0/B0I・q1/2/4/8。metadata checkerのみ実装、H8 science adapter/decoder検証・SOURCE sealは未完了。新科学/数値再解析/compile/有効grant0、Phaseごとの別承認が必要。[V2現行fitは利用者指示でCLOSED_NO_GO](../docs/research/track_a_v2_residual_design_closure_20261016.md)。832理論行の代替利用は技術的可否だけ、旧条件の緩和なし。公開後mandatory STOP。以下は各時点の履歴。

`scripts/check_track_a_h8_strengthened_plan.py`（repo rootから）：stdlibの提案metadata検査のみ。科学runnerではない。旧library・provider・V2 runnerは変更しない。

## 2026-10-16 V2保存RZ残差：人工検証完了・数値実行未承認

[残差runner](track_a_v2_residual_runner.py)、[metadata準備](prepare_track_a_v2_residual.py)。[報告](../docs/research/track_a_v2_residual_implementation_20261016.md)。262 scope entriesは全件来歴未解決でfit対象0。実データfit/残差/compile0、未承認draft・公開後mandatory STOP。primary agent自身のlocal checksであり独立レビューではない。以下は各時点の履歴。

## 2026-10-16 V1：p7実測費用の限定比較完了・STOP

[結果・算術・保全監査](../docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md)。H6/1Å/STO-3G/DF19/T=.8/B0+I p7 q1 S2/ε=.05/P=0/RZ。保存shot×実測2軸で決定論356,097,424 RZ、保存確認PR403,423,200 RZ、PR削減率−13.29012029%（倍率1/10）。通常bias・経験的allowanceの保存点比較で逆転を維持。全候補再選択・母平均/厳密bias保証は未実施/未認定。旧grant等663/compile raw22不変、新compile/signal/sampling/shot/fit/V2なし、mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1：p7/q1 S2の2軸実測PASS・完了STOP

[実測・終了監査](../docs/research/track_a_v1_p7_compile_results_20261016.md)。SOURCE84b9ea22…、H6/1Å/STO-3G/DF19/T=.8/B0+I p7 q1 S2。cosine/sine各14,492 RZ、予測14,524.2から各0.222191554%でPASS。wall73.241秒/native exit0/全回収、SOURCE21/provider230/runtime34273/input15/旧grant等660不変。登録compile2のみ、再fit/retry/新signal/費用再会計/V2なし。総費用比較は別承認待ち、mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1：p7/q1 S2の2軸実測を準備・未承認

[実装・90人工tests・入力監査・実行契約](../docs/research/track_a_v1_p7_compile_readiness_20261016.md)。指定結果25bbdd33…から専用branchを分離。B0+I p7/q1 S2 cosine/sine最大2軸、予測14524.2 RZ/軸を固定、各軸2%以上で即STOP。旧SOURCE17/provider230/runtime34273・15入力・H5/H7含む660ファイル不変。新compile/build/probe/signal・有効grant発行0。未承認draft/CPU[]/command null、実測後の費用比較も別承認。準備公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1：5精度費用再評価・完了STOP

[結果・全4320会計行・60比較・監査](../docs/research/track_a_v1_reaccount_results_20261016.md)。SOURCE2261a3ad…、H6/1Å/STO-3G/DF19/T=.8/P=0/RZ。通常biasと固定費用modelの条件下で、細格子＋S4後のPR削減率はε=.05/.01/.005/.001/.0005で−13.04/30.89/34.97/31.17/10.03%。新最良対照はmodel推定、p7は外挿。緩精度の優位反転・中間10pp以上低下を報告。非相殺診断は旧B0I12候補の分解欠測で全domain最良未解決。旧2170照合一致、wall6.024秒/native exit0/全回収、旧653ファイル不変、新signal/compile/fit/V2なし、mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1：5精度の保存費用再評価を準備

[比較・実行契約・実装・人工検証](../docs/research/track_a_v1_reaccount_readiness_20261016.md)。利用者の継続指示を受け、Phase Bの固定modelと実測4軸を入力とするone-shot再会計を準備。旧32→S2追加96→S4追加136候補を比較し、確認PR union8/n32と探索72/n8を区別。実測優先・新compile/fit/V2なし。pinned71人工tests PASS、厳密bias保証と全gridモデル誤差保証は未解決のまま、fresh gate後実行・終了時mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1 Phase B run01：登録全4軸PASS・完了STOP

[結果・全表・終了監査](../docs/research/track_a_v1_phase_b_run01_results_20261016.md)。SOURCEfcf37e67…、H6/1Å/STO-3G/DF19/T=.8のS4登録4軸を検証。B0 p11q1の両軸誤差0.0983742%、B0+I p15q8の両軸0%、全軸2%未満。wall3037.384秒/native exit0/全回収。固定model・650旧grant/ledger等・runtime34273不変。再fit/追加compile/5精度再会計/V2実行0、mandatory STOP。全grid保証・保守的bias保証・旧B0 binary同値は未認定。以下は各時点の履歴。

## 2026-10-16 V1 Phase B：費用モデル固定・検証起動準備

[契約・モデル・実装・人工検証](../docs/research/track_a_v1_phase_b_readiness_20261016.md)。Phase A追加104候補と保存28費用軸を照合、4独立S2 fitの残差0。S4予測を固定し最大4軸を各軸2%以上で即STOP、再fit/追加compileなし。pinned57人工tests PASS。旧B0 binary profile未確認・経験的u・bias保証未解決を保持。後続の利用者開始指示を受けfresh gate後one-shot起動へ、V2/5精度再会計は保留。以下は各時点の履歴。

## 2026-10-16 V1 Phase A run02：追加signal完了STOP

[結果・全表・終了監査](../docs/research/track_a_v1_phase_a_run02_results_20261016.md)。SOURCE26b9fde6…、新104候補＋PR診断8件を取得。wall95.102秒/native exit0/全27worker回収、factor1/10の保存適格性は一致。経験的u・保守的bias保証未解決を保持。旧grant/ledger等646不変、compile/費用model/V2/次Phase0、mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1 Phase A run01：科学予約前STOP

[実行・回収・保全・診断](../docs/research/track_a_v1_phase_a_run01_stop_20261016.md)。利用者の起動承認後、fresh gate通過。最初の監視でOBSERVER_RSS_CEILING、driver exit -15/全回収、科学lock/output/予約0。旧grant/ledger等644ファイル不変。実測RSSは未記録で原因未確定。監視修正を47人工＋pinned影響17件で検証、再実行は未承認、自動retryなし・V2保留・mandatory STOP。以下は各時点の履歴。

## 2026-10-16 Track A V1 Phase A並列準備・V2保存結果照合

[Phase A資料](../docs/research/track_a_v1_phase_a_parallel_readiness_20261016.md)。V1は最大16worker・44限定人工tests PASS、SOURCE/input/profile静的照合一致。新科学計算・新grant0、未承認・mandatory STOP。V2は指定commit7e416705…のH6/H8計832理論行を読み取り専用照合し、重複生成・計算を保留。以下は各時点の履歴。

## 2026-10-16 H6/H8 metadata接続・人工schema検査

## 2026-10-16 Track A V1 実行準備

[報告・実装・人工テスト・証跡](../docs/research/track_a_v1_h6_deterministic_readiness_20261016.md)。科学計算未実施、未認可draft、mandatory STOP。

`check_native_df_metadata_binding.py` は人工20件だけを実行。実データ引数なし、Gate/回路生成・cost評価・科学importなし。[報告](../docs/research/track_a_native_df_h6_h8_metadata_binding_20261016.md)。

# scripts 索引

## 2026-10-16 Native DFモデル：Qiskit 1.3.0接続検証

[人工接続runner](check_native_df_qiskit_connection.py)／[固定SOURCE loader・独立Qiskit分解](native_df_qiskit_validation_support.py)。新規installなし。分子I/O・sampling・transpile入口を拒否する。
実分子評価・保存H6/H8再解析・sampling・transpile・server接続・新grant・原稿改訂0。公開後mandatory STOP。

## 2026-10-16 Native DF理論ゲート数モデル（人工検証のみ）

[standalone人工runner](check_native_df_gate_model_synthetic.py)。`python3 -B scripts/check_native_df_gate_model_synthetic.py`。標準ライブラリのみ。[設計](../docs/research/track_a_native_df_theoretical_gate_model_20261016.md)。
新しい実H6/H8数値評価・sampling・compile・server操作・grant発行は0。既存結果・科学SOURCEは保持し、公開後mandatory STOP。

## 2026-10-15 H6 fingerprint並列化・cleanup修正

[修正・人工61件・再実行案](../docs/research/track_a_h6_family_fingerprint_parallel_fix_20261015.md)。親の重複走査をworker前後SHA証跡へ集約し、停止後のsignal割込によるreap中断を修正。旧証拠保持、2worker案、資源/PSI条件不変、今回の再起動0。以下は過去の各時点の記録。

## 2026-10-15 H6 native B0+I 全12構成の実行準備

[全12構成・旧6軸再利用・新規最大18軸の実装と契約案](../docs/research/track_a_h6_native_b0i_full_family_readiness_20261015.md)。SOURCE731826b3…、44人工tests PASS、別担当の限定技術監査PASS_WITH_LIMITS。旧3構成は元SOURCE2e733d8a…の実測としてREUSE_WITH_DECLARED_LIMITS。新9の全10条件screen後、全selected correctness合格時のみ最大6並列compileへ進む経路を準備した。旧4h実績を残し、新wall-null/CPU別予算・資源は未承認案。CPU[]/false/未seal/commandnull、新実H6数値・native build/probe/compile・grant・venv/他job変更0、mandatory STOP。以下は各時点の履歴。

## 2026-10-15 H6 B0+I blueprint結合修正

[原因・修正・検査・契約案](../docs/research/track_a_h6_b0i_blueprint_fix_20261015.md)。run01停止原因の旧preparation hashと現在profileの不一致を3caseで再現。旧hash保持＋現profile/input/providerに固定した3blueprintの厳密照合、実/期待field差の保存を実装。SOURCE6b4d3026…、新12＋影響70の82人工tests PASS。数学/compiler/最大6並列・合計6compileは維持。サブエージェント0、限定snapshot診断2/preparation6以外のsignal/native build/probe/compile・新grant・本計算再起動0。未認可draftへ再結合し、旧run/carryを返却せず保全。 以下は各時点の履歴。

## 2026-10-15 H6 native B0+I pilot6 run01：Phase A STOP

[起動・停止・回収・carry監査](../docs/research/track_a_h6_native_b0i_pilot6_run01_20261015.md)。利用者の明示承認を固定し、SOURCE7300b39d…/最大6並列・合計6compileのpilotを一度起動。Phase AでB0_BLUEPRINT_FINGERPRINT_MISMATCH、native exit2/約8.60秒、snapshot1・overlay1予約、native build/control/compile0。driver/launcher退出、SOURCE/input不変、失敗分をrefund/resetせず保持。どのblueprint fieldが違うかは現メッセージで未特定、guard迂回・自動retryなし、mandatory STOP。 以下は各時点の履歴。

## 2026-10-15 H6 native B0+I 6軸compile並列化準備

[並列化実装・契約・検証入口](../docs/research/track_a_h6_native_b0i_parallel_readiness_20261015.md)。既存検査済み6wrapperをfork継承し、A/B全3case合格後のcompileだけ最大6processへ並列化。SOURCE7300b39d…、新24＋影響46の70人工tests PASS、軽い独立差分レビューPASS。再load/build/QPY転送を追加せず総compile6、各thread1、owned cleanup/PDEATHSIG/総CPU会計を実装。実fork/科学/compile/高速化実測0。各AS32/RSS9GiB、stage総CPU・wall4hを保持し、全role込みadmission240.5GiBを未承認案へ。CPU[]/false/未seal/commandnull、明示実行承認とfresh binding待ちSTOP。 以下は各時点の履歴。

## 2026-10-15 Track A H6 native B0+I pilot準備・承認待ちSTOP

[実装・契約・人工検証・独立監査入口](../docs/research/track_a_h6_native_b0i_pilot_readiness_20261015.md)。独立pilot runner/adapter/owned launcherを追加。SOURCE b4c12d7a…、author23＋独立23＋旧bridge17の63限定人工tests PASS。旧75結果・科学provider/bridge/shot規則は保持。5入力bytes/SHA・provider230/profile34,273 files一致、実H6数値読込・新scalar/signal/native build/probe/compile・新grant0。実行は未承認、CPU[]/未seal/commandnull。明示承認とfresh source/input/profile/carry/CPU・資源/quota/output binding後に3case6軸だけone-shot、matched B0/全面mapへの自動進行なし。 以下は各時点の履歴。

## 2026-10-15 Track A H8 D0 one-shot完了STOP

[結果・全table・監査入口](../docs/research/track_a_h8_d0_saved_value_result_20261015.md)。H8 D0 one-shot完了。SOURCE20f89bc3…、旧precision944/cost3576 EXACT PASS後に5精度/factor1・10/6metric/Pを評価。確認32のε=.0005/P=0/RZ PR/取得済B0-B1最良.761667659、探索Pのunion外coverageを分離。native exit0/wall2.97095秒/監視RSS52.3359MiB、source/input/profile不変、新science/compile/H4-H6再実行0。H6は別承認待ち、mandatory STOP。以下は各時点の履歴。

## 2026-10-15 Track A 独立技術監査・最小修正

[独立監査入口](../docs/research/track_a_independent_technical_audit_20261015.md)。著者文脈を引き継がない別担当が監査し、6制御不具合を再現・最小修正。SOURCE20f89bc3…、新22＋既存41＋変更adapter旧12の75限定人工tests PASS。旧D0/shot/math/native科学SOURCE・73旧worktree/grant/venv不変、実科学/再会計/build/compile/新grant0。empirical u・実一致/native・監視/運用bindingの限界を残し、未認可・未seal、GPT科学判断へ戻してmandatory STOP。以下は各時点の履歴。

## 2026-10-15 Track A D0 one-shot保存値再会計・完了STOP

[結果・完全一致・6metric/P・終了監査](../docs/research/track_a_d0_saved_value_reanalysis_result_20261014.md)。SOURCE fa75c799…、旧precision1,088/費用2,659件exact PASS後、5精度×factor1/10×6metric/Pを一度再会計。ε=.0005/P=0のPR/full-PF RZ点比はH4固定panel .429664、H6確認union8 .496333。未確認CX_depth/P候補と標本・政策差を明記。wall2.223秒/peak54.14MiB、exit0、SOURCE/入力/venv/他job/H8変更・新signal/trajectory/compile0、科学判断未採択・mandatory STOP。以下は各時点の履歴。

## 2026-10-14 Track A D0：保存scalar再会計の技術準備

[入力・code・人工検査・実行契約の入口](../docs/research/track_a_d0_saved_value_reanalysis_preflight_20261014.md)。H4 native fixed16／H6 exploration92／確認union8を分離し、ε=.0005・6指標・P感度を準備した。33限定人工tests PASS、SOURCE fa75c7997e135da7b7a0c4e23949e47252bd7a2e。実データの新再会計・新科学計算・H8変更・新grant0、別実行承認待ちSTOP。以下は各時点の履歴。

## 2026-10-14 H4 B0+I map240準備

新namespace h4_b0i_map、future runner run_h4_b0i_map240.py、byte-only prepare_h4_b0i_map240.py、stdlib19 mock検査。[閉じた234 compile＋6 numeric照合再利用の準備](../docs/research/track_a_h4_b0i_map_preparation_20261014.md)。実行条件は新grant待ち、計算未起動。

## 2026-10-14 H4 B0+I pilot6結果

固定SOURCEの代表6 wrapper実compile完了。[実測・保存shot再会計・終了監査](../docs/research/track_a_h4_b0i_pilot6_results_20261014.md)から公開verificationへ辿れる。科学source/既存テストの変更・追加compile・次stage起動はない。

## 2026-10-06 H4 signal/compile INPUT_BOUND草案・容量不足STOP

run02の6凍結入力へplanを結合し、source19/218 templates/compiler/run ID/outputを不変に保った。
stage必要3.5 GiB/560000 inodesに対しavailable3.419376 GiB、約82.6 MiB不足。CPU候補6 core/memory/quota観測はPASS。
全72h監視・74784 records・149569 ledger deltas・全8KiB worker logs・1308 signal files・journal/temp/directory余裕を含む。
CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は次段の提案、review=false、利用者の別stage承認/明示launch未取得。
signal/seed/sampling/build/compile/transpile/taskset/worker/GPU/共有環境・他job変更0、入力再生成0。
[次段scope・容量](../docs/research/track_a_h4_signal_compile_plan_review.md)と[資料入口・承認対象](../artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06/README.md)を参照する。
`H4_SIGNAL_COMPILE_PLAN_PREPARED_STORAGE_BLOCKED_STOP`。容量と別認可成立後もfresh検査不合格なら起動せず、map後MAP_COMPLETE_STOP。
scientific runtimeはcommitしない。旧10GiB案/旧source/旧bundle/入力生成完了と以下の履歴は保持する。

## 2026-10-06 H4入力生成run02・6入力freeze完了STOP

利用者の明示再実行指示でworker bootstrapのstdlib signal shadowを-Pで修正し、run01を保存してrun02を別固定した。
CPU [3,5,6,7,8,9]・6 workers・own-run限定、fresh resource/容量/quota検査PASS。
H4 linear/STO-3G/DF12の追加6距離入力を一度生成しfreeze完了。NPZ6 bytes SHAを照合し、own driver/worker残存0。
sourceは049e69919af16ad29a67a217dc7a407d6b1754a6。科学/seed/compiler/resource条件は不変、source19変更はgates/worker起動だけ。
`INPUTS_FROZEN_STOP`、next_stage_authorized=false、mandatory_stop=true。signal/compile/GPU/追加transpile0、共有環境・他job変更0。
[修正・完了scope](../docs/research/track_a_h4_worker_bootstrap_run02.md)と[完了報告](../artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/COMPLETION_REPORT_v1.md)を参照する。scientific runtimeはcommitしない。以下は各時点の履歴。

## 2026-10-06 H4入力生成stage容量確認・最終承認待ち

容量準備の判定は「足りる」。6入力生成→freeze→STOPの必要量3GiB/260000 inodesに対し、
2026-10-06 16:58:53 JSTのnonroot available約3.615GiB、225817022 inodes、user/group/project quota非有効をread-only確認。
32保存配列/NPY・ZIP overhead/64MiB IPC上限/temp-final/72h監視259202 files/journal/metadata余裕を含む。
全campaign10GiBはcharge capとして維持し、旧全量空き確保案を履歴に保存した上でstage-specific補足を追加した。
source19/plan/auth/approved=false reviewはbyte-identical、CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は未承認。
CPU使用許可/独立最終review/明示launch/fresh CPU・memory・pressure/OOM・容量検査が残る。signal/compile容量は別認可。
[容量根拠](../docs/research/track_a_h4_input_generation_stage_storage_review.md)と[最終承認資料入口](../artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_STAGE_STORAGE_CONFIRMED_AWAITING_FINAL_APPROVAL`で公開後STOP。
新科学/追加transpile/taskset/worker/GPU/共有環境・他job変更0。旧10GiB案を含む以下は当時の履歴。

## 2026-10-06 H4入力生成 CPU/launch最終案・利用者承認未取得

提案CPU[3,5,6,7,8,9]、6 worker、異なる6 physical core・NUMA0。約3秒の受動負荷sampleで各core busy0%。
source19 pathsとplan v2 bytes/source_rootは不変。authは候補CPU集合だけ、reviewはauth digestだけ変更しapproved=falseを保持。
own新規runだけにtaskset maskを指定する未実行commandを用意した。CPU許可/専有予約/独立review/明示launchは未取得。
memory/context read-only確認は成功。filesystem空き約3.717GiBは総上限10GiB全量確保案に未達で、launch容量条件は未解決。
12 metadata gate tests PASS、fail/error/skip0。観測01/02の失敗logと03の容量未解決記録を保持し、追加transpile0/旧28件不変。
[提案資料](../docs/research/track_a_h4_input_generation_cpu_launch_proposal.md)と[bundle・一括承認判断](../artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL`で公開後STOP。taskset/科学/worker/GPU/共有環境・他job変更0。

## 2026-10-06 H4入力生成 resource observer修正・未承認草案再固定

真のv2 hierarchy rootをnamespace/mount/所属から判定し、rootの非root memory interface要求を修正した。
全可視非root祖先の制限・pressure/OOMは保持し、非root欠測/不明namespace/hidden mountはSTOP、host-only fallbackなし。
observer33 zero-science tests PASS、実read-only観測成功。準備観測available約981.826GiB、PSI/OOM0はlaunch成立ではない。
production変更はresources observerとgateのnew audit pathだけ。科学/並列/seed/compiler source15件は不変。
[実装資料](../docs/research/track_a_h4_geometry_resource_observer_fix.md)と[source bundle](../artifacts/resource_applicability/track_a_h4_geometry_resource_observer_fix/2026-10-06/README.md)、
[new認可草案v2](../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/README.md)を参照する。source固定→別草案commit、binding検査結果はv2へ記録。
requested workers6、allowed_cpus=[]、approved=false。CPU/launch context・独立最終review・明示launch未解決、実行準備完了とはしない。
`H4_INPUT_GENERATION_RESOURCE_FIX_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/追加transpile/GPU/本番起動/共有環境・他job変更0。
旧bundle/科学証拠/原稿/Track Bと旧監査履歴を保存し、系列transpile28/64を維持する。

## 2026-10-06 H4 geometry 入力生成専用認可草案・実行未承認

凍結science source6a121725（17 Python＋親2件）を変えず、入力生成source-bound planとresult-prior認可草案を追加した。
requested workers6、inputs/freeze digestはnull、218 templatesを機械転記。reviewはapproved=false、allowed_cpus=[]。
CPU許可は利用者指示で未確定のまま。現在process CPU0–255を許可とみなさず、launch contextとmemory観測は未解決。
既存observerはroot cgroup memory.max欠落で停止し、実行準備完了とはしない。source/共有設定を緩和しない。
新57 zero-science gate tests PASS、fail/error/skip0。合格経路はメモリ内模擬承認だけ、追加transpile0・旧累積28/64不変。
[実装・停止条件](../docs/research/track_a_h4_geometry_input_generation_authorization_draft.md)と[bundle・最終レビュー入口](../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_AUTHORIZATION_DRAFT_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/GPU/本番起動/共有環境・他job変更0。
有効execution authorization0、final review/利用者の明示launch未実施。入力生成・本計算・signal/compile認可・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry compile並列source再固定・科学未実行

compileの逐次waitをadmitted worker数以下のbounded投入・回収へ変更した。処理中ownerを追跡し、
COMPLETEとidentity/digest検査後だけ再利用する。trajectory/axis順・weight、科学scope・seed/compilerは不変。
[実装資料](../docs/research/track_a_h4_geometry_parallel_source_implementation.md)と[new bundle](../artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06/README.md)を現在の入口とする。
既存94＋並列回帰17＝111 synthetic tests PASS、fail/error/skip0。今回transpile3、旧25＋新3＝28/64。
fake futures/mock workersだけで制御を検査し、実worker/production性能は未検証。旧bundle/audit/契約・保存証拠は不変。
SOURCEとsourceを変更しないREVIEWの2 commitを分ける。分子アクセス/科学処理/GPU/本番起動/認可発行/共有環境・他job変更0。
`H4_GEOMETRY_PARALLEL_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。入力生成plan/auth作成・本計算・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry server-native source固定・科学未実行

利用者の新指示で契約v2 D1〜D4を実装条件へ採用し、旧未承認履歴を保存した。
新namespace `src/trottertracks/resource_applicability/h4_geometry/`、二つのfuture runner、専用synthetic testsの入口は
[実装資料](../docs/research/track_a_h4_geometry_server_native_source_implementation.md)と
[bundle](../artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06/README.md)。
最終94 tests pass、fail/error/skip0。失敗・再検査込みsynthetic transpile25/64、旧benchmark128再実行0。
SOURCE_COMMITとsourceを変えないREVIEW_BUNDLE_COMMITを分離し、actual blob/hashは別監査で固定する。
旧247 source・v1/v2 bundle・準備25 files・保存6 JSONは不変。分子入力/科学処理/本番runner launch/GPU/環境・他job変更/認可発行0。
`H4_GEOMETRY_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。別入力生成authorizationの作成へ進めるかをレビューし、今回は発行・実行しない。
以下は各milestone当時の履歴。


## 2026-10-06 H4 geometry契約v2・レビュー待ちSTOP

現在の入口は[契約v2 bundle](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/README.md)。v1 commit `7c1a3d43f61c5501a9e79206b7c60933f94b1077`を保存し、
D1〜D4を具体的な採用案、memory admissionを8+8w+16 GiB、認可を入力生成→freeze STOP→別signal/compile認可へ分離した。
H4 linear/STO-3G/DF rank12、6距離・218 template・32 paired trajectories・74,784上限は不変。8 system＋ancilla1(index8)、合計9 qubits。
[pure JSON validator](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/contract_validator_v2.py)と[専用合成検査](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/run_contract_tests_v2.py)はreview用で、science source/runnerではない。
新規320件pass（fail/skip0）、旧129件は保存・runner再実行0。旧v1 manifestはbase blobで照合し書き換えない。
D1〜D4レビュー承認は未解決、science/source port/input generation/next stage認可false、plan未seal、mandatory STOP。
今回の公開指示は軽量契約bundleと関連文書だけのcommit/non-force push。以下は各段階当時の履歴。

原稿用の表示・照合script：

- [build_track_a_manuscript_figures.py](resource_applicability/build_track_a_manuscript_figures.py)：
  固定9 CSV/JSONのcommit blob/SHAを照合して4主図・補足図1と表示CSVを新規outputへ生成する。
- [verify_track_a_manuscript_bundle.py](resource_applicability/verify_track_a_manuscript_bundle.py)：
  22assetのmanifest、表示値、原稿リンク・hashと専用21 synthetic/local testsを検査する。

[原稿付録の再表示手順](../docs/manuscripts/track_a_resource_study_supplement_v0_1.md)を参照。
科学module、分子snapshot、sampler、compiler、GPUは使わない。新しい科学runnerではない。

`resource_applicability/run_pr2_pm2_precision_analysis.py`は明示指示後に一回完了し、
[PM-2結果照合](../docs/pr2_pm2_precision_resource_result_validation.md)へ戻った。source不変、保存値解析のみ。
固定outputは作成済みで、上書き/resume/retryはしない。追加science/次段は未認可。
以下のsource固定・launch待ち記述は当時の履歴である。

PM-2解析sourceの入口は`resource_applicability/run_pr2_pm2_precision_analysis.py`と
`resource_applicability/run_pr2_pm2_implementation_tests.py`（synthetic-only、real saved evidenceも禁止）。
[実装と停止条件](../docs/research/pr2_pm2_precision_analysis_implementation.md)を参照する。
source `324435d77b6642dbd44e8d1f178420daf62e77ed`、62 synthetic tests passed。本解析は明示launch待ち、固定outputは未作成。
以下の契約準備・未実装記述は当時の履歴である。

PM-2の入口は`resource_applicability/run_pr2_pm2_precision_contract.py`（保存JSON4件の準備bundleをstdoutへ出すのみ）と
`resource_applicability/run_pr2_pm2_preparation_tests.py`（保護path guard付き専用contract tests）。
[契約/schema/input inventory](../docs/research/pr2_pm2_precision_resource_contract_v1.md)を参照する。
精度解析runnerは未実装、ε sweep・順位・P envelope・新signal/compileは未実行、解析は未認可。

PM-1は最終review・authorization一項目確定・利用者launch後に一回完了した。
[結果と照合](../docs/pr2_pm1_discard_result_validation.md)：8 signals/16 wrappers、pre/post201 local tests passed。
source不変、`PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`でmandatory STOP、retry/resume/次段なし。
下のdraft/future記述は実装・準備時点の履歴であり、現在の実行状態はこの節を優先する。

`resource_applicability/run_pr2_post_m2_evidence_attribution.py`はTrack A PM-0の保存JSON/source再集計専用。
stdlibのみ、artifact内pathを辿らず、bundleをstdoutへ出すだけでファイルを書き換えない。
[報告・対応source/test/artifact](../docs/research/pr2_post_m2_evidence_attribution.md)を参照する。
PM-1準備は`resource_applicability/run_pr2_pm1_discard_contract.py`（stdoutのみのsealed plan生成）、
`resource_applicability/run_pr2_pm1_preparation_tests.py`（科学データguard付き限定7-file tests）。
`resource_applicability/run_pr2_pm1_discard.py`は別committed authorization後のfuture科学入口であり、今回は呼ばない。
[契約・source・preparation audit](../docs/research/pr2_pm1_nearby_discard_contract_v1.md)を参照する。本実行未認可。

`scripts/` はコマンドラインから実行する入口を置く。研究ロジックの本体は原則として
`src/trotterlib/` にあり、runnerは条件の読込み、呼出し、成果物保存を担当する。

最新の研究段階は[`../docs/research/研究概要・現状.md`](../docs/research/研究概要・現状.md)、
各runnerと成果物の対応は[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)を
参照する。下記の「現行」は、最終科学結論を意味せず、現在の近似検証で使う経路を表す。

## 現行の検証runner

### PF誤差係数

| runner | 用途 |
|---|---|
| `run_pf_delta_validation.py` | 小規模系でPF係数の定義・delta依存・参照値を比較。cross-validation結合には`--snapshot --n-electrons`を使用可能 |
| `run_pf_c_system_size_validation.py` | H-chainの系サイズ方向とstate-action経路を検証 |
| `run_df_ground_state.py` | DF Hamiltonianの基底状態snapshotを作成 |
| `benchmark_df_ground_state.py` | 基底状態計算経路の計測 |
| `validate_pauli_partial_cgs_all_prefixes.py` | Pauli partial回路係数のprefix検査 |

### finite RTE

| runner | 用途 |
|---|---|
| `run_finite_rte_signal_validation.py` | 演算子誤差、状態上の複素期待値、半径、位相上界を検証 |
| `run_fr_revision_nonuniform.py` | FR-R1bの凍結expected生成と非一様4×4・負時間・K4 controlの実行を分離し、R0--R7を判定 |
| `run_pr2_s0_s1_validation.py` | 結果前v3に従い、非上書きの独立`S0`/`S1` commandだけを提供する。S2/S3 commandと自動進行は持たない |
| `run_pr2_new_series_v1_v3.py` | 旧STOPを保持した別snapshot系列のV1–V3だけを非上書き実行し、V4/S1′未承認で停止する |
| `run_pr2_v4_correctness.py` | 別snapshot系列のcorrected/raw signal、Re/Im Hadamard semantics、controlled full-wrapper compileを検査する |
| `run_pr2_s2_development.py` | V4 PASS後の固定development-only resource comparisonをserial実行し、mandatory STOPする |
| `run_pr2_s2_development_parallel.py` | 同じS2 cell、seed、32→96拡張barrierを維持し、full-wrapper compileを最大8 CPU workersで並列実行する。既定4 workers、persistent SQLite cache、別outputを使用する |
| `run_pr2_matched_accuracy_m1_contract.py` | M1の208候補identity、最大4境界候補、16-cell selectorをsynthetic fixtureで検査するzero-compute runner。NPZ、signal、trajectory、circuit、compileを開かずmandatory STOPする |
| `run_pr2_matched_accuracy_m1_precompile_barrier.py` | M1-A/M1-B hard barrierのlimited/clear synthetic分岐を検査するzero-compute runner。limited時のcompile job生成を拒否し、科学計算を承認しない |
| `run_pr2_matched_accuracy_m1_a.py` | 結果前authorizationに従い、development H4 1.00 Åだけで最大212候補のcompile-free signal/selectorを実行する。実結果は210候補、52未選択frontierにより`SELECTION_LIMITED`、compile 0で停止 |
| `run_pr2_matched_accuracy_m1_b1_contract.py` | byte-fixed M1-A結果からrandom 194＋baseline 16 cellと12,448 wrapperのcache identityを生成するstandard-library-only zero-compute runner。snapshot、trajectory、circuit、compiler、GPUを開かず停止する |
| `run_pr2_matched_accuracy_m1_b1.py` | actual execution source commitに結合したzero-compute planをfreezeし、別commitのresult-prior authorization後だけ12,448-wrapper compile mapを最大6 workersで実行する。研究四分岐を自動選択せずreview待ちで停止する |
| `run_pr2_matched_accuracy_m1_b1_result_validation.py` | 保存済みM1-A/M1-B1 result、210 checkpoint、210 candidate別SQLite cacheをread-only再検査し、actual frontier、fixed-q=8差、旧selector、proxy、状態準備感度と外部review判断をJSONへ出す。分子snapshotとheld-outは読まない |
| `run_pr2_matched_accuracy_m2_transfer_contract.py` | commit済みM1-B1 result/validationだけを読み、5構成、usable B2限定のtransfer判定、seed、196-wrapper上限を固定するstandard-library-only zero-compute runner。v2正式planはsource commit blob照合必須、未commitは `--draft` のみ。held-out pathをresolve/stat/hash/loadせず、科学実行を認可しない |
| `run_pr2_matched_accuracy_m2_transfer.py` | actual source commitに結合したzero-science `plan`と、別のcommit済みauthorizationを要求するone-shot `run`を分離する。最大5 workers、固定5構成、196 wrappers。一回の実行は`TRANSFER_SUPPORTED`で完了し、[結果照合](../docs/pr2_matched_accuracy_m2_transfer_result_validation.md)後mandatory STOP。再実行/resume/別output救済・追加科学計算は認可しない |

### コンパイル後回路コスト

| runner | 用途 |
|---|---|
| `run_random_circuit_cost_validation.py` | イベント単純加算と一体コンパイルを比較 |
| `run_rte_boundary_cost_validation.py` | 2・3イベントの境界補正を検証 |
| `run_rte_boundary_cost_replication.py` | 境界分類を別乱数seedで再検証 |
| `run_hierarchical_cost_validation.py` | 局所補正、長さ、制御付き反復をまとめて検証 |
| `run_rte_order_stratified_cost_validation.py` | 低確率Taylor次数を条件付き抽出して検証 |
| `run_rte_connected_cluster_cost_validation.py` | 1--3イベント境界補正の運用推定を検証 |
| `run_rte_connected_cluster_transfer_validation.py` | $L_D$・短時間幅が異なる条件へ移した場合を検証 |
| `run_rte_connected_cluster_holdout_supplement.py` | 独立検証の標本を補充 |
| `run_rte_connected_cluster_k4_calibration.py` | 不通過条件の4イベント係数を較正 |
| `run_rte_connected_cluster_k4_extrapolation_diagnostic.py` | 4イベント補正の外挿診断 |
| `run_rte_paired_k4_l8_validation.py` | 同一イベント列で4イベント補正の構造残差を検証 |
| `run_controlled_repetition_holdout.py` | 制御付き反復回路の反復数依存を検証 |
| `run_rte_cost_angle_invariance_validation.py` | 回転角をまたぐmetric cache再利用可否を検証 |
| `run_h5_system_size_cost_validation.py` | H5の対応あり構造検証 |
| `run_h5_independent_cost_validation_batch.py` | H5の独立係数較正・未使用回路検証 |

### RPE信号・測定回数・1 shotコストへの接続

| runner | 用途 |
|---|---|
| `run_rpe_round_cost_connection_validation.py` | 短いRPE段で信号半径、shot数、回路costを接続 |
| `run_rpe_hadamard_failure_validation.py` | 仮想Hadamard測定と失敗確率を検証 |
| `run_rpe_hadamard_proxy_resource_validation.py` | $q=8$の1 shot cost proxyと資源集計を検証 |
| `run_rpe_allocation_sensitivity_validation.py` | 位相誤差・失敗確率配分の感度を比較 |
| `run_rpe_four_round_accounting_validation.py` | $q=1,2,4,8$の限定4段を集計 |
| `run_rpe_four_round_phase_validation.py` | $q=8$物理信号と4段の分枝付き位相復元を検証 |
| `run_rpe_target_round_horizon_validation.py` | 目標精度から必要round数を決め、固定設定の長$q$可否を行列診断 |
| `run_rpe_delta_round_schedule_validation.py` | 実行済み$\delta$をPFでscreeningし、round別$(r_m,K_m)$ scheduleを行列検証 |
| `run_rpe_delta_compiled_cost_validation.py` | 検証済み局所境界係数をround scheduleへ接続し、中央RTEブロックのcompiled-cost proxyを比較 |
| `run_research_direction_prevalidation.py` | WP00比較契約、WP02 round-horizon coverage、WP01-Sの条件付き小系比較を順に生成 |
| `run_research_direction_ablation.py` | WP04のschedule、$\beta$、$\alpha$、cost-provider寄与分解と信号・統計bound診断を生成 |
| `run_research_direction_pf_sensitivity.py` | WP03でPF係数だけを差し替え、$L_D$・$\delta$選択とcandidate regretの感度を生成 |
| `run_research_direction_gate_s1.py` | WP00/WP02/WP01-S/WP04/WP03のfingerprint済み結果をGate S1の研究方向判断へ統合 |
| `run_research_direction_structure_pilot.py` | WP06-aでfull/support限定Gaussian basis、basis融合、control、relative phaseを代表Z/ZZと短列で比較 |
| `run_research_direction_sequence_policy.py` | WP06-bでsequence-aware basis policyを独立training/holdoutし、中央RTE差を既存$q$ slopeへ接続 |
| `run_research_direction_full_scope.py` | WP05-aで選択済みbasis policyをcomplete controlled partial-$S_2$／Hadamard wrapperへ接続し、$q=1,2$較正と未使用$q=4$を直接transpile |
| `run_research_direction_full_scope_extension.py` | WP05-bで$q=8$と比較対照$\delta=0.01$へfull-scope較正を拡張 |
| `run_research_direction_full_scope_replication.py` | WP05-bRで境界条件$\delta=0.02,r=32,q=8$を独立32 trajectoryで再検証 |
| `run_research_direction_decision_cost.py` | WP01-D/C07でfull-scope proxyを使い、候補ごとの誤差配分とshot数を再最適化 |
| `run_research_direction_decision_synthesis.py` | WP01-D/C07のlocal区間と移送感度区間を分離して方向判断を生成 |
| `run_research_direction_round_dominance.py` | G08でround別cost・proxy不確かさ・PF/RTE riskを分解しM08対象を固定 |
| `run_research_direction_proxy_precision.py` | M08で支配的$r=32$の未使用$q=16,32$ full-wrapper holdoutを直接transpile |
| `run_research_direction_m08_reaggregation.py` | M08実測幅、従来5%、移送25%でWP01-D/C07判断区間を再集計 |
| `run_research_direction_compiler_transfer_compute.py` | M06/L08用に同一trajectoryをoptimization level 2で再compileする計算専用runner |
| `run_research_direction_compiler_transfer_analysis.py` | M06/L08のoptimization level 2結果を解析し、固定planでfocused再集計するrunner |
| `run_research_direction_uncertainty_break_even.py` | N07不確かさ台帳とP03状態準備break-evenを既存artifactから再集計するrunner |
| `run_research_direction_wp11_synthesis.py` | WP11で実施済みartifactをT1--T7へ統合し、次の検証を一件だけ選ぶrunner |
| `run_research_direction_full_opt2_compute.py` | M06-Fのall-r opt2 cell manifest、dry-run、checkpoint/resume computeを担当するrunner |
| `run_research_direction_full_opt2_analysis.py` | 完了済みM06-F computeだけを解析し、coherent opt2再最適化と必要時のfresh 32-trajectory manifestを生成するrunner |
| `run_research_direction_full_opt2_completion.py` | M06-Fのtask/worker/checkpoint/aggregate完全性と解析gateを再監査し、既存結果を上書きせず日付付きartifactを生成するrunner |
| `run_research_direction_full_opt2_extension_analysis.py` | 初回36とfresh-32 15 taskを統合監査し、両gate通過時だけcoherent opt2再最適化とbreak-even再集計を行うrunner |
| `run_research_direction_proxy_lineage_reconciliation.py` | A0として最新fresh `q=1,2` RZ proxyを旧opt2 `q=16,32`固定holdoutへ再適用し、lineageと適用domainをfingerprint付きで記録 |
| `run_research_direction_signal_weight_pilot.py` | P-Bとして既存H4 PF artifactのenergy bias、target weight、q別signalを再集計し、signal-aware選択差の有無を記録 |
| `run_research_direction_geometry_energy_difference_pilot.py` | P-CとしてH4 5 geometryのsigned PF係数、geometry/delta holdout、差分energy biasを再集計 |
| `run_research_direction_geometry_tracking_breakdown.py` | 事前固定したH4 8 geometryでindependent/tracked prefix、signed PF bias、blind/pair予測、continuity診断を評価 |
| `run_research_direction_joint_synthesis_pilot.py` | P-Aとしてfull共有、event-support、現行policy、interval-union DPを独立event列で同一compile比較 |
| `run_research_direction_theme_selection.py` | fingerprint済みP-B/P-C/P-Aを4問で比較し、暫定主題と次の停止点を記録 |
| `run_research_direction_energy_tail_pareto.py` | P-Dとして固定PF familyの次数、exact two-block energy bias、`Gamma_R`、finite-RTE解析負担をdevelopment/blindで比較 |
| `run_research_direction_pd_realization.py` | P-D Go/No-Goとして負時間finite-RTE/control位相、fragment内部`H_D`誤差、fresh `L_D=5`選択差を固定条件で検証 |
| `run_research_direction_pd_fair_comparison.py` | P-D S1として共通時間・位相予算でB0/B1a/B1b/B2/B4をnested/native比較し、regret、限定K4、境界規則でCase A--Dを判定 |
| `run_research_direction_pd_s1_posthoc.py` | 固定S1 v2 artifactだけを再集計し、主baseline、5%近傍、B1aのm_D依存、nested/native内訳を事後解釈として保存 |
| `run_finite_rte_phase_amplitude.py` | FR-1の固定2×2 gridでphase/radius境界、semantic、G0--G4を評価しfingerprint付きartifactを保存 |
| `run_fr_revision_fr1a_posthoc.py` | 完了済みFR-1の33条件・99状態を再構成し、正scalarと強いnorm/FR baselineを事後再集計。旧判断とFR-R1b事前登録は変更しない |
| `run_research_direction_joint_synthesis_blind_validation.py` | P-A v1を未使用H5 snapshotとH4 opt2へ移す事前登録済みblind検証。compile前manifest、per-task checkpoint、resume、status対応 |
| `run_research_direction_joint_synthesis_formalization.py` | 完了済みpilot/blind artifactを再監査し、P-A v1の目的関数、DP、計算量、同値性条件と機構coverageをfingerprint付きで記録 |
| `run_research_direction_joint_synthesis_mechanism_validation.py` | P-A v1を明示的一区間baselineとforced-support order-2 streamで比較し、固定gateに従って継続／停止を判定 |

## 長時間・複数条件のbatch runner

次のrunnerは、上記の検証を複数jobで実行・再開するためのもの。単独の数値だけで結論を
判断せず、対応する検証文書とmanifestを読む。

- `run_parallel_validation_batch.py`：共有サーバー向けtask manifest生成、dry-run、bounded実行、resume、status
- `run_rte_cost_data_batch.py`
- `run_rte_cost_followup_batch.py`
- `run_df_anchor_refinement_pipeline.py`
- `run_df_ld_window_cgs.py`
- `run_df_screening_anchor_cgs_h3_h14.py`
- `run_h5_df_cgs_gpu_slopes.py`
- `run_h9_ld5_df_cgs_ground_cache_slope_check.py`

## 診断・変換・可視化

| script | 用途 |
|---|---|
| `check_validation_manifest.py` | manifestのschemaと参照パスを検査 |
| `check_partial_randomized_diagnostics.py` | partial-randomized計算の診断値を確認 |
| `analyze_df_ld_direction_trend.py` | $L_D$方向の傾向を解析 |
| `compare_df_error_budget_rules.py` | 誤差予算規則を比較 |
| `compare_df_error_budget_rules_from_summaries.py` | 保存済み要約から誤差予算規則を比較 |
| `cache_grouping_step_costs.py` | grouping step costをキャッシュ |
| `export_df_cgs_cost_table.py` | CGS cost表を書き出す |
| `plot_df_optimized_costs.py` | DF最適化costを可視化 |
| `plot_df_partial_grouping_cost_comparison.py` | partial grouping間のcostを比較 |
| `profile_h5_8th_cgs.py` | 旧高次PF経路のH5 profiling |

## 旧経路・現在の結論に直接使わないrunner

次は履歴・比較・再現のため残している。出力を現行結果として使う前に
[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)を確認する。

- `run_partial_randomized_pf.py`
- `run_df_screening_cost_minimization.py`
- `run_grouped_uwc_pf_qpe.py`
- `run_grouped_uwc_theta_sweep.py`
- `run_uwc_grouped_alpha_h2_h6.py`

特に旧DF screeningの入力表は基底状態不一致のため失効しており、UWC数値は参照raw artifactが
未登録である。runnerが存在することと、研究結果として利用可能であることを混同しない。

## runnerを追加するとき

新しい検証runnerは、可能なら次の同名セットを用意する。

```text
src/trotterlib/<validation_name>.py
scripts/run_<validation_name>.py
tests/test_<validation_name>.py
docs/<validation_name>.md
artifacts/<validation_name>/
```

結果を研究上の証拠へ加える場合は、`VALIDATION_STATUS.md`、研究概要、研究ノート、
`artifacts/validation_manifest.json`も更新する。


## H4 geometry 契約準備 v1（2026-10-06・local未commit）

[準備bundle](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06/README.md)は契約schema・zero-compute plan・pure JSON validatorと129合成検査の入口。
6距離、218 template/点、74,784 wrapper、最大12 workersを固定し、生成/seed/memory/wall/outputはreview待ち。
science source/runnerの追加ではなく、旧公開draft・source・科学結果は不変。本計算・port・commit/pushは未認可、STOP。


## H4候補間compile投入・12 worker明示再実行

利用者の増員再実行指示により[run03 source・認可・検査記録](../docs/research/track_a_h4_cross_candidate_run03.md)を追加した。候補内2回路の完了待ちで4 workerがidleとなる問題を、候補間bounded queueで修正。旧run02はworker failure STOPで全証跡を保持し、旧6入力を再生成せず利用する。49人工job/metadata testsはlocal PASSで科学的結果ではない。12 workerのfresh CPU/memory/容量/hash検査後だけ一度起動しMAP_COMPLETE_STOP。旧累積bytes/wall/actual invocationsを引継ぎ、科学条件・compiler・上限は不変。


## H4 run04：identity hash分割・5秒監視維持

[run04固定sourceと再実行binding](../docs/research/track_a_h4_streaming_monitor_run04.md)を追加した。run03はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。65 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/7 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## H4 run05：identity hash分割・5秒監視維持

[run05固定sourceと再実行binding](../docs/research/track_a_h4_lazy_identity_run05.md)を追加した。run04はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。75 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/12 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## 2026-10-07 H4新host A案 source/人工検証固定・本計算未認可

[監視修正・32人工tests](../docs/research/track_a_h4_new_server_monitor_fix_a_20261007.md)。既存private venv不変で準備用A案採用。
旧source19は3変更/16不変、新module込み25 closure。256×256人工matrix＋9-qubitの旧/new byte/digest一致、独立observerのGIL/GC観測・I/O delay/EOF/所有・資源境界を検証。
SOURCE `b2a5ad89e8b39d72716f7ddb17d263bd0cdedb45`、production/追加transpile0、旧28/64・benchmark128保持。環境18 version差、旧45 raw-reference RECORD差保持・normalized22差を明記。入力6/freeze/runtime/control未受領。
observer AS256MiB/RSS64MiB/admission120.25GiB、容量5.5625GiB案は未承認。allowed_cpus=[]/approved=false/runtime_authorization=false/launch=null、STOP。科学成果/原稿/Track Bと旧資料を保持。

人工runner `scripts/resource_applicability/run_h4_monitor_fix_a_tests.py`、auditor `scripts/resource_applicability/audit_h4_monitor_fix_a.py`。実runnerを起動しない。


## 2026-10-08 H4本計算前準備・最終承認待ちSTOP

[統合入口・最終承認案](../docs/research/track_a_h4_prelaunch_preparation_20261008.md)。新host schema/profile/observer/CPU/one-shot/累積budget bindingを整備。
SOURCE `ad57d1639133f7158cce58d767b8e0aa179bf044`、32 closure、57限定人工tests PASS。core quota read-only確認、worker12＋driver/observer各1別coreを提案。
carry20/165214360 bytes/5466.188392877579秒、残74764を保持。全74784 logicalを保証する最小actual cap+20→74804案は未承認。
候補環境はprivate venv不変、18 version/旧参照対raw45・normalized22差。追加output5GiB/301000 inodes、charge約8.29GiB、copy前は暫定6GiB。
入力6/freeze/native stop proof未受領、allowed_cpus=[]/approved=false/runtime_authorization=false/未seal、science/追加transpile/GPU/共有環境・他job変更0。明示承認・final review・launch前にSTOP。


人工runner `resource_applicability/run_h4_prelaunch_tests.py`、source-bound auditor `resource_applicability/audit_h4_prelaunch_preparation.py`。本番runnerを起動しない。

## 2026-10-09 H4 cleanup ESRCH修正・独立再review PASS

[新SOURCE・独立再review・残る承認条件](../docs/research/track_a_h4_cleanup_esrch_fix_20261009.md)。旧992c09d6から独立worktreeでP1を修正し、SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33を固定。
両pidfd送信経路はESRCHだけ既退出扱いで後続cleanupを継続し、他の送信error・所有検証を保持する。
限定人工39件PASS、別担当13純mock件PASSとbinding照合でP1_SCOPE_TECHNICAL_PASS。wait/reap/FD/pipe/first STOP理由保持まで確認。
新test `tests/tracks/resource_applicability/test_h4_cleanup_esrch.py` と既存人工runnerで追跡し、source/profile/plan/auth/reviewを新SHAへ再結合した。
入力0/6・freeze/native停止証拠未受領はNOT_EVALUABLE、未seal。環境/CPU/observer/74804案は未承認、carry20/165214360 bytes/5466.188392877579秒・現actual cap74784不変。
approved=false、runtime_authorization=false、allowed_cpus=[]。追加transpile/科学actual/本番起動0。旧資料を保存して本計算STOP。

## 2026-10-09 H4全byte受領・carry合格・native終端proof待ち

[受領・binding・最終承認案](../docs/research/track_a_h4_byte_receipt_binding_20261009.md)。packet86691840B/全2150files/既知9SHAをbyte-only照合しPASS、NPZ6/freeze受領完了。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変。ledger chain/cumulative journalからcarry20/165214360 bytes/5466.188392877579秒を保持、現cap74784・残74764。
run05 log/exact旧sourceからworker cleanup到達は推認可能だが、driver/12 workers停止後identity/残存0 native proofが不足。古いrun02停止監査をrun05proofへ流用しない。
input/profile/source/output/carryとv6 plan/auth/reviewを結合し、control82件のbasename・単一link・sender manifest mappingを照合。source条件は緩めていない。
sealed=false/approved=false/runtime_authorization=false/allowed_cpus=[]。環境/CPU/observer/累積74804案/一度のmap launchは未承認。科学array読込/新科学actual/追加transpile/共有設定変更0でSTOP。

## 2026-10-09 H4追加native停止proof合格・technical再seal

[再seal・独立最終整合review・一括承認案](../docs/research/track_a_h4_native_proof_seal_20261009.md)。追加JSON17521B/SHA一致、旧host/run05の13identity・2回残存0・元3証拠hashを照合して現在のnative停止条件PASS。
過去のexit code/正確な終了・reap時刻/原boot IDは未記録のままnull。今回の観測で補完せず、連続監視や歴史cleanup順の証明とも扱わない。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変、profile/input/carry/control83件の固定validator合格でplan再seal、sealed=true。
approved=false/runtime_authorization=false/allowed_cpus=[]。carry20/165214360 bytes/5466.188392877579秒、現cap74784・残74764を保持。
候補environment/compiler・CPU・observer・累積actual74804案・一度のmap launchは未承認。承認による最終artifact/digest再結合とfresh resource/CPU/fs/inode/quota gateをlaunch前に確認。
科学array読込/新science actual/追加transpile/source変更/共有設定変更/本計算0でSTOP。

## 2026-10-09 H4一度のmap実行を利用者承認・最終artifact固定

[実行認可・直前gate・起動報告入口](../docs/research/track_a_h4_authorized_launch_20261009.md)。利用者の明示認可で候補environment/compiler採用、worker12 CPUs2/4–6/8–15・driver16・observer18、observerAS256MiB/RSS64MiB/admission120.25GiBを認可。
carry20/165214360 bytes/5466.188392877579秒を保持し、累積actualだけ74804へ+20改定。SOURCE6bd1ba01・science/compiler/options/他caps不変。
sealed/approved/runtime_authorization=true、allowed_cpusはexact14role集合。独立v8 review PASS。artifact commit後fresh CPU/memory/PSI/OOM/FS/inode/quotaとSOURCE/profile/input/carry/unusedrootを確認しPASSなら追加承認なし一度起動。
既存proof/回帰/benchmark再実行0、追加準備campaign/transpile0。oldpartial/cache/GPU/共有環境・venv・他job変更なし、完了またはfail-closed STOP後終了・retry/次stageなし。
これは認可artifact固定時点のsnapshot。実起動/PID/状態は入口への追記・外部runtime receiptで別記録する。

## 2026-10-09 H4 library cache保存先修正・再実行予算不合格

[修正・再実行条件](../docs/research/track_a_h4_library_cache_fix_20261009.md)。前回のOpenFermion→Matplotlib mkdir拒否を、homeの新private library cacheへprocess限定MPLCONFIGDIRを結合して修正。
driver/workerとも既存directoryのEEXIST probe以外のcache writeを拒否、29816BのSHA固定。47限定回帰＋3 import case PASS、科学array/transpile/実worker/affinity/GPU0。
SOURCE `b8b3ce6e8c98f1ec0419a7af79c5d7c5f3a3b9bb`、36 closure、science/compiler/options不変。前回費用を返却せずcarry20/4428938712B/5472.345380863175sへ結合。
累積worst charge13165893832B=12.261694GiB>承認10GiBで未seal/approved=false/runtime_authorization=false、再起動0。13GiBは未承認proposalのみ。
既承認environment/CPU/observer/actual74804を保持。cap改定・新SOURCE/gate binding/review・fresh gate後の一度再実行が残る。
private homeは共有systemと区別し、旧run/one-shot/失敗証拠・全予約課金を保持する。

## 2026-10-09 H4軽量高速化・限定同等性確認

限定[runner](resource_applicability/run_h4_lightweight_speedup_tests.py)・[tests](../tests/tracks/resource_applicability/test_h4_lightweight_speedup.py)、実装[signal](../src/trottertracks/resource_applicability/h4_geometry/signal.py)・[ledger](../src/trottertracks/resource_applicability/h4_geometry/ledger.py)、[bundle](../artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/README.md)。

[変更・限定検証・binding](../docs/research/track_a_h4_lightweight_speedup_20261009.md)。driverの距離内共通準備を再利用し、one/DF block呼出を静的13×218→13、全prepareを218→10種類に削減。ledger deltaは変更entry/reservation各最大1だけを参照し、全件走査・saved-historyコピーを除いた。
SOURCE `4d2d1492fc23d0736c305533d78967cc1db8a7c8`、closure38。限定48人工PASS、全218prep/代表8wrapper+dense256case1/代表4signal/ledger13fileの旧new bytes・digest一致。12workersはmock、単一test process内部thread1、science array/transpile/GPU/affinity/production0。
実Gaussian/旧compiler output/実速度・H4本体成功は未検証。monitor/caps/compiler/science/carry不変、旧partial/cacheと混合しない。
carry20/4428938712B/5472.345380863175s、actual74804既承認、worst charge12.261694GiB>承認10GiBは残る。未seal/approved=false/runtime_authorization=false、absolute_launch_command=null、本計算0。

## 2026-10-09 H4利用者が13GiB累積charge・一度の再実行を明示認可

[認可・source・一度の起動入口](../docs/research/track_a_h4_approved_relaunch_20261009.md)。利用者の「これについては問題ないので再実行して」を、既存13GiB cumulative charge案と一度のmap再実行の承認として反映。
SOURCE `a7b617600cd7063f7870f2059d5694ef00283f0e`/closure39、output改定schema/gate/実OutputBudget capとmarginのみ変更。legacy10GiB default・科学/compiler/その他caps・prepare再利用/ledger保存は維持。
限定22pure gate PASS、旧48speedup/library/cleanup/native証拠campaign再実行なし。carry20/4428938712B/5472.345380863175s返却なし、actual74804・新残74784。
worst13165893832B <= 新cap13958643712B、余裕792749880B。sealed/approved/runtime_authorization=trueの認可artifactへ再結合。
独立review/artifact固定後fresh SOURCE/profile/input/carry・CPU/memory/PSI/OOM/fs/block/inode/quota/unusedroot/one-shot合格時にそのまま一度起動。実run状態はruntime証跡へ記録。
既承認12workers CPUs2/4–6/8–15、driver16/observer18、thread1・observerAS256MiB/RSS64MiB/admission120.25GiB保持。自動retry/入力再生成/旧partial/cache/次stage/GPU/共有設定変更なし。


## H4 worker failure限定回帰

[run_h4_worker_failure_tests.py](resource_applicability/run_h4_worker_failure_tests.py)はhomeの事前人工plan/evidenceを要求する単一process検査。実child/科学入力/transpile/affinity0、12workersはmock、publication raceだけ2fixture threads。[scopeと上限](../docs/research/track_a_h4_worker_error_fix_20261009.md)。


## H4 run03純粋binding検証

[run_h4_retry_binding_tests.py](resource_applicability/run_h4_retry_binding_tests.py)は1 process/内部thread1、事前home planの資源上限で予算・carry・draft拒否だけを検査する。[scope](../docs/research/track_a_h4_production_run03_20261009.md)。人工回路/compile/実childなし。

## H4 entrypoint cache修正（2026-10-09）

[原因保存・private memory cache・8限定metadata検証](../docs/research/track_a_h4_entrypoint_cache_fix_20261009.md)。別namespaceのlibrary_cache v2 policy、runner run_h4_entrypoint_cache_tests.py。科学compile/追加本計算なし。


## 2026-10-09 H4 run04：carry0で本計算、STOP後も修正・再実行を継続

利用者が過去失敗分を次回へ加算しない方針と、STOP後チャットを終了せず原因調査・修正・再実行の継続、本体起動を明示認可。
新SOURCE d31b51080665a7eea806ff1adc17f3e0d19151fc/closure49、library cache v2保持、新run04。純粋11 tests PASS、人工compile/追加transpile0。
carry0（actual/charge/wall）・既承認17GiB/74805/72h、全map8.136939GiB/74784。環境/CPU12workers/driver16/observer18/資源caps/科学compiler不変。
旧証拠・入力・one-shotを保持し、各fresh attemptを独立認可binding。停止した同一runのresumeと旧partial科学cache混合は禁止。
独立review・immutable artifact・直前fresh gatesを経て本計算、一度のSTOPで作業を終了せず必要修正とfresh再起動を続ける。次stage未認可。
[最新入口](../docs/research/track_a_h4_production_run04_20261009.md)。


## 2026-10-09 H4 run05：carry0で本計算、STOP後も修正・再実行を継続

利用者が過去失敗分を次回へ加算しない方針と、STOP後チャットを終了せず原因調査・修正・再実行の継続、本体起動を明示認可。
新SOURCE 9d1471aff5840a76fa1579f4e14e71e62b8497a3/closure52、library cache v2保持、新run05。純粋16 tests PASS、人工compile/追加transpile0。
carry0（actual/charge/wall）・既承認17GiB/74805/72h、全map8.136939GiB/74784。環境/CPU12workers/driver16/observer18/資源caps/科学compiler不変。
旧証拠・入力・one-shotを保持し、各fresh attemptを独立認可binding。停止した同一runのresumeと旧partial科学cache混合は禁止。
独立review・immutable artifact・直前fresh gatesを経て本計算、一度のSTOPで作業を終了せず必要修正とfresh再起動を続ける。次stage未認可。
[最新入口](../docs/research/track_a_h4_production_run05_20261009.md)。


## 2026-10-10 H4 run06：carry0で本計算、STOP後も修正・再実行を継続

利用者が過去失敗分を次回へ加算しない方針と、STOP後チャットを終了せず原因調査・修正・再実行の継続、本体起動を明示認可。
新SOURCE 697843fbbd2a2aa7224261aa26f8da141ca57687/closure55、library cache v2保持、新run06。純粋21 tests PASS、人工compile/追加transpile0。
carry0（actual/charge/wall）・既承認17GiB/74805/72h、全map8.136939GiB/74784。環境/CPU4workers/driver16/observer18/資源caps/科学compiler不変。
旧証拠・入力・one-shotを保持し、各fresh attemptを独立認可binding。停止した同一runのresumeと旧partial科学cache混合は禁止。
独立review・immutable artifact・直前fresh gatesを経て本計算、一度のSTOPで作業を終了せず必要修正とfresh再起動を続ける。次stage未認可。
[最新入口](../docs/research/track_a_h4_production_run06_20261010.md)。


## 2026-10-10 H4 run06もhost PSI0.18でSTOP、inactive修正案を固定

run06は4workersで約856.585秒、host PSI0.18/nonroot全0/OOM増分0/available約445GiB/role8GiB内でもmemory_pressure STOP。
4actual予約/完了0・signal0、6identity残存0を2回確認。STOPの発生元や正確な終了時刻は補完しない。
元D4 contractのmemory PSI full avg10>0即STOPを変更するには明示契約改定が必要。通常の修正/再実行認可をpressure変更に拡張解釈しない。
SOURCE41d2b61f640937255c5fa6cab9549ae67b83c462/closure58、旧55 source不変、inactive evaluator・16pure tests PASS。
提案はhost-only PSI<1%を完全なnonroot PSI0/OOM0delta/effective120.25GiB/fresh5秒/既存role8GiB guardで条件付き許容。nonroot>0/host>=1%/欠測等はSTOP。
threshold最適性/共有負荷安全性/本計算成功は未検証。production未接続、approved/runtime false・allowed[]・未seal・commandnull。
carry0/17GiB/74805/72h/科学/compiler/凍結入力維持。本体再起動0、GPU/共有環境/venv/他job変更0。
[最新入口](../docs/research/track_a_h4_host_pressure_policy_fix_20261010.md)。


## 2026-10-10 H4 run07：host-only PSI例外の明示承認・本番接続

利用者が「この条件を承認して再実行」と明示承認。host-only PSI<1%は全nonroot0/OOM増分0/effective120.25GiB/fresh5秒でのみ許容、他pressure/欠測/8GiB/5秒等はfail-closed維持。
SOURCE1dbdd1be2133a59ab81c7b81a6074af0880f72a3/closure62、33pure tests PASS、初回失敗ログ保持。actual kernel baselineと原driver OOM/hierarchyをobserver子へ固定transport。
run06原STOP/費用/6×2ABSENT/10filesSHAを保持し、新run07/carry0/CPU4subset[2,4,5,6]/driver16/observer18で再実行。17GiB/74805/72h・科学/compiler/凍結6入力不変。
独立review・immutable artifact・直前fresh/未使用lock PASS後に起動。共有環境/venv/他job/GPU変更0。開始確認後chat終了可、observer監視は継続。
[最新入口](../docs/research/track_a_h4_production_run07_20261010.md)。


## 2026-10-10 H4 worker AS/RSS32GiBを明示承認・SOURCE固定

利用者の「上限を３２Gで修正して」でnewhost4workersを32GiB、driver8GiB/observer256MiB・64MiB/head16GiBを維持。両fresh/startup gateは152.25GiB。
親soft8でhardをchild準備まで保持し、workerは認可/checkout後に32、全ready後driverhard8。observerはtrusted driver PIDとnative登録workerを8/32で区別。
独立review P2 cleanup signal errorでの後続掃除skipを修正、最終53pure/mock PASS。実child/science/transpile/affinity/本体0。
SOURCE b652fff9d2907015d3b8b7b23ce8bf0fe4fce33a/closure65、旧SOURCE変更7/不変55+new3。科学/compiler/凍結入力/旧pressure/carry0/17GiB/74805/72h不変。
SOURCE/profile/input/carry/capsの準備binding、memory承認true・overalllaunchfalse/allowed[]/未seal/commandnull。fresh run/plan/auth/review/freshgatesは次回起動へ。
旧run07 one-shot/partial/cacheは保持し再利用しない。元native診断2gapsと全map完走は未検証。共有環境/venv/他job/GPU変更0。
[修正資料入口](../docs/research/track_a_h4_worker_memory32_fix_20261010.md)。


## 2026-10-10 H4 run08：worker32GiBで明示再実行

利用者が本計算再実行と開始確認後chat終了を明示指示。SOURCE8c917a5af7943969d4abede8b0bb012880efeee0/closure67、新run08/carry0。
worker4 AS/RSS32GiB、driver8GiB、observer256MiB/64MiB、admission152.25GiB、CPU2/4/5/6・driver16・observer18。科学/compiler/旧pressure/凍結6入力/17GiB/74805/72h不変。
最新run07 native6×2ABSENT/元18fileSHA/cost保持、旧partial/control/one-shotを再利用しない。新11metadata/byte tests PASS、旧53campaignと人工compile再実行0。
独立reviewとimmutable artifact・直前SOURCE/profile/input/carry/CPU/memory/pressure/OOM/容量/inode/quota/未使用lock合格後、一度mapを起動。実状態はruntime receiptへ別記録。
共有環境/既存venv/他job/GPU変更0、home内だけ。全map完走は未検証、次stage未認可。
[起動資料入口](../docs/research/track_a_h4_production_run08_20261010.md)。


## 2026-10-10 H4 run08 host1.26% STOP・限定猶予の未承認案

run08は約32分でhostPSI1.26%によりSTOP、全nonroot0/OOM増分なし/available約426GiB/role32GiB内。compile2/signal1、6 owned identities×2ABSENT、旧費用保持・次carry0。
SOURCE736bdc15ccfec6b2a715e1723850a481f0f2ff6b/closure69、旧production67不変、新inactive2files/14pure PASS。毎秒監視は維持しhost1〜5%だけ152.25GiB/nonroot0/OOM0/fresh5秒の下で最大30秒猶予、5%以上等は即STOP案。
既承認1%即STOPの変更なので利用者承認前は本番接続/再起動しない。flagsfalse/allowed[]/未seal/commandnull。科学/compiler/worker4・32GiB/共有環境/venv/GPU変更0。
[限定修正案・検査](../docs/research/track_a_h4_host_pressure_grace_fix_20261010.md)。


## 2026-10-10 H4 run09：host PSI猶予の明示承認・再実行

利用者が「猶予案を承認して再実行」と承認。host1〜5%だけnonroot0/OOM同一/available152.25GiB/fresh5秒の下で30秒猶予、5%以上・30秒継続・nonroot/OOM/欠測は即STOP、毎秒監視継続。
SOURCE251993785ef1dab2a3891bdbb5d079f5d2184f4d/closure72、新16pure回帰PASS、旧14/53/11campaign・人工compile反復0。worker4/32GiB・driver8GiB・observer256/64MiB、CPU2/4/5/6・driver16・observer18不変。
最新run08原17fileSHA/6identity×2ABSENT/cost6・complete2保持、carry0/17GiB/74805/72h/科学/compiler/凍結6入力維持。新run09、旧partial/one-shot/control/output混合なし。
限定差分review・固定artifact・直前freshgates合格後一度map、開始確認後chat終了可。共有/venv/他job/GPU変更0。
[認可・検証・起動資料](../docs/research/track_a_h4_production_run09_20261010.md)。


## 2026-10-10 H4 realistic runtime：Gaussian構造化の未承認候補

現dense256×256/8qgenericQSDで初期q1/q2回路273万/507万operations、全map72h完走は現速度で厳しい。run09を維持した別worktreeで最大28two-mode Givens+8numberphase候補を実装。
SOURCE0cfdca65ed4a19d05e0cd886c9bbfe2bc26bee57/closure74、旧production72不変、新2files。12人工数値PASS、fullFock256/exterior-minors/oldlog-expm/fermionic signs/controlledphase1e-12一致、Qiskitbuild/transpile/NPZ/worker0。
compiledgatecountsは旧denseと変わる新cost系列案、実speedupと全map72hは未保証。adoption未承認/flagsfalse、run09に接続しない。科学/compileroptions/資源・共有環境不変案。
[候補・数値検証・採用範囲](../docs/research/track_a_h4_gaussian_structure_proposal_20261010.md)。


## 2026-10-10 H4 run10：Gaussian新cost系列の明示承認・切替再実行

利用者が「新方式へ切替・再実行」と承認。ownrun09を停止し8予約/4complete/2signal/charge4263948142B・26fileSHA・6identity×2ABSENTを保持、旧partial再利用0。
SOURCE5aa6c3685afb9c3afcfbdc36f80244da0a7fe525/closure77、新7paths接続12tests PASS、旧数学12campaign反復0。structured Gaussian最大28two-mode+8numberphase、vacuum/JW/相対位相保持、新semantics/seed/cost系列へ分離。
科学H4/PF/RTE/compileroptions/input/環境/CPU4/32GiB/pressure1/5/30/17GiB/74805/72h/carry0不変。実Qiskit/compile速度・72h完走は未測定、人工compileは省略して実本計算で確認。
限定差分review・immutable artifact・freshgates全合格後一度run10起動、開始確認後chat終了可、監視継続。共有/venv/他job/GPU変更0。
[承認・新cost系列・起動資料](../docs/research/track_a_h4_production_run10_20261010.md)。


## 2026-10-11 H4 encoding再利用：run10を継続

run10は変更せず、独立worktreeで小行列の値をkeyにcanonical textを再利用する実装を固定。
SOURCE 3beb3870369eeb8ef76b4ead965c886eba7d16e9 / closure80、candidate内cosine/sine・trajectoryで共有しcache最大4MiB/1024 entries。
12限定人工tests PASS、全bytes/digest・同じ回路のgate counts/depth一致。人工encodingのみ0.310→0.135秒（約2.29倍）、production全体の速度は未測定。
科学/compiler/回路構成/資源契約不変、追加transpile・実NPZ読込・新本番起動0。将来の新SOURCE採用ではseed/binding再結合が必要。
[実装・限定検証・差分review・資料入口](../docs/research/track_a_h4_encoding_speedup_20261010.md)。


## 2026-10-11 H4 run10：高速化版への再開処理を準備

SOURCE 11bba8f0669a264c0e1ef1d71b4d9d14bb394a12 / closure85、dataset identity SOURCE5aa6c368を保持し新実行SOURCEを別記録。
COMPLETEのみledger/bytes/nativeで照合し、完了wrapperのbuild/compileと検証済みsignalの再計算を省略。未完了/孤立は再利用せずcarry保持。
18限定mock PASS、独立差分review PASS、3件のbinding/root検証漏れ修正。実worker/transpile/科学array/新launch0。
run10は継続し変更0。最終停止proof/carry/receiptは未確定、allowed[]/approvedfalse/runtimefalse/未seal/commandnull。
[資料入口](../docs/research/track_a_h4_resume_preparation_20261011.md)。


## 2026-10-11 H4 run10→resume01：完了分を保持した高速化継続

利用者「引き継いで」でown run10停止・native6×2残存0を確認。SOURCE 941749b8d4aedb43fd5f47b824116082493df393 / closure85、dataset seed SOURCE5aa保持。
COMPLETE8150/signal142、carry7878/4320481456B/35474.12562095933s、未完了2を保持。worst74512≤74805、chargebound約12.16GiB≤17GiB。
既知sealed停止ログ名の限定修正と19mock PASS。科学/compiler/CPU4/32GiB/observer/pressure/17GiB/74805/72h不変。
最終binding独立差分review・artifact固定・fresh gate後、一度mapを再開する。実状態はprivate receiptへ記録。
[資料入口](../docs/research/track_a_h4_production_resume01_20261011.md)。


## 2026-10-11 H4 inverse・命令/定義encoding再利用：resume01を継続

SOURCE def7d48f4af6c07f2d4e361fae868f3b01db8915 / closure89、独立worktreeでgeometry-owned基底inverseを一度保持し、小命令/定義のcanonical fragmentを値keyで再利用。
cache2MiB/2048 entries、key16KiB/384 nodes、definition16命令まで。large/lazy fallback・mutation/phase/control/ordering/nonfinite STOPを保持。
14限定人工tests PASS、旧/new bytes/digest・uncompiled gate counts/depth一致。人工inverse9→0回、encoding0.446→0.320秒（約1.39倍）、production速度未測定。
簡単なauthor差分review PASS、独立launch review/campaign反復0。sampling/seed/signal関数とcompiler/worker/observer/資源契約は不変。
稼働resume01のsource85件SHA一致、runを維持し反映0。新採用bindingはfalse/allowed[]/未seal/commandnull、将来のSOURCE/seed/carry等再結合が必要。
追加transpile・実NPZ読込・新本体起動0。
[実装・限定検査・軽量review・資料入口](../docs/research/track_a_h4_inverse_instruction_speedup_20261011.md)。


## 2026-10-11 H4 resume01→resume02 対応：現在runを継続

SOURCE 05a37cebaf18fa28e3a1e8582b86f54e32ba2a66 / closure92。run10由来importとresume01新規COMPLETEを区別してbyte/source/digest/seed/metricsを照合し、次runで完了済みbuild/compileを省略する。
actual=祖先carry+当run予約、ID carry+1、journal sumは過去chargeを含むため二重加算しない。未完了予約・orphan費用と累積72hを保持。
24限定mock PASS、receipt CLI/cache-owner順序/閉じた認可scopeも確認。CLI test環境のPython2変数不足を補正し全試行証跡を保持。
軽量author差分review PASS、科学/高速化/compiler/worker/observer/資源上限は不変。実worker/NPZ/追加transpile/新launch0。
親の6 owned identityをread-only確認しclock630bytesを固定。現在resume01を止めずsource85件一致、停止proof/最終carry/receiptは停止後に確定する。
準備bindingはfalse/allowed[]/未seal/commandnull。将来の明示引継ぎ指示後にreceipt/binding/fresh gateを完成して一度移行する。
[実装・限定検査・binding・将来の手順・資料入口](../docs/research/track_a_h4_resume02_preparation_20261011.md)。


## 2026-10-11 H4 resume02：高速化へ引継ぎ・72h STOP解除

SOURCE d0726e946f89f8c30a08d383fb8ff48b363629ec / closure95。利用者の明示引継ぎ・可能なら時間上限解除指示を根拠に、own resume01 driverのみpidfd SIGINT、native6×2残存0を確認。
全80,157files/2世代ledger/bytes SHAを照合しCOMPLETE29388/signal521、carry28731/8788058342B/104895.53687566705sを保持。旧未完了費用返却0。
新wall capnullをdriver/observer両方へ接続し、累積wall記録・trace2GiB/総17GiB/actual74805/5秒/ASRSS/pressureOOM/own-stopを保持。候補環境/compiler/科学/seed/CPU4・32GiBは不変。
12新mock PASS、旧24/人工compile/campaign反復0。独立最終binding・private launcher差分PASS、保守的actual74127、chargebound17573199062Bで承認内。
SOURCE無変更のartifact固定後、直前freshCPU/memory/PSI/OOM/FS/inode/quota/未使用roots/one-shotを合格して一度mapへ。実開始状態はprivate receiptへ別記録。
[資料入口・承認・独立review・起動条件](../docs/research/track_a_h4_production_resume02_20261011.md)。

## 2026-10-13 H4 geometry map 全件完了・軽量結果公開

H4 linear/STO-3G/DF12、T=0.8、二次DF-prefix PF/finite RTE、6距離の1,308候補・74,784 wrappersがMAP_COMPLETE_STOPで完了。
全record/signalの保存identity・paired metrics・coverageを照合。公開projectionだけから302精度点×6距離の点最小と18比較行を追跡できる。
ε=0.05/P=0ではB2 L_D=3が全距離の点推定最小、最小deterministic候補比27.9〜42.7%低い。全302点でもB2点最小。
近接B2の±2SEは重なり厳密winner未認定、energy/RPE/H6/noise/state prepへ一般化しない。旧dense/compiler系列と混合しない。
累積wall74.53h、compile消費/予約73023/74805、charge12.68/17GiB。own6役退出、追加science0・次stage未認可。
[結果・projection照合script](../docs/research/track_a_h4_geometry_results_20261013.md)。本体library/source/runnerの変更はなく、公開saved JSONのstdlib集計だけ。

## 2026-10-13 H4 B0+I保存値診断・科学判断待ちSTOP

優先Bでtail scalarのsource符号/UPLO=L規約と6入力SHA、120保存complex signalを照合し、13人工算術PASS。
追加compiled RZ/shotをゼロと置いた参考モデルで397/1812点が登録B2点最小を下回り得る。ε=.05では0.70/0.80/1.40 Å。
B0+Iの実compile費用は未測定。元登録集合の結果は不変で、主要claim変更の可能性という利用者停止条件に従いGPT判断へ戻す。
原稿本体・A/C/Dの全面統合は未実施、新eigh/SCF/DF/state/量子signal/trajectory/build/compile0、H8/共有環境/他job変更0。
[限定scalar解析と公開値照合の入口](../docs/research/track_a_h4_geometry_scalar_diagnostic_20261013.md)。本体science source/runnerは不変。

## H4 geometry保存値の技術統合（2026-10-13）

実行science runnerを追加せず、[bundle内解析・図・検証script](../artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13/README.md)を収録。
stdlib保存スカラー解析、Matplotlib表示、Fraction照合、bounded event metadata読取。新trajectory/compileなし。

## H4 B0+I preparation-only

[新資料](../docs/research/track_a_h4_b0i_compile_preparation_20261013.md)。
`prepare_h4_b0i_pilot6.py` はbyte/profile/hostだけの未承認草案、`run_h4_b0i_preparation_tests.py` は44限定人工/mock検査。
`run_h4_b0i_pilot6.py` は新source-bound grant/reviewを要求するfuture runnerで、現在のdraftを実行しない。
科学SOURCE・既存B0・H8を変更せず、実/人工transpile0。

## H4 B0+I原稿v0.4の保存値・図・照合

科学library/sourceを変更せず、[保存値統合](resource_applicability/integrate_h4_b0i_manuscript.py)、[図生成](resource_applicability/plot_h4_b0i_manuscript.py)、[bundle読取照合](resource_applicability/verify_h4_b0i_manuscript_bundle.py)を分離した。入力は公開CSV/JSONのみ、新signal/trajectory/compile0。限定testは`tests/tracks/resource_applicability/test_h4_b0i_manuscript_integration.py`、条件と再生成は`docs/research/track_a_h4_b0i_manuscript_integration_20261014.md`から参照。

## Track Aサーバー研究情報の暫定共有索引

[共通入口](../docs/research/track_a_shared_research_status.md)と`docs/research/track_a_server_inventory.md`から固定result/source/レビューへ辿れる。`scripts/resource_applicability/verify_track_a_shared_status.py`はGit blob/公開bytes・JSON単位・内部リンクの読取照合のみ。科学library/import/計算、H8操作を行わない。ローカル照合待ち。


## 2026-10-15 Track A：H8 D0 driver／H6 native B0+I bridge

[H8 D0 driver](h8_d0_saved_execution_v1.py)、[逐次launcher](launch_h8_d0_saved_once_v1.sh)、[共通guard](final_pre_review_execution_v1.py)、[H6 bridge](h6_native_b0i_bridge_v1.py)を独立追加。未認可ではinput loader前に拒否し、旧944precision/3576map完全一致後だけ追加解析へ進む。H6はscalar/control phaseを一度だけ元builderへ渡す。[技術資料](../docs/research/track_a_final_pre_review_implementation_20261015.md)。人工41PASS、実H8再会計/実H6 build/compile0、原scientific module/runner変更0。


## 2026-10-15 H6 独立資源観測の修正

H6 pilot run02はnative正しさ検証中のcallback stalenessでSTOP、compile0。既存metadata launcherの独立資源観測へ修正し、SOURCEb73580dd…/closure13、限定84testsとGIL保持C約12秒の2process実動作検査PASS。追加本番role/process0、5秒・資源・科学/compiler条件維持、旧消費/unknown保持、subagent0。新run03へ結合して再実行。
[修正・検証・新実行binding](../docs/research/track_a_h6_external_monitor_fix_20261015.md)。


## 2026-10-15 H6 /proc退出競合修正・run04-v3起動

H6 run03は全native検証後の退出/proc競合でSTOP、1 worker PASS/受理0・全6 reap。SOURCE2e733d8a…でlive欠測を成功扱いせず5秒内の再観測/reapへ修正、92限定＋最小2childの退出/FD検査PASS。旧消費保持。CPU gateで未起動2grantを廃止、5秒idle選定・fresh gate後にrun04-v3起動。科学/profile/compiler/資源上限不変、subagent0。完了は未主張。
[修正・検証・新実行の入口](../docs/research/track_a_h6_proc_exit_fix_20261015.md)。

## H6 fixed-delta T panel

`prepare_h6_fixed_delta_v1.py` はbyte/metadataだけ、`h6_fixed_delta_runner_v1.py` は新grant必須。`h6_fixed_delta_plan_v1.py` / `h6_fixed_delta_science_v1.py` が固定15セル・lazy科学adapter。[準備資料](../docs/research/track_a_h6_fixed_delta_t_dependence_readiness_20261015.md)。

## H6 fixed-delta saved-result audit

`summarize_h6_fixed_delta_v1.py --verify <public-bundle>` は公開scalarのみからshot/paired6metricsを再計算する。`--run/--output` はprivate完了証拠からexclusiveな軽量bundleを作る。`plot_h6_fixed_delta_result_v1.py` は保存値のみでSVGを作成。[結果](../docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md)。科学計算・compile0。

`track_a_v1_p7_measured_reaccount.py`：stdlib限定、default plan-only、新grantでp7実測×保存shotと保存PRの1比較だけ。参照は上記結果報告。

H8 selected-six handover: `h8_key_compile_handover.py` binds old validated QPY and completed-prefix results after old native cleanup; the new monitored driver waits for an exclusive atomic release. See [running handover record](../docs/research/track_a_h8_key_compile_parallel_handover_20261017.md).


## H8 selected-six saved-cost reaccounting (2026-10-17)

`h8_six_axis_reaccount_contract.py` / `run_h8_six_axis_reaccount.py` / `witness_h8_six_axis_reaccount.py`：H8の保存shot・PR費用/SEを固定し、6軸単価のみ反映する新one-shot。SOURCE公開・新grant必須。`--check`はmetadata/保存値の結合検査のみ。

## H4/H6 new_2追加検証

`h46_new2_native.py`は明示XY Gaussian loweringと独立occupation oracle。`run_h46_new2_native.py`/`launch_h46_new2_native.py`は群別検証、`run_h46_new2_native_shard.py`/`launch_h46_new2_native_shards.py`は同じ科学kernel/seedのH6軌道並列。`h46_new2_coefficient_audit.py`は70桁toyの係数丸め診断。実行済み結果は[報告](../docs/research/track_a_h46_new2_native_validation_20261018.md)。新しい実行は独立output/明示scopeで扱い、完了済みrun/旧grantを再利用しない。

## H4/H6 高次PF比較の準備（2026-10-18、未承認）

`audit_h46_higher_pf_readiness.py` は固定証拠metadataのみを監査する。`h46_higher_pf_adapter.py` と `run_h46_higher_pf_comparison.py` は保存値のq比較接続。既定はscope表示、公開無効grantは実値読込前に拒否する。新sampling/compile経路なし。[契約・限界](../docs/research/track_a_h46_higher_pf_readiness_20261018.md)。

## 2026-10-18 高次PF native取得

`h46_acquire_native.py` / `launch_h46_acquire.py` は固定12群96trajectoryのnative取得。`h46_acquire_integration.py` / `run_h46_acquire_comparison.py` は別承認用の統合比較binding/CLI。[結果・契約](../docs/research/track_a_h46_higher_pf_acquire_run01_results_20261018.md)。

## H4/H6 higher-PF saved-data reader (2026-10-20)

[h46_higher_pf_dataset.py](h46_higher_pf_dataset.py) reads SHA-bound Git JSON/CSV and validates exact metadata joins; it does not import scientific modules or dispatch science. See [dataset](../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/README.md), [protocol](../docs/research/track_a_h46_higher_pf_supplementary_protocol_v1.md), and [31 metadata tests](../tests/test_h46_higher_pf_dataset.py).

## H4/H6 outer32 design pilot (2026-10-20)

[h46_pr_design_pilot.py](h46_pr_design_pilot.py) proposes n_P and separates DESIGN_PILOT seeds; statistics are tested on artificial values only. [Contract](h46_pr_pilot_contract.py), [native adapter](h46_pr_pilot_native.py), [launcher](run_h46_pr_design_pilot.py), [disabled preparation](prepare_h46_pr_design_pilot.py), [report](../docs/research/track_a_h46_pr_design_pilot_readiness_20261020.md), [tests](../tests/test_h46_pr_design_pilot.py). No main exploration or confirmation dispatch.

## H4/H6 pilot lossless storage prototype (2026-10-20)

[h46_pilot_lossless_storage.py](h46_pilot_lossless_storage.py) copies opaque bytes into lossless verified archives and restores into new files. A forward-only serializer sink is tested with artificial QPY; no existing science runner uses it. See [report](../docs/research/track_a_h46_pr_pilot_storage_optimization_20261020.md), [bundle](../artifacts/resource_applicability/track_a_h46_pr_pilot_storage_optimization/2026-10-20/README.md), [tests](../tests/test_h46_pilot_lossless_storage.py). No source deletion or scientific dispatch.

## H2/H4/H6 higher-PF confirmation readiness (2026-10-21)

[Design arithmetic](h246_confirmation_design.py), [contract and future activation](h246_confirmation_contract.py), [native runtime reuse](h246_confirmation_runtime.py), [disabled-pack preparer](prepare_h246_confirmation.py), [stage CLI](run_h246_confirmation.py), [fresh joint controller](launch_h246_confirmation.py), [fixed paired evaluation](h246_confirmation_evaluation.py). [Report](../docs/research/track_a_h246_higher_pf_confirmation_readiness_20261021.md), [artifact index](../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_readiness/2026-10-21/README.md), [design tests](../tests/test_h246_confirmation_design.py), [artificial execution tests](../tests/test_h246_confirmation_readiness.py). Exact saved shots and candidates; independent CONFIRMATION cohorts; no real scientific execution before separate approval.
