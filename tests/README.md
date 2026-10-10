# tests の役割

## 2026-10-10 Track A：H4限定一回実行・coverage interface STOP

最新入口は[H4限定実行報告](../docs/research/track_a_ax2b_h4_limited_execution_v1.md)。固定source/manifestの一回実行はACTUAL_COVERAGE_CHANGEDでSTOP。
correctness0/8、reference/primitive/control/sampling/compile counter0。input/native準備は制御フローから推論。
保存schedule list対runtime tupleの静的interface差を確認。actual bounds全体は未保存。
source/manifest/旧証拠を編集せず、raw STOPと欠測・保存監査を別inventoryへ公開する。
grantは消費済み、retry/resume0。次の修正・新manifest/科学実行は今回未実施。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定、mandatory STOP。
既存dirty/未追跡・Track Bを保全。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4限定science manifest固定・seal

最新入口は[H4限定seal報告](../docs/research/track_a_ax2b_h4_limited_seal_v1.md)。保存再監査済みboundsと旧8 cell・capsを固定する。
science source/input/env・CPU3/worker1/BLAS1・専用future outputをmetadataとして結合。
35合成tests pass。新科学計算/array decode/native準備/probe/sampling/build/compile0。
execution_plan_sealed=trueは条件固定だけ。science_authorized=false / launch_allowed=false。
H4_LIMITED_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。
旧source/freezes/STOP/結果・dirty差分を保持。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4-P保存read gate v2・再監査

最新入口は[H4-P保存再監査](../docs/research/track_a_ax2b_h4_native_receipt_reaudit_v2.md)。新gate/runner・36合成testsを追加。
原16MiB aggregate budget内で4MiB超JSONを読める保存専用経路。凍結v1は変更しない。
元source/STOP/receiptのbytesを保持し、実行時sourceと新audit sourceを別に固定する。
新分子計算/native準備/signal/probe/sampling/wrapper build/compile0。H4-P再実行なし。
science manifestのseal/launchなし。`H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED`。
`DRAFT_NOT_AUTHORIZATION`、mandatory STOP。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4-P一回実行・親監査STOP

最新入口は[H4-P実行報告](../docs/research/track_a_ax2b_h4_native_receipt_execution_v1.md)。保存H4 load1/native準備8とreceipt保存を実施。
親はB3 JSON4,443,419 bytesを4MiB読込gateで拒否しSTOP。総output9,821,513 bytesは16MiB以内。
保存JSONのstdlib補助監査は一致。原STOP・source・結果のbytesを維持する。
signal/probe/sampling/wrapper build/compile0。retry/resume/source修正/science sealなし。
`H4_NATIVE_RECEIPT_STOP` / `H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED`。
`DRAFT_NOT_AUTHORIZATION`、mandatory STOP。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4-P runner・実行前固定 v1

最新入口は[H4-P準備契約](../docs/research/track_a_ax2b_h4_native_receipt_preparation_v1.md)。専用source・runner・48合成testsを追加した。
CPU3・900秒・AS8GiB・output16MiB・load1/prepare8を未来の計画へ指定する。
今回の実分子load/native準備/signal/sampling/wrapper build/compileは0。
H4-P取得planのsealは認可ではなく、H4 science manifestは未sealのまま。
`H4_NATIVE_RECEIPT_NOT_AUTHORIZED` / `H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED`。
`DRAFT_NOT_AUTHORIZATION`、mandatory STOP。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4実行前契約・metadata固定 v3

最新入口は[H4契約・metadata preflight](../docs/research/track_a_ax2b_h4_prelaunch_contract_v3.md)。
保存入力/source/旧8 cellを照合し、179 primitive-time組/537 probesを固定した。
専用metadata tests12 passed。新科学計算/array load/sampling/circuit/compile0。
native instruction receipt、CPU実割当、別grantは未固定。manifestは未sealを維持。
`H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
以下は各stage当時の履歴。旧結果・sourceと既存dirty差分を保全する。

## 2026-10-10 Track A：H4/H6 backend接続準備 v2

最新入口は[接続・実行gate・合成検証の報告](../docs/research/track_a_ax2b_bound_ports_preparation_v2.md)。
H4独立MP/stage/event port、専用H6 sector/native backendと別grant必須launcherを追加した。
新39＋前回49の88 local synthetic/mock tests pass。分子の正しさ・総u・CI証拠ではない。
source固定のみ。actual input/coverage、CPU/別認可は未seal。新科学計算/sampling/circuit/compile0。
`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。旧証拠・既存dirty差分を保全。
以下は各stage当時の履歴。

## 2026-10-10 Track A：独立レビュー後の準備

[GPT独立レビュー](../docs/research/track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md)を受け、[準備追補](../docs/research/track_a_ax2b_post_independent_review_amendment_v1.md)と[H6準備契約 v2](../docs/research/track_a_ax2b_h6_pilot_preparation_contract_v2.md)を追加。
H4-N/A/E/Mの限定計画、独立small reference・u-aware会計・tol-only adapter、別H6 controller/caps/watchdogを準備した。
専用49 local synthetic tests pass。分子H4/H6検証の新結果・総u認定ではない。
H6 molecular backend/science launcher、H4全stage検証port、input/CPU/別認可は未完了。
旧source/results/freeze/manifestと既存dirty差分を保全。`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
以下は各stage当時の履歴。

原稿用の[表示tests](tracks/resource_applicability/test_manuscript_figures.py)（12件）と
[bundle helper tests](tracks/resource_applicability/test_manuscript_bundle.py)（9件）は合計21 passed、fail/skip0。
合成値・一時Markdown/CSV・sourceだけでmissing、identity、固定集合、affine描画、unsafe link拒否を検査する。
元artifactの表示値照合は別の[原稿audit](../artifacts/resource_applicability/track_a_manuscript_audit/2026-10-05/verification.json)。
local testsであり科学run再実行・immutable CI・独立投稿可レビューではない。full suiteは実行しない。

PM-2保存値解析実行のpre/postでsynthetic-only 62 testsがそれぞれpassed、fail/skip0。
[実データの保存値照合](../docs/pr2_pm2_precision_resource_result_validation.md)は別監査で、
test件数へ水増しせず、immutable CIや新しいscience evidenceとはしない。

PM-2解析の`tracks/resource_applicability/test_pm2_precision_analysis.py`はsynthetic-only 62 tests。
`scripts/resource_applicability/run_pr2_pm2_implementation_tests.py`のguardはreal saved evidenceのaccessも禁止する。
起動gate・ε境界・軸別cost・paired covariance・ties・CSV/schema・STOPを検査し、実データgate PASSとは区別する。
[実装監査](../docs/research/pr2_pm2_precision_analysis_implementation.md)を参照する。full suite・旧science testsは実行しない。

PM-2は`tracks/resource_applicability/test_pm2_precision_contract.py`のcontract/schema・保存JSON inventory tests。
`scripts/resource_applicability/run_pr2_pm2_preparation_tests.py`で分子データ/runtime/registryのguardをpytest import前に入れる。
precision sweep、新signal/trajectory/circuit/compile、旧科学tests、full repository suiteは呼ばない。
[契約](../docs/research/pr2_pm2_precision_resource_contract_v1.md)にscopeとSTOPを固定する。

PM-1 authorization draftの再検査も201 passed、fail/skip0。
[最終review監査](../artifacts/resource_applicability/pr2_pm1_discard_authorization/2026-10-04/authorization_review_audit_v1.json)には
actual draftの最終boolean=false拒否とconditional mock positive・9改変拒否を区別して保存した。
追加監査をunit test件数へ水増しせず、production gate PASS・科学結果とは呼ばない。

Track A PM-0は`tracks/resource_applicability/test_pm0_evidence_attribution.py`の18 local tests。
synthetic bookkeeping・入力allowlist・改変拒否と保存JSON regressionだけを実行する。
分子NPZ/runtime/cache、旧science/validation runner、compilerを呼ばず、immutable CIとはしない。

PM-1は`tracks/resource_applicability/test_pm1_discard.py`の49 local tests。
固定8候補、authorization-before-data gateのmock、上限・partial/null ledger・no retry、
旧state-actionとのsynthetic q回帰、tiny2-qubit wrapper compileと保存JSONのみを検査する。
`scripts/resource_applicability/run_pr2_pm1_preparation_tests.py`でPM-0 18＋helper134と合わせて
201 passed、fail/skip0。NPZ/NPY/pickle/runtimeのopen/stat/lstatをimport前に拒否し、禁止アクセス試行0。
guardは診断でありOS sandboxではない。H4 science、PM-1結果、immutable CIは0。
[契約と監査資料](../docs/research/pr2_pm1_nearby_discard_contract_v1.md)へ対応する。

`tests/` は、ライブラリAPI、数値恒等式、成果物schema、ガード条件の回帰検査を置く。
基本的に `src/trotterlib/<name>.py`、`scripts/run_<name>.py`、
`tests/test_<name>.py`、`docs/<name>.md`、`artifacts/<name>/`を一組として読む。

テスト通過は、実装が期待した規約を満たすことを示す。一方で、次を単独では意味しない。

- 大きな系でも同じ近似精度になること
- 実量子backendやnoise下で成立すること
- 未検証のパラメータ範囲への外挿が正しいこと
- 最終的な総コストや手法間の優位性が確定したこと

研究上の証拠statusは[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)と
[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)を参照する。

最新のRPE長round検証は`test_rpe_target_round_horizon_validation.py`と
`test_rpe_delta_round_schedule_validation.py`、`test_rpe_delta_compiled_cost_validation.py`を、
対応する検証文書・artifactと一組で読む。最後の検証は中央RTEブロックだけを扱い、
Hadamard 1 shot全体や最終総コストの検証ではない。

P-A v1 blind transferは`test_research_direction_joint_synthesis_blind_validation.py`で、固定source/artifact
hash、事前登録gate、checkpointと最終artifactのtamper検出を確認する。H5 physical transferと
H4 optimization-level-2 compiler transferは両方とも完了している。
`test_research_direction_joint_synthesis_formalization.py`では、DPと全列挙の一致、syntheticな非退化分割、
完成済み54 recordの一区間退化・order 0 coverage、artifact tamperとscope guardを検査する。
`test_research_direction_joint_synthesis_mechanism_validation.py`では、計算前expected-task manifest、
全30 taskのorder-2 coverage、明示的一区間baselineとの0 split・0 plan差・0 RZ改善、固定停止判断、
artifact tamperとscope guardを検査する。
`test_research_direction_geometry_tracking_breakdown.py`では、16 taskのcompile-before manifestとsource hash、
訂正済み先行P-Cとのtraining再現、追跡prefix不変、stretch側blind予測破れ、固定停止判断、
artifact tamperとscope guardを検査する。
`test_research_direction_pd_fair_comparison.py`では、共通時間のexpected manifest、source freeze、finite K2/K4 cost、配分不能点のinfeasible記録、S1後の強制停止とscope guardを検査する。
`test_research_direction_pd_s1_posthoc.py`では、固定S1 fingerprint、一次Case Bの保存、主baselineの事後解釈、B1aのm_D診断、構成内訳、artifact改変拒否を検査する。

`test_pr2_s0_s1_validation.py`では、generation-prefix adapter、snapshot改ざん検出、corrected finite-RTE
mean、$\mathcal B^2$ shot式、q=1/8 controlled wrapperのRe/Im規約、stage gate、非上書きを検査する。

`test_pr2_new_series_validation.py`では、固定amendment/input hash、development二回load、held-out非開封、
raw/internal tamper、rank 3/6/9のpartition・sampling sign・確率和・再構成、counter、fingerprint、
非上書き、V4非承認guardを検査する。

`test_pr2_v4_s2_development_validation.py`ではV4/S2のnormalization、full-wrapper compile、
32/96 pooling、resource decision、mandatory STOPを検査する。
`test_pr2_v4_s2_parallel_execution.py`では、同じcompile cellのserial/parallel完全一致、canonical
result order、persistent SQLite cache再利用、worker上限、cell identity付き例外、atomicで非上書きの
failure reportをtoy Hamiltonianで検査する。実H4 S2の並列再実行結果や速度倍率を示すものではない。

`test_pr2_matched_accuracy_m1_contract.py`では208候補のexact countとfingerprint uniqueness、
`q*delta=T`、occurrence seed独立性、最大4件のr64境界、16-cell selectorと`SELECTION_LIMITED`経路、
source hash、schema、全科学counter 0、非上書きrunnerを検査する。M1 signalまたはcompileのtestではない。

`test_pr2_matched_accuracy_m1_precompile_barrier.py`では、selector理由と未解決集合の整合性、limited時の
compile-plan生成拒否、clear時の最大16+16 cell plan、source hash、non-overwrite dry-run、全科学counter 0を
検査する。M1-A signalまたはM1-B compileを実行するtestではない。

`test_pr2_matched_accuracy_m2_transfer_contract.py`では、M1-B1からの5構成固定、future seed衝突0、
196-wrapper上限、primary予測と10%重大underestimate、4 terminal status、held-out/transfer未認可、schemaを
検査する。held-out signalまたはcompileを実行するtestではない。
