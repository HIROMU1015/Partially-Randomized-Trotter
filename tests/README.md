## 2026-10-24 H6保存済み4方式の回収・再計算省略

[保存値・出典・保全・任意rank契約](../docs/research/track_a_h6_saved_four_method_20261024.md)。H6の4方式すべてのC/N/G・PR paired SEを固定保存資料から回収し、利用者指示に従い再計算を省いた。H6/1 Å/STO-3G/tol-only DF1e-8/rank19、Full prefix19 q8/q4、PR prefix5 q8/q4各n33。H2/H4/H6の12行CSVは保存fieldの転記のみ。人工10件＋封印runtime人工3 event PASS、971 file seal・既知1,305 file変更0。旧Full中断stageのparent exit/reap不明を保持。任意の同時順位ranking1だけを未承認契約で準備し、実H6科学call/新規grant0、旧strict COMPARE BLOCKED、mandatory STOP。

人工検査：`test_h6_saved_four_method.py`。実H6数値処理を使用しない。

## 2026-10-24 H4高次PR/PF 4方式の探索的会計 run01完了

[結果・SOURCE・保全・STOP](../docs/research/track_a_h4_four_method_exploratory_run01_20261024.md)。承認済みH4固定4候補の保存値会計を一回正常終了。4/4適格、Full/PRともYoshida4が低費用、次数の方向逆転なし、4点最小はPR Yoshida4。exact39 event・約1.18秒・失敗/再試行0、compile/sampling/simulation0。H2完了出力と並列表示するCSVを保存し、混合cohort・追加数値解析・詳細科学レビューなし。既知1141 file変更0、旧strict COMPARE BLOCKED、mandatory STOP。

## 2026-10-24 H4高次PR/PF 4方式の探索的会計準備

[契約・SOURCE・承認ゲート](../docs/research/track_a_h4_four_method_exploratory_ready_20261024.md)。固定H4 DF10のFull prefix10 S2/q8・Y4/q4とPR prefix3 S2/q8(n48)・Y4/q4(n33)を準備。保存166軸・signal/shot/allowanceを再利用し、追加取得不要。SOURCE/input/runtime1065 file、人工11件＋封印runtime人工39 event PASS。実H4新規会計・順位/compile/sampling/simulation/grant0。保存reference末尾差・条件付きempirical/有限標本を保持し、別途科学実行承認待ち、mandatory STOP。

## 2026-10-24 H2探索的4方式の保存総費用比較 run01完了

[結果・SOURCE・保全・STOP](../docs/research/track_a_h2_archived_exploratory_run01_20261024.md)。ユーザー承認後、固定H2 4候補の保存費用・保存DIAGNOSTICSから探索的会計を一回完了。4/4適格、FullはS2・PRはY4が低費用で次数順位逆転候補、4点の最小はFull S2。exact43 event、約1.206秒、exit0/回収、compile/sampling/simulation0、中断/再計算0。有限標本・empirical allowance・A/B未解決を保持し、旧strict COMPARE BLOCKED、mandatory STOP。

## 2026-10-24 H2保存費用の探索的4方式比較：実行準備

[固定契約・承認ゲート・STOP](../docs/research/track_a_h2_archived_exploratory_ready_20261024.md)。保存PR134軸・共通Full4軸と受入済みDIAGNOSTICSを用いる探索的会計を別SOURCE/契約で準備。元allowance/account/rankとpaired費用統計を再利用し、外側15＋内部28＝43 ledger eventを固定。科学call0、SOURCE/input/runtime996 fileを封印、人工108PASS＋封印runtime人工smoke PASS。科学実行の別途明示承認待ち、旧strict COMPARE BLOCKED、grant発行0、mandatory STOP。

## 2026-10-23 H2保存PR/PF費用の静的結合 B-2

[証拠範囲・未解決・STOP](../docs/research/track_a_h2_saved_cost_static_join_b2_20261023.md)。保存PR134軸・共通Full4軸を固定証拠へ結合。Python1/1・主要runtime8/8・binary21/21、provider relative path235/236一致。Full原本byteのGPU01確認とGPU02の直接確認を区別。Full全parameter stream/provider非干渉/過去load/A独立角度は未解決。人工133PASS、16,804保護file変化0。政策2a・代替契約未採択、COMPARE BLOCKED、新科学/compile/uNG/転送/push0、ローカルcommit後mandatory STOP。

## 2026-10-23 H2共通Full PF原本の限定回収

[REPORT・SOURCE・SHA・handoff・保全・STOP](../docs/research/track_a_h2_common_pf_original_recovery_20261023.md)。GPU01で対象2候補4軸の原plan・実行入力・原result・QPY/archiveのbyteを固定結果と照合。12 archive/8静的QPY parse、宣言closure365 entry一致、固定GitHub13資料全文一致、保全2,141 file変化0。保存1655/2463 RZの別cost objectを保持。完全過去実loadと現在共通native全streamのbinding等は未認定。人工8PASS。科学call/compile/simulation/COMPARE/正式採択0、原本転送0、公開push0、local commit後mandatory STOP。

人工境界検査：[test_h2_common_pf_original_recovery.py](test_h2_common_pf_original_recovery.py)。gzip破損/連結/size/SHA、候補重複、private path、科学import不在を確認。

## 2026-10-16 V2保存RZ残差：人工検証完了・数値実行未承認

[新規36人工tests](test_track_a_v2_residual.py)＋既存observer/cleanup限定7件＝43 PASS。[receipt](../artifacts/resource_applicability/track_a_v2_residual_implementation/2026-10-16/artificial_validation.json)・[報告](../docs/research/track_a_v2_residual_implementation_20261016.md)。262 scope entriesは全件来歴未解決でfit対象0。実データfit/残差/compile0、未承認draft・公開後mandatory STOP。primary agent自身のlocal checksであり独立レビューではない。以下は各時点の履歴。

# tests の役割

## 2026-10-16 Native DFモデル：Qiskit 1.3.0接続検証

[実Qiskit接続15件](test_native_df_qiskit_connection.py)＋既存回帰27件＝42 PASS。[最終receipt](../artifacts/resource_applicability/track_a_native_df_qiskit_validation/2026-10-16/connection_validation_v1.json)。fallback27件は別環境の部分判定であり加算しない。
実分子評価・保存H6/H8再解析・sampling・transpile・server接続・新grant・原稿改訂0。公開後mandatory STOP。

## 2026-10-16 Native DF理論ゲート数モデル（人工検証のみ）

[27人工tests](test_native_df_gate_model.py)、[専用runner](../scripts/check_native_df_gate_model_synthetic.py)、[receipt](../artifacts/resource_applicability/track_a_native_df_gate_model/2026-10-16/synthetic_validation_v1.json)。解析unitary oracleとDP/全列挙を比較。local technical checksでありCI・独立科学レビューではない。
新しい実H6/H8数値評価・sampling・compile・server操作・grant発行は0。既存結果・科学SOURCEは保持し、公開後mandatory STOP。

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

## 2026-10-23 H2 runtime差：保存3.12費用の再利用調査

[完了報告・比較政策・未解決条件](../docs/research/track_a_h2_runtime_cost_reuse_20261023.md)。元PR134軸と同じPython3.12 binaryの共通Full4軸を発見（S2/q4各1655、Y4/q2各2463 RZ）。主要compiler binary21/21、runtime8/8、Full preparation hashも現在の共通診断と一致。原PF QPY/plan・関連runtimeの静的結合を先行し、追加compileの必要性は未認定。人工82PASS・16,824保護file変化0。代替契約未採択、旧runtime出力同等性/A独立diagonal未証明、COMPARE BLOCKED。新科学計算/u/N/G/grant0、local commit後mandatory STOP。
