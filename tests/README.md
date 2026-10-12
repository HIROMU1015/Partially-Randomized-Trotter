# tests の役割

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

## Track B G1 source準備の限定tests

[test_g1_source_preparation.py](tracks/algorithm_codesign/test_g1_source_preparation.py)は59件のlocal
off-domain testsで、generic algebra、mock proof分類、source gate、marker、順序、cap、初回STOP、
旧guardによる模擬process、grandchild回収を確認する。本構造監査の呼出しは禁止patchで0と確認し、
固定8入力のsolve・実backend呼出しは0。
[source review](../docs/tracks/algorithm_codesign/g1_source_review_20261009.md)と
[保存結果](../artifacts/track_b_g1_source_preparation/2026-10-09/focused_test_results_v1.json)を対応させる。


## Track B G3 focused tests（2026-10-09）

`tracks/algorithm_codesign/test_g3_finite_law.py`は17 PASS、
`tracks/algorithm_codesign/test_g3_return_comparator.py`は10 PASS。
synthetic/off-domain mean、finite reweight/confidence、zero-cost、controlled phaseとsource reuseだけを検査。
testsでregistered profile採点やsynthesisを行わず、full suite未実行。
[結果](../docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md)。

## DF sparse correction audit focused controls

`test_df_sparse_correction_audit.py`：15 PASS。可換quotient、pure-D、tiny η、exact階層確率、identity、controlled phase、非Gaussian basis、独立drawのnegative control。
[監査](../docs/tracks/algorithm_codesign/df_sparse_correction_independent_audit_20261011.md)。新分子入力/本番run/full suiteなし、mandatory STOP。


## Track B DF correction comparison v2

`tracks/algorithm_codesign/test_df_correction_comparison_v2.py`: off-domain finite-bit accounting, Pauli mean/control phase, frame/proxy/angle enclosures, partial T multiplicity and external resource/launch guards. No registered CD/E recipe or gridsynth; one small off-domain native lowering compile per focused-suite invocation. Source-bound local checks, not immutable CI/scientific comparison.


## DF correction comparison integrity source v3 (2026-10-11)

[限定修正・GPT引き継ぎ](../docs/tracks/algorithm_codesign/df_correction_comparison_integrity_repairs_v3_20261011.md)：科学契約v2不変、512 KiB分割とinventory SHA結合、88 focused PASS。登録比較/T合成0、authorization=false、mandatory STOP。


## Track B fingerprint診断source準備（2026-10-12）

[GPT handoff](../docs/tracks/algorithm_codesign/df_correction_fingerprint_diagnostics_preparation_and_gpt_handoff_20261012.md)：B-local観測adapter、lossless matrix bytes・recursive payload・None/exception保存、登録外2-mode旧/new controls。
最終focused tests104 PASS。controlledで同じ行列でもrecursive自動名によるhash差を保存した。登録r3停止の原因は未確定。
既存moduleの数式・guard・expected・contract不変。旧marker/result/auth/STOP保持、登録science/T合成0、RUN_READY=false。
新source snapshotと全失敗test記録を公開し、mandatory STOP。次scopeの判断はGPT/利用者側。


## Track B E-r3 identity-only診断準備（2026-10-12）

[scope/source handoff](../docs/tracks/algorithm_codesign/e_r3_identity_diagnostic_source_review_and_gpt_handoff_20261012.md)：保存r3 uncontrolledのみ、current→legacy各一回の診断専用runner・新launch/manifest/result namespaceを準備。
native transpile/CD/r6/Tなし。definition内部計算はobserverとCPU/RSS/output guardで会計。新off-domain tests33 PASS。
registered診断0、旧source/authorization/marker/result/STOP保持。新authorization pending・runs0。
source/scope確認と別明示one-shot承認までmandatory STOP。旧科学分類TECHNICAL_INCONCLUSIVE不変。

source固定前の最終確認：profiler context違反のtechnical STOPを追加し、new focused testsは34 PASS。初回33 PASSとdraft checkpointは保持。
