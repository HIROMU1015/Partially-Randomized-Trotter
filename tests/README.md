# tests の役割

## 2026-10-13 mixed-access v1 source・環境準備完了／科学認可前STOP

[技術準備・数式・資源template](../docs/research/qc_mixed_access_source_preparation_v1_20261013.md)、[次の認可境界](../artifacts/qc_mixed_access_source_v1/2026-10-13/launch_readiness.json)。新runner/checker、dependency wheel hash lock、22登録外synthetic PASS。full-Fock/phase/DF/adjoint/2worker保存監査を検証。
`src/trottertracks/representation_exploration/qc_mixed_access_{math,resource,execution,pipeline,checker}_v1.py`、`scripts/run_qc_mixed_access_feasibility_v1.py`、`scripts/verify_qc_mixed_access_feasibility_v1.py`、`scripts/check_qc_mixed_access_{synthetic,preparation}_v1.py`、`tests/test_qc_mixed_access_feasibility_v1.py`、artifacts `artifacts/qc_mixed_access_source_v1/2026-10-13/`。
固定H4/7法/T1,4/r1…1024の契約は不変。今回分子生成・SCF・登録科学PF・compile・GPTレビュー0、function/full energy費用UNKNOWN。次はユーザーの固定一回実行承認。技術準備だけでGPT再レビューを要求しない。旧122 entries/科学source/認可/STOPとglobal description欠落FAIL、N1STOP/N2HOLD/N3CLOSEDを保持。以下は過去milestone。

## 2026-10-12 N3 OLD-family v1 one-shot results

[固定S1二例の有理dual分離・独立監査・STOP](../docs/research/n3_old_family_v1_results_20261012.md)。同じ保存U/c/OCC有理δでNORM＋全旧κ/μ族（FIVE包含）を分離。
source/module/runner/checker/testsは凍結済み準備版のまま。artifacts `artifacts/n3_old_family_v1/2026-10-12/diagnostic1/` と `execution_audit/`。
一回/retry0、posthoc development、元primary/gate/STOP不変、GPTの新結果判断へ。以下は過去milestone。

## 2026-10-12 N3 OLD-family v1 preparation
+
+[数学契約・実装・実行前STOP](../docs/research/n3_old_family_v1_preparation_20261012.md)。
+modules `n3_old_family_contract_v1` / `n3_old_family_v1` / `n3_old_family_check_v1` / `n3_old_family_inputs_v1`、scripts prepare/run/verify_n3_old_family_v1、tests test_n3_old_family_v1。
+artifacts `artifacts/n3_old_family_v1/2026-10-12/`。23 synthetic PASS、新S1診断0、別実行指示待ち。元結果/STOP不変。以下は過去milestone。

## 2026-10-12 N3 saved-width v1 preparation

[証拠整理・式・synthetic・数値適用前STOP](../docs/research/n3_saved_width_v1_preparation_20261012.md)。
module `n3_saved_evidence_v1.py`/`n3_saved_width_v1.py`/`n3_saved_width_analysis_v1.py`; scripts `prepare_n3_saved_width_v1.py`/`run_n3_saved_width_v1.py`/`verify_n3_saved_width_v1.py`;
tests `test_n3_saved_width_v1.py`; artifacts `artifacts/n3_saved_width_v1/2026-10-12/`. 実8入力の新幅は未評価。


## 2026-10-12 N3 OCC v1 fixed eight-input result

[一回結果・監査・STOP](../docs/research/n3_occupation_transfer_v1_results_20261012.md)。強対照/OCC各8/8、固有gain0、保存check5,008,702 PASS。
Frozen generator/runner/checker/tests are unchanged; artifacts `artifacts/n3_occupation_transfer_v1/2026-10-12/run1/` and `execution_audit/`.


## 2026-10-12 N3 OCC v1 implementation preparation

[実装・検証・実行前STOP](../docs/research/n3_occupation_transfer_v1_source_preparation_20261012.md)。新8入力batch未認可。
module `src/trottertracks/representation_exploration/n3_occupation_transfer_v1.py`;
independent `n3_occupation_transfer_check_v1.py`; runner `scripts/run_n3_occupation_transfer_v1.py`;
saved verifier `scripts/verify_n3_occupation_transfer_v1.py`; tests `tests/test_n3_occupation_transfer_v1.py`;
artifacts `artifacts/n3_occupation_transfer_v1/2026-10-12/preparation/`.


## 2026-10-12 N3 occupation certificate preparation

[数学検証・比較設計](../docs/research/n3_occupation_redesign_mathematical_audit_20261012.md)。新科学batch未認可。
module `src/trottertracks/representation_exploration/n3_occupation_certificate_audit.py`;
runner `scripts/verify_n3_occupation_redesign_preparation.py`;
coefficient-only preparation `scripts/prepare_n3_occupation_transfer_inputs.py`;
test `tests/test_n3_occupation_certificate_audit.py`;
artifacts `artifacts/n3_occupation_redesign_preparation/2026-10-12/`.


2026-10-11 N1/N2構成・N3係数下界の限定feasibilityを完了。
[結果・GPT判断事項](../docs/research/hamiltonian_algorithm_design_results_20261011.md)、[固定scope](../docs/research/hamiltonian_algorithm_design_scope.md)を入口とする。
N1：JW4・ν2・signed DF rank2・決定論S2、T=.6、delta=.6/.3/.15/.075、εH<=.02。
N2：occupation4・full-ij J・k<=2 signed charge、T=.7、係数予算.0021。共通εcomplex=.04。
N1はframe費用を減らせたが元H精度のshot負担でexactより高く、N2は加算込みでdirectより高い。
N3：6 modes・ν2・初期active4、gapped例で係数δ=.001549636、小gapではfull spaceへ戻す。
science source ed335a0、85 local tests、280 native IR/120 rows/177 source blobsの保存照合PASS。immutable CIではない。
module `src/trottertracks/representation_exploration/algorithm_design.py`、runner `scripts/run_hamiltonian_algorithm_design.py`、
保存集約器 `scripts/finalize_hamiltonian_algorithm_design.py`、verifier `scripts/verify_hamiltonian_algorithm_design.py`、
構成・選択・保存監査 `scripts/audit_hamiltonian_algorithm_design_construction.py`、
tests `tests/test_hamiltonian_algorithm_design.py`、artifacts `artifacts/hamiltonian_algorithm_design/2026-10-11/`。
中心テーマnull・次段false・mandatory STOP、GPT判断待ち。以下の旧stage/他Trackの履歴・契約・STOPは保持する。

2026-10-11整理：A寄与分解・B′物理pairの限定batch（計算系列日付2026-10-10）を完了。
[結果とGPT判断事項](../docs/research/representation_attribution_pair_results_20261010.md)、[scope](../docs/research/representation_attribution_pair_scope.md)を入口とする。
Aは3-mode JW全Fock8、synthetic exact DF rank2、native L_D0/1/2＋whole-Pauli＋旧/完全占有core、T=.4、delta=.4/.2/.1。
Bはphysical3+aux1 isometry fixture、JW、L_D0/r1/K2、T=.2、delta=.2/.1/.05。geometry/化学basis/分子fittingなし。
費用は固定q1、meanはq1/2/4、epsilon=.05/.02、full-physical-Fock bias診断で揃える。
Aは安いwhole-Pauliを加えるとcore改善を支持せず、完全占有coreの追加効果はRZ約.452%減・CX約.202%増。
B′はlambda/shot数を下げてもbasis/native費用でdirect-Pauliより高い。C追加scan0。
source3975650、85 focused local tests、1558 compiles、177 source/input blobsと全IRの別保存監査PASS。
local開発証拠でimmutable CI/外部科学再現/新規性認定ではない。
`ATTRIBUTION_PAIR_COMPLETE_AWAITING_GPT_REVIEW`、中心仮説null、次段false、mandatory STOP。
以下の旧stage/Track A/Bの履歴・science/契約/STOPを変更しない。


2026-10-10 A寄与分解・B′物理pair限定batchの[固定scope](../docs/research/representation_attribution_pair_scope.md)。
module `src/trottertracks/representation_exploration/attribution_pair.py`、runner `scripts/run_representation_attribution_pair.py`、
tests `tests/test_representation_attribution_pair.py`、保存監査 `scripts/verify_representation_attribution_pair.py`、
artifacts `artifacts/representation_attribution_pair/2026-10-10/`を一組に読む。旧系列は当時の履歴として保持する。


限定構成比較の[test_representation_construction_comparison.py](test_representation_construction_comparison.py)は新31件。
analytic JW、input構成とcore不変性、有限RTE列挙、Gaussian/controlled位相、degree2 mixed access、
invalid graph拒否、反復境界、isometry quartic接続、shots/paired SE、保存IRを検査する。
旧探索23＋既存finite-RTE16と合計70 local tests passed、fail/skip0、18 warnings（Qiskit16、旧ComplexWarning2）。
[結果・限界・pre-freeze failure log](../docs/research/representation_construction_comparison_results_20261010.md)を参照。
保存された全374 IRの別実装照合と3改変拒否は別監査で、pytest件数には加算しない。full suiteは実行しない。

独立表現探索の[test_representation_exploration.py](test_representation_exploration.py)は23 local tests。
非可換square恒等式、独立JW参照、反射finite-RTE平均、制御位相、mixed oracleを検査し、
既存[test_rte.py](test_rte.py)16件と合計39 passed（real入力に関するComplexWarning2件をlogへ保存）。
[結果・限界](../docs/research/representation_exploration_initial_validation_20261010.md)を参照。
全イベントの保存値照合と3改変拒否は別監査で、科学実験数やtest件数に加算しない。full suiteは実行しない。

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

## 2026-10-12 N3理論・幅指向縮約taskの統合準備

文書入口：`docs/research/n3_theory_task_integration_v1_20261012.md`。
数学：`docs/research/n3_theory_claim_map_v1_20261012.md`、文献：`docs/research/n3_literature_correspondence_v1_20261012.md`、将来仕様：`docs/research/n3_width_reduction_v2_spec_20261012.md`。
保存帰結checker：`src/trottertracks/representation_exploration/n3_theory_implications_v1.py`、CLI：`scripts/verify_n3_theory_implications_v1.py`、合成tests：`tests/test_n3_theory_implications_v1.py`。
証拠・review/identity：`artifacts/n3_theory_task_integration_v1/2026-10-12/`。
30 synthetic PASS／保存全幅補題二例を独立照合。新science0、幅generator未実装、次はGPT研究判断、Mandatory STOP。

## N3幅比較 v2：比較実装・未使用入力の科学実行準備

文書入口 `docs/research/n3_width_comparison_v2_preparation_20261012.md`、数学 `n3_width_comparison_v2_contract_20261012.md`、結果前設計 `n3_width_comparison_v2_preregistration_20261012.md`、文献 `n3_width_comparison_v2_literature_amendment_20261012.md`（いずれもdocs/research）。
module群 `src/trottertracks/representation_exploration/n3_width_{contract,algebra,generator,check,pipeline,execution}_v2.py`。
runner `scripts/run_n3_width_comparison_v2.py`、保存verifier `scripts/verify_n3_width_comparison_v2.py`、入力literal `scripts/prepare_n3_width_inputs_v2.py`、tests `tests/test_n3_width_comparison_v2.py`。
artifacts `artifacts/n3_width_comparison_v2/2026-10-12/preparation/`。32 synthetic PASS、科学入力の数値評価0、実行認可待ち・Mandatory STOP。

2026-10-13 N3幅比較v2：[一回pilotの一次結果](../docs/research/n3_width_comparison_v2_results_20261013.md)。固定source13不変、science6H/504 rows、independent1224証拠PASS、認可消費済み・Mandatory STOP・GPT科学判断待ち。

N3保存帰属v1：[資料・事後評価準備](../docs/research/n3_saved_attribution_v1_posthoc_contract_20261013.md)。module n3_saved_attribution_v1/n3_saved_attribution_execution_v1、runner run_n3_saved_attribution_v1、tests test_n3_saved_attribution_v1、artifacts/n3_saved_attribution_v1/2026-10-13/preparation。23 synthetic PASS、新登録数値評価/PSD再認証0、明示posthoc認可待ち。

- 2026-10-13 N3保存帰属v1 [結果・限界](../docs/research/n3_saved_attribution_v1_results_20261013.md)、[GPT判断引継ぎ](../docs/research/n3_saved_attribution_v1_results_review_handoff_20261013.md)。保存データ限定の独立監査 `scripts/audit_n3_saved_attribution_v1_results.py`、artifact `artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1{,_audit}`。新producer/solverなし、STOP。

## 2026-10-13 量子化学relative blockの事前検証

[構成・方法差・誤差・費用](../docs/research/qc_relative_block_prevalidation_v1_20261013.md)、[科学判断引継ぎ](../docs/research/qc_relative_block_review_handoff_v1_20261013.md)。module `src/trottertracks/representation_exploration/qc_relative_block_v1.py`、synthetic runner `scripts/check_qc_relative_block_synthetic_v1.py`、tests `tests/test_qc_relative_block_v1.py`、artifacts `artifacts/qc_relative_block_prevalidation/2026-10-13/`。13 fixed synthetic PASS、実分子/eig/compile0。可逆関数回路とmethod delta/具体費用は未確定。pilot未認可・Mandatory STOP、旧Track/N3 source/results保全。
