# 文書索引

2026-10-11 N1/N2構成・N3係数下界の限定feasibilityを完了。
[結果・GPT判断事項](research/hamiltonian_algorithm_design_results_20261011.md)、[固定scope](research/hamiltonian_algorithm_design_scope.md)を入口とする。
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
[結果とGPT判断事項](research/representation_attribution_pair_results_20261010.md)、[scope](research/representation_attribution_pair_scope.md)を入口とする。
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


2026-10-10の独立系列第2 batchは
[限定構成・同一信号精度比較の結果](research/representation_construction_comparison_results_20261010.md)、
[固定scope](research/representation_construction_comparison_scope.md)、
[利用者提供独立reviewとZIP inputs](research/representation_construction_inputs/)。
Aの入力-derived core/実finite-RTE、Cの対称S2対照、Bのisometry接続を小系で閉じ、GPTレビュー待ちSTOP。
全136 precision rowsと374 native IR、source/runner/tests/保存監査は報告書の証拠表から辿れる。
中心仮説null、次段false、旧初期探索・Track A/Bのscience/契約/STOPは不変。以下は各milestone時点の履歴。

2026-10-10の独立Hamiltonian表現探索は
[初期検証結果・GPTレビュー資料](research/representation_exploration_initial_validation_20261010.md)、
[固定scope](research/representation_exploration_scope.md)、[dated note](research/研究ノート/2026-10-10_representation_exploration.md)。
source/runner/test/artifact/監査は報告書の証拠表に対応する。A/B/Cの小規模比較を完了してSTOP。
中心仮説・新規性・精度一致総costは未確定。以下は基点までの既存Track A/Bの履歴で、変更しない。

2026-10-06のTrack Aは[H4正式契約準備指示](research/gpu_server_track_a_h4_geometry_contract_preparation_prompt.md)と
[scope JSON](research/track_a_h4_geometry_contract_preparation_scope_v1.json)。
利用者確認済みの追加6距離・最大12 workers、将来74,784-wrapper上限を契約準備へ反映する。
checkpoint数値回路identity修正とzero-compute検査まで。本計算・science port・新benchmarkは未認可、旧証拠不変。
利用者指示で今回の契約準備資料だけをcommit/pushする。以下は各milestone当時の履歴。

最新のTrack Aは、利用者の新しい意向により原稿作成を保留し、
[H4 geometryと要求精度の全候補map](research/track_a_geometry_precision_extension_proposal_v0.md)を準備する段階。
設計案・予算・未確定条件は[別JSON](research/track_a_geometry_precision_extension_proposal_v0.json)へ分離した。
[GPUサーバー側準備指示](research/gpu_server_track_a_h4_geometry_resource_preparation_prompt.md)は既存server環境を優先し、
環境inventory・synthetic CPU benchmark・契約草案まで。本計算はこのhandoffだけでは起動しない。
距離・実行場所・worker・新science source/authorizationは未固定で、本計算は未実行。
旧原稿・証拠・STOPは保存する。以下は各milestone当時の履歴。

Track Aの現在の入口は[原稿v0.1・Supplement・claim audit・review依頼](manuscripts/README.md)。
固定artifactから4主図と補足図1を生成し、日本語通し原稿を作成した。科学計算0、科学STOPを維持する。
原稿のscope監査は執筆者によるもので、独立投稿可能性reviewは次の段階。以下は各stage当時の履歴。

最新のTrack AはPM-2後reviewを採用した[主張・証拠対応表](research/track_a_post_pm2_claim_evidence_map.md)と
[原稿・主要4図の設計](research/track_a_post_pm2_manuscript_design.md)。
追加計算をせず限定case studyとして原稿化する。現在は設計までで、図生成・通し原稿は未実施。
根拠path・固定commit・証拠階層・先行研究との差分・非claimを整理した。
既存result/status/manifest不変、科学計算STOP、Track B変更0。以下のreview待ち等は各stage当時の履歴である。

最新のTrack Aは[PM-2精度・資源境界結果と照合](pr2_pm2_precision_resource_result_validation.md)。
保存JSONだけの302点・67,346行、元ε再現、paired uncertainty、P envelopeを照合し、研究方針review待ちSTOP。
新しい科学計算0、利用者指示でresult commitへ収録するPOSTHOC local evidence。以下は各milestone当時の履歴として読む。

現在は[PM-2保存値解析実装](research/pr2_pm2_precision_analysis_implementation.md)と
[source固定監査](../artifacts/resource_applicability/pr2_pm2_precision_implementation/2026-10-05/source_freeze_v1.json)の段階。
62 local synthetic tests合格、source `324435d77b6642dbd44e8d1f178420daf62e77ed`、本解析は未実行。
研究契約は不変で、別の明示解析指示後も全statusでmandatory STOPする。

最新のTrack A準備は[PM-2精度と資源境界契約](research/pr2_pm2_precision_resource_contract_v1.md)。
全218 development候補とM2元5構成を別集合で固定し、schema・input identity・成果物・STOP条件まで準備した。
保存JSON/input coverageだけを照合し、精度解析は未実施・未認可。旧結果は変更しない。

Track Aの最新は[PM-1実行結果・照合](pr2_pm1_discard_result_validation.md)。
H4 1.00 Å、STO-3G、DF rank12、T=0.8、B0 rank4/5 × q=1/2/4/8の8件がaccuracy適格、
8 signals/16 wrappersを一回完了。pre/post201 local tests passed。
新B0最小rank5・q1は保存B2 r4のprimary点推定の1.75659倍、旧rank6 discardより9.34%低い。
`PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`、mandatory STOP、研究判断null。
結果と監査は利用者指示によるresult commitへ収録するlocal evidence。旧draft/preparation/finalizationの未実行記述は当時の履歴として保持する。

最新のTrack A claim限定は[PM-0 POSTHOC証拠帰属・機構解析](research/pr2_post_m2_evidence_attribution.md)。
元のM1/M2結果と区別し、同一domain、selector指標別regret、N×cost、same-R、欠測の表を辿る。
後続の[PM-1近接discard契約・実装](research/pr2_pm1_nearby_discard_contract_v1.md)は
8 signal＋16 wrapperのfuture上限とsealed planを準備した段階。本計算・authorizationは未実施。
[GPTへのPM-1準備bundleレビュー依頼](research/pr2_pm1_preparation_external_review_request_fd7552e.md)から
PM-0の根拠、固定source、plan、監査へ辿れる。

このディレクトリには、研究方針の正本、実装規約、検証報告、発表資料の案内が共存する。
研究全体を初めて読む場合は、先に[`../PROJECT_MAP.md`](../PROJECT_MAP.md)と
[`research/研究概要・現状.md`](research/研究概要・現状.md)を読む。

PR-2別系列の最新結果は
[`M2 held-out結果照合`](pr2_matched_accuracy_m2_transfer_result_validation.md)。
固定5構成、196 wrappersの一回実行は`TRANSFER_SUPPORTED`、研究方針全面review待ちで停止している。
developmentの根拠は[`M1-B1結果検証`](pr2_matched_accuracy_m1_b1_result_validation.md)。
旧S0 STOPとS2結果を保持した
S2後reviewでは、[`matched-accuracy先行研究gate`](research/pr2_matched_accuracy_prior_art_gate_v1.md)、
[`M1前resource-map契約`](research/pr2_matched_accuracy_resource_contract_v1.md)、
[`M1実装契約`](research/pr2_matched_accuracy_m1_implementation_contract_v1.md)、
[`M1前最終amendment`](research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md)を固定した。続いて
[`M1-A validation`](pr2_matched_accuracy_m1_a_validation.md)でdevelopment 1.00 Åの210候補を評価した。
64 proxy-frontier候補中52件が16-cell cap外に残り、`SELECTION_LIMITED`でcompile 0のまま停止した。
外部review後、[`M1-B1 bounded compile契約`](research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md)で
accuracy適格random 194 cell×32 trajectory×2軸とB0/B1 16 cell×2軸、計12,448 wrapperの有限grid、
cache identity、B1後STOPをzero-compute固定した。
実行前外部reviewの修正要求は、[`M1-B1 execution contract amendment v2`](research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md)で
科学実行sourceをauthorizationより先に固定し、runnerのterminal statusをcompile map完成review待ちまたは
implementation failureだけに限定した。
actual execution source commit `33f436b`、source-bound plan v2、
[`M1-B1 execution authorization v1`](research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md)を固定して
12,448-wrapper mapを完了した。検証後の判断は`CONTINUE_RESOURCE_STUDY`で、その時点ではheld-out未認可だった。
後続M2だけを別契約・source・authorization・最終reviewと利用者指示に従って実行した。追加96、S3は未実行・未認可。

## 研究方針と現在地

- [`research/研究概要・現状.md`](research/研究概要・現状.md)：最新の短い全体要約
- [`research/prevalidation_catalog_evidence_map.md`](research/prevalidation_catalog_evidence_map.md)：事前検証カタログの実施IDと文書・artifact・testの対応
- [`research/README.md`](research/README.md)：研究文書内の索引
- [`research/研究目的・研究課題.md`](research/研究目的・研究課題.md)：目的と研究課題
- [`research/研究方法・解析手順.md`](research/研究方法・解析手順.md)：採用する解析手順
- [`research/数値実験・評価計画.md`](research/数値実験・評価計画.md)：検証と評価の計画
- `research/研究ノート/`：時系列の判断記録。現在の仕様ではない

## 現在の主な検証文書

### PF係数

- [`pf_delta_validation.md`](pf_delta_validation.md)
- [`pf_c_system_size_validation.md`](pf_c_system_size_validation.md)

### finite RTE

- [`rte_conventions.md`](rte_conventions.md)
- [`rte_truncation_budget.md`](rte_truncation_budget.md)
- [`finite_rte_signal_validation.md`](finite_rte_signal_validation.md)
- [`research/finite_rte_phase_amplitude_contract.md`](research/finite_rte_phase_amplitude_contract.md)：FR-0の補正後演算子・平均演算子、位相・信号半径境界、比較契約
- [`research/finite_rte_phase_amplitude_prior_art.md`](research/finite_rte_phase_amplitude_prior_art.md)：finite RTEと近接解析のscoped先行研究監査
- [`research/finite_rte_phase_amplitude_fr1_preregistration.md`](research/finite_rte_phase_amplitude_fr1_preregistration.md)：FR-1の非可換toy入力・gate・停止規則と実行後status
- [`finite_rte_phase_amplitude_validation.md`](finite_rte_phase_amplitude_validation.md)：G0/G1/G3/G4通過、G2不通過、mechanism-only停止結果
- [`fr_revision_fr1a_posthoc.md`](fr_revision_fr1a_posthoc.md)：既存FR-1を正scalarと強いbaselineで再解析し、scalar-only説明となった事後監査
- [`fr_revision_nonuniform.md`](fr_revision_nonuniform.md)：非一様4×4でFR固有の境界改善を確認したが、固定予算の選択差がなくmechanism-onlyで停止したFR-R1b結果
- [`research/fr_revision_scalar_structure_contract.md`](research/fr_revision_scalar_structure_contract.md)：FR-R0の正scalar分離、情報層、強いbaseline、FR-R1事前登録要件
- [`research/fr_revision_fr1a_posthoc_plan.md`](research/fr_revision_fr1a_posthoc_plan.md)：既存FR-1を再分類しない正scalar事後解析計画
- [`research/fr_revision_nonuniform_preregistration.md`](research/fr_revision_nonuniform_preregistration.md)：非一様4×4の固定grid、状態、比較、GO/STOP
- [`research/fr_research_claim_and_manuscript.md`](research/fr_research_claim_and_manuscript.md)：FR-R1b後の中核主張C1/C2、条件付きC3、先行研究監査、証明義務、完成判定
- [`research/pr2_s0_s1_execution_amendment_v3.md`](research/pr2_s0_s1_execution_amendment_v3.md)：S0実行と、S0通過時だけのS1 correctness実行を許可し、S1 summary後のmandatory STOPを固定
- [`research/pr2_codex_validation_policy_d3e1723.md`](research/pr2_codex_validation_policy_d3e1723.md)：Codexが実装・実行してよいS0/S1範囲とS2/S3禁止を定める方針
- [`research/pr2_s0_reproduction_stop_c644925.md`](research/pr2_s0_reproduction_stop_c644925.md)：development hash不一致による`STOP_INPUT_REPRODUCTION_MISMATCH`、S1未実行、証拠hashと再試行条件
- [`research/pr2_s0_external_review_request_c644925.md`](research/pr2_s0_external_review_request_c644925.md)：S0 terminal STOP後に、終了または新しい結果前amendmentの要否をGPTへ確認するレビュー依頼
- [`research/pr2_v4_s2_development_authorization_v5.md`](research/pr2_v4_s2_development_authorization_v5.md)：V4 correctness、development-only S2、S2後mandatory STOPを結果前固定
- [`pr2_v4_s2_development_validation.md`](pr2_v4_s2_development_validation.md)：V4/S2の実行結果、B2/B3 frontier、rank control、方針review判断
- [`pr2_matched_accuracy_m1_a_validation.md`](pr2_matched_accuracy_m1_a_validation.md)：210候補のcompile-free signal/selector、52未選択frontier、`SELECTION_LIMITED`、全compile counter 0を記録するM1-A結果
- [`pr2_matched_accuracy_m1_b1_result_validation.md`](pr2_matched_accuracy_m1_b1_result_validation.md)：12,448 wrapper、全checkpoint/cache再集計、actual B2 rank-3 frontier、旧selector監査、fixed-q=8比較、状態準備感度と`CONTINUE_RESOURCE_STUDY`を記録するM1-B1結果
- [`pr2_matched_accuracy_m2_transfer_result_validation.md`](pr2_matched_accuracy_m2_transfer_result_validation.md)：固定5構成のH4 1.30 Å transfer、196 wrapper照合、`TRANSFER_SUPPORTED`、primary ratio/paired uncertainty、資源・pre/post testsと研究方針reviewへのmandatory STOP
- [`research/pr2_matched_accuracy_prior_art_gate_v1.md`](research/pr2_matched_accuracy_prior_art_gate_v1.md)：M1前のclaim-level先行研究比較と`PROCEED_RESOURCE_STUDY`判定
- [`research/pr2_matched_accuracy_resource_contract_v1.md`](research/pr2_matched_accuracy_resource_contract_v1.md)：matched-accuracy baseline、可変q correctness、compile選抜、held-out前freezeを定める研究契約
- [`research/pr2_matched_accuracy_m1_implementation_contract_v1.md`](research/pr2_matched_accuracy_m1_implementation_contract_v1.md)：M1の候補identity、seed、selector、schema、zero-compute guardと科学計算未承認を固定する実装契約
- [`research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md`](research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md)：2026年の近接研究二件との最終claim照合と、M1-A limited時にcompile job 0で停止するhard barrierを追加する現行amendment
- [`research/pr2_matched_accuracy_m1_execution_authorization_v1.md`](research/pr2_matched_accuracy_m1_execution_authorization_v1.md)：development-only M1-Aの入力、source、最大212 signal、compile 0、held-out access 0を結果前固定する実行承認
- [`research/pr2_matched_accuracy_m1_execution_authorization_v1_1.md`](research/pr2_matched_accuracy_m1_execution_authorization_v1_1.md)：v1のresult未作成停止後、固定KのRTEConfig self-consistencyだけを修正して同じM1-Aを再認可
- [`research/pr2_m1_a_selection_limited_external_review_request_3c1831e.md`](research/pr2_m1_a_selection_limited_external_review_request_3c1831e.md)：commit `3c1831e`のM1-A結果を固定し、compile上限拡張・technical note・停止の三択をGPTへ依頼するレビュー文
- [`research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md`](research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md)：194 random＋16 baseline cell、12,448 wrapper上限、cache/checkpoint identity、B1後STOPを固定し、科学実行を未承認に保つ契約
- [`research/pr2_m1_b1_preexecution_external_review_request_1228168.md`](research/pr2_m1_b1_preexecution_external_review_request_1228168.md)：source commitとzero-compute planを固定し、result-prior M1-B1 authorization作成前のGPTレビュー項目と回答形式を定める依頼文
- [`research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md`](research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md)：execution source先行固定と、compile map完成後に研究判断を外部reviewへ戻すterminal status修正
- [`research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md`](research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md)：source commit `33f436b`、plan v2、12,448 wrapper、6 workers、2 terminal status、held-out禁止を結果前固定する一回限りの実行認可
- [`research/pr2_m1_b1_execution_authorization_external_review_request_8fc2400.md`](research/pr2_m1_b1_execution_authorization_external_review_request_8fc2400.md)：authorization bundle commit `8fc2400`を固定し、本計算開始前の最終GPT reviewと三択回答形式を指定する依頼文
- [`df_rte_tail_extraction.md`](df_rte_tail_extraction.md)
- [`df_rte_event_circuit_api.md`](df_rte_event_circuit_api.md)

### コンパイル後回路コスト

- [`random_circuit_cost_validation.md`](random_circuit_cost_validation.md)
- [`rte_boundary_cost_validation.md`](rte_boundary_cost_validation.md)
- [`rte_boundary_pair_validation.md`](rte_boundary_pair_validation.md)
- [`hierarchical_cost_validation.md`](hierarchical_cost_validation.md)
- [`rte_connected_cluster_cost_validation.md`](rte_connected_cluster_cost_validation.md)
- [`rte_compiled_cost_validation_summary.md`](rte_compiled_cost_validation_summary.md)
- [`rte_compiled_event_cost.md`](rte_compiled_event_cost.md)
- [`df_partial_s2_compiled_cost.md`](df_partial_s2_compiled_cost.md)
- [`df_partial_s2_repeated_compiled_cost.md`](df_partial_s2_repeated_compiled_cost.md)

### RPEへの接続

- [`rpe_resource_accounting.md`](rpe_resource_accounting.md)
- [`rpe_hadamard_interrogation.md`](rpe_hadamard_interrogation.md)
- [`rpe_round_cost_connection_validation.md`](rpe_round_cost_connection_validation.md)
- [`rpe_hadamard_failure_validation.md`](rpe_hadamard_failure_validation.md)
- [`rpe_hadamard_proxy_resource_validation.md`](rpe_hadamard_proxy_resource_validation.md)
- [`rpe_allocation_sensitivity_validation.md`](rpe_allocation_sensitivity_validation.md)
- [`rpe_four_round_accounting_validation.md`](rpe_four_round_accounting_validation.md)
- [`rpe_four_round_phase_validation.md`](rpe_four_round_phase_validation.md)
- [`rpe_target_round_horizon_validation.md`](rpe_target_round_horizon_validation.md)
- [`rpe_delta_round_schedule_validation.md`](rpe_delta_round_schedule_validation.md)
- [`rpe_delta_compiled_cost_validation.md`](rpe_delta_compiled_cost_validation.md)
- [`research_direction_prevalidation.md`](research_direction_prevalidation.md)
- [`research_direction_ablation.md`](research_direction_ablation.md)
- [`research_direction_pf_sensitivity.md`](research_direction_pf_sensitivity.md)
- [`research_direction_gate_s1.md`](research_direction_gate_s1.md)
- [`research_direction_structure_pilot.md`](research_direction_structure_pilot.md)
- [`research_direction_sequence_policy.md`](research_direction_sequence_policy.md)
- [`research_direction_full_scope.md`](research_direction_full_scope.md)
- [`research_direction_full_scope_extension.md`](research_direction_full_scope_extension.md)
- [`research_direction_decision_cost.md`](research_direction_decision_cost.md)
- [`research_direction_late_round_proxy.md`](research_direction_late_round_proxy.md)
- [`research_direction_compiler_transfer.md`](research_direction_compiler_transfer.md)
- [`research_direction_uncertainty_break_even.md`](research_direction_uncertainty_break_even.md)
- [`research_direction_wp11_synthesis.md`](research_direction_wp11_synthesis.md)
- [`research_direction_full_opt2.md`](research_direction_full_opt2.md)：WP11選択M06-Fの事前固定条件、51/51完全性監査、direct-RZ測定、coherent opt2再最適化、A0 proxy-lineage再照合
- [`research_direction_signal_weight_pilot.md`](research_direction_signal_weight_pilot.md)：P-Bのenergy bias・target weight・q別signal再解析とテーマ選定判断
- [`research_direction_geometry_energy_difference_pilot.md`](research_direction_geometry_energy_difference_pilot.md)：P-Cのgeometry依存signed PF error、未使用geometry/delta、差分bias予測
- [`research/pc_geometry_tracking_breakdown_preregistration.md`](research/pc_geometry_tracking_breakdown_preregistration.md)：P-Cの8 geometry、追跡規則、blind region、7 gate、停止規則を計算前に固定
- [`research_direction_geometry_tracking_breakdown.md`](research_direction_geometry_tracking_breakdown.md)：追跡prefix不変、stretch予測破れ、固定gateによるcurrent H4 P-C停止
- [`research/pd_energy_tail_pareto_preregistration.md`](research/pd_energy_tail_pareto_preregistration.md)：P-Dの固定5公式、development/blind、7 gate、停止規則
- [`research_direction_energy_tail_pareto.md`](research_direction_energy_tail_pareto.md)：energy-onlyとtail-aware選択のblind逆転、P-D条件付き候補、次のsigned-time/internal-H_D gate
- [`research/pd_realization_go_no_go_preregistration.md`](research/pd_realization_go_no_go_preregistration.md)：P-D現実化の負時間finite-RTE、fragment内部`H_D`誤差、fresh `L_D=5`、Go/No-Go停止規則
- [`research_direction_pd_realization.md`](research_direction_pd_realization.md)：D1--D3全通過、P-D正式候補化と研究再設計停止点
- [`research/pd_primary_research_contract.md`](research/pd_primary_research_contract.md)：P-D S0の主RQ、比較契約、Case A--D、強制停止
- [`research/pd_prior_art_and_baselines.md`](research/pd_prior_art_and_baselines.md)：既知absolute-tail-time modelと新規性候補の境界
- [`research/pd_s1_fair_comparison_preregistration.md`](research/pd_s1_fair_comparison_preregistration.md)：固定時間・位相予算、B0/B1a/B1b/B2/B4、K4・境界規則
- [`research_direction_pd_fair_comparison.md`](research_direction_pd_fair_comparison.md)：B1b/B2/B4一致、Case C/D不成立、B1a境界未解消のS1停止結果
- [`research/pd_s1_posthoc_reanalysis_plan.md`](research/pd_s1_posthoc_reanalysis_plan.md)：固定S1 artifactだけを使う事後再解析の入力、5%近傍、解釈規則、停止条件
- [`research_direction_pd_s1_posthoc.md`](research_direction_pd_s1_posthoc.md)：一次Case Bを保存した主baseline再解釈、B1a診断、nested/native内訳
- [`research/r3_prior_art_and_minimal_contract.md`](research/r3_prior_art_and_minimal_contract.md)：広いR3の重複、R3-S0不通過、実行しない条件付き最小検証契約
- [`../pd_s1_review_5c331f0.md`](../pd_s1_review_5c331f0.md)：S1 snapshotに対する外部レビュー。正式方針ではなく事後再解析の入力資料
- [`research_direction_joint_synthesis_pilot.md`](research_direction_joint_synthesis_pilot.md)：P-Aのinterval-aware DF回路列合成、強いbaseline、未使用列holdout
- [`research_direction_theme_selection.md`](research_direction_theme_selection.md)：P-B/P-C/P-A比較と後続停止点を含む選定履歴。現行判断はA/B/Cに確認済み主題なし
- [`research/pa_joint_synthesis_prior_art_audit.md`](research/pa_joint_synthesis_prior_art_audit.md)：P-A v1のscoped prior-art audit、限定novelty statement、v2境界
- [`research/pa_joint_synthesis_blind_validation_preregistration.md`](research/pa_joint_synthesis_blind_validation_preregistration.md)：H5 physical transferとH4 opt2 compiler transferの事前登録、compile前task manifest、固定gate
- [`research_direction_joint_synthesis_blind_validation.md`](research_direction_joint_synthesis_blind_validation.md)：P-A v1のH5 physical transferとH4 opt2 compiler transferの完了結果、固定gate、判断、scope
- [`research/pa_joint_synthesis_v1_formalization.md`](research/pa_joint_synthesis_v1_formalization.md)：P-A v1の有限候補、DP最適性・計算量・同値性条件、一区間退化と次の機構識別
- [`research/pa_joint_synthesis_mechanism_validation_preregistration.md`](research/pa_joint_synthesis_mechanism_validation_preregistration.md)：明示的一区間baseline、forced support変化、order 2 stream、固定gate・停止規則の事前登録
- [`research_direction_joint_synthesis_mechanism_validation.md`](research_direction_joint_synthesis_mechanism_validation.md)：P-A非退化mechanism検証の0 split・0 plan差・0 RZ改善とP-C復帰判断

## 実行・運用

- [`server_parallel_validation_execution.md`](server_parallel_validation_execution.md)：共有CPU/GPUサーバー向けのbounded実行、checkpoint、resume、dry-run
- [`pr2_s2_parallel_execution.md`](pr2_s2_parallel_execution.md)：固定PR-2 S2のcell-level CPU並列化、段階barrier、persistent compile cache、serial同値性test
- [`examples/parallel_validation_h4_q1_manifest.json`](examples/parallel_validation_h4_q1_manifest.json)：H4 q=1のdry-run用manifest例

## 発表資料と参考文献

- [`presentations/README.md`](presentations/README.md)：発表資料・構成案の位置づけ
- [`references/README.md`](references/README.md)：同梱した論文PDFの位置づけ
- [`rte_source_versions.md`](rte_source_versions.md)：RTE一次資料の版管理

## 状態の読み方

個別文書に数値があっても、それだけで現在利用可能とは判断しない。
再現可能性、失効、成果物の有無は[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)と
[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)で確認する。

## M2 usable B2契約修正 v2（2026-10-04）

外部reviewの修正要求を[amendment v2](research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)へ反映した。
Pareto supportとprimary ratioは共にaccuracy-eligibleかつprimary重大underestimateのないB2だけを使う。
v1証拠・固定5構成・seed・196-wrapper上限を維持し、科学実行とheld-out accessは未認可である。
moduleは`src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`、runnerは
`scripts/run_pr2_matched_accuracy_m2_transfer_contract.py`、testは
`tests/test_pr2_matched_accuracy_m2_transfer_contract.py`、schema/planは
`artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`から辿れる。

## PR-2 M2科学実行コードの入口

[実装資料](research/pr2_matched_accuracy_m2_transfer_execution_implementation.md)に、固定5構成・196-wrapper
上限、usable B2、paired-axis covariance、one-shot停止、source-bound planと別authorizationを記録した。
sourceとsynthetic検証を固定する段階であり、held-out開封・科学実行・次段階は未認可である。

## PR-2 M2最終実行前レビュー

- [実行authorization](research/pr2_matched_accuracy_m2_transfer_execution_authorization_v1.md)：actual source/plan、固定5構成、196 wrappers、最大5 workers、一回限りを固定する。
- [最終review依頼](research/pr2_m2_execution_authorization_external_review_request_90a9f24.md)：最終承認と利用者の実行指示までheld-out未開封・本計算未実行で停止する。
- `artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/authorization_audit_v1.json`：local zero-science gateとtimed tests。科学結果・immutable CIではない。
