> **M1-S γ依存V1採択監査**：[非数値checker](tracks/algorithm_codesign/check_m1s_gamma_adoption_static.py)／[正式契約](../docs/tracks/algorithm_codesign/m1s_v1_gamma_formally_adopted_contract_20261017.md)。既存runner/SOURCEは変更せず、正式machine契約のschema・許可差分・hash・authority未取得での拒否だけを検査。science runなし。

> **M1-S V1 γ依存**：[runner](tracks/algorithm_codesign/run_m1s_saved_signal_gamma_v1.py)／[static checker](tracks/algorithm_codesign/check_m1s_gamma_preparation_static.py)／[登録外benchmark](tracks/algorithm_codesign/benchmark_m1s_gamma_offdomain.py)／[契約候補](../docs/tracks/algorithm_codesign/m1s_v1_gamma_dependent_contract_candidate_20261017.md)。`status`はmetadataのみ。未採択候補でrun拒否、別途C/Aが必要。benchmarkは旧artifactをopenしない。

> **M1-S V1 V0**：[future runner](tracks/algorithm_codesign/run_m1s_saved_signal_v1.py)／[static checker](tracks/algorithm_codesign/check_m1s_v1_v0_static.py)／[契約候補](../docs/tracks/algorithm_codesign/m1s_v1_saved_signal_contract_candidate_20261017.md)。`status`はmetadataのみ。`run`は候補で拒否し、新しい採択Cと別途authority-only Aが必要。forecast未実行、mandatory STOP。

> **M1-S RPE適用条件の非数値整理（2026-10-17）：[Decision Packet](../docs/tracks/algorithm_codesign/m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md)／[非数値checker](tracks/algorithm_codesign/check_m1s_rpe_conditions_static.py)。** 最新GPTレビューを引き継ぎ、位相bias予算・visibility/shot式・全wrapperとstate準備費用・総額条件を記号式で整理。D1–D11未決、契約候補未採択、科学計算/数値再解析/forecast/native生成/新authority・marker0。旧科学status/結果/SOURCEを保全、mandatory STOP。以下の本文は保持。

> **M1-S有限step v2計算完了（2026-10-16）：[結果・修正来歴](../docs/tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md)。** ユーザーの修正・完了承認の下で試行01の精度STOPを保全、Phiの既存Frobenius収縮性を誤差伝播へ適用し試行02で108/108行完了。H4 1.00 Å/STO-3G/8 modes/DF rank12・G3/G6/G9・保存oracle・delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1不変。correctness/全3twirl/精度PASS、条件付き未認証。G3/G6は各16絶対対照resolved higher、G9は16 unresolved。科学GO/native総資源未認定、旧成果・各試行・marker・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step v2起動前technical STOP（2026-10-16）：[不具合・未適用修正案](../docs/tracks/algorithm_codesign/m1s_finite_signal_v2_pre_execution_configuration_stop_20261016.md)。** 別途ユーザー実施承認を受領したが、固定SOURCE fbc2edeaの精度文字列2^-24/2^-18をFractionへ直接渡す経路がValueErrorとなることをmetadata確認。H4 1.00 Å/STO-3G/8 modes/DF rank12・G3/G6/G9・delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1・保存oracle・108行・精度/runtimeは不変。科学/初回correctness/pilot起動0、one-shot未開始・未消費、authorization未発行・marker0。旧成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step信号pilot v2正式採択（2026-10-16）：[正式契約・採択記録](../docs/tracks/algorithm_codesign/m1s_finite_step_complex_signal_pilot_v2_formally_adopted_20261016.md)。** ユーザーが68fa5580候補の条件付きcoverage・モデル差B/abs(T)B会計・108行grid・比較・runtime・one-shot/failure STOPを採択。H4 linear1.00 Å/STO-3G/8 modes/DF rank12・G3/G6/G9・保存数値oracle、delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1。SOURCE fbc2edea不変、platform/数値correctness未認証。採択契約は実行を許可しない。科学実施は明示未承認、新科学/数値test/authorization/grant/marker0、旧成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step信号pilot v2モデル差会計候補（2026-10-16）：[新契約](../docs/tracks/algorithm_codesign/m1s_finite_step_complex_signal_pilot_contract_v2_candidate_20261016.md)／[v2 SOURCE](../src/trottertracks/algorithm_codesign/m1s_finite_signal_v2)／[runner](tracks/algorithm_codesign/run_m1s_finite_signal_v2.py)／[静的checker](tracks/algorithm_codesign/check_m1s_finite_signal_v2_static.py)。** H4 linear1.00 Å/STO-3G/8modes/DF rank12・G3/G6/G9・保存数値oracle、delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1・zero込み108行を維持。raw H_fullとH_tildeを区別、Bとabs(T)Bを別台帳、総raw-H biasと純アルゴリズム差の区間を区別。SOURCE fbc2edea。条件付き未認証・未採択・実施未承認、数値correctness未検証、新科学/数値test/authorization/grant/marker0、旧成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step信号pilot準備（2026-10-16）：[契約候補・SOURCE](../docs/tracks/algorithm_codesign/m1s_finite_step_complex_signal_pilot_contract_candidate_20261016.md)。** 受領GPT独立結果レビューを原本保存。H4 linear1.00 Å/STO-3G/8spin orbitals/DF rank12・G3/G6/G9・保存数値oracle。delta候補1/8,1/16 Ha^-1、R2/4、T1/4 Ha^-1、同じ独立draw数のcoherent shared/ordinary/independent比較。SOURCE9758aeaa固定、複素信号/visibility/phaseとlambda²絶対尺度の設計。条件付きcoverage/精度/runtime・新target選択は未採択、新科学/Pauli/数値test/finite signal/authorization/grant/marker0。旧結果・STOP・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S独立one-shot完了（2026-10-16）：[結果・停止](../docs/tracks/algorithm_codesign/m1s_independent_one_shot_results_and_stop_20261016.md)。** ユーザー別途実施承認・専用A2a55ba01、SOURCE4787cf90・採択C3ec5721bでrun1/retry0。H4 linear1.00 Å/STO-3G/8 spin orbitals/保存DF rank12・G3/G6/G9・deltaなし・保存数値oracle。全3分割の照合可能範囲PASS、support/sign差0、W/v/rho固定許容値内。未保存full-H/HDによる全scope UNDETERMINEDを保持。ローカル事後独立再現性照合でありplatform認証/科学GO/finite-step/native総資源評価ではない。旧v4/v5結果・SOURCE・契約・消費済marker・親/他worktree保全、追加科学なし、mandatory STOP。以下は履歴。

> **M1-S独立correctness契約を正式採択（2026-10-16）：[正式契約・採択記録](../docs/tracks/algorithm_codesign/m1s_gate0_v5_independent_correctness_contract_adopted_20261016.md)。** ユーザーが2ba0858cの候補を採択。H4 linear1.00 Å/STO-3G/8 spin orbitals/保存DF rank12・G3/G6/G9・deltaなし・保存数値oracle、SOURCE4787cf90固定。照合可能範囲PASS/FAIL/UNDETERMINEDと、未保存full-H/HDを含む全scope UNDETERMINED（FAIL優先）を区別。精度・算術・runtime/caps不変、数値correctness未検証。科学実施は明示未承認、新科学/Pauli/数値test/W-v-rho/authority/grant/marker0、旧v4/v5成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5独立correctness準備（2026-10-16）：[検証契約候補・独立性と限界](../docs/tracks/algorithm_codesign/m1s_gate0_v5_independent_correctness_contract_candidate_20261016.md)。** 公開草稿d6d35cfbを別worktreeで再利用、SOURCE4787cf90固定。H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/G6/G9・deltaなし・保存数値oracle。occupation/CAR/全Pauli trace/tensor再構成、exact rational W/v/rho、結果JSONはreference固定後のみ照合。旧full-H/HD行列未保存分はUNDETERMINED、全scope PASS不可。契約未採択・実施未承認・数値correctness未検証、新科学/実入力・模擬input数値test/authority/grant/marker0、旧v4/v5成果・marker・元Codex/親worktree保持、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5初回one-shot（2026-10-16）：[結果・停止](../docs/tracks/algorithm_codesign/m1s_gate0_v5_one_shot_results_and_stop_20261016.md)。** 別ユーザー実施承認・固定A8da63d67から一回実行。H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・保存HD G3/G6/G9・deltaなし。全3行で局所W/v予算とrho精度を通過、W/v/rho取得、到達1,239gate PASS。HEADROOM_DIAGNOSTICS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。同一development条件での事後技術改訂評価、platform/true-ground/finite-step/native総資源/科学GOは未認証・未判断。旧v4 STOP/null/marker保持、one-shot1/retry0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5正式採択（2026-10-16）：[正式契約と採択記録](../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v5_formally_adopted_20261016.md)。** ユーザーがH4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/G6/G9・deltaなし、局所alpha_R=gamma_(713+r)S_R・17精度条件・1e-8 Ha+B_res・条件付きcoverage・runtime/resource・all-splits/one-shot/failure STOPを採択。SOURCE69de022/proof b5a185cc固定、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。採択flagはplatform認証ではない。契約完備性と実行許可を分離、科学計算は明示未承認。新科学/予算評価/Pauli/数値test/authority/grant/marker0、旧v4 STOP/null/marker保持、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5 SOURCE接続（2026-10-16）：[最終実験契約候補](../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v5_candidate_20261016.md)／[v5 SOURCE](../src/trottertracks/algorithm_codesign/m1s_gate0_v5)／[runner](../scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v5.py)／[static checker](../scripts/tracks/algorithm_codesign/check_m1s_gate0_v5_static.py)／[監査・機械契約](../artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1)。** H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9・deltaなし。tail alpha_R=gamma_(713+r)S_R resolver接続、global/local ledger分離、17条件/runtime維持。SOURCE69de022固定、122 static PASS、実入力新式/Pauli/数値test/W-v-rho/Gate0評価0。候補未採択・coverage採択false・実施未承認、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。旧v4 TECHNICAL_STOP/null/消費済marker保持、新authority/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 tail局所上界（2026-10-16）：[条件付き証明](../docs/tracks/algorithm_codesign/m1s_gate0_tail_local_bound_proof_20261016.md)／[v5精度設計候補](../docs/tracks/algorithm_codesign/m1s_gate0_tail_local_precision_contract_v5_design_candidate_20261016.md)／[宣言的spec](../src/trottertracks/algorithm_codesign/m1s_gate0_tail_bound_spec.py)／[静的checker](../scripts/tracks/algorithm_codesign/check_m1s_gate0_tail_bound_static.py)／[証拠台帳](../artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1)。** H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9・deltaなし。SOURCEのgroup長とDAGからalpha_R=gamma_(713+r)S_Rを条件付き導出、index722/719/716。旧2^-24を維持できる解析的十分領域を証明。53 static PASS、数値評価0。v5はSOURCE未接続・非実行の未採択設計候補、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。旧v4 TECHNICAL_STOP/null/marker、SOURCE/契約/精度ceiling/runtime保持、Gate0再実行0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 精度原因調査（2026-10-16）：[項別予算・改善候補](../docs/tracks/algorithm_codesign/m1s_gate0_precision_budget_cause_and_options_20261016.md)／[保存データscalar診断](../artifacts/track_b_m1s_gate0_precision_cause/2026-10-16/v1)／[診断SOURCE](../scripts/tracks/algorithm_codesign/analyze_m1s_gate0_precision_saved.py)。** H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9・deltaなし。承認済み保存再解析1、4 NPZ配列のexact尺度と3×8 scalarケースを評価。全H共通係数上界とlambda正規化が支配、tail局所化だけでは不足、局所DAGとの組合せに未証明の改善余地。新Pauli/state/matrix/W-v-rho/Gate0実行0。旧TECHNICAL_STOP/null・17条件・SOURCE/契約/authorization/marker保持、候補未採択、CONDITIONAL_ENGINEERING_NOT_CERTIFIED、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v4初回one-shot（2026-10-16）：[結果・技術停止](../docs/tracks/algorithm_codesign/m1s_gate0_v4_one_shot_results_and_stop_20261016.md)／[raw result](../artifacts/track_b_m1s_gate0_result/2026-10-16/v4_one_shot/result.json)／[保存監査](../artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot)／[saved-only auditor](../scripts/tracks/algorithm_codesign/audit_m1s_gate0_v4_saved_result.py)。** 別ユーザー実施承認・固定Aから一回実行。H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9、delta評価なし。到達132gate PASS、残差品質PASS。全splitで事前W/v誤差予算超過、W/v/rho未評価null、TECHNICAL_STOP / PRECISION_UNRESOLVED。17条件・SOURCE・入力は不変。CONDITIONAL_ENGINEERING_NOT_CERTIFIED、metric correctness未到達、科学的GO/no-go未判断。one-shot1/retry0、marker消費済み、旧成果・親worktree保持、mandatory STOP。以下は各段階の履歴。

> **M1-S Gate 0 v4正式採択（2026-10-16）：[正式v4契約](../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v4_formally_adopted_20261016.md)／[正式machine契約](../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/execution_contract_v4_adopted.json)／[ユーザー採択記録](../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/user_final_contract_adoption_record.json)／[採択監査](../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1)／[採択static checker](tracks/algorithm_codesign/check_m1s_gate0_v4_adoption_static.py)。** ユーザーが17項目・1e-8 Ha+B_res・条件付きcoverage・runtime/caps・one-shot/failure STOPを正式採択。H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・saved G3/6/9、delta評価なし、SOURCE/入力/精度条件は候補と同一。CONDITIONAL_ENGINEERING_NOT_CERTIFIED、runtime/数値correctness未検証。契約完備性と実行許可を分離し、科学計算は明示的に未承認。新science/数値test/formula評価/実行authorization/grant/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v4契約候補（2026-10-16）：[v4最終契約候補](../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v4_candidate_20261016.md)／[品質/coverage説明](../docs/tracks/algorithm_codesign/m1s_gate0_v4_oracle_residual_quality_and_coverage_20261016.md)／[GPT原本](../docs/research/track_b_m1s_gate0_v3_gpt_limited_scientific_review_20261016.md)／[v4 SOURCE](../src/trottertracks/algorithm_codesign/m1s_gate0_v4)／[runner](tracks/algorithm_codesign/run_m1s_gate0_headroom_v4.py)／[static checker](tracks/algorithm_codesign/check_m1s_gate0_v4_static.py)／[監査資料](../artifacts/track_b_m1s_gate0_v4_contract/2026-10-16/v1)。** H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・saved G3/6/9、delta評価なし。残差gateを1e-8 Ha+B_resへ改訂、他8式/8scalar/runtime保持。全17項目が候補化、未採択。CONDITIONAL_ENGINEERING_NOT_CERTIFIED、99非数値static PASS、runtime/数値correctness未検証。新science/Pauli/数値test/formula評価/authority/grant/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 誤差算定式の契約準備（2026-10-16）：[v3最終契約候補](../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v3_candidate_20261016.md)／[式・coverage条件](../docs/tracks/algorithm_codesign/m1s_gate0_preregistered_error_formula_design_20261016.md)／[v3 SOURCE](../src/trottertracks/algorithm_codesign/m1s_gate0_v3)／[runner](tracks/algorithm_codesign/run_m1s_gate0_headroom_v3.py)／[static checker](tracks/algorithm_codesign/check_m1s_gate0_formula_static.py)／[監査資料](../artifacts/track_b_m1s_gate0_formula_contract/2026-10-16/v1)。** H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・saved G3/6/9、delta評価なし。未決9件中8件を条件付き式候補として具体化、oracle残差品質ηは未決。17項目・runtime/caps維持、68非数値static PASS、数値correctness/coverage未検証、全契約未採択・実行不可。新science/Pauli表/数値test/formula評価/authority/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 scope正式採択・精度契約案（2026-10-16）：[最終契約v2候補](../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v2_candidate_20261016.md)／[精度17項目](../docs/tracks/algorithm_codesign/m1s_gate0_precision_error_budget_design_20261016.md)／[runtime/caps](../docs/tracks/algorithm_codesign/m1s_gate0_runtime_and_resource_contract_candidate_20261016.md)／[v2 module](../src/trottertracks/algorithm_codesign/m1s_gate0_v2)／[runner](tracks/algorithm_codesign/run_m1s_gate0_headroom_v2.py)／[追加static checker](tracks/algorithm_codesign/check_m1s_gate0_precision_static.py)／[scope採択記録](../artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/gate0_scientific_scope_adoption_record.json)。** H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/6/9・Pauli law/oracle比較のみユーザー採択。精度8値を候補化、9絶対値未決。追加67静的PASS、旧SOURCE/成果保持、数値correctness未検証、全契約未採択・科学未承認、新科学/Pauli表/authority/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 SOURCE準備（2026-10-16）：[実験契約最終候補](../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_candidate_20261016.md)／[専用module](../src/trottertracks/algorithm_codesign/m1s_gate0)／[runner](tracks/algorithm_codesign/run_m1s_gate0_headroom.py)／[非科学的checker](tracks/algorithm_codesign/check_m1s_gate0_source_static.py)／[監査資料](../artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1)。** 保存H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/6/9のadapterとW/v/ρ SOURCEを実装。30静的確認PASS、数値correctness未検証。IMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED、契約未採択・実行未承認、新科学/Pauli表/authority/marker0、mandatory STOP。以下は履歴。

> Parent allocation入口はS4補完contract/manifestを検査し、訂正KKT/halfway最大4構成のみを対象とする。pending authorityで拒否、登録実行0。[補完準備](../docs/tracks/algorithm_codesign/parent_allocation_s4_kkt_repair_and_supplement_preparation_20261015.md)、[zero-compute preflight](tracks/algorithm_codesign/preflight_parent_allocation_experiment.py)。以下は履歴。

> Parent allocation v3は一回実行完了・marker消費済み。[結果/保存監査](../docs/tracks/algorithm_codesign/parent_allocation_v3_results_and_gpt_handoff_20261015.md)。追加runner起動は未認可、mandatory STOP。以下は準備履歴。

> Parent allocationの既存one-shot入口はv3 sourceの固定contract/manifestを検査し、Parent専用buffered writerを使う。新S3/A3を固定後、一回実行・retry0・STOP。

> Parent allocationの[preflight](tracks/algorithm_codesign/preflight_parent_allocation_experiment.py)はv2 contractを検査する。[限定修正・78 tests](../docs/tracks/algorithm_codesign/parent_allocation_v2_owner_repair_source_preparation_20261015.md)。登録runner/workerは今回未起動、別明示承認待ち。

> [Parent allocation新zero-compute preflight](tracks/algorithm_codesign/preflight_parent_allocation_experiment.py)、[将来one-shot入口](tracks/algorithm_codesign/run_parent_allocation_one_shot.py)、[一key worker](tracks/algorithm_codesign/run_parent_allocation_synthesis_key.py)。[採択条件/71 tests/実行上限](../docs/tracks/algorithm_codesign/parent_allocation_execution_source_preparation_20261015.md)。pending authorityで登録処理を拒否、今回科学実行0。

> Parent allocationの[最終技術準備](../docs/tracks/algorithm_codesign/parent_allocation_scientific_design_decision_packet_20261014.md)でnative/cache/guardを接続。既存preflightはmetadata専用のまま。科学worker/launch authorityは作成していない。

> [Parent allocation zero-compute preflight](tracks/algorithm_codesign/preflight_parent_allocation.py)：JSON/bytes/SHAのみ、algorithm sourceをimportしない。常にscience execution不可。実行launch gateではない。[引き継ぎ](../docs/tracks/algorithm_codesign/parent_allocation_source_preparation_and_gpt_handoff_20261014.md)。

> [D3 saved-only analyzer](tracks/algorithm_codesign/analyze_d3_saved_mechanisms.py)：stdlibと固定commitから取得したscalar/既存incidenceだけ。budget/runner/event/pair再評価なし。

> **D3 v2保存値監査**：[audit_d3_v2_saved_outputs.py](tracks/algorithm_codesign/audit_d3_v2_saved_outputs.py)。stdlibでidentity・保存scalar・chunksを照合。backend/science/pairの再実行なし。[結果](../docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md)。

> **D3 v2準備**：[read-only verifier](tracks/algorithm_codesign/verify_d3_source_preparation.py)／[新source引き継ぎ](../docs/tracks/algorithm_codesign/d3_v2_precision_repair_and_source_preparation_20261014.md)。同じone-shot入口を新contract/namespaceへbindし、旧markerを流用しない。今回登録実行0。

> **D3失敗保存値監査**：[audit_d3_failed_saved_outputs.py](tracks/algorithm_codesign/audit_d3_failed_saved_outputs.py)。stdlibでraw bytes/identity/静的call順序を確認。runner/backend/scienceを実行しない。[報告](../docs/tracks/algorithm_codesign/d3_failed_execution_and_gpt_handoff_20261014.md)。

> D3専用[owner/observer](tracks/algorithm_codesign/run_d3_one_shot.py)、[内部ledger](tracks/algorithm_codesign/run_d3_ledger_job.py)、[内部key worker](tracks/algorithm_codesign/run_d3_synthesis_key.py)、[静的source verifier](tracks/algorithm_codesign/verify_d3_source_preparation.py)。ownerもsource-bound別明示認可まで起動拒否。内部CLIを独立起動しない。[契約・報告](../docs/tracks/algorithm_codesign/d3_source_preparation_and_execution_handoff_20261013.md)。

> Track B D3準備： [saved key/static word verifier](tracks/algorithm_codesign/prepare_d3_guarantee_price_plan.py) は保存metadataだけを読む。budget/price再会計、synthesis、pair評価、provider/circuit/matrix生成なし。[計画](../docs/tracks/algorithm_codesign/d3_guarantee_price_preregistration_v1_20261013.md)。本番runnerは未実装。

> 2026-10-07 Track B RA-D0 v3：**READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION**（実行承認ではない）。
> [source review](../docs/tracks/algorithm_codesign/ra_d0_source_review_v3_20261007.md) / [GPT handoff](../docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_v3_20261007.md) / [manifest](../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/evidence_manifest_v3.json)。
> [focused verifier](tracks/algorithm_codesign/verify_ra_d0_source_review_v3.py) / [v3 tests](../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v3.py)。
> exact-certified B2 minimum infeasibilityを正常outcomeに修正。pointをfreezeへ記録しbudget/queryを空にして次nへ進む。
> uncertified failureはtechnical STOP。数値・candidate・grid・call/resource capは維持。登録最適化0、authorizationなし、mandatory STOP。

> 2026-10-07 Track B RA-D0 v2：**READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW**。
> [source review](../docs/tracks/algorithm_codesign/ra_d0_source_review_v2_20261007.md) / [GPT handoff](../docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_20261007.md) / [evidence manifest](../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/evidence_manifest_v2.json)。
> [future runner](tracks/algorithm_codesign/run_ra_d0_one_shot.py) / [focused verifier](tracks/algorithm_codesign/verify_ra_d0_source_review_v2.py) / [v2 tests](../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v2.py)。
> B0_saved/ideal分離、数値B1⊂B2⊂B3、profile-paired budget、batch freeze-before-B3、
> anchor-first、main LP 55,275 / auxiliary込み110,550、resource/launch guardを固定。
> 登録最適化・実budget/minimum/witness取得0、authorizationなし。旧本文・旧STOP・Track Aは保持。mandatory STOP。

## Track B RA-D0 static preparation / mandatory STOP（2026-10-06）

[source review](../docs/tracks/algorithm_codesign/ra_d0_source_preparation_review_v1.md)：21 columns/x、18 sign pairs一致、35 focused tests PASS。
B専用namespace `src/trottertracks/algorithm_codesign/ra_d0/`、static generator／verifier、
`artifacts/track_b_ra_d0_preparation/2026-10-06/`に候補・grid・semantic audit・provenanceを保存。
ideal nestingと数値membershipを区別し、query実行scope／資源上限をGPTへ返す。
登録最適化／新合成／新science0、RUN_READY=false、既存分類・Track A・旧STOP保持。
**mandatory STOP。以下の既存本文を全文保持する。**

## Track B RA-RTE統合数学監査・mandatory STOP（2026-10-06）

R1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942` 基点、GPT設計案へのDOCS_SYMBOLIC_ONLY_MATHEMATICAL_AUDIT。
[命題別監査](../docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_v1.md)と[GPT handoff](../docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_gpt_handoff_20261006.md)：一block／finite table／canonicalのfixed-n LPは仮定付きで成立。
shot-gridの固定total cap保存には反例。log/root・sampler認証、Delta=0、peak workspaceの規約を実行前修正へ返す。
[stdlib人工bookkeeping](tracks/algorithm_codesign/check_ra_rte_mathematical_bookkeeping.py)の[50 checks](../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_checks_v1.json)を一般証明と分離した。
[manifest](../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/evidence_manifest_v1.json)、[dated note](../docs/research/研究ノート/2026-10-06_track_b_ra_rte_mathematical_audit.md)。science/synthesis/solver/資源再採点0、共通API変更0。
既存科学分類・結果・STOPは不変。性能・新規性・algorithm採択・次実装／R2 authorizationは未確定。
**mandatory STOP。次の採択・実装・pilotの必要性／範囲はGPT判断。以下の既存本文を全文保持する。**

## Track B R1.5保存値帰属・mandatory STOP（2026-10-06）

input R1 commit `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b` の保存値だけを用いたPOSTHOC attribution / design input。
[帰属報告](../docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md)と[GPT handoff](../docs/tracks/algorithm_codesign/r1p5_gpt_handoff_20261006.md)：新science/synthesis/compile/候補追加0、R1科学分類は不変。
primaryは2-qubit finite P₃、distinct-basis controlled、x={1/8,1/4}、登録native三precision。
Aは登録(G_T,G_CX,G_1Q) frontにx=1/8の1e-4、x=1/4の1e-3/1e-4で残る。
normalizationだけでなくnative費用とbias/shotの関係を整理し、固定合成列への依存も保存した。
[stdlib保存値解析](tracks/algorithm_codesign/analyze_r1p5_saved_attribution.py)、[全summary](../artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)、[provenance manifest](../artifacts/track_b_r1p5_saved_attribution/2026-10-06/evidence_manifest_v1.json)、
[日付note](../docs/research/研究ノート/2026-10-06_track_b_r1p5_saved_attribution.md)。共通library変更・独立validation・新algorithm採択はない。
限定診断SUPPORTS_RA_RTE_DESIGNは設計入力のみ。eta探索/R2/DF接続/追加scienceは未認可。
**mandatory STOP。次の数学設計・研究方針判断はGPT側。以下の既存本文を全文保持する。**

## Track B R1一回結果・mandatory STOP（2026-10-06）

固定S `d43d64a821a0249a0dfab12a2472bd3a72fdee74` →直接子authorization-only A
`411f08f768244fe87b600d82308c3851847fe9e4`からrun1/retry0。
[結果照合](../docs/tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md)：126 keys / 264 rows / 132 controlled tasks完了、全task適格。
terminal R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW、[保存field専用監査](tracks/algorithm_codesign/audit_r1_saved_result.py) PASS。
primary distinct-basis controlledではB²改善とnative/shot資源のtrade-offを保存し、自動研究GOはない。
[全264 rows CSV](../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)、[evidence manifest](../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)、
[GPT判断への入口](../docs/tracks/algorithm_codesign/r1_post_run_gpt_review_request_20261006.md)。原result/marker/source、既存証拠・共通API・Track A保持。
science終了後mandatory STOP、追加合成/target/grid/分子/DF/trajectory/GPUは行わない。
研究方針・RQ・新規性・着地点・追加検証の必要性/範囲はGPT側。
以下のpending/最新記述は当時の履歴として本文をそのまま保持する。

# scripts 索引

## Track B R1 native-resource runner: source review only

[run_r1_rte_reallocation.py](tracks/algorithm_codesign/run_r1_rte_reallocation.py)のplanはstatic key inventory照合だけ。
runはsource Sのdirect authorization-only child Aと新明示指示を要求する。現在pending、科学実行なし。
[契約/source review](../docs/tracks/algorithm_codesign/rte_reallocation_r1_source_review_request_20261006.md)、
[source manifest](../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/source_manifest_v1.json)。
27 focused tests、126 planned keys、mandatory STOP。


## Track B R0.5 symbolic checker

[check_rte_reallocation_r05_symbolic.py](tracks/algorithm_codesign/check_rte_reallocation_r05_symbolic.py)：
固定9条件/18 signのexact Fraction/Pauli比較を一回実施。science runnerではない。
[監査](../docs/tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md)、
[保存artifact](../artifacts/track_b_rte_reallocation_r05/2026-10-06/exact_symbolic_comparison_v1.json)。
追加比較・再実行・科学実行の認可なし、mandatory STOP。


## Track B R0 technical checker

[check_rte_reallocation_symbolic.py](tracks/algorithm_codesign/check_rte_reallocation_symbolic.py) verifies fixed free-word/Fraction identities. It is not a sampler, solver or science runner; no trotterlib imports. [Review/provenance](../docs/tracks/algorithm_codesign/rte_reallocation_r0_review_packet_20261006.md) records one local technical execution and STOP.


Track B [BS-0.5設計監査](../docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md)はdocs-only。
runner/solver/synthesis/testsを追加・実行していない。ordinaryの式とledgerは実装済みsourceではない。
RUN_READY=false、mandatory STOP、次実装/検証scopeはGPT判断。以下は既存実行履歴。

Track B [block合成の次仕様案](../docs/tracks/algorithm_codesign/block_synthesis_design_review_20261006.md)はdocs-only。
新runner/solver/library生成/合成/testsは追加・実行していない。旧SP one-shotを再使用しない。
API/対照/pilot案の採否はGPT reviewへ戻し、RUN_READY=false／mandatory STOP。以下は既存実行履歴。

Track B [SP-1一回結果](../docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)は
SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW、mandatory STOP、retry0。science runnerのone-shotはconsumed。
[audit_sp1_saved_result.py](tracks/algorithm_codesign/audit_sp1_saved_result.py)はstdlibの保存field照合専用。
source/authorization/hash/inventory、保存interval算術／分類predicate／duplicate／capを監査する。
science moduleやrunnerをimportせず、matrix・guard・係数・Bernstein shots・resourceを再評価しない。
[audit／evidence manifest](../artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/)。保存値監査PASS、次stage未認可。
以下のrunner準備・pending記述は結果前履歴。

Track B [SP-1 runner](tracks/algorithm_codesign/run_sp1_wrapper_pilot.py)を追加した。
`plan --source-commit <full S>`は静的角度/key/fusionとsource identityだけを照合する。
資源/信号採点・合成・marker作成は0。`run`は別の明示authorizationとclean direct-child Aを要求する。
[採用契約](../docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)、
[source manifest](../artifacts/track_b_sp1_wrapper_source/2026-10-06/source_manifest_v1.json)。
focused testsは`test_sp1_wrapper_*.py`だけ。science未認可、最終source reviewへSTOP。以下は履歴。

Track B [SP-1 wrapper契約案](../docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_proposal_v1.md)は準備段階。
science runnerはまだ作成していない。共通会計kernelと17 artificial focused testsだけを追加し、
science sweep／SP-0.5 rerunを行わない。実adapter／fusion監査／source freeze／別authorization前にSTOP。
限定test入口：`PYTHONPATH=src python3 -m unittest discover -s tests/tracks/algorithm_codesign -p test_sp1_wrapper_accounting.py -v`。
以下のSP-0.5 runnerは別のconsumed one-shotで、SP-1の入口ではない。

Track Bの[SP-0.5一回結果](../docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md)は
mandatory STOP、retry0、wrapper pilot未認可。[audit_sp05_saved_result.py](tracks/algorithm_codesign/audit_sp05_saved_result.py)は
保存field／hash／counts／分類predicate／provenanceだけをstdlibで照合する。
synthesis／PAI／J／error-guard関数を再実行しない。旧runner runは一回consumed、追加runしない。


Track Bの[SP-0.5専用runner](tracks/algorithm_codesign/run_sp05_synthesis_economics.py)は
[結果前契約](../docs/tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md)に沿ったprimitive合成経済性gateの入口。
planは合成0、runはsource S→別authorization-only child A・明示指示・fresh one-shot markerを要求する。
準備tests34件pass、登録target合成0、現在run未認可。16-cell wrapper／DF／分子／trajectoryは扱わない。


Track B [BM-0.5](../docs/tracks/algorithm_codesign/bm05_review_packet_20261005.md)の
[形式word監査](tracks/algorithm_codesign/audit_bm05_symbolic_equivalence.py)は抽象非可換letters／Fractionだけの
degree3比較。physical inputやscience runnerを呼ばない。保存reportとscopeは[専用索引](tracks/algorithm_codesign/README.md)。
現adapterは同値でnew-method gate不通過、BM-1は実行しない。以下のBF記述は既存履歴。

Track Bの[BF1-R0一回replay結果](../docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md)は
recovery complete、事後primary BF-A。R0 markerもconsumed、retryなし。以下のR0準備説明は結果前履歴。

Track Bの`tracks/algorithm_codesign/run_bf1_read_only_recovery.py`は[BF1-R0契約](../docs/tracks/algorithm_codesign/bf1_read_only_recovery_contract_v1.md)
に沿った一回の保存値replay専用入口。科学runnerを呼ばず、cache不足で停止する。

Track Bの`tracks/algorithm_codesign/prepare_bf1.py`はformula-only domain列挙と限定synthetic testsの入口。
`run_bf1.py`は別authorization・commit-bound sourceを必要とするone-shot runner。
[BF-1規範](../docs/tracks/algorithm_codesign/bf1_preregistration_v1.md)。固定source/authorizationで一回実行後、
JSON保存例外により[INCONCLUSIVEで停止](../docs/tracks/algorithm_codesign/bf1_one_shot_result_validation_20261005.md)。
one-shot markerはconsumed、retryなし。
[別branchの保存型修正](../docs/tracks/algorithm_codesign/bf1_serialization_repair_review_20261005.md)では
runnerを変更・実行せず、限定synthetic testsのみを行う。
[v2 amendment](../docs/tracks/algorithm_codesign/bf1_execution_gate_revision_v2.md)はpost-search cross-scoreと
authorization-only child commit方式を固定する。
[v3 amendment](../docs/tracks/algorithm_codesign/bf1_assembly_guard_revision_v3.md)はassemblyの逐次誤差伝播を固定する。
prepareは既存v1 domainを参照して新規v3 packetを作り、旧v1/v2を保持する。

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

## Track B G1限定診断のsource

[run_g1_decision_packet.py](tracks/algorithm_codesign/run_g1_decision_packet.py)はread-only source確認と、
別明示指示後の一回だけのdecision packetを分離する。
[専用module](tracks/algorithm_codesign/g1_decision_packet/README.md)、
[source review](../docs/tracks/algorithm_codesign/g1_source_review_20261009.md)、
[focused tests](../tests/tracks/algorithm_codesign/test_g1_source_preparation.py)、
[証拠manifest](../artifacts/track_b_g1_source_preparation/2026-10-09/evidence_manifest_v1.json)。
旧guard/verifier/binaryは不変。本構造監査・固定8人工LPは未実行。全outcomeでSTOP、retry=0。

## Track B G2 saved-value diagnosis (2026-10-09)

`tracks/algorithm_codesign/g2_saved_diagnostic.py` is stdlib-only post-hoc arithmetic for the fixed
21-column tables, 252 profiles/x. See [handoff](../docs/tracks/algorithm_codesign/g2_saved_diagnostic_handoff_20261009.md).
Completed; exclusive diagnostic markers are retained. Do not rerun or infer production authorization.
It imports no old execution controller or shared library and performs no solver/synthesis/circuit work.


## Track B G3（2026-10-09）

`tracks/algorithm_codesign/g3_finite_law.py`（消費済みPhase A）、
`tracks/algorithm_codesign/g3_return_comparator.py`（消費済みPhase B）、
`tracks/algorithm_codesign/audit_g3_saved_outputs.py`（保存値のread-only照合）。
[仕様と結果](../docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md)、
[manifest](../artifacts/track_b_g3_finite_law/2026-10-09/evidence_manifest_v1.json)。
各Phase一回/retry0。full suite/旧run/registered LP/CTS/DFへの自動進行なし、mandatory STOP。


## Track B G4

`tracks/algorithm_codesign/g4_independent_certificate.py`（旧helperなしsaved-value認証）、
`g4_cts_specialization.py`（exact Pauli代数）、`g4_matched_cts.py`（条件付き12合成one-shot）、
`audit_g4_saved_outputs.py`（STOP後の保存値照合）。
[G4 handoff](../docs/tracks/algorithm_codesign/g4_results_and_gpt_handoff_20261009.md)を先に読む。
両marker consumed、retry不可。Cは設計のみ。次science stage未認可。


## Track B G5 fixed-dictionary closure（2026-10-10）

[Saved-only stdlib verifier](tracks/algorithm_codesign/g5_fixed_dictionary_closure.py)：fixed G5 sourceで一回完了。
x1/4・6頂点252 profiles、strict rational/dyadic CTS照合とdigital bridge。新LP/synthesis/matrix/circuitなし。
[scope](../docs/tracks/algorithm_codesign/g5_fixed_dictionary_closure_scope_20261010.md)、
[result](../docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md)、
[manifest](../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/evidence_manifest_v1.json)。marker消費済み、再実行しない。mandatory STOP。


## Track B G6 formal technical audit（2026-10-10）

[G6 checker](tracks/algorithm_codesign/g6_return_generator_audit.py)は独立off-domain形式testsのみ。
本回technical bundle完了、science runnerではない。新technical marker/STOPを保存し、再呼出ししない。
[結果・認可境界](../docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md)。


## Track B G7（2026-10-10）

[限定費用runner](tracks/algorithm_codesign/g7_budget_control_economics.py)、
[契約](../docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md)。
fixed source/full SHA/clean HEAD/new marker/24 keysのみ。再実行不可、bundle後STOP。


## 2026-10-10 Track B G7：取得完了・mandatory STOP（最新追記）

[G7結果/GPT引継ぎ](../docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md)。
`G7_LIMITED_IMPLEMENTATION_ECONOMICS_COMPLETE_AWAITING_GPT_REVIEW`。source ab2549f41b3546fb3940342a2162dd9ee93699c4、24 keys/8 rows、strict error PASS、retry0。
固定P5では登録3対照後にもconditional期待T減少、P3ではclosed-form対照が小さい。
hard shot cap・classical generation/angle acquisition・未指定provider costを併記。
新規性/主method/次stage未採択、G5閉鎖/G6原証拠とmarker保持、GPT判断へ戻す。


## Track B G8（2026-10-10、source preparation）

[G8 proof/contract](../docs/tracks/algorithm_codesign/g8_proof_contract_and_on_demand_scope_20261010.md)：採用GPT G7 review §11に基づく限定確認。
2 known development inputs/4 same production laws、finite-provider parameter、分離failure配分、
on-demand strict Rzと対称bounded cache。G7のstatus/point comparison/consumed markerを保持。
17 off-domain focused tests、source preparation時点のnew native acquisition0。一束後mandatory STOP。


- `scripts/tracks/algorithm_codesign/audit_g8_saved_outputs.py`: stdlib saved-output/hash audit; no generator/native reacquisition. [G8 handoff](../docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md).


## Track B G9 source preparation（2026-10-10）

[Fixed proof/scope](../docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md)：GPT G8 review §14を採用。known P5/指定3-qubit provider、6 direct+5 helper診断、19 keys/新CTS1 key、23 focused tests。source固定後一束のみ、終了後mandatory STOP。

## G9 one-shot STOP（2026-10-10）

`G9_TECHNICAL_INCONCLUSIVE`：epsilon引数のFraction→mp.mpf変換で停止、native比較0 row。
新規helper attempts1 / 新規sequence取得0 / retry0。source・contract・過去941pathは不変。
23 focused testsと23 saved-output checksは準備/整合証拠で、native資源の科学結果ではない。
[G9 failure・GPT handoff](tracks/algorithm_codesign/audit_g9_saved_outputs.py)。mandatory STOP、次の研究判断はGPTへ返す。

## G9 v2 API-boundary source preparation（2026-10-10）

[G9 v2準備/入口](tracks/algorithm_codesign/g9_p5_matched_native_v2.py)。精度値の型接続を最小修正、19 stub-only/launch tests PASS。
46科学条件・同19-key inventory・旧v1 source/result/markerは保持。
新実合成・登録matrix/予算/科学実行0、v2 marker absent、別authorization pending。
独立branchで資料公開後STOPし、新source-bound明示認可を待つ。

## G9 v2 one-shot completed / STOP（2026-10-10）

[G9 v2 results/GPT handoff](tracks/algorithm_codesign/audit_g9_v2_saved_outputs.py)。`G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`、11 rows/22 axes、19 keys（new1/reuse18）、retry0。
known P5/指定3-qubit providerの登録direct6方式でclosed P5のT intercept/Kが小さく、CTSのCXは小さい。
1,866 event accounting、saved-only25 checks PASS。旧v1結果/marker、critical80/protected982は不変。
実量子shots/trajectory/DF/分子/NPZ/GPU/LPは0、source-bound local evidence。次の科学実行・採択はGPT判断、mandatory STOP。


## G10 saved-policy audit / degree-comparison source preparation（2026-10-10）

[G10 source review / GPT入口](../docs/tracks/algorithm_codesign/g10_source_review_20261010.md)。保存P5 direct945 bindingsのT再計数と任意proposal固定policy下界を独立確認。
ordinary/partial/closed P3/CTSは保存closed P5と分離、general full/closed P5はこの下界では未分離。原G9分類は不変。
同p/x/3-qubit providerのm3/5/7（17 rows/34 axes）を固定。m5はsaved-only共通policy再会計、
登録P3/P7は未取得。41 off-domain focused tests PASS、固定runtime/未認可拒否を確認。
実synthesis0、G10 marker absent、authorization pending。旧source/result/auth/marker/STOP・Track Aを保持。
[evidence manifest](../artifacts/track_b_g10_degree_preparation/2026-10-10/evidence_manifest_v1.json) / [contract](../artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json) / [future runner](tracks/algorithm_codesign/g10_degree_matched_native.py) / [focused tests](../tests/tracks/algorithm_codesign/test_g10_degree_preparation.py)。
次は固定source review→別authorization-only child→新one-shot指示。全結果STOP、研究判断はGPT。


## G10 one-shot technical STOP（2026-10-10、最新追記）

[結果/GPT引継ぎ](../docs/tracks/algorithm_codesign/g10_results_and_gpt_handoff_20261010.md)。`G10_TECHNICAL_INCONCLUSIVE`：固定RSS cap512 MiB超過、guard peak539.546875 MiB。
source `05c5ef23` / authorization-only child `f5cd0755`、一回のみ、retry0。new synthesis27 / reuse19、17行保存。
technical prefixは科学判断に使用せず、原結果/marker/STOPと旧1241 pathを保持。
保存hash/sequence/会計45,605条件の監査PASS。critical113のfull source prefix保持、追記のみ。
追加science/matrix/synthesis/sampler/budget/lower処理なし。mandatory STOP、次の研究/実行scopeはGPT判断。
保存値監査script：[audit_g10_saved_outputs.py](tracks/algorithm_codesign/audit_g10_saved_outputs.py)（stdlib-only、exclusive audit receipt作成済み）。


## G10 RSS技術調査・STOP（2026-10-10、最新追記）

[技術報告/GPT引継ぎ](../docs/tracks/algorithm_codesign/g10_rss_failure_technical_investigation_20261010.md)。原分類 `G10_TECHNICAL_INCONCLUSIVE` は不変。
保存JSONのI/O検算で全serial copy＋JSON chunk list/joinのメモリ増幅を確認。
同原bytes/hashのmaterialized peak508.90625 MiB / stream253.25 MiB。元G10 heap/例外位置の完全再現ではない。
独立調査branch、既存source/result/audit/marker/STOP保持。production修正・cap変更・science再実行0。
typed streaming/lifetime/failure保存の限定scopeをGPTへ返し、mandatory STOP。
[saved JSON I/O診断script](tracks/algorithm_codesign/audit_g10_rss_saved_json.py)はstdlib-only、G10 runnerを呼ばない。


## 2026-10-10 Track B G10 RSS修正source S2準備

future runner: `scripts/tracks/algorithm_codesign/g10_degree_matched_native_v2.py`。検査は `verify_g10_rss_repair_preparation.py`、I/O-only memory検算は `validate_g10_streaming_saved_io.py`。

報告: `docs/tracks/algorithm_codesign/g10_rss_repair_source_and_gpt_review_20261010.md`。契約・根拠: `artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/`。
旧source・結果・marker・STOPを保護する。科学実行/A2/fresh production markerは未認可・未実施。mandatory STOP。


## G10 v2 post-STOP saved failure audit

`tracks/algorithm_codesign/audit_g10_v2_saved_failure.py`：stdlib-only、原failure/marker/IO telemetry、outer process、source/protected identityを確認する。科学再実行・再採点なし、既存receipt上書きなし。資料は `docs/tracks/algorithm_codesign/g10_v2_results_and_gpt_handoff_20261010.md`、raw evidenceは `artifacts/track_b_g10_degree_result/2026-10-10/v2/`。mandatory STOP。


## G10 S3 preparation entry points

`tracks/algorithm_codesign/g10_degree_matched_native_v3.py` は未認可future runner。`verify_g10_key_compatibility_preparation_v3.py` はstatic/saved-only verifier、`validate_g10_key_compatibility_io_v3.py` と `g10_json_compatibility_fixtures.py` は非科学temporary IO検査専用。source reviewは `docs/tracks/algorithm_codesign/g10_v3_key_compatibility_source_and_gpt_review_20261010.md`、artifactは `artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/`。mandatory STOP、production実行禁止。


## Track B G10 v3 completed / mandatory STOP（2026-10-10、最新追記）

[結果/GPT handoff](../docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md)：`G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`。
S3 b9ed014→直接authorization-only A3 53a7bc4、一回/retry0。17rows/34axes、46keys（new27/reuse19）、10,936bindings。
stream/token/STOPとouter正常COMPLETE status一致。outer wall22.2261s/CPU20.0291s/peak240.015625MiB、512MiB cap不変。
保存監査のfield仮定/report組立の訂正と初回記録を保存、本番再実行0。旧結果/marker/STOP、critical180/protected1352保持。
`artifacts/track_b_g10_degree_result/2026-10-10/v3/evidence_manifest_v3.json` とsaved-only `audit_g10_v3_saved_outputs.py` を参照。
known synthetic providerの次数内有限operator比較のみ、science完了後mandatory STOP、追加科学/採択/新規性判断はGPTへ返す。

## Track B G10 v3：文書化の保存値照合・静的図（2026-10-11）

`tracks/algorithm_codesign/summarize_g10_v3_review_saved_values.py`は固定exact fieldの差・比・affine境界だけをstdoutへ出すstdlib helper。
`tracks/algorithm_codesign/plot_g10_v3_saved_review_figures.py`はその保存出力からmatplotlib/Aggで静的図を作る。
どちらもscience runnerではなく、sampling/budget/lower/合成/行列を再生成しない。表示h範囲は新しい科学条件ではない。
資料は`artifacts/track_b_g10_v3_scientific_review_intake/2026-10-10/`、説明は`docs/manuscripts/track_b_return_aggregation_native_resource_note_v0_1.md`。
新規本番実行・G11は未認可、mandatory STOP維持。

## Track B DF sparse correction independent audit

`tracks/algorithm_codesign/audit_df_sparse_correction.py --phase A|B|CD`はexact log／synthetic CAR／有限平均境界の検査。`audit_df_sparse_correction_native.py`はsource Gaussian bridgeと5代表compileのみ。
[監査](../docs/tracks/algorithm_codesign/df_sparse_correction_independent_audit_20261011.md)、artifacts=`artifacts/track_b_df_sparse_correction_audit/2026-10-11/`。outputはexclusive create。本番runnerではなく、終了済みreceiptを上書きしない。新実行は今回で終了、mandatory STOP。

## DF correction comparison planning metadata validator

`tracks/algorithm_codesign/validate_df_correction_comparison_plan.py` はstdlibだけで入力hash/ZIP CRC/固定params/予算整合/source不変性を確認する。matrix/signal/compile/shot/T合成を行わず、新science PASSではない。receiptはexclusive create、既存receiptを上書きしない。 [契約](../docs/tracks/algorithm_codesign/df_correction_comparison_preregistration_20261011.md)。比較runner・実行認可は未準備、mandatory STOP。


## DF correction comparison v2 (Track B)

`tracks/algorithm_codesign/run_df_correction_comparison_v2.py`: future source-bound one-shot coordinator; current authorization false. `tracks/algorithm_codesign/verify_df_correction_comparison_source_v2.py`: stdlib metadata verifier, no registered fixture. See docs/tracks/algorithm_codesign/df_correction_comparison_execution_plan_v2_20261011.md.


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


## E-r3 identity-only observer修正 v2（2026-10-12）

[限定修正・技術検証](../docs/tracks/algorithm_codesign/e_r3_identity_observer_repair_preparation_20261012.md)：observer設置／復元失敗をtechnical STOPへ伝播。77 focused PASS、contract全bytes・expected・capsは不変。新sourceとv2 namespaceを固定し、READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION。旧S/科学結果/marker/authorization/STOP保持、登録診断/A作成0。追加GPT詳細研究レビューは不要、別明示one-shot承認までmandatory STOP。


## Track B E/T resumption v4（2026-10-12）

`tracks/algorithm_codesign/run_df_correction_et_resumption_v4.py` is a separately authorized E/T-only one-shot coordinator; CD saved-only, fresh marker/namespace, retry0/STOP. See [source report](../docs/tracks/algorithm_codesign/df_correction_et_resumption_preparation_20261012.md). No registered execution during preparation.


## Track B E/T storage v5（2026-10-12）

- `tracks/algorithm_codesign/run_df_correction_et_resumption_v5.py`：future source-bound E/T one-shot、別明示承認待ち。
- `tracks/algorithm_codesign/verify_df_correction_et_v5_storage_replay.py`：保存値／literal controlのI/Oだけ。登録matrix/compile/synthesisなし。
- `tracks/algorithm_codesign/verify_df_correction_et_v5_preparation.py`：contract/AST/旧証拠/pending拒否/source manifestのread-only照合。

[固定上限・validation・未検証事項](../docs/tracks/algorithm_codesign/df_correction_et_storage_v5_preparation_and_handoff_20261012.md)。
登録E/Tは未実行、旧v4 marker/STOP保持、mandatory STOP。


## Track B E/T v5実行後監査（2026-10-12）

`artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v2/audit_saved_outputs.py`は保存bytes/JSON/IRだけを照合する。
[結果報告](../docs/tracks/algorithm_codesign/df_correction_et_v5_results_and_gpt_handoff_20261012.md)：numerical guard失敗でTECHNICAL_INCONCLUSIVE、native10/T0、retry0。
旧/新consumed marker保持、runner再実行禁止、mandatory STOP。


## E/T numerical observer v6

`tracks/algorithm_codesign/run_df_correction_et_resumption_v6.py`：source-bound fresh version、guard式不変、失敗前観測情報保存。既定native22/T64、caps不変。[準備記録](../docs/tracks/algorithm_codesign/df_correction_et_numeric_v6_preparation_20261012.md)。


## E/T v7

`tracks/algorithm_codesign/run_df_correction_et_resumption_v7.py`：raw-phase accounting修正、新result v4。native22/T64/caps不変。


## E/T saved recovery

`tracks/algorithm_codesign/run_df_correction_et_saved_recovery.py`はv7の保存値だけをpostprocessする。新native/T/matrixを生成しない。元run分類/marker保持、別recovery namespaceで完了後STOP。


## Track B E/T 限定計算・保存値集計の完了（2026-10-12）

[最終技術監査・GPT handoff](../docs/tracks/algorithm_codesign/df_correction_et_completed_saved_accounting_and_gpt_handoff_20261012.md)：`SAVED_E_T_ACCOUNTING_COMPLETE`。
元v7でnative22／T key39（backend33、Clifford fast path6）を取得。大きい辞書の保存停止後、同じpost-T ASTをsaved-onlyで復元した。
元v7のTECHNICAL_INCONCLUSIVE／markerは不変。復元で新science0、815 saved-output checks PASS、全Aのactual Tはunknownを維持する。
CDは保存参照のみ、別E fixtureとの混合で全PR勝敗を作らない。研究継続・新規性・資源的意義はGPT判断、mandatory STOP。
復元結果：`artifacts/track_b_df_correction_et_saved_recovery_result/2026-10-12/v2/`、監査／manifest：`artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/`。


## Track B DF三次補正 E/T：GPT科学判断・現系列の区切り（2026-10-12）

[採択済みGPTレビュー](../docs/research/track_b_df_correction_et_scientific_review_20261012.md)／[系列の区切り・証拠索引](../docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md)。
現generic whole実装を新規PRアルゴリズムの主線として拡張せず、この検証系列を区切る。
3条件の局所固定加法planでA completion下界>B完全event T上界を保持。A全T、4行目、全PR優劣、一般whole優劣は未確定。
有限stepのA指数平均とB weighted線形平均は区別する。Bの自動採択・Track B全体の終了を意味しない。
元v7 TECHNICAL_INCONCLUSIVE／復元SAVED_E_T_ACCOUNTING_COMPLETE／旧結果・marker・sourceは不変。
今回のCodex作業は文書・索引・provenance・公開確認のみ、新科学計算0、mandatory STOP。
同じ科学レビューは反復しない。新候補の研究判断・scope採択はGPT側、新実行は別途明示認可が必要。

## Track B D1 development preparation（2026-10-13）

`tracks/algorithm_codesign/verify_rte_development_d1_source.py`は登録外focused validation。将来の別承認実行入口は`tracks/algorithm_codesign/observe_rte_development_d1.py`（内部runner `run_rte_development_d1.py`、worker `run_rte_development_synthesis_key.py`）。[技術報告](../docs/tracks/algorithm_codesign/rte_development_d1_source_preparation_20261013.md)。pending authorizationのため登録評価・合成を実行しない。

## D1保存scalar追加分析（2026-10-13）

`tracks/algorithm_codesign/analyze_d1_saved_resources.py`はpinned D1 JSONのFraction演算だけで順位・Pareto・factor分解・r差・affine感度を保存。scientific generator/runner/compile/synthesisを呼ばない。[技術報告](../docs/tracks/algorithm_codesign/d1_saved_resource_analysis_and_gpt_handoff_20261013.md)。

## Track B D2準備（2026-10-13）

`tracks/algorithm_codesign/analyze_d1_saved_key_mechanisms.py`は保存event/keyのみのT単価分解。`verify_d2_boundary_source.py`は登録外tests。`run_d2_saved_boundary_one_shot.py`と`observe_d2_saved_boundary_one_shot.py`は別の明示指示とsource直接子authorizationが必要。[契約/報告](../docs/tracks/algorithm_codesign/d2_key_mechanisms_and_boundary_preparation_20261013.md)。登録D2未実行、mandatory STOP。

## D2 saved-output audit

`tracks/algorithm_codesign/audit_d2_saved_boundary_result.py`はD2保存スカラー・hash/来歴だけの監査。pair kernel/科学runnerを再呼出ししない。[結果報告](../docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md)。

## Track B D2 final saved integration

`tracks/algorithm_codesign/analyze_d2_final_saved_integration.py`はD1r1/D2r2の保存scalar感度とPareto・depth分析だけ。[技術報告](../docs/tracks/algorithm_codesign/d2_final_saved_integration_and_gpt_handoff_20261013.md)。科学runner・boundary kernelを呼ばない。
