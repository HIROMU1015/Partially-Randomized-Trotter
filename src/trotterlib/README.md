> **M1-S Stage S技術準備（2026-10-17、未採択・実行未承認）**：[cap-edge契約候補](../../docs/tracks/algorithm_codesign/m1s_stage_s_cap_edge_contract_candidate_20261017.md)。linear H4 1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4。G3の16/G6 T4の8を主対象、G6 T1不成立8/G9絶対未解像16を保持。連続supremumと開端点floor+1を区別する新SOURCE `src/trottertracks/algorithm_codesign/m1s_stage_s_v1`、runner `scripts/tracks/algorithm_codesign/run_m1s_stage_s_v1.py`、checker `scripts/tracks/algorithm_codesign/check_m1s_stage_s_preparation_static.py`、人工tests `tests/tracks/algorithm_codesign/test_m1s_stage_s_preparation.py`、証拠 `artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1`。CPU/RAM/GPU/worker/timeに人工capなし、output/call/instruction bounds候補。保存再解析/科学shot/native/新authority・grant・marker0、guard closed、mandatory STOP。以下は履歴。

> **M1-S 次期整数shot・costの非数値準備（2026-10-17、未採択・未実行）**：[Decision Packet](../../docs/tracks/algorithm_codesign/m1s_shot_cost_feasibility_decision_packet_20261017.md)。linear H4 1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4。G3の16成立、G6 T4の8成立・T1の8不成立、G9の16絶対差未解像を保持。位相予算・整数ceil・全wrapper/HD/状態準備を含むbreak-evenを記号式で整理。source spec `src/trottertracks/algorithm_codesign/m1s_shot_cost_design_v1.py`、checker `scripts/tracks/algorithm_codesign/check_m1s_shot_cost_design_static.py`、証拠 `artifacts/track_b_m1s_shot_cost_preparation/2026-10-17/v1`。数値engine未実装、D1–D7未採択、科学計算/数値再解析/shot forecast/native compile/新authority・grant・marker0。旧成果保全、mandatory STOP。以下は履歴。

> **M1-S π/3限定照合完了（2026-10-17）**：[結果・監査](../../docs/tracks/algorithm_codesign/m1s_gamma_pi3_completed_results_and_stop_20261017.md)。別途ユーザー実施承認・専用Aa1dd87a1から初回完了。H4 linear1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4不変。G3の16、G6の8が成立、G6の8不成立、閾値未解像0、G9の16は絶対差未解像・未照合を保持。SOURCE2f800f17/Cbf97fcb2不変、root/pi/signal/整数shot/native/RPE総資源の新計算なし。条件付き未認証、科学GO未判断、旧成果保全、mandatory STOP。checker `scripts/tracks/algorithm_codesign/audit_m1s_gamma_pi3_saved_result.py`、artifact `artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01`。以下は履歴。

> **M1-S π/3限定gate正式採択（2026-10-17、実行未承認）**：[正式契約・採択記録](../../docs/tracks/algorithm_codesign/m1s_gamma_pi3_formally_adopted_contract_20261017.md)。ユーザーの次段階承認を受け、公開e3828965候補を別machine契約へ固定。SOURCE2f800f17・入力・32対象/G9の16未解像保持・判定・条件付きcoverage・resource・one-shot/failure STOPは不変。H4 linear1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4。contract_grants_execution=false、別明示承認が必要。新数値評価・authorization/grant/marker0、mandatory STOP。以下は履歴。

> **M1-S π/3限定gate準備（2026-10-17、未採択・未実行）**：[契約候補](../../docs/tracks/algorithm_codesign/m1s_gamma_pi3_interval_gate_contract_candidate_20261017.md)。保存G3/G6のγ境界区間と保存π区間の厳密比較SOURCE／closed guard／schemaを準備。G9未解像性を保持。H4 linear1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4は不変。新数値評価・閾値代入・authorization/grant/marker0。runner `scripts/tracks/algorithm_codesign/run_m1s_gamma_pi3_v1.py`、checker `scripts/tracks/algorithm_codesign/check_m1s_gamma_pi3_preparation_static.py`、SOURCE `src/trottertracks/algorithm_codesign/m1s_gamma_pi3_v1`、artifact `artifacts/track_b_m1s_gamma_pi3_preparation/2026-10-17/v1`。gate／契約採択と別実行承認が必要、mandatory STOP。以下は履歴。

> **M1-S γ依存V1完了（2026-10-17）**：[結果・修正来歴](../../docs/tracks/algorithm_codesign/m1s_v1_gamma_completed_results_and_stop_20261017.md)。ユーザーの修正・完了承認で旧失敗を保全し別S2/C2/A2から試行02が完了。H4 linear1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4・α0.05/p128/CPU160/wall320不変。72行評価・36zero保持・48対照分類、G3/G6各16tradeoff境界、G9絶対差未解像を保持。絶対shot/native/RPE総資源/科学GOなし。SOURCE `src/trottertracks/algorithm_codesign/m1s_saved_signal_gamma_v2`、runner `scripts/tracks/algorithm_codesign/run_m1s_saved_signal_gamma_v2.py`、tests `tests/tracks/algorithm_codesign/test_m1s_gamma_completion.py`、artifact `artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02`。条件付き未認証、旧成果保全、mandatory STOP。以下は履歴。

> **M1-S V1 γ依存**：[新SOURCE](../trottertracks/algorithm_codesign/m1s_saved_signal_gamma_v1)／[契約](../../docs/tracks/algorithm_codesign/m1s_v1_gamma_dependent_contract_candidate_20261017.md)／[登録外tests](../../tests/tracks/algorithm_codesign/test_m1s_gamma_preparation.py)。旧V0を保全した新namespace。連続十分shot因子のγ境界を扱い、整数strict利得・最適shot・native/RPE費用は保証しない。

> **M1-S V1 V0**：新SOURCEは[独立Track namespace](../trottertracks/algorithm_codesign/m1s_saved_signal_v1)、[契約候補](../../docs/tracks/algorithm_codesign/m1s_v1_saved_signal_contract_candidate_20261017.md)、[tests](../../tests/tracks/algorithm_codesign/test_m1s_v1_v0_static.py)。既存RPE helperのunit-radius／effective-eigenstate／wrapper価格を流用しない。保存ideal信号のみ、数値correctness未検証、execution closed。

> **M1-S有限step v2計算完了（2026-10-16）：[結果・修正来歴](../../docs/tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md)。** ユーザーの修正・完了承認の下で試行01の精度STOPを保全、Phiの既存Frobenius収縮性を誤差伝播へ適用し試行02で108/108行完了。H4 1.00 Å/STO-3G/8 modes/DF rank12・G3/G6/G9・保存oracle・delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1不変。correctness/全3twirl/精度PASS、条件付き未認証。G3/G6は各16絶対対照resolved higher、G9は16 unresolved。科学GO/native総資源未認定、旧成果・各試行・marker・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step v2起動前technical STOP（2026-10-16）：[不具合・未適用修正案](../../docs/tracks/algorithm_codesign/m1s_finite_signal_v2_pre_execution_configuration_stop_20261016.md)。** 別途ユーザー実施承認を受領したが、固定SOURCE fbc2edeaの精度文字列2^-24/2^-18をFractionへ直接渡す経路がValueErrorとなることをmetadata確認。H4 1.00 Å/STO-3G/8 modes/DF rank12・G3/G6/G9・delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1・保存oracle・108行・精度/runtimeは不変。科学/初回correctness/pilot起動0、one-shot未開始・未消費、authorization未発行・marker0。旧成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step信号pilot v2正式採択（2026-10-16）：[正式契約・採択記録](../../docs/tracks/algorithm_codesign/m1s_finite_step_complex_signal_pilot_v2_formally_adopted_20261016.md)。** ユーザーが68fa5580候補の条件付きcoverage・モデル差B/abs(T)B会計・108行grid・比較・runtime・one-shot/failure STOPを採択。H4 linear1.00 Å/STO-3G/8 modes/DF rank12・G3/G6/G9・保存数値oracle、delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1。SOURCE fbc2edea不変、platform/数値correctness未認証。採択契約は実行を許可しない。科学実施は明示未承認、新科学/数値test/authorization/grant/marker0、旧成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step信号pilot v2モデル差会計候補（2026-10-16）：[新契約](../../docs/tracks/algorithm_codesign/m1s_finite_step_complex_signal_pilot_contract_v2_candidate_20261016.md)／[v2 SOURCE](../trottertracks/algorithm_codesign/m1s_finite_signal_v2)／[runner](../../scripts/tracks/algorithm_codesign/run_m1s_finite_signal_v2.py)／[静的checker](../../scripts/tracks/algorithm_codesign/check_m1s_finite_signal_v2_static.py)。** H4 linear1.00 Å/STO-3G/8modes/DF rank12・G3/G6/G9・保存数値oracle、delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1・zero込み108行を維持。raw H_fullとH_tildeを区別、Bとabs(T)Bを別台帳、総raw-H biasと純アルゴリズム差の区間を区別。SOURCE fbc2edea。条件付き未認証・未採択・実施未承認、数値correctness未検証、新科学/数値test/authorization/grant/marker0、旧成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step信号pilot準備（2026-10-16）：[契約候補・SOURCE](../../docs/tracks/algorithm_codesign/m1s_finite_step_complex_signal_pilot_contract_candidate_20261016.md)。** 受領GPT独立結果レビューを原本保存。H4 linear1.00 Å/STO-3G/8spin orbitals/DF rank12・G3/G6/G9・保存数値oracle。delta候補1/8,1/16 Ha^-1、R2/4、T1/4 Ha^-1、同じ独立draw数のcoherent shared/ordinary/independent比較。SOURCE9758aeaa固定、複素信号/visibility/phaseとlambda²絶対尺度の設計。条件付きcoverage/精度/runtime・新target選択は未採択、新科学/Pauli/数値test/finite signal/authorization/grant/marker0。旧結果・STOP・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S独立one-shot完了（2026-10-16）：[結果・停止](../../docs/tracks/algorithm_codesign/m1s_independent_one_shot_results_and_stop_20261016.md)。** ユーザー別途実施承認・専用A2a55ba01、SOURCE4787cf90・採択C3ec5721bでrun1/retry0。H4 linear1.00 Å/STO-3G/8 spin orbitals/保存DF rank12・G3/G6/G9・deltaなし・保存数値oracle。全3分割の照合可能範囲PASS、support/sign差0、W/v/rho固定許容値内。未保存full-H/HDによる全scope UNDETERMINEDを保持。ローカル事後独立再現性照合でありplatform認証/科学GO/finite-step/native総資源評価ではない。旧v4/v5結果・SOURCE・契約・消費済marker・親/他worktree保全、追加科学なし、mandatory STOP。以下は履歴。

> **M1-S独立correctness契約を正式採択（2026-10-16）：[正式契約・採択記録](../../docs/tracks/algorithm_codesign/m1s_gate0_v5_independent_correctness_contract_adopted_20261016.md)。** ユーザーが2ba0858cの候補を採択。H4 linear1.00 Å/STO-3G/8 spin orbitals/保存DF rank12・G3/G6/G9・deltaなし・保存数値oracle、SOURCE4787cf90固定。照合可能範囲PASS/FAIL/UNDETERMINEDと、未保存full-H/HDを含む全scope UNDETERMINED（FAIL優先）を区別。精度・算術・runtime/caps不変、数値correctness未検証。科学実施は明示未承認、新科学/Pauli/数値test/W-v-rho/authority/grant/marker0、旧v4/v5成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5独立correctness準備（2026-10-16）：[検証契約候補・独立性と限界](../../docs/tracks/algorithm_codesign/m1s_gate0_v5_independent_correctness_contract_candidate_20261016.md)。** 公開草稿d6d35cfbを別worktreeで再利用、SOURCE4787cf90固定。H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/G6/G9・deltaなし・保存数値oracle。occupation/CAR/全Pauli trace/tensor再構成、exact rational W/v/rho、結果JSONはreference固定後のみ照合。旧full-H/HD行列未保存分はUNDETERMINED、全scope PASS不可。契約未採択・実施未承認・数値correctness未検証、新科学/実入力・模擬input数値test/authority/grant/marker0、旧v4/v5成果・marker・元Codex/親worktree保持、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5初回one-shot（2026-10-16）：[結果・停止](../../docs/tracks/algorithm_codesign/m1s_gate0_v5_one_shot_results_and_stop_20261016.md)。** 別ユーザー実施承認・固定A8da63d67から一回実行。H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・保存HD G3/G6/G9・deltaなし。全3行で局所W/v予算とrho精度を通過、W/v/rho取得、到達1,239gate PASS。HEADROOM_DIAGNOSTICS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。同一development条件での事後技術改訂評価、platform/true-ground/finite-step/native総資源/科学GOは未認証・未判断。旧v4 STOP/null/marker保持、one-shot1/retry0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5正式採択（2026-10-16）：[正式契約と採択記録](../../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v5_formally_adopted_20261016.md)。** ユーザーがH4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/G6/G9・deltaなし、局所alpha_R=gamma_(713+r)S_R・17精度条件・1e-8 Ha+B_res・条件付きcoverage・runtime/resource・all-splits/one-shot/failure STOPを採択。SOURCE69de022/proof b5a185cc固定、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。採択flagはplatform認証ではない。契約完備性と実行許可を分離、科学計算は明示未承認。新科学/予算評価/Pauli/数値test/authority/grant/marker0、旧v4 STOP/null/marker保持、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5 SOURCE接続（2026-10-16）：[最終実験契約候補](../../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v5_candidate_20261016.md)／[v5 SOURCE](../../src/trottertracks/algorithm_codesign/m1s_gate0_v5)／[runner](../../scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v5.py)／[static checker](../../scripts/tracks/algorithm_codesign/check_m1s_gate0_v5_static.py)／[監査・機械契約](../../artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1)。** H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9・deltaなし。tail alpha_R=gamma_(713+r)S_R resolver接続、global/local ledger分離、17条件/runtime維持。SOURCE69de022固定、122 static PASS、実入力新式/Pauli/数値test/W-v-rho/Gate0評価0。候補未採択・coverage採択false・実施未承認、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。旧v4 TECHNICAL_STOP/null/消費済marker保持、新authority/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 tail局所上界（2026-10-16）：[条件付き証明](../../docs/tracks/algorithm_codesign/m1s_gate0_tail_local_bound_proof_20261016.md)／[v5精度設計候補](../../docs/tracks/algorithm_codesign/m1s_gate0_tail_local_precision_contract_v5_design_candidate_20261016.md)／[宣言的spec](../../src/trottertracks/algorithm_codesign/m1s_gate0_tail_bound_spec.py)／[静的checker](../../scripts/tracks/algorithm_codesign/check_m1s_gate0_tail_bound_static.py)／[証拠台帳](../../artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1)。** H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9・deltaなし。SOURCEのgroup長とDAGからalpha_R=gamma_(713+r)S_Rを条件付き導出、index722/719/716。旧2^-24を維持できる解析的十分領域を証明。53 static PASS、数値評価0。v5はSOURCE未接続・非実行の未採択設計候補、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。旧v4 TECHNICAL_STOP/null/marker、SOURCE/契約/精度ceiling/runtime保持、Gate0再実行0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 精度原因調査（2026-10-16）：[項別予算・改善候補](../../docs/tracks/algorithm_codesign/m1s_gate0_precision_budget_cause_and_options_20261016.md)／[保存データscalar診断](../../artifacts/track_b_m1s_gate0_precision_cause/2026-10-16/v1)／[診断SOURCE](../../scripts/tracks/algorithm_codesign/analyze_m1s_gate0_precision_saved.py)。** H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9・deltaなし。承認済み保存再解析1、4 NPZ配列のexact尺度と3×8 scalarケースを評価。全H共通係数上界とlambda正規化が支配、tail局所化だけでは不足、局所DAGとの組合せに未証明の改善余地。新Pauli/state/matrix/W-v-rho/Gate0実行0。旧TECHNICAL_STOP/null・17条件・SOURCE/契約/authorization/marker保持、候補未採択、CONDITIONAL_ENGINEERING_NOT_CERTIFIED、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v4初回one-shot（2026-10-16）：[結果・技術停止](../../docs/tracks/algorithm_codesign/m1s_gate0_v4_one_shot_results_and_stop_20261016.md)／[raw result](../../artifacts/track_b_m1s_gate0_result/2026-10-16/v4_one_shot/result.json)／[保存監査](../../artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot)／[saved-only auditor](../../scripts/tracks/algorithm_codesign/audit_m1s_gate0_v4_saved_result.py)。** 別ユーザー実施承認・固定Aから一回実行。H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9、delta評価なし。到達132gate PASS、残差品質PASS。全splitで事前W/v誤差予算超過、W/v/rho未評価null、TECHNICAL_STOP / PRECISION_UNRESOLVED。17条件・SOURCE・入力は不変。CONDITIONAL_ENGINEERING_NOT_CERTIFIED、metric correctness未到達、科学的GO/no-go未判断。one-shot1/retry0、marker消費済み、旧成果・親worktree保持、mandatory STOP。以下は各段階の履歴。

> **M1-S Gate 0 v4正式採択（2026-10-16）：[正式v4契約](../../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v4_formally_adopted_20261016.md)／[正式machine契約](../../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/execution_contract_v4_adopted.json)／[ユーザー採択記録](../../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/user_final_contract_adoption_record.json)／[採択監査](../../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1)／[採択static checker](../../scripts/tracks/algorithm_codesign/check_m1s_gate0_v4_adoption_static.py)。** ユーザーが17項目・1e-8 Ha+B_res・条件付きcoverage・runtime/caps・one-shot/failure STOPを正式採択。H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・saved G3/6/9、delta評価なし、SOURCE/入力/精度条件は候補と同一。CONDITIONAL_ENGINEERING_NOT_CERTIFIED、runtime/数値correctness未検証。契約完備性と実行許可を分離し、科学計算は明示的に未承認。新science/数値test/formula評価/実行authorization/grant/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v4契約候補（2026-10-16）：[v4最終契約候補](../../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v4_candidate_20261016.md)／[品質/coverage説明](../../docs/tracks/algorithm_codesign/m1s_gate0_v4_oracle_residual_quality_and_coverage_20261016.md)／[GPT原本](../../docs/research/track_b_m1s_gate0_v3_gpt_limited_scientific_review_20261016.md)／[v4 SOURCE](../trottertracks/algorithm_codesign/m1s_gate0_v4)／[runner](../../scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v4.py)／[static checker](../../scripts/tracks/algorithm_codesign/check_m1s_gate0_v4_static.py)／[監査資料](../../artifacts/track_b_m1s_gate0_v4_contract/2026-10-16/v1)。** H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・saved G3/6/9、delta評価なし。残差gateを1e-8 Ha+B_resへ改訂、他8式/8scalar/runtime保持。全17項目が候補化、未採択。CONDITIONAL_ENGINEERING_NOT_CERTIFIED、99非数値static PASS、runtime/数値correctness未検証。新science/Pauli/数値test/formula評価/authority/grant/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 誤差算定式の契約準備（2026-10-16）：[v3最終契約候補](../../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v3_candidate_20261016.md)／[式・coverage条件](../../docs/tracks/algorithm_codesign/m1s_gate0_preregistered_error_formula_design_20261016.md)／[v3 SOURCE](../trottertracks/algorithm_codesign/m1s_gate0_v3)／[runner](../../scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v3.py)／[static checker](../../scripts/tracks/algorithm_codesign/check_m1s_gate0_formula_static.py)／[監査資料](../../artifacts/track_b_m1s_gate0_formula_contract/2026-10-16/v1)。** H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・saved G3/6/9、delta評価なし。未決9件中8件を条件付き式候補として具体化、oracle残差品質ηは未決。17項目・runtime/caps維持、68非数値static PASS、数値correctness/coverage未検証、全契約未採択・実行不可。新science/Pauli表/数値test/formula評価/authority/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 scope正式採択・精度契約案（2026-10-16）：[最終契約v2候補](../../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v2_candidate_20261016.md)／[精度17項目](../../docs/tracks/algorithm_codesign/m1s_gate0_precision_error_budget_design_20261016.md)／[runtime/caps](../../docs/tracks/algorithm_codesign/m1s_gate0_runtime_and_resource_contract_candidate_20261016.md)／[v2 module](../trottertracks/algorithm_codesign/m1s_gate0_v2)／[runner](../../scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v2.py)／[追加static checker](../../scripts/tracks/algorithm_codesign/check_m1s_gate0_precision_static.py)／[scope採択記録](../../artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/gate0_scientific_scope_adoption_record.json)。** H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/6/9・Pauli law/oracle比較のみユーザー採択。精度8値を候補化、9絶対値未決。追加67静的PASS、旧SOURCE/成果保持、数値correctness未検証、全契約未採択・科学未承認、新科学/Pauli表/authority/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 SOURCE準備（2026-10-16）：[実験契約最終候補](../../docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_candidate_20261016.md)／[専用module](../trottertracks/algorithm_codesign/m1s_gate0)／[runner](../../scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom.py)／[非科学的checker](../../scripts/tracks/algorithm_codesign/check_m1s_gate0_source_static.py)／[監査資料](../../artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1)。** 保存H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/6/9のadapterとW/v/ρ SOURCEを実装。30静的確認PASS、数値correctness未検証。IMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED、契約未採択・実行未承認、新科学/Pauli表/authority/marker0、mandatory STOP。以下は履歴。

> Track Bの[KKT訂正・S4補完](../../docs/tracks/algorithm_codesign/parent_allocation_s4_kkt_repair_and_supplement_preparation_20261015.md)は別namespace `parent_experiment/supplement.py`。共通libraryは不変、旧S3と全result保持、science承認待ち。

> Track Bの[Parent allocation v3結果/監査](../../docs/tracks/algorithm_codesign/parent_allocation_v3_results_and_gpt_handoff_20261015.md)は別namespace。共通libraryは保持、科学的採否は未判断、mandatory STOP。

> Track B実験のParent専用保存helperは `src/trottertracks/algorithm_codesign/parent_experiment/evidence.py`。既存共通writer/科学会計を保持する。

> Track Bの[Parent allocation v2準備](../../docs/tracks/algorithm_codesign/parent_allocation_v2_owner_repair_source_preparation_20261015.md)は別namespaceのowner metadata修正のみ。共通library・科学条件は不変。

> Track Bの[Parent allocation実行source](../trottertracks/algorithm_codesign/parent_experiment/README.md)は別namespace。共通library・旧adapter/sourceは保持。[実行準備報告](../../docs/tracks/algorithm_codesign/parent_allocation_execution_source_preparation_20261015.md)、登録比較は別承認待ち。

> Track Bの[Parent allocation native bridge](../trottertracks/algorithm_codesign/parent_allocation_native.py)は別namespace。旧cache metadataとstrict phase/errorを接続し、登録sourceは変更しない。[Decision Packet](../../docs/tracks/algorithm_codesign/parent_allocation_scientific_design_decision_packet_20261014.md)。

> Track Bの[Parent allocation adapter](../trottertracks/algorithm_codesign/parent_allocation.py)は別namespace `trottertracks.algorithm_codesign` にある。共通libraryは変更しない。ordinary/closed P5共通rootと登録外検証、科学runner/合成はなし。[技術報告](../../docs/tracks/algorithm_codesign/parent_allocation_source_preparation_and_gpt_handoff_20261014.md)。

> **Track B D3 v2結果**：[独立trackの保証×価格resource map](../../docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md)。共有libraryのAPI/実装は変更せず、追加scienceなし、mandatory STOP。

> **Track B D3 v2**：[独立track namespaceの型adapter/source準備](../../docs/tracks/algorithm_codesign/d3_v2_precision_repair_and_source_preparation_20261014.md)。この共有libraryのAPI/実装は変更していない。別明示実行承認待ち、mandatory STOP。

> Track B D3の新sourceは[専用namespace](../trottertracks/algorithm_codesign/rte_robustness/README.md)にある。共通API変更なし。[技術報告](../../docs/tracks/algorithm_codesign/d3_source_preparation_and_execution_handoff_20261013.md)、科学実行0、別指示までSTOP。

> Track B D3は既存B sourceの静的照合と[結果前計画](../../docs/tracks/algorithm_codesign/d3_guarantee_price_preregistration_v1_20261013.md)のみ。共通library/Bの旧科学source変更なし、新execution codeは未実装。

> 2026-10-07 Track B RA-D0 v3：**READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION**（実行承認ではない）。
> [source review](../../docs/tracks/algorithm_codesign/ra_d0_source_review_v3_20261007.md) / [GPT handoff](../../docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_v3_20261007.md) / [manifest](../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/evidence_manifest_v3.json)。
> [focused verifier](../../scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v3.py) / [v3 tests](../../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v3.py)。
> exact-certified B2 minimum infeasibilityを正常outcomeに修正。pointをfreezeへ記録しbudget/queryを空にして次nへ進む。
> uncertified failureはtechnical STOP。数値・candidate・grid・call/resource capは維持。登録最適化0、authorizationなし、mandatory STOP。

> 2026-10-07 Track B RA-D0 v2：**READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW**。
> [source review](../../docs/tracks/algorithm_codesign/ra_d0_source_review_v2_20261007.md) / [GPT handoff](../../docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_20261007.md) / [evidence manifest](../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/evidence_manifest_v2.json)。
> [future runner](../../scripts/tracks/algorithm_codesign/run_ra_d0_one_shot.py) / [focused verifier](../../scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v2.py) / [v2 tests](../../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v2.py)。
> B0_saved/ideal分離、数値B1⊂B2⊂B3、profile-paired budget、batch freeze-before-B3、
> anchor-first、main LP 55,275 / auxiliary込み110,550、resource/launch guardを固定。
> 登録最適化・実budget/minimum/witness取得0、authorizationなし。旧本文・旧STOP・Track Aは保持。mandatory STOP。

## Track B RA-D0 static preparation / mandatory STOP（2026-10-06）

[source review](../../docs/tracks/algorithm_codesign/ra_d0_source_preparation_review_v1.md)：21 columns/x、18 sign pairs一致、35 focused tests PASS。
B専用namespace `src/trottertracks/algorithm_codesign/ra_d0/`、static generator／verifier、
`artifacts/track_b_ra_d0_preparation/2026-10-06/`に候補・grid・semantic audit・provenanceを保存。
ideal nestingと数値membershipを区別し、query実行scope／資源上限をGPTへ返す。
登録最適化／新合成／新science0、RUN_READY=false、既存分類・Track A・旧STOP保持。
**mandatory STOP。以下の既存本文を全文保持する。**

## Track B RA-RTE統合数学監査・mandatory STOP（2026-10-06）

R1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942` 基点、GPT設計案へのDOCS_SYMBOLIC_ONLY_MATHEMATICAL_AUDIT。
[命題別監査](../../docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_v1.md)と[GPT handoff](../../docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_gpt_handoff_20261006.md)：一block／finite table／canonicalのfixed-n LPは仮定付きで成立。
shot-gridの固定total cap保存には反例。log/root・sampler認証、Delta=0、peak workspaceの規約を実行前修正へ返す。
[stdlib人工bookkeeping](../../scripts/tracks/algorithm_codesign/check_ra_rte_mathematical_bookkeeping.py)の[50 checks](../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_checks_v1.json)を一般証明と分離した。
[manifest](../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/evidence_manifest_v1.json)、[dated note](../../docs/research/研究ノート/2026-10-06_track_b_ra_rte_mathematical_audit.md)。science/synthesis/solver/資源再採点0、共通API変更0。
既存科学分類・結果・STOPは不変。性能・新規性・algorithm採択・次実装／R2 authorizationは未確定。
**mandatory STOP。次の採択・実装・pilotの必要性／範囲はGPT判断。以下の既存本文を全文保持する。**

## Track B R1.5保存値帰属・mandatory STOP（2026-10-06）

input R1 commit `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b` の保存値だけを用いたPOSTHOC attribution / design input。
[帰属報告](../../docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md)と[GPT handoff](../../docs/tracks/algorithm_codesign/r1p5_gpt_handoff_20261006.md)：新science/synthesis/compile/候補追加0、R1科学分類は不変。
primaryは2-qubit finite P₃、distinct-basis controlled、x={1/8,1/4}、登録native三precision。
Aは登録(G_T,G_CX,G_1Q) frontにx=1/8の1e-4、x=1/4の1e-3/1e-4で残る。
normalizationだけでなくnative費用とbias/shotの関係を整理し、固定合成列への依存も保存した。
[stdlib保存値解析](../../scripts/tracks/algorithm_codesign/analyze_r1p5_saved_attribution.py)、[全summary](../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)、[provenance manifest](../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/evidence_manifest_v1.json)、
[日付note](../../docs/research/研究ノート/2026-10-06_track_b_r1p5_saved_attribution.md)。共通library変更・独立validation・新algorithm採択はない。
限定診断SUPPORTS_RA_RTE_DESIGNは設計入力のみ。eta探索/R2/DF接続/追加scienceは未認可。
**mandatory STOP。次の数学設計・研究方針判断はGPT側。以下の既存本文を全文保持する。**

## Track B R1一回結果・mandatory STOP（2026-10-06）

固定S `d43d64a821a0249a0dfab12a2472bd3a72fdee74` →直接子authorization-only A
`411f08f768244fe87b600d82308c3851847fe9e4`からrun1/retry0。
[結果照合](../../docs/tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md)：126 keys / 264 rows / 132 controlled tasks完了、全task適格。
terminal R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW、[保存field専用監査](../../scripts/tracks/algorithm_codesign/audit_r1_saved_result.py) PASS。
primary distinct-basis controlledではB²改善とnative/shot資源のtrade-offを保存し、自動研究GOはない。
[全264 rows CSV](../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)、[evidence manifest](../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)、
[GPT判断への入口](../../docs/tracks/algorithm_codesign/r1_post_run_gpt_review_request_20261006.md)。原result/marker/source、既存証拠・共通API・Track A保持。
science終了後mandatory STOP、追加合成/target/grid/分子/DF/trajectory/GPUは行わない。
研究方針・RQ・新規性・着地点・追加検証の必要性/範囲はGPT側。
以下のpending/最新記述は当時の履歴として本文をそのまま保持する。

# trotterlib モジュール索引

## Track B R1 builder is a separate namespace

[R1専用実装](../trottertracks/algorithm_codesign/rte_reallocation/)を新設し、共有library/既存even RTE invariantを変更しない。
[preregistration/source review](../../docs/tracks/algorithm_codesign/rte_reallocation_r1_source_review_request_20261006.md)。
27 focused synthetic semantic/accounting/launch tests。DF controlled wrapperのvalidationではない。
登録synthesis/resource取得0、science未認可、mandatory STOP。


## Track B R0.5: shared API unchanged

[R0.5監査](../../docs/tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md)と
[GPT handoff](../../docs/tracks/algorithm_codesign/rte_reallocation_r05_gpt_handoff_20261006.md)を追加。
新checkerはdocs/symbolic技術監査専用。library/共通API/既存RTE実装は変更しない。
odd eventのnative/controlled実装は未検証。R1未認可、mandatory STOP。


## Track B R0 boundary (2026-10-06)

[Finite-mean reallocation audit](../../docs/tracks/algorithm_codesign/rte_reallocation_r0_review_packet_20261006.md) uses a standalone exact checker. Shared RTE/DF APIs are unchanged; RTEEvent remains even-order only. Odd-event algebra is not a validated builder. No track-specific copy of trotterlib is created.


Track B [BS-0.5 ordinary形式仕様](../../docs/tracks/algorithm_codesign/bs05_ordinary_finite_rte_baseline_v1.md)はdocs-only。
固定rte.pyをtext照合しただけで、共有implementation/APIを変更・実行していない。
operator/channelは別target、現candidateの独立delta未定義。実装/science/tests0、mandatory STOP。以下は履歴。

Track B [SP-1後block合成の数学/API仕様](../../docs/tracks/algorithm_codesign/block_synthesis_design_review_20261006.md)はdocs-only。
channel用Gateとoperator LCUを別型にする設計案。新namespace/コードは未作成、共有library/API変更なし。
current rte.pyは固定git blobのtext参照だけで呼び出さない。実装/science/tests0、mandatory STOP。以下は既存履歴。

Track B [SP-1一回結果・保存値監査](../../docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)は
固定`../trottertracks/algorithm_codesign/synthesis_placement/`の実行記録。
48 rows／96 axesを完了、mandatory STOP、retry0。共有trotterlib実装/APIは変更していない。
[保存field audit](../../scripts/tracks/algorithm_codesign/audit_sp1_saved_result.py)はscience moduleをimportせず照合する。
actual DF/RTE wrapperやcompiled費用の検証ではない。次の研究判断はGPT側。
以下のsource準備・未実行記述は結果前履歴として保持する。

Track B [SP-1結果前契約](../../docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)の
sequence/adapter/accounting/launch/result modulesは`../trottertracks/algorithm_codesign/synthesis_placement/`。
共有trotterlibをimport/変更せず、保存B primitiveを入力にしたsynthetic adapterと59 focused tests。
登録science sweep0、実行authorization pending。共通APIの採択やDF wrapper検証ではない。
以下の未完成記述は前段履歴として保持する。

Track B [SP-1準備](../../docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_proposal_v1.md)の
[wrapper_accounting.py](../trottertracks/algorithm_codesign/synthesis_placement/wrapper_accounting.py)は
共通libraryから独立したexact Fraction会計fixture kernel。17 focused testsはlocal pass。
共有API変更なし、science入力／合成呼出し0。actual wrapper adapter／interval guard／science runnerは未完成。
以下は元のSP-0.5準備履歴。現在のSP-0.5結果は別の固定result commitを参照する。

Track Bの[SP-0.5経済性gate](../../docs/tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md)は
`../trottertracks/algorithm_codesign/synthesis_placement/`の新namespaceに置く。
gate-string合成／interval guard／T×weight会計だけを扱い、共有library実装・DF wrapperをimportしない。
34 focused technical tests pass、登録target計測は未実行・未認可。


Track Bの[BM-0.5記号監査](../../docs/tracks/algorithm_codesign/bm05_review_packet_20261005.md)は
共有libraryをimportせず、[専用script](../../scripts/tracks/algorithm_codesign/audit_bm05_symbolic_equivalence.py)の
抽象words／Fractionだけで実施する。library module・API・science runnerの変更なし、BM-1未認可。

Track Bのexperimental codeは共有APIを変更せず、`../trottertracks/algorithm_codesign/`へ置く。
[BF-1 preparation](../../docs/tracks/algorithm_codesign/README.md)のCPU mean adapterであり、
DF controlled wrapperの検証・scientific run・algorithm採択は未実施。

このディレクトリが実装本体である。`scripts/` は主にここにある関数を呼び出す実行入口、
`tests/` は回帰検査である。研究全体の入口は[`../../PROJECT_MAP.md`](../../PROJECT_MAP.md)を参照する。

## 現行研究の中心実装

### Hamiltonian・DF分割・基底状態

- `chemistry_hamiltonian.py`、`df_hamiltonian.py`：分子HamiltonianとDF表現
- `df_partial_randomized_pf.py`：DF分割、fragment変換、state-action PF係数計算の共通基盤
- `df_trotter/`：決定論DF回路とstate-actionの低レベル実装
- `df_gpu_statevector.py`：大きめの系を想定したstate-action支援
- `io_cache.py`：入力・計算結果の保存と読込み

### partial-$S_2$・finite RTE

- `df_partial_s2.py`：決定論half sweepとRTE中央部を持つpartial-$S_2$。通常weight-ranked prefixに加え、
  保存順・fragment内容を検証する結果前explicit ordered-partition adapterを提供
- `df_partial_s2_repeated.py`：partial-$S_2$の反復
- `df_rte_tail.py`：ランダム側HamiltonianのI/Z/ZZ表現
- `df_rte_circuit.py`、`df_rte_qiskit.py`：RTEイベント回路構築
- `rte.py`：有限RTEの確率、打切り、誤差計算の基礎

### コンパイル後回路コスト

- `rte_compiled_cost.py`：イベント列の構築、コンパイル、metric集計
- `df_partial_s2_cost.py`、`df_partial_s2_repeated_cost.py`：partial-$S_2$のcost集計
- `rte_connected_cluster_cost_validation.py`：局所境界補正の較正・予測・独立検証
- `rte_boundary_cost_validation.py`、`rte_boundary_pair_validation.py`：境界補正の基礎検証
- `rte_order_stratified_cost_validation.py`：Taylor次数の条件付き抽出
- `rte_system_size_cost_validation.py`：系サイズ方向の検証
- `rte_cost_angle_invariance_validation.py`：回転角とmetric cacheの診断

### RPE

- `rpe_hadamard_interrogation.py`：X/Y Hadamard interrogation回路
- `df_rpe_resource.py`、`rpe_resource_accounting.py`：誤差・shot・1 shot costの資源集計
- `rpe_short_round_optimization.py`：短いRPE段の候補選択
- `df_rpe_hadamard_compiled_cost.py`：Hadamard回路のコンパイル後cost
- `rpe_hadamard_compiled_cost_proxy.py`：反復数方向のcost proxy
- `rpe_hadamard_validated_proxy_provider.py`：検証範囲を守ってproxyを供給

## 共有サーバー向け実行基盤

- `parallel_validation_executor.py`：決定論的task ID、資源guard、bounded subprocess、
  atomic checkpoint、resume、GPU job割当、決定論的集約

科学ロジックや科学的判定は含めない。運用方法と安全上の既定値は
[`../../docs/server_parallel_validation_execution.md`](../../docs/server_parallel_validation_execution.md)を参照する。

## 検証モジュール

次のモジュールは、研究用APIそのものではなく、条件固定、比較、判定、成果物生成を担当する。

- `finite_rte_signal_validation.py`
- `pr2_s0_s1_validation.py`：PR-2のsnapshot freeze、prefix identity、corrected estimator、
  S1 correctness-only artifactと強制停止
- `pr2_new_series_validation.py`：別snapshot系列のV1 integrity/model、V2 rank 3/6/9構造、
  operation counter、V4非承認guard
- `pr2_v4_s2_development_validation.py`：別snapshot系列のV4 correctnessとdevelopment-only
  S2 resource comparison、32→96拡張規則、mandatory STOP
- `pr2_v4_s2_parallel_execution.py`：同じS2 compile cellと段階barrierを維持するbounded
  spawned-process実行、canonical result order、persistent compiled-cost cache
- `pr2_matched_accuracy_m1_contract.py`：standard-library-onlyのM1候補台帳、candidate fingerprint、
  occurrence seed、16-cell selector、zero-compute dry-run validator。科学計算moduleをimportしない
- `pr2_matched_accuracy_m1_precompile_barrier.py`：M1-A selectorを検証し、`SELECTION_LIMITED`なら
  M1-B compile plan生成前に停止するstandard-library-only hard barrier
- `pr2_matched_accuracy_m1_execution.py`：保存済みdevelopment H4だけを一回読み、Qiskit circuitを作らず
  base 208＋r64最大4候補のsignal、normalization、analytic shot、selector、hard barrierを評価するM1-A経路。
  実結果は210候補、52未選択frontierで`SELECTION_LIMITED`となり、M1-Bを開始していない
- `pr2_matched_accuracy_m1_b1_contract.py`：byte-fixed M1-A結果からaccuracy適格random 194 cellとB0/B1
  16 cellを固定し、paired-axis trajectory seed、source/compiler/candidate/axis/trajectoryを含む12,448 wrapper
  cache identityを生成・検証するstandard-library-only経路。M1-B1科学実行は行わない
- `pr2_matched_accuracy_m1_b1_execution.py`：source commit固定後のplan/authorizationだけを受け、194 random
  cell×32 trajectory×二軸と16 baseline cell×二軸を最大6 spawned workersでcompileするM1-B1経路。
  candidate別cacheとexact checkpoint identityを用い、compile map完成後は研究判断をせずreview待ちで停止する
- `pr2_matched_accuracy_m1_b1_result_validation.py`：保存済みM1-A/M1-B1 artifact、全checkpoint、
  candidate別SQLite cacheをread-only検査し、全集約値、actual Pareto、fixed-q=8差、旧selector、proxy相関、
  状態準備感度を再計算する結果検証。development/held-out分子snapshotを読まない
- `pr2_matched_accuracy_m2_transfer_contract.py`：検証済みM1-B1 JSONだけからB2 actual Pareto二件と
  B0/B1/B3代表を固定し、primary RZ、6指標Pareto、10% materiality、重大underestimate、4 terminal status、
  future seedと196-wrapper上限をzero-compute固定する。v2はusable B2だけでPareto support/ratioを判定し、
  v1証拠を保存する。held-out snapshotは開かず、M2実行を認可しない
- `pr2_matched_accuracy_m2_transfer_execution.py`：別commitのsource-bound authorization照合後だけ、
  固定5構成のheld-out signal/paired-axis full-wrapper costを最大5 workers、196-wrapper上限で評価する。
  usable B2判定、paired covariance、checkpoint identityを検査し、全status後mandatory STOP。
  最終review承認と利用者指示を得た一回実行は`TRANSFER_SUPPORTED`で完了し、
  [保存済み結果照合](../../docs/pr2_matched_accuracy_m2_transfer_result_validation.md)後に停止した。追加計算・再実行は未認可
- `pf_delta_validation.py`
- `pf_c_system_size_validation.py`
- `pauli_partial_cgs_validation.py`
- `random_circuit_cost_validation.py`
- `hierarchical_cost_validation.py`
- `rpe_round_cost_connection_validation.py`
- `rpe_hadamard_failure_validation.py`
- `rpe_hadamard_proxy_resource_validation.py`
- `rpe_allocation_sensitivity_validation.py`
- `rpe_four_round_accounting_validation.py`
- `rpe_four_round_phase_validation.py`
- `rpe_target_round_horizon_validation.py`
- `rpe_delta_round_schedule_validation.py`
- `rpe_delta_compiled_cost_validation.py`
- `rpe_hadamard_compiled_cost_benchmark.py`
- `research_direction_prevalidation.py`
- `research_direction_ablation.py`
- `research_direction_pf_sensitivity.py`
- `research_direction_gate_s1.py`
- `research_direction_structure_pilot.py`
- `research_direction_sequence_policy.py`
- `research_direction_full_scope.py`
- `research_direction_full_scope_extension.py`
- `research_direction_full_scope_replication.py`
- `research_direction_decision_cost.py`
- `research_direction_decision_synthesis.py`
- `research_direction_round_dominance.py`
- `research_direction_proxy_precision.py`
- `research_direction_m08_reaggregation.py`
- `research_direction_compiler_transfer_compute.py`
- `research_direction_compiler_transfer_analysis.py`
- `research_direction_uncertainty_break_even.py`
- `research_direction_wp11_synthesis.py`
- `research_direction_full_opt2.py`
- `research_direction_full_opt2_completion.py`
- `research_direction_full_opt2_extension_analysis.py`
- `research_direction_proxy_lineage_reconciliation.py`
- `research_direction_signal_weight_pilot.py`
- `research_direction_geometry_energy_difference_pilot.py`
- `research_direction_geometry_tracking_breakdown.py`
- `research_direction_joint_synthesis_pilot.py`
- `research_direction_theme_selection.py`
- `research_direction_joint_synthesis_blind_validation.py`
- `research_direction_joint_synthesis_formalization.py`
- `research_direction_joint_synthesis_mechanism_validation.py`

- `research_direction_energy_tail_pareto.py`
- `research_direction_pd_realization.py`
- `research_direction_pd_fair_comparison.py`
- `research_direction_pd_s1_posthoc.py`
- `finite_rte_phase_amplitude.py`
- `fr_revision_fr1a_posthoc.py`
- `fr_revision_nonuniform.py`
同名のrunner、test、文書、artifactと合わせて読む。モジュールが実装済みでも、対象範囲が
科学的に検証済みとは限らない。

## 解析モデルと旧経路が同居するモジュール

- `partial_randomized_pf.py`：誤差配分・解析cost modelの共通関数と旧Pauli screening経路
- `df_screening_cost.py`、`df_cost_plotting.py`：DF screening・可視化

これらは現行コードから共通関数を利用する場合がある一方、保存済みの旧screening結果は
失効している。モジュール全体を旧式とみなすのでも、既存出力を現行結果とみなすのでもなく、
利用する関数とartifactのstatusを個別に確認する。

## 旧高次PF・比較用実装

次はプロジェクトの出発点である高次積公式、UWC、旧screening経路を支える。現行の
DF部分ランダム化研究と混同しない。

- `optimal_trotter.py`、`product_formula.py`、`pf_decomposition.py`
- `Almost_optimal_grouping.py`
- `qiskit_time_evolution_grouping.py`、`qiskit_time_evolution_ungrouped.py`
- `qiskit_time_evolution_pyscf.py`、`qiskit_time_evolution_utils.py`
- `plots_timeevo_error.py`、`cost_extrapolation.py`、`rz_layers.py`
- `uwc.py`、`grouped_uwc_comparison.py`、`grouped_uwc_theta_sweep.py`

これらの結果を引用するときは、先に
[`../../VALIDATION_STATUS.md`](../../VALIDATION_STATUS.md)で失効・欠落状態を確認する。

## 共通支援

- `config.py`：共通設定
- `analysis_utils.py`、`plot_utils.py`：解析・可視化支援
- `matrix_multiply.py`、`matrix_pf_build.py`：行列ベースの参照計算
- `eig_error.py`、`qpe_beta.py`：固有値誤差・位相推定の旧来支援

新しい機能は、runnerへ直接大きな処理を書くのではなく、ここへテスト可能な関数として置く。

## Track B G1限定技術sourceへの参照

G1のexact symbolic kernel・audit・controllerは
[専用script namespace](../../scripts/tracks/algorithm_codesign/g1_decision_packet/README.md)へ置いた。
共有library/APIを変更する機能ではない。
[source review](../../docs/tracks/algorithm_codesign/g1_source_review_20261009.md)、
[入口](../../scripts/tracks/algorithm_codesign/run_g1_decision_packet.py)、
[focused tests](../../tests/tracks/algorithm_codesign/test_g1_source_preparation.py)、
[準備artifact](../../artifacts/track_b_g1_source_preparation/2026-10-09/evidence_manifest_v1.json)を参照する。
本構造監査・固定8人工LPは別明示指示まで未実行。科学的B2/B3比較、backend採用は認可しない。

## Track B G2 evidence entry (2026-10-09)

[Saved-value diagnosis](../../docs/tracks/algorithm_codesign/g2_saved_diagnostic_handoff_20261009.md)
is isolated in `scripts/tracks/algorithm_codesign/g2_saved_diagnostic.py`; no shared library API was changed.
The saved-table diagnostic is complete and STOPped; it is not a new registered RA-D0 certificate.


## Track B G3への参照（2026-10-09）

共有library APIは変更していない。Track B限定scripts
`../../scripts/tracks/algorithm_codesign/g3_finite_law.py`、`g3_return_comparator.py`と
[結果/handoff](../../docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md)、
[manifest](../../artifacts/track_b_g3_finite_law/2026-10-09/evidence_manifest_v1.json)から辿る。
旧R1 native/numeric/accountingをidentity固定して再利用。288 finite profilesと固定return12合成を完了、mandatory STOP。


## Track B G4との境界

[G4 handoff](../../docs/tracks/algorithm_codesign/g4_results_and_gpt_handoff_20261009.md)。
独立certificateとCTS specializationは`scripts/tracks/algorithm_codesign/g4_*.py`。
CTS native/numeric/accountingは既存`src/trottertracks/algorithm_codesign/rte_reallocation/`をreadonly共用。
共有`src/trotterlib`実装/APIに変更なし。両one-shot完了・mandatory STOP。


## 2026-10-10 Track B G5：shared APIは変更しない

[G5保存値終了認証](../../docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md)と
[static access inventory](../../docs/tracks/algorithm_codesign/g5_static_access_inventory_20261010.md)。
Track B verifierはstdlibのみ、shared libraryのimport・変更・copyなし。
既存even paired DF wrapperの存在をodd/complement/reallocationの検証済み能力として転用しない。
[runner](../../scripts/tracks/algorithm_codesign/g5_fixed_dictionary_closure.py)、
[tests](../../tests/tracks/algorithm_codesign/test_g5_fixed_dictionary_closure.py)、
[manifest](../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/evidence_manifest_v1.json)。mandatory STOP。


## 2026-10-10 Track B G6との境界

shared library/API実装は変更しない。[新局所formal prototype](../trottertracks/algorithm_codesign/return_aggregation.py)はTrack B namespace。
[一般証明・access・停止記録](../../docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md)を参照。
controlled native実装、Hamiltonian、PR/QPE接続の検証ではない。


## Track B G7（2026-10-10）

共有API変更なし。実験実装は別namespaceの
[g7_generator.py](../trottertracks/algorithm_codesign/g7_generator.py)、
[g7_provider.py](../trottertracks/algorithm_codesign/g7_provider.py)、
[g7_reference.py](../trottertracks/algorithm_codesign/g7_reference.py)、
[g7_launch.py](../trottertracks/algorithm_codesign/g7_launch.py)。
productionと小support referenceを分離。conditional controlled-Q前提、一般分子実装ではない。


## 2026-10-10 Track B G7：取得完了・mandatory STOP（最新追記）

[G7結果/GPT引継ぎ](../../docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md)。
`G7_LIMITED_IMPLEMENTATION_ECONOMICS_COMPLETE_AWAITING_GPT_REVIEW`。source ab2549f41b3546fb3940342a2162dd9ee93699c4、24 keys/8 rows、strict error PASS、retry0。
固定P5では登録3対照後にもconditional期待T減少、P3ではclosed-form対照が小さい。
hard shot cap・classical generation/angle acquisition・未指定provider costを併記。
新規性/主method/次stage未採択、G5閉鎖/G6原証拠とmarker保持、GPT判断へ戻す。


## Track B G8（2026-10-10、source preparation）

[G8 proof/contract](../../docs/tracks/algorithm_codesign/g8_proof_contract_and_on_demand_scope_20261010.md)：採用GPT G7 review §11に基づく限定確認。
2 known development inputs/4 same production laws、finite-provider parameter、分離failure配分、
on-demand strict Rzと対称bounded cache。G7のstatus/point comparison/consumed markerを保持。
17 off-domain focused tests、source preparation時点のnew native acquisition0。一束後mandatory STOP。
共有API変更なし。実験namespaceは `src/trottertracks/algorithm_codesign/g8_*.py`。


- Track B G8 evidence: [conditional on-demand result](../../docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md). Library code unchanged; experimental modules remain in `trottertracks.algorithm_codesign`.


## Track B G9 source preparation（2026-10-10）

[Fixed proof/scope](../../docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md)：GPT G8 review §14を採用。known P5/指定3-qubit provider、6 direct+5 helper診断、19 keys/新CTS1 key、23 focused tests。source固定後一束のみ、終了後mandatory STOP。

## G9 one-shot STOP（2026-10-10）

`G9_TECHNICAL_INCONCLUSIVE`：epsilon引数のFraction→mp.mpf変換で停止、native比較0 row。
新規helper attempts1 / 新規sequence取得0 / retry0。source・contract・過去941pathは不変。
23 focused testsと23 saved-output checksは準備/整合証拠で、native資源の科学結果ではない。
[G9 failure・GPT handoff](../../docs/tracks/algorithm_codesign/g9_results_and_gpt_handoff_20261010.md)。mandatory STOP、次の研究判断はGPTへ返す。

## G9 v2 API-boundary source preparation（2026-10-10）

[G9 v2準備/入口](../../docs/tracks/algorithm_codesign/g9_v2_api_boundary_source_review_20261010.md)。精度値の型接続を最小修正、19 stub-only/launch tests PASS。
46科学条件・同19-key inventory・旧v1 source/result/markerは保持。
新実合成・登録matrix/予算/科学実行0、v2 marker absent、別authorization pending。
独立branchで資料公開後STOPし、新source-bound明示認可を待つ。

## G9 v2 one-shot completed / STOP（2026-10-10）

[G9 v2 results/GPT handoff](../../docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md)。`G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`、11 rows/22 axes、19 keys（new1/reuse18）、retry0。
known P5/指定3-qubit providerの登録direct6方式でclosed P5のT intercept/Kが小さく、CTSのCXは小さい。
1,866 event accounting、saved-only25 checks PASS。旧v1結果/marker、critical80/protected982は不変。
実量子shots/trajectory/DF/分子/NPZ/GPU/LPは0、source-bound local evidence。次の科学実行・採択はGPT判断、mandatory STOP。


## G10 saved-policy audit / degree-comparison source preparation（2026-10-10）

[G10 source review / GPT入口](../../docs/tracks/algorithm_codesign/g10_source_review_20261010.md)。保存P5 direct945 bindingsのT再計数と任意proposal固定policy下界を独立確認。
ordinary/partial/closed P3/CTSは保存closed P5と分離、general full/closed P5はこの下界では未分離。原G9分類は不変。
同p/x/3-qubit providerのm3/5/7（17 rows/34 axes）を固定。m5はsaved-only共通policy再会計、
登録P3/P7は未取得。41 off-domain focused tests PASS、固定runtime/未認可拒否を確認。
実synthesis0、G10 marker absent、authorization pending。旧source/result/auth/marker/STOP・Track Aを保持。
[evidence manifest](../../artifacts/track_b_g10_degree_preparation/2026-10-10/evidence_manifest_v1.json) / [contract](../../artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json) / [future runner](../../scripts/tracks/algorithm_codesign/g10_degree_matched_native.py) / [focused tests](../../tests/tracks/algorithm_codesign/test_g10_degree_preparation.py)。
次は固定source review→別authorization-only child→新one-shot指示。全結果STOP、研究判断はGPT。


## G10 one-shot technical STOP（2026-10-10、最新追記）

[結果/GPT引継ぎ](../../docs/tracks/algorithm_codesign/g10_results_and_gpt_handoff_20261010.md)。`G10_TECHNICAL_INCONCLUSIVE`：固定RSS cap512 MiB超過、guard peak539.546875 MiB。
source `05c5ef23` / authorization-only child `f5cd0755`、一回のみ、retry0。new synthesis27 / reuse19、17行保存。
technical prefixは科学判断に使用せず、原結果/marker/STOPと旧1241 pathを保持。
保存hash/sequence/会計45,605条件の監査PASS。critical113のfull source prefix保持、追記のみ。
追加science/matrix/synthesis/sampler/budget/lower処理なし。mandatory STOP、次の研究/実行scopeはGPT判断。
shared API code変更なし。G10実験source/contractを保持。


## 2026-10-10 Track B G10 RSS修正source S2準備

Track B固有の `trottertracks.algorithm_codesign.g10_io` を追加。既存trotterlib API・scientific modulesは変更していない。

報告: `docs/tracks/algorithm_codesign/g10_rss_repair_source_and_gpt_review_20261010.md`。契約・根拠: `artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/`。
旧source・結果・marker・STOPを保護する。科学実行/A2/fresh production markerは未認可・未実施。mandatory STOP。


## Track B G10 S3 IO-only compatibility preparation

Shared scientific APIs are unchanged. The repair is confined to `src/trottertracks/algorithm_codesign/g10_io.py`; future v3 runner and preparation/IO verifiers live in `scripts/tracks/algorithm_codesign/`. Source review: `docs/tracks/algorithm_codesign/g10_v3_key_compatibility_source_and_gpt_review_20261010.md`; tests: `tests/tracks/algorithm_codesign/test_g10_key_compatibility_v3.py`; artifacts: `artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/`. No production execution authorized, mandatory STOP.


## Track B G10 v3 completed / mandatory STOP（2026-10-10、最新追記）

[結果/GPT handoff](../../docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md)：`G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`。
S3 b9ed014→直接authorization-only A3 53a7bc4、一回/retry0。17rows/34axes、46keys（new27/reuse19）、10,936bindings。
stream/token/STOPとouter正常COMPLETE status一致。outer wall22.2261s/CPU20.0291s/peak240.015625MiB、512MiB cap不変。
保存監査のfield仮定/report組立の訂正と初回記録を保存、本番再実行0。旧結果/marker/STOP、critical180/protected1352保持。
`artifacts/track_b_g10_degree_result/2026-10-10/v3/evidence_manifest_v3.json` とsaved-only `audit_g10_v3_saved_outputs.py` を参照。
known synthetic providerの次数内有限operator比較のみ、science完了後mandatory STOP、追加科学/採択/新規性判断はGPTへ返す。

## Track B G10 v3科学レビュー採用・構成ノート（2026-10-11）

一般fullの固定入力性能探索を区切り、既存return familyとnative限界を
`docs/manuscripts/track_b_return_aggregation_native_resource_note_v0_1.md`へ整理した。
保存値照合・図のhelperは`scripts/tracks/algorithm_codesign/`にあり、library/API/source moduleは変更していない。
過去の一般数学構成・finite-bit証拠、原COMPLETEとmandatory STOPを保持。新規科学実行・G11は未認可。

## DF sparse correction audit source references（2026-10-11）

[限定独立監査](../../docs/tracks/algorithm_codesign/df_sparse_correction_independent_audit_20261011.md)は`df_partial_s2.py`/`rte.py`/`df_rte_tail.py`/`df_trotter/ops.py`/`rte_compiled_cost.py`をread-only照合した。
新audit scripts/testsはlibrary外、API/code変更なし。Gaussian orbitalsとsmall controlled回転はsyntheticで照合、thin-isometry/digital/Clifford+T production経路は未実装。科学的GOなし、mandatory STOP。


## Track B DF correction comparison source v2

共有trotterlib実装/APIは変更していない。B固有sourceは `../trottertracks/algorithm_codesign/df_correction_comparison/`、文書は docs/tracks/algorithm_codesign/df_correction_comparison_preregistration_v2_20261011.md、runnerは scripts/tracks/algorithm_codesign/run_df_correction_comparison_v2.py、testsは tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py。登録実行は別authorizationが必要。


## DF correction comparison integrity source v3 (2026-10-11)

[限定修正・GPT引き継ぎ](../../docs/tracks/algorithm_codesign/df_correction_comparison_integrity_repairs_v3_20261011.md)：科学契約v2不変、512 KiB分割とinventory SHA結合、88 focused PASS。登録比較/T合成0、authorization=false、mandatory STOP。


## Track B fingerprint診断source準備（2026-10-12）

[GPT handoff](../../docs/tracks/algorithm_codesign/df_correction_fingerprint_diagnostics_preparation_and_gpt_handoff_20261012.md)：B-local観測adapter、lossless matrix bytes・recursive payload・None/exception保存、登録外2-mode旧/new controls。
最終focused tests104 PASS。controlledで同じ行列でもrecursive自動名によるhash差を保存した。登録r3停止の原因は未確定。
既存moduleの数式・guard・expected・contract不変。旧marker/result/auth/STOP保持、登録science/T合成0、RUN_READY=false。
新source snapshotと全失敗test記録を公開し、mandatory STOP。次scopeの判断はGPT/利用者側。


## Track B E-r3 identity-only診断準備（2026-10-12）

[scope/source handoff](../../docs/tracks/algorithm_codesign/e_r3_identity_diagnostic_source_review_and_gpt_handoff_20261012.md)：保存r3 uncontrolledのみ、current→legacy各一回の診断専用runner・新launch/manifest/result namespaceを準備。
native transpile/CD/r6/Tなし。definition内部計算はobserverとCPU/RSS/output guardで会計。新off-domain tests33 PASS。
registered診断0、旧source/authorization/marker/result/STOP保持。新authorization pending・runs0。
source/scope確認と別明示one-shot承認までmandatory STOP。旧科学分類TECHNICAL_INCONCLUSIVE不変。

source固定前の最終確認：profiler context違反のtechnical STOPを追加し、new focused testsは34 PASS。初回33 PASSとdraft checkpointは保持。


## E-r3 identity-only observer修正 v2（2026-10-12）

[限定修正・技術検証](../../docs/tracks/algorithm_codesign/e_r3_identity_observer_repair_preparation_20261012.md)：observer設置／復元失敗をtechnical STOPへ伝播。77 focused PASS、contract全bytes・expected・capsは不変。新sourceとv2 namespaceを固定し、READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION。旧S/科学結果/marker/authorization/STOP保持、登録診断/A作成0。追加GPT詳細研究レビューは不要、別明示one-shot承認までmandatory STOP。


## Track B E/T layered acceptance preparation（2026-10-12）

Shared library APIs and fingerprint policy remain unchanged. The separate `src/trottertracks/algorithm_codesign/df_correction_et_resumption/` adapter reuses existing numeric/native/synthesis helpers. See [source report](../../docs/tracks/algorithm_codesign/df_correction_et_resumption_preparation_20261012.md). No registered E/T run in preparation.


## Track B E/T storage v5 の実験namespace

共有trotterlibは変更せず、[df_correction_et_storage_v5](../trottertracks/algorithm_codesign/df_correction_et_storage_v5/README.md)に
lossless blob/snapshot codecとversion-specific実行保護を追加した。
[技術報告](../../docs/tracks/algorithm_codesign/df_correction_et_storage_v5_preparation_and_handoff_20261012.md)と
[証拠manifest](../../artifacts/track_b_df_correction_et_storage_preparation/2026-10-12/v5/source_manifest_v5.json)を参照。
科学部分はv4と同一、登録E/Tは別承認まで未実行、mandatory STOP。


## Track B E/T v5の停止記録（2026-10-12）

[新結果・saved-only監査](../../docs/tracks/algorithm_codesign/df_correction_et_v5_results_and_gpt_handoff_20261012.md)：r6 controlled numerical guard失敗、TECHNICAL_INCONCLUSIVE。
共有library・frozen algorithmは変更していない。native attempts10、T0、一回/retry0、旧/新marker保持、mandatory STOP。


## E/T observer v6

共有libraryは変更せず、`../trottertracks/algorithm_codesign/df_correction_et_numeric_v6/`で既存guardの観測を追加。[準備](../../docs/tracks/algorithm_codesign/df_correction_et_numeric_v6_preparation_20261012.md)、focused testsはv6 observer/integration。科学条件・caps不変。


## E/T phase v7

新namespace `../trottertracks/algorithm_codesign/df_correction_et_phase_v7/`内でraw/W phase会計を修正。共有API不変。phase/observer/integration/storage focused testsと[報告](../../docs/tracks/algorithm_codesign/df_correction_et_phase_v7_preparation_20261012.md)参照。


## E/T saved recovery

新namespace `../trottertracks/algorithm_codesign/df_correction_et_saved_recovery/`、metadata AST/accountingとlossless chunk fixだけ。共有library/元science source不変、追加science0。[準備](../../docs/tracks/algorithm_codesign/df_correction_et_saved_recovery_preparation_20261012.md)。


## Track B E/T 限定計算・保存値集計の完了（2026-10-12）

[最終技術監査・GPT handoff](../../docs/tracks/algorithm_codesign/df_correction_et_completed_saved_accounting_and_gpt_handoff_20261012.md)：`SAVED_E_T_ACCOUNTING_COMPLETE`。
元v7でnative22／T key39（backend33、Clifford fast path6）を取得。大きい辞書の保存停止後、同じpost-T ASTをsaved-onlyで復元した。
元v7のTECHNICAL_INCONCLUSIVE／markerは不変。復元で新science0、815 saved-output checks PASS、全Aのactual Tはunknownを維持する。
CDは保存参照のみ、別E fixtureとの混合で全PR勝敗を作らない。研究継続・新規性・資源的意義はGPT判断、mandatory STOP。
復元結果：`artifacts/track_b_df_correction_et_saved_recovery_result/2026-10-12/v2/`、監査／manifest：`artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/`。


## Track B DF三次補正 E/T：GPT科学判断・現系列の区切り（2026-10-12）

[採択済みGPTレビュー](../../docs/research/track_b_df_correction_et_scientific_review_20261012.md)／[系列の区切り・証拠索引](../../docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md)。
現generic whole実装を新規PRアルゴリズムの主線として拡張せず、この検証系列を区切る。
3条件の局所固定加法planでA completion下界>B完全event T上界を保持。A全T、4行目、全PR優劣、一般whole優劣は未確定。
有限stepのA指数平均とB weighted線形平均は区別する。Bの自動採択・Track B全体の終了を意味しない。
元v7 TECHNICAL_INCONCLUSIVE／復元SAVED_E_T_ACCOUNTING_COMPLETE／旧結果・marker・sourceは不変。
今回のCodex作業は文書・索引・provenance・公開確認のみ、新科学計算0、mandatory STOP。
同じ科学レビューは反復しない。新候補の研究判断・scope採択はGPT側、新実行は別途明示認可が必要。

## Track B D1実験用source（2026-10-13）

B固有namespaceは`src/trottertracks/algorithm_codesign/rte_development/`。共有trotterlibの実装を変更せず、既存generator/native数学を参照。[技術報告](../../docs/tracks/algorithm_codesign/rte_development_d1_source_preparation_20261013.md)、[source manifest](../../artifacts/track_b_rte_development_source/2026-10-13/v1/source_manifest_v1.json)。57登録外tests PASS、登録計算・合成未実行。

## D1保存値分析（2026-10-13）

[追加技術報告](../../docs/tracks/algorithm_codesign/d1_saved_resource_analysis_and_gpt_handoff_20261013.md)。analyzerはscripts側のstdlib Fraction処理であり、共有library・既存Track B science実装を変更しない。新science/compile/synthesis0、mandatory STOP。

## Track B D2専用kernel（2026-10-13）

共有trotterlib APIを変更せず、`trottertracks.algorithm_codesign.rte_development.key_mechanisms`と`boundary`/`boundary_launch`へ保存価格・exact literal相殺を準備。[技術報告](../../docs/tracks/algorithm_codesign/d2_key_mechanisms_and_boundary_preparation_20261013.md)。登録費用取得は別明示one-shot認可、mandatory STOP。

## D1/D2最終保存分析（2026-10-13）

共有library・science sourceの変更なし。新analyzerはscripts側のstdlib exact affine処理。[技術報告](../../docs/tracks/algorithm_codesign/d2_final_saved_integration_and_gpt_handoff_20261013.md)。旧科学分類保持、mandatory STOP。
