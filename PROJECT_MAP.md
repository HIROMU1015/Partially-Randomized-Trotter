> **M1-S統合研究記録（2026-10-17）**： [統合サマリー](docs/research/m1s_integration_20261017/integrated_research_summary.md)／[証拠索引](docs/research/m1s_integration_20261017/evidence_and_results_index.md)／[Codex再開ハンドオフ](docs/research/m1s_integration_20261017/codex_restart_handoff.md)。Stage Sと先行研究調査を固定provenanceで連結し、最新Claude/GPT原本を保存。新しい科学判断・実行・研究終了/継続採択なし、mandatory STOP。以下の旧進行記述は履歴として読む。

> **M1-S Stage S one-shot完了（2026-10-17）**：[結果・監査・STOP](docs/tracks/algorithm_codesign/m1s_stage_s_one_shot_results_and_stop_20261017.md)。H4 linear1.00 Å/STO-3G/DF rank12・G3/G6/G9・δ1/8,1/16・R2,4・T1,4。主対象24評価・全48報告。G3 T4の8対照にcap-edge整数削減の存在証拠、G3 T1/G6 T4の16対照はcap直下で同数。G6 T1旧不成立8/G9絶対未解像16をcarry。S7a338c4e/Cb4b4c128/A437e6bbc、初回正常完了・SOURCE修正/retry0。条件付き十分設計のみ、RPE/native総資源GOなし。保存監査448/保全14 PASS、旧成果保全、mandatory STOP。以下は履歴。

> **M1-S Stage S正式採択・計算開始承認受領（2026-10-17）**：[採択・実施scope](docs/tracks/algorithm_codesign/m1s_stage_s_formal_adoption_and_execution_scope_20261017.md)。H4 linear1.00 Å/STO-3G/DF rank12・G3/G6/G9・δ1/8,1/16・R2,4・T1,4。公開0dd660ec候補とSOURCE7a338c4eを変更せず正式契約へ固定。ユーザーの今回の開始指示を専用Aへ別結合して一回実行する。契約単独は実行をgrantしない。failure消費・retry禁止・最終mandatory STOP維持。以下は履歴。

> **M1-S Stage S技術準備（2026-10-17、未採択・実行未承認）**：[cap-edge契約候補](docs/tracks/algorithm_codesign/m1s_stage_s_cap_edge_contract_candidate_20261017.md)。linear H4 1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4。G3の16/G6 T4の8を主対象、G6 T1不成立8/G9絶対未解像16を保持。連続supremumと開端点floor+1を区別する新SOURCE `src/trottertracks/algorithm_codesign/m1s_stage_s_v1`、runner `scripts/tracks/algorithm_codesign/run_m1s_stage_s_v1.py`、checker `scripts/tracks/algorithm_codesign/check_m1s_stage_s_preparation_static.py`、人工tests `tests/tracks/algorithm_codesign/test_m1s_stage_s_preparation.py`、証拠 `artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1`。CPU/RAM/GPU/worker/timeに人工capなし、output/call/instruction bounds候補。保存再解析/科学shot/native/新authority・grant・marker0、guard closed、mandatory STOP。以下は履歴。

> **M1-S 次期整数shot・costの非数値準備（2026-10-17、未採択・未実行）**：[Decision Packet](docs/tracks/algorithm_codesign/m1s_shot_cost_feasibility_decision_packet_20261017.md)。linear H4 1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4。G3の16成立、G6 T4の8成立・T1の8不成立、G9の16絶対差未解像を保持。位相予算・整数ceil・全wrapper/HD/状態準備を含むbreak-evenを記号式で整理。source spec `src/trottertracks/algorithm_codesign/m1s_shot_cost_design_v1.py`、checker `scripts/tracks/algorithm_codesign/check_m1s_shot_cost_design_static.py`、証拠 `artifacts/track_b_m1s_shot_cost_preparation/2026-10-17/v1`。数値engine未実装、D1–D7未採択、科学計算/数値再解析/shot forecast/native compile/新authority・grant・marker0。旧成果保全、mandatory STOP。以下は履歴。

> **M1-S π/3限定照合完了（2026-10-17）**：[結果・監査](docs/tracks/algorithm_codesign/m1s_gamma_pi3_completed_results_and_stop_20261017.md)。別途ユーザー実施承認・専用Aa1dd87a1から初回完了。H4 linear1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4不変。G3の16、G6の8が成立、G6の8不成立、閾値未解像0、G9の16は絶対差未解像・未照合を保持。SOURCE2f800f17/Cbf97fcb2不変、root/pi/signal/整数shot/native/RPE総資源の新計算なし。条件付き未認証、科学GO未判断、旧成果保全、mandatory STOP。checker `scripts/tracks/algorithm_codesign/audit_m1s_gamma_pi3_saved_result.py`、artifact `artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01`。以下は履歴。

> **M1-S π/3限定gate正式採択（2026-10-17、実行未承認）**：[正式契約・採択記録](docs/tracks/algorithm_codesign/m1s_gamma_pi3_formally_adopted_contract_20261017.md)。ユーザーの次段階承認を受け、公開e3828965候補を別machine契約へ固定。SOURCE2f800f17・入力・32対象/G9の16未解像保持・判定・条件付きcoverage・resource・one-shot/failure STOPは不変。H4 linear1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4。contract_grants_execution=false、別明示承認が必要。新数値評価・authorization/grant/marker0、mandatory STOP。以下は履歴。

> **M1-S π/3限定gate準備（2026-10-17、未採択・未実行）**：[契約候補](docs/tracks/algorithm_codesign/m1s_gamma_pi3_interval_gate_contract_candidate_20261017.md)。保存G3/G6のγ境界区間と保存π区間の厳密比較SOURCE／closed guard／schemaを準備。G9未解像性を保持。H4 linear1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4は不変。新数値評価・閾値代入・authorization/grant/marker0。runner `scripts/tracks/algorithm_codesign/run_m1s_gamma_pi3_v1.py`、checker `scripts/tracks/algorithm_codesign/check_m1s_gamma_pi3_preparation_static.py`、SOURCE `src/trottertracks/algorithm_codesign/m1s_gamma_pi3_v1`、artifact `artifacts/track_b_m1s_gamma_pi3_preparation/2026-10-17/v1`。gate／契約採択と別実行承認が必要、mandatory STOP。以下は履歴。

> **M1-S γ依存V1完了（2026-10-17）**：[結果・修正来歴](docs/tracks/algorithm_codesign/m1s_v1_gamma_completed_results_and_stop_20261017.md)。ユーザーの修正・完了承認で旧失敗を保全し別S2/C2/A2から試行02が完了。H4 linear1.00 Å/STO-3G/DF rank12・G3/G6/G9・保存δ1/8,1/16・R2,4・T1,4・α0.05/p128/CPU160/wall320不変。72行評価・36zero保持・48対照分類、G3/G6各16tradeoff境界、G9絶対差未解像を保持。絶対shot/native/RPE総資源/科学GOなし。SOURCE `src/trottertracks/algorithm_codesign/m1s_saved_signal_gamma_v2`、runner `scripts/tracks/algorithm_codesign/run_m1s_saved_signal_gamma_v2.py`、tests `tests/tracks/algorithm_codesign/test_m1s_gamma_completion.py`、artifact `artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02`。条件付き未認証、旧成果保全、mandatory STOP。以下は履歴。

> **M1-S γ依存V1 one-shot技術停止（2026-10-17）**：[失敗・保全報告](docs/tracks/algorithm_codesign/m1s_v1_gamma_one_shot_technical_stop_20261017.md)／[保存監査](artifacts/track_b_m1s_v1_gamma_execution_audit/2026-10-17/v1)。S=f2a5611c、C=9aafa8b7、A=b0b48836、run1/retry0。固定Python3.11.0rc1にsys.set_int_max_str_digitsがなく起動prefixで停止。one-shot消費済み、108行/48対照未評価、科学数値処理0。旧成果／marker／SOURCE／契約保持、mandatory STOP。

> **M1-S γ依存V1正式採択（2026-10-17）**：[正式契約・採択記録](docs/tracks/algorithm_codesign/m1s_v1_gamma_formally_adopted_contract_20261017.md)／[machine契約・監査](artifacts/track_b_m1s_v1_gamma_formal_adoption/2026-10-17/v1)。ユーザーが候補fcd1c464をPER_METHOD_ROW、α0.05、p128、CPU160/wall320秒等で正式採択。SOURCE f2a5611cは不変。条件付きengineering採用でありplatform未認証。科学実施未承認、新計算/authorization/grant/marker0、mandatory STOP。以下は履歴。

> **M1-S V1 γ依存の技術準備（2026-10-17）**：[改訂契約候補](docs/tracks/algorithm_codesign/m1s_v1_gamma_dependent_contract_candidate_20261017.md)／[SOURCE](src/trottertracks/algorithm_codesign/m1s_saved_signal_gamma_v1)／[runner](scripts/tracks/algorithm_codesign/run_m1s_saved_signal_gamma_v1.py)／[tests](tests/tracks/algorithm_codesign/test_m1s_gamma_preparation.py)／[証拠](artifacts/track_b_m1s_v1_gamma_preparation/2026-10-17/v1)。方向変更のみ採択済み。旧V0を保持し、全γ領域の条件・境界式を新namespaceへ接続。登録外scalar testsのみ、保存H4再解析0、改訂契約未採択・実行guard closed、mandatory STOP。

> **M1-S V1 V0技術準備（2026-10-17）**：[契約候補](docs/tracks/algorithm_codesign/m1s_v1_saved_signal_contract_candidate_20261017.md)／[SOURCE](src/trottertracks/algorithm_codesign/m1s_saved_signal_v1)／[runner](scripts/tracks/algorithm_codesign/run_m1s_saved_signal_v1.py)／[static tests](tests/tracks/algorithm_codesign/test_m1s_v1_v0_static.py)／[evidence](artifacts/track_b_m1s_v1_v0_preparation/2026-10-17/v1)。Task A先行・Task B最終目標・raw無補正・保存全cellの方向採択を記録。γ/α/policy未採択、数値解析0、実行guard closed、mandatory STOP。以下は履歴。

> **M1-S RPE適用条件の非数値整理（2026-10-17）：[Decision Packet](docs/tracks/algorithm_codesign/m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md)。** 最新GPTレビューを引き継ぎ、位相bias予算・visibility/shot式・全wrapperとstate準備費用・総額条件を記号式で整理。D1–D11未決、契約候補未採択、科学計算/数値再解析/forecast/native生成/新authority・marker0。旧科学status/結果/SOURCEを保全、mandatory STOP。以下の本文は保持。

> **M1-S有限step v2計算完了（2026-10-16）：[結果・修正来歴](docs/tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md)。** ユーザーの修正・完了承認の下で試行01の精度STOPを保全、Phiの既存Frobenius収縮性を誤差伝播へ適用し試行02で108/108行完了。H4 1.00 Å/STO-3G/8 modes/DF rank12・G3/G6/G9・保存oracle・delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1不変。correctness/全3twirl/精度PASS、条件付き未認証。G3/G6は各16絶対対照resolved higher、G9は16 unresolved。科学GO/native総資源未認定、旧成果・各試行・marker・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step v2起動前technical STOP（2026-10-16）：[不具合・未適用修正案](docs/tracks/algorithm_codesign/m1s_finite_signal_v2_pre_execution_configuration_stop_20261016.md)。** 別途ユーザー実施承認を受領したが、固定SOURCE fbc2edeaの精度文字列2^-24/2^-18をFractionへ直接渡す経路がValueErrorとなることをmetadata確認。H4 1.00 Å/STO-3G/8 modes/DF rank12・G3/G6/G9・delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1・保存oracle・108行・精度/runtimeは不変。科学/初回correctness/pilot起動0、one-shot未開始・未消費、authorization未発行・marker0。旧成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step信号pilot v2正式採択（2026-10-16）：[正式契約・採択記録](docs/tracks/algorithm_codesign/m1s_finite_step_complex_signal_pilot_v2_formally_adopted_20261016.md)。** ユーザーが68fa5580候補の条件付きcoverage・モデル差B/abs(T)B会計・108行grid・比較・runtime・one-shot/failure STOPを採択。H4 linear1.00 Å/STO-3G/8 modes/DF rank12・G3/G6/G9・保存数値oracle、delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1。SOURCE fbc2edea不変、platform/数値correctness未認証。採択契約は実行を許可しない。科学実施は明示未承認、新科学/数値test/authorization/grant/marker0、旧成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step信号pilot v2モデル差会計候補（2026-10-16）：[新契約](docs/tracks/algorithm_codesign/m1s_finite_step_complex_signal_pilot_contract_v2_candidate_20261016.md)／[v2 SOURCE](src/trottertracks/algorithm_codesign/m1s_finite_signal_v2)／[runner](scripts/tracks/algorithm_codesign/run_m1s_finite_signal_v2.py)／[静的checker](scripts/tracks/algorithm_codesign/check_m1s_finite_signal_v2_static.py)。** H4 linear1.00 Å/STO-3G/8modes/DF rank12・G3/G6/G9・保存数値oracle、delta1/8,1/16 Ha^-1・R2/4・T1/4 Ha^-1・zero込み108行を維持。raw H_fullとH_tildeを区別、Bとabs(T)Bを別台帳、総raw-H biasと純アルゴリズム差の区間を区別。SOURCE fbc2edea。条件付き未認証・未採択・実施未承認、数値correctness未検証、新科学/数値test/authorization/grant/marker0、旧成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S有限step信号pilot準備（2026-10-16）：[契約候補・SOURCE](docs/tracks/algorithm_codesign/m1s_finite_step_complex_signal_pilot_contract_candidate_20261016.md)。** 受領GPT独立結果レビューを原本保存。H4 linear1.00 Å/STO-3G/8spin orbitals/DF rank12・G3/G6/G9・保存数値oracle。delta候補1/8,1/16 Ha^-1、R2/4、T1/4 Ha^-1、同じ独立draw数のcoherent shared/ordinary/independent比較。SOURCE9758aeaa固定、複素信号/visibility/phaseとlambda²絶対尺度の設計。条件付きcoverage/精度/runtime・新target選択は未採択、新科学/Pauli/数値test/finite signal/authorization/grant/marker0。旧結果・STOP・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S独立one-shot完了（2026-10-16）：[結果・停止](docs/tracks/algorithm_codesign/m1s_independent_one_shot_results_and_stop_20261016.md)。** ユーザー別途実施承認・専用A2a55ba01、SOURCE4787cf90・採択C3ec5721bでrun1/retry0。H4 linear1.00 Å/STO-3G/8 spin orbitals/保存DF rank12・G3/G6/G9・deltaなし・保存数値oracle。全3分割の照合可能範囲PASS、support/sign差0、W/v/rho固定許容値内。未保存full-H/HDによる全scope UNDETERMINEDを保持。ローカル事後独立再現性照合でありplatform認証/科学GO/finite-step/native総資源評価ではない。旧v4/v5結果・SOURCE・契約・消費済marker・親/他worktree保全、追加科学なし、mandatory STOP。以下は履歴。

> **M1-S独立correctness契約を正式採択（2026-10-16）：[正式契約・採択記録](docs/tracks/algorithm_codesign/m1s_gate0_v5_independent_correctness_contract_adopted_20261016.md)。** ユーザーが2ba0858cの候補を採択。H4 linear1.00 Å/STO-3G/8 spin orbitals/保存DF rank12・G3/G6/G9・deltaなし・保存数値oracle、SOURCE4787cf90固定。照合可能範囲PASS/FAIL/UNDETERMINEDと、未保存full-H/HDを含む全scope UNDETERMINED（FAIL優先）を区別。精度・算術・runtime/caps不変、数値correctness未検証。科学実施は明示未承認、新科学/Pauli/数値test/W-v-rho/authority/grant/marker0、旧v4/v5成果・親/他worktree保全、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5独立correctness準備（2026-10-16）：[検証契約候補・独立性と限界](docs/tracks/algorithm_codesign/m1s_gate0_v5_independent_correctness_contract_candidate_20261016.md)。** 公開草稿d6d35cfbを別worktreeで再利用、SOURCE4787cf90固定。H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/G6/G9・deltaなし・保存数値oracle。occupation/CAR/全Pauli trace/tensor再構成、exact rational W/v/rho、結果JSONはreference固定後のみ照合。旧full-H/HD行列未保存分はUNDETERMINED、全scope PASS不可。契約未採択・実施未承認・数値correctness未検証、新科学/実入力・模擬input数値test/authority/grant/marker0、旧v4/v5成果・marker・元Codex/親worktree保持、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5初回one-shot（2026-10-16）：[結果・停止](docs/tracks/algorithm_codesign/m1s_gate0_v5_one_shot_results_and_stop_20261016.md)。** 別ユーザー実施承認・固定A8da63d67から一回実行。H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・保存HD G3/G6/G9・deltaなし。全3行で局所W/v予算とrho精度を通過、W/v/rho取得、到達1,239gate PASS。HEADROOM_DIAGNOSTICS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。同一development条件での事後技術改訂評価、platform/true-ground/finite-step/native総資源/科学GOは未認証・未判断。旧v4 STOP/null/marker保持、one-shot1/retry0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5正式採択（2026-10-16）：[正式契約と採択記録](docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v5_formally_adopted_20261016.md)。** ユーザーがH4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/G6/G9・deltaなし、局所alpha_R=gamma_(713+r)S_R・17精度条件・1e-8 Ha+B_res・条件付きcoverage・runtime/resource・all-splits/one-shot/failure STOPを採択。SOURCE69de022/proof b5a185cc固定、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。採択flagはplatform認証ではない。契約完備性と実行許可を分離、科学計算は明示未承認。新科学/予算評価/Pauli/数値test/authority/grant/marker0、旧v4 STOP/null/marker保持、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v5 SOURCE接続（2026-10-16）：[最終実験契約候補](./docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v5_candidate_20261016.md)／[v5 SOURCE](./src/trottertracks/algorithm_codesign/m1s_gate0_v5)／[runner](./scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v5.py)／[static checker](./scripts/tracks/algorithm_codesign/check_m1s_gate0_v5_static.py)／[監査・機械契約](./artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1)。** H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9・deltaなし。tail alpha_R=gamma_(713+r)S_R resolver接続、global/local ledger分離、17条件/runtime維持。SOURCE69de022固定、122 static PASS、実入力新式/Pauli/数値test/W-v-rho/Gate0評価0。候補未採択・coverage採択false・実施未承認、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。旧v4 TECHNICAL_STOP/null/消費済marker保持、新authority/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 tail局所上界（2026-10-16）：[条件付き証明](./docs/tracks/algorithm_codesign/m1s_gate0_tail_local_bound_proof_20261016.md)／[v5精度設計候補](./docs/tracks/algorithm_codesign/m1s_gate0_tail_local_precision_contract_v5_design_candidate_20261016.md)／[宣言的spec](./src/trottertracks/algorithm_codesign/m1s_gate0_tail_bound_spec.py)／[静的checker](./scripts/tracks/algorithm_codesign/check_m1s_gate0_tail_bound_static.py)／[証拠台帳](./artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1)。** H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9・deltaなし。SOURCEのgroup長とDAGからalpha_R=gamma_(713+r)S_Rを条件付き導出、index722/719/716。旧2^-24を維持できる解析的十分領域を証明。53 static PASS、数値評価0。v5はSOURCE未接続・非実行の未採択設計候補、CONDITIONAL_ENGINEERING_NOT_CERTIFIED。旧v4 TECHNICAL_STOP/null/marker、SOURCE/契約/精度ceiling/runtime保持、Gate0再実行0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 精度原因調査（2026-10-16）：[項別予算・改善候補](./docs/tracks/algorithm_codesign/m1s_gate0_precision_budget_cause_and_options_20261016.md)／[保存データscalar診断](./artifacts/track_b_m1s_gate0_precision_cause/2026-10-16/v1)／[診断SOURCE](./scripts/tracks/algorithm_codesign/analyze_m1s_gate0_precision_saved.py)。** H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9・deltaなし。承認済み保存再解析1、4 NPZ配列のexact尺度と3×8 scalarケースを評価。全H共通係数上界とlambda正規化が支配、tail局所化だけでは不足、局所DAGとの組合せに未証明の改善余地。新Pauli/state/matrix/W-v-rho/Gate0実行0。旧TECHNICAL_STOP/null・17条件・SOURCE/契約/authorization/marker保持、候補未採択、CONDITIONAL_ENGINEERING_NOT_CERTIFIED、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v4初回one-shot（2026-10-16）：[結果・技術停止](./docs/tracks/algorithm_codesign/m1s_gate0_v4_one_shot_results_and_stop_20261016.md)／[raw result](./artifacts/track_b_m1s_gate0_result/2026-10-16/v4_one_shot/result.json)／[保存監査](./artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot)／[saved-only auditor](./scripts/tracks/algorithm_codesign/audit_m1s_gate0_v4_saved_result.py)。** 別ユーザー実施承認・固定Aから一回実行。H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12・HD G3/G6/G9、delta評価なし。到達132gate PASS、残差品質PASS。全splitで事前W/v誤差予算超過、W/v/rho未評価null、TECHNICAL_STOP / PRECISION_UNRESOLVED。17条件・SOURCE・入力は不変。CONDITIONAL_ENGINEERING_NOT_CERTIFIED、metric correctness未到達、科学的GO/no-go未判断。one-shot1/retry0、marker消費済み、旧成果・親worktree保持、mandatory STOP。以下は各段階の履歴。

> **M1-S Gate 0 v4正式採択（2026-10-16）：[正式v4契約](docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v4_formally_adopted_20261016.md)／[正式machine契約](artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/execution_contract_v4_adopted.json)／[ユーザー採択記録](artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/user_final_contract_adoption_record.json)／[採択監査](artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1)／[採択static checker](scripts/tracks/algorithm_codesign/check_m1s_gate0_v4_adoption_static.py)。** ユーザーが17項目・1e-8 Ha+B_res・条件付きcoverage・runtime/caps・one-shot/failure STOPを正式採択。H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・saved G3/6/9、delta評価なし、SOURCE/入力/精度条件は候補と同一。CONDITIONAL_ENGINEERING_NOT_CERTIFIED、runtime/数値correctness未検証。契約完備性と実行許可を分離し、科学計算は明示的に未承認。新science/数値test/formula評価/実行authorization/grant/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 v4契約候補（2026-10-16）：[v4最終契約候補](docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v4_candidate_20261016.md)／[品質/coverage説明](docs/tracks/algorithm_codesign/m1s_gate0_v4_oracle_residual_quality_and_coverage_20261016.md)／[GPT原本](docs/research/track_b_m1s_gate0_v3_gpt_limited_scientific_review_20261016.md)／[v4 SOURCE](src/trottertracks/algorithm_codesign/m1s_gate0_v4)／[runner](scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v4.py)／[static checker](scripts/tracks/algorithm_codesign/check_m1s_gate0_v4_static.py)／[監査資料](artifacts/track_b_m1s_gate0_v4_contract/2026-10-16/v1)。** H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・saved G3/6/9、delta評価なし。残差gateを1e-8 Ha+B_resへ改訂、他8式/8scalar/runtime保持。全17項目が候補化、未採択。CONDITIONAL_ENGINEERING_NOT_CERTIFIED、99非数値static PASS、runtime/数値correctness未検証。新science/Pauli/数値test/formula評価/authority/grant/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 誤差算定式の契約準備（2026-10-16）：[v3最終契約候補](docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v3_candidate_20261016.md)／[式・coverage条件](docs/tracks/algorithm_codesign/m1s_gate0_preregistered_error_formula_design_20261016.md)／[v3 SOURCE](src/trottertracks/algorithm_codesign/m1s_gate0_v3)／[runner](scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v3.py)／[static checker](scripts/tracks/algorithm_codesign/check_m1s_gate0_formula_static.py)／[監査資料](artifacts/track_b_m1s_gate0_formula_contract/2026-10-16/v1)。** H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・saved G3/6/9、delta評価なし。未決9件中8件を条件付き式候補として具体化、oracle残差品質ηは未決。17項目・runtime/caps維持、68非数値static PASS、数値correctness/coverage未検証、全契約未採択・実行不可。新science/Pauli表/数値test/formula評価/authority/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 scope正式採択・精度契約案（2026-10-16）：[最終契約v2候補](docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v2_candidate_20261016.md)／[精度17項目](docs/tracks/algorithm_codesign/m1s_gate0_precision_error_budget_design_20261016.md)／[runtime/caps](docs/tracks/algorithm_codesign/m1s_gate0_runtime_and_resource_contract_candidate_20261016.md)／[v2 module](src/trottertracks/algorithm_codesign/m1s_gate0_v2)／[runner](scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v2.py)／[追加static checker](scripts/tracks/algorithm_codesign/check_m1s_gate0_precision_static.py)／[scope採択記録](artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/gate0_scientific_scope_adoption_record.json)。** H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/6/9・Pauli law/oracle比較のみユーザー採択。精度8値を候補化、9絶対値未決。追加67静的PASS、旧SOURCE/成果保持、数値correctness未検証、全契約未採択・科学未承認、新科学/Pauli表/authority/marker0、mandatory STOP。以下は履歴。

> **M1-S Gate 0 SOURCE準備（2026-10-16）：[実験契約最終候補](docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_candidate_20261016.md)／[専用module](src/trottertracks/algorithm_codesign/m1s_gate0)／[runner](scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom.py)／[非科学的checker](scripts/tracks/algorithm_codesign/check_m1s_gate0_source_static.py)／[監査資料](artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1)。** 保存H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12・G3/6/9のadapterとW/v/ρ SOURCEを実装。30静的確認PASS、数値correctness未検証。IMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED、契約未採択・実行未承認、新科学/Pauli表/authority/marker0、mandatory STOP。以下は履歴。

> **M1-S Checkpoint 2後（2026-10-16）：[理論addendum草稿](docs/tracks/algorithm_codesign/m1s_g0t1_post_checkpoint2_theory_addendum_draft_20261016.md)／[Gate 0事前調査・登録案](docs/tracks/algorithm_codesign/m1s_gate0_headroom_preflight_and_prereg_draft_20261016.md)／[追加文献照合](docs/tracks/algorithm_codesign/m1s_checkpoint2_prior_art_delta_20261016.md)。** GPT2のT1b限定受理・非縮退Jensen floor・rho総資源STOP不採用を整理。H4 NPZ所在/hash/header確認のみ、非冗長Pauli変換と実験条件は未採択。新科学/新証明0、mandatory STOP。以下は履歴。

> **M1-S G0-T1 T1b（2026-10-16）：[部分乱択の解析的移送](docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md)／[Checkpoint 2資料](docs/tracks/algorithm_codesign/m1s_g0t1_t1b_checkpoint2_decision_packet_20261016.md)。** GPT1はT1a限定受理、今回ユーザーがT1bを明示承認。全H C-sec一様上界と条件付きrhoを著者提出、共鳴single-rate一般化に解析反例。T1b独立受理・新規性・科学GOは未判断、数値/compile0、mandatory STOP。以下は履歴。

> **M1-S G0-T1 T1a（2026-10-16）：[一様C-sec上界・全証明](docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md)／[Checkpoint 1資料](docs/tracks/algorithm_codesign/m1s_g0t1_t1a_checkpoint1_decision_packet_20261016.md)。** T1a実施承認に基づく解析的証明の著者提出。固定有限次元・有限減衰予算で全N一様O(abs(theta))、独立科学的受理・新規性・Checkpoint 1判断は未実施。数値/compile0、T1b未実施、mandatory STOP。以下は履歴・既存索引。

> **Track B M1-S S0正式採択（2026-10-16）：[G0-T1正式理論契約](docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md)。** ユーザーがG0-T1/A′・T1aのC-sec・T1bの固定r/小step系列を正式採択。契約採択と実施承認を分離し、T1a/T1b証明・Gate 0・Q4/Q5は未承認。新証明/科学計算/compile0、次はT1a実施の別途明示承認、mandatory STOP。以下は履歴・既存案内。

> **Track B M1-S S0候補修正（2026-10-16）：[G0-T1理論契約候補](docs/tracks/algorithm_codesign/m1s_g0t1_s0_repaired_theory_contract_candidate_20261016.md)。** G0-T1/A′は暫定承認済み、契約最終採択・T1a/T1b証明実施は未承認。局所高次誤差・exact-half Ψ・joint scaling・dressing/相対誤差・Checkpointを整理。新証明/科学計算0、mandatory STOP。以下は履歴・既存案内。

> **Track B M1-S契約準備（2026-10-15）：[G0証明義務・Gate 0候補契約](docs/tracks/algorithm_codesign/m1s_g0_proof_and_gate0_precontract_20261015.md)。** 最新GPT完成レビュー原本をbyte保存・remote照合。Q1–Q5、ordered protocol未決定欄、データ所在、rho適用域、理論先行／探索先行を整理。契約・M1-S未採択、新証明・科学計算0、mandatory STOP。以下は従来の案内。

> **Track B（2026-10-15・M1-S G0-pre完了）：[先行研究・protocol・費用比較条件](docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md)。** 最新GPTレビュー原本をbyte保存。回文PF／coherent共有乱数の既知部分と未確立のlaw・RPE保証を整理。M1-S未採択、G0・新証明・科学計算0、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-15・M1 P0完了）：[採択前の先行研究・制御実装・資源資料Decision Packet](docs/tracks/algorithm_codesign/m1_p0_prior_art_control_and_resource_decision_packet_20261015.md)。** 固定GitHub21資料・一次PDF9件の関連箇所を整理。M1正式採択・G0証明・V1科学計算は未開始、新規計算0。Parent allocation D＋Eを保持。次は明示開始承認後のGPT採択判断、mandatory STOP。[履歴](docs/research/研究ノート/2026-10-15_track_b_M1_P0_evidence_preflight.md)。以下は過去の状態。

> **Track B現在（2026-10-15・D＋E正式採択）：[Parent allocationの限定的成果と現在探索の区切り](docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md)。** [資源構造調査](docs/tracks/algorithm_codesign/track_b_resource_structure_and_bottlenecks_20261015.md)と[次期方向Decision Packet](docs/tracks/algorithm_codesign/track_b_next_algorithm_research_decision_packet_20261015.md)を追加。4候補群は未採択・順位未設定。新科学計算0、旧成果不変、GPTレビュー開始／科学実行は別途明示承認、mandatory STOP。[判断履歴](docs/research/研究ノート/2026-10-15_track_b_D_plus_E_and_next_direction_preparation.md)。以下は過去の状態。

> **Track B現在（2026-10-15）：[S4訂正KKT補完one-shot完了](docs/tracks/algorithm_codesign/parent_allocation_s4_supplement_results_and_gpt_handoff_20261015.md)。** 4構成/20新law・12合成・24root診断、旧50対照の保存値を再利用。371 saved-only checks PASS、S4/A4・旧20proxy結果/marker保持、retry0。分類PARENT_ALLOCATION_KKT_SUPPLEMENT_COMPLETE_AWAITING_GPT_REVIEW、科学的採否未判断、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-15）：[KKT目的不一致のS4訂正・補完準備](docs/tracks/algorithm_codesign/parent_allocation_s4_kkt_repair_and_supplement_preparation_20261015.md)。** 採択sqrt(C)目的へ戻し、114登録外tests PASS。旧20 lawは別proxyの原記録として保持、50対照は保存再利用。新登録計算0、authorization pending、別S4 one-shot明示承認までmandatory STOP。以下は履歴。

> **Track B現在（2026-10-15）：[Parent allocation v3一回実行完了：15構成/70 law、新16合成/72 root診断、918保存監査PASS。科学条件/caps不変、旧証拠保持、retry0。科学的採否は未判断、mandatory STOP。](docs/tracks/algorithm_codesign/parent_allocation_v3_results_and_gpt_handoff_20261015.md)。** 分類はPARENT_ALLOCATION_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW。次は開始承認後のGPT科学判断。以下は履歴。

> **Track B現在（2026-10-15）：[Parent allocation v3保存処理修正](docs/tracks/algorithm_codesign/parent_allocation_v3_buffered_source_preparation_20261015.md)。** 科学条件/caps同一、Parent専用16 KiB保存、93登録外tests PASS、人工bytes比較PASS。旧marker/結果保持。利用者の継続承認に基づき、新S3/A3を固定後に一回実行する準備。以下は履歴。

> **Track B現在（2026-10-15）：[Parent allocation v2技術停止・保存監査](docs/tracks/algorithm_codesign/parent_allocation_v2_results_and_gpt_handoff_20261015.md)。** Parent allocation v2は総CPU上限3600秒でTECHNICAL_INCONCLUSIVE。新合成16、root診断72、15構成/61 prefix行を保存、完成resource mapなし。148 read-only checks PASS、旧資料・親worktree保持、retry0、mandatory STOP。 以下は履歴。

> **Track B現在（2026-10-15）：[Parent allocation v2 PID限定修正・source準備](docs/tracks/algorithm_codesign/parent_allocation_v2_owner_repair_source_preparation_20261015.md)。** 旧失敗source/結果/markerを保持、科学条件/caps同一、実acquire＋worker gateを通る新7件を含む78登録外tests PASS。登録評価・新合成・A2/marker0、別one-shot承認待ち、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-15）：Parent allocation v1 one-shotはtechnical inconclusive。** [停止原因・保存監査・引き継ぎ](docs/tracks/algorithm_codesign/parent_allocation_v1_failed_execution_and_gpt_handoff_20261015.md)。初回key jobのparent_pid重複で停止、新合成/比較law0、旧guard20・候補freeze保持、read-only23 checks PASS。marker消費済み、retry0、科学的勝敗未判断、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-15）：[Parent allocation実行source準備](docs/tracks/algorithm_codesign/parent_allocation_execution_source_preparation_20261015.md)完了。** GPT科学設計採択を固定contractへ反映、ordinary/P5と三参照・最大15構成/75 law、登録外71 tests PASS。科学実行・新authorization・marker0、READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-14）：[Parent allocation近接先行研究対応](docs/tracks/algorithm_codesign/parent_allocation_prior_art_correspondence_20261014.md)調査完了。** 指定5本文の関連箇所、Euler/ISの直接帰結、SPS embedding、R0/G1/G5限定差を整理。完全な実装同値/priorityは未認定、科学条件7領域は不変、新science/source/tests0、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-14）：[Parent allocation科学設計Decision Packet](docs/tracks/algorithm_codesign/parent_allocation_scientific_design_decision_packet_20261014.md)完成。** native/cache/strict guard互換性を準備、新26＋既存影響6 tests PASS。内部配分とroot-local/global proposalの境界、7未採択領域を整理。runtime metadata一致、登録科学/保存列operator再検証0、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-14）：[Parent allocationの汎用source・技術検証](docs/tracks/algorithm_codesign/parent_allocation_source_preparation_and_gpt_handoff_20261014.md)完了。** ordinary/closed P5共通root・全端点・controlled位相・common-angle復帰、62登録外tests PASS。判定TECHNICALLY_READY_FOR_SCIENTIFIC_DESIGN_REVIEW。[比較技術案v2](docs/tracks/algorithm_codesign/parent_allocation_comparison_technical_options_v2_20261014.md)は未採択、本番worker/合成runtimeは未準備、新科学/authority/marker0、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-14）：parent係数のchild別配分を[限定技術監査](docs/tracks/algorithm_codesign/parent_allocation_technical_audit_20261014.md)。** 理想mean・位相・norm最小性・固定cost解析を確認、13登録外形式tests PASS。実行source未作成、native利益未取得、比較案は未採択、旧数値closure/marker保持、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-14）：最新GPT科学レビュー完了。G系列〜D1/D2/D3を[研究引継ぎノート](docs/research/研究ノート/2026-10-14_track_b_return_aggregation_G_D_research_note.md)へ統合。構成・生成法と資源条件／限界を保持し、現在の数値探索は一区切り。新計算・原稿作成・追加レビューは開始せずmandatory STOP。** 以下は履歴。

> **D1〜D3一次文献照合（2026-10-14）：13 claimの固定証拠・既知原理・未確立部分を整理。8論文の関連本文を確認、旧G系列監査は参照のみ、新科学取得/再計算0。科学判断は未実施、mandatory STOP。** [GPT引き継ぎ](docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md)。

> **D3保存値の最終機構分析（2026-10-14）：36 keyの価格寄与をexactに分解し、保証によるr選択・primaryと全方式の差を整理。369追加算術checksと9人工tests PASS。旧分類/証拠は不変、新科学取得0、mandatory STOP。** [技術報告](docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md)。以下は履歴。

> **Track B現在（2026-10-14）：D3 v2一回実行・保存監査完了、GPT科学判断待ち。** [結果/一次データ/引き継ぎ](docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md)。36新合成・48行・四ケース、outer exit0、2579 saved-only checks PASS。native会計1e-6固定、旧証拠保持、retry0、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-14）：D3 v2型adapter修正・source準備完了、別one-shot承認待ち。** [限定差分・81 focused tests・実行条件](docs/tracks/algorithm_codesign/d3_v2_precision_repair_and_source_preparation_20261014.md)。科学条件/精度会計/36 keys/48行/資源上限は同じ。旧v1失敗とmarker保持、実合成/登録比較0、authorization pending、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-14）：D3 v1一回実行はD3_TECHNICAL_INCONCLUSIVE。** [原因・保存監査・GPT引き継ぎ](docs/tracks/algorithm_codesign/d3_failed_execution_and_gpt_handoff_20261014.md)。precisionのFraction→mpmath変換で最初の2 jobsが停止、完成sequence0、登録rows0/48。999 saved-only checks PASS、source/旧証拠保持、retry0、marker消費済み、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-13）：D3専用source準備完了、別one-shot承認待ち。** [source/guard/63 focused tests](docs/tracks/algorithm_codesign/d3_source_preparation_and_execution_handoff_20261013.md)。固定保証×価格4ケース/48行、native会計1e-6固定、登録新N/価格/合成/pair0。旧証拠保持、mandatory STOP。以下は履歴。

> **Track B現在（2026-10-13）：D3数学確認・保証×価格の比較計画固定。登録再会計/新合成0、新execution source/authorizationは未作成、mandatory STOP。** [D3計画](docs/tracks/algorithm_codesign/d3_guarantee_price_preregistration_v1_20261013.md)。旧D1/D2保持、次はCodexの技術source準備。以下は履歴。

> **現在のTrack B（2026-10-13）：D1/D2保存値の最終統合分析完了。** [全方式感度・Pareto・r選択・相殺構造/GPT引き継ぎ](docs/tracks/algorithm_codesign/d2_final_saved_integration_and_gpt_handoff_20261013.md)。D2 CX最小包絡線はCTS→partial→P5、joint12のT最小はr2 P5。exact支配領域と帰属限界を整理、16人工scalar checks PASS、新science/pair再評価0。旧証拠保持、mandatory STOP、次は開始承認後のGPT科学レビュー。以下は履歴。

> **現在のTrack B（2026-10-13）：D2共通境界相殺一回実行完了。** [結果/保存値監査/GPT引き継ぎ](docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md)。r=2・六方式219,303 pairs、retry0、outer exit0、saved-only1218 checks PASS。P5対partial T差は−5.8358%→−5.8557%（固定nativeモデル）。旧source/結果/markersを保持、新compile/合成/matrix0、mandatory STOP。研究判断はGPT側。以下は履歴。

> **現在のTrack B（2026-10-13）：D1 key/event単価分析・D2境界相殺準備完了。** [技術報告・次の実行承認待ち](docs/tracks/algorithm_codesign/d2_key_mechanisms_and_boundary_preparation_20261013.md)。GPT科学レビュー採択に従い、12 row/1890 eventのliteral Tと合成key価格を分解。D2はr=2全六方式・旧law/N/bias/価格を固定するexact境界相殺対照。45登録外tests PASS、登録pair評価/compile/合成0、authorization pending、mandatory STOP。以下の過去表記は履歴。

> **現在のTrack B（2026-10-13）：D1保存値の追加技術分析完了。** [全6方式・要因分解・r差・感度・不足整理](docs/tracks/algorithm_codesign/d1_saved_resource_analysis_and_gpt_handoff_20261013.md)／[分析manifest](artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/analysis_manifest_v1.json)。90 pair-metric分解、18 r-contrast、全hの量子Pareto関係。原D1分類・source/results/markersは不変、科学実行0、mandatory STOP。以下の過去表記は履歴。

> **現在のTrack B（2026-10-13）：D1 one-shot正常完了、GPT科学レビュー待ち。** [結果・引き継ぎ](docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md)／[保存値監査](artifacts/track_b_rte_development_d1_result/2026-10-13/v1/saved_output_audit/saved_output_audit_v1.json)。12 rows / 1890 events、合成17新規+19固定reuse、retry=0、内外status整合。source/旧成果不変、mandatory STOP。以下の同日source準備・設計・休止・旧承認待ちは履歴。

> **現在のTrack B（2026-10-13）：D1比較source準備完了、別実行承認待ち。** [技術報告](docs/tracks/algorithm_codesign/rte_development_d1_source_preparation_20261013.md)／[固定contract](artifacts/track_b_rte_development_source/2026-10-13/v1/contract_v1.json)。P5・同provider・総時間5/7、r=1,2、六方式。57登録外tests PASS、登録評価・合成0、authorization pending、mandatory STOP。以下の同日設計・休止・旧承認表記は履歴。

> **現在のTrack B（2026-10-13、後続GPT判断）：既存成果の発展RQの技術設計を優先。** [科学レビュー原本](docs/research/track_b_existing_results_development_broad_scientific_review_20261013.md)／[比較計画v1](docs/tracks/algorithm_codesign/existing_results_development_technical_plan_v1_20261013.md)。同じfinite-RTE時間発展とconfidenceで表現・normalizer情報・finite-bit・native価格を比較する設計。新実装・科学実行は未認可、旧closureとSTOPは保持。以下の同日休止記録・過去の最新表記は判断履歴。

> **現在のTrack B（2026-10-13）：新規手法探索を一時休止。** [GPT判断原本](docs/research/track_b_overall_direction_and_research_pause_review_20261013.md)／[休止記録](docs/tracks/algorithm_codesign/track_b_research_pause_record_20261013.md)。既存成果・終了判断は保持し、closed P5は将来候補のみ。新実装・科学実行へ進まずmandatory STOP。以下の過去の承認待ち・最新表記は判断履歴。

> 2026-10-07 Track B RA-D0 v3：**READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION**（実行承認ではない）。
> [source review](docs/tracks/algorithm_codesign/ra_d0_source_review_v3_20261007.md) / [GPT handoff](docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_v3_20261007.md) / [manifest](artifacts/track_b_ra_d0_source_review_v3/2026-10-07/evidence_manifest_v3.json)。
> [focused verifier](scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v3.py) / [v3 tests](tests/tracks/algorithm_codesign/test_ra_d0_source_review_v3.py)。
> exact-certified B2 minimum infeasibilityを正常outcomeに修正。pointをfreezeへ記録しbudget/queryを空にして次nへ進む。
> uncertified failureはtechnical STOP。数値・candidate・grid・call/resource capは維持。登録最適化0、authorizationなし、mandatory STOP。

> 2026-10-07 Track B RA-D0 v2：**READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW**。
> [source review](docs/tracks/algorithm_codesign/ra_d0_source_review_v2_20261007.md) / [GPT handoff](docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_20261007.md) / [evidence manifest](artifacts/track_b_ra_d0_source_review_v2/2026-10-07/evidence_manifest_v2.json)。
> [future runner](scripts/tracks/algorithm_codesign/run_ra_d0_one_shot.py) / [focused verifier](scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v2.py) / [v2 tests](tests/tracks/algorithm_codesign/test_ra_d0_source_review_v2.py)。
> B0_saved/ideal分離、数値B1⊂B2⊂B3、profile-paired budget、batch freeze-before-B3、
> anchor-first、main LP 55,275 / auxiliary込み110,550、resource/launch guardを固定。
> 登録最適化・実budget/minimum/witness取得0、authorizationなし。旧本文・旧STOP・Track Aは保持。mandatory STOP。

## Track B RA-D0 static preparation / mandatory STOP（2026-10-06）

[source review](docs/tracks/algorithm_codesign/ra_d0_source_preparation_review_v1.md)：21 columns/x、18 sign pairs一致、35 focused tests PASS。
B専用namespace `src/trottertracks/algorithm_codesign/ra_d0/`、static generator／verifier、
`artifacts/track_b_ra_d0_preparation/2026-10-06/`に候補・grid・semantic audit・provenanceを保存。
ideal nestingと数値membershipを区別し、query実行scope／資源上限をGPTへ返す。
登録最適化／新合成／新science0、RUN_READY=false、既存分類・Track A・旧STOP保持。
**mandatory STOP。以下の既存本文を全文保持する。**

## Track B RA-RTE統合数学監査・mandatory STOP（2026-10-06）

R1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942` 基点、GPT設計案へのDOCS_SYMBOLIC_ONLY_MATHEMATICAL_AUDIT。
[命題別監査](docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_v1.md)と[GPT handoff](docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_gpt_handoff_20261006.md)：一block／finite table／canonicalのfixed-n LPは仮定付きで成立。
shot-gridの固定total cap保存には反例。log/root・sampler認証、Delta=0、peak workspaceの規約を実行前修正へ返す。
[stdlib人工bookkeeping](scripts/tracks/algorithm_codesign/check_ra_rte_mathematical_bookkeeping.py)の[50 checks](artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_checks_v1.json)を一般証明と分離した。
[manifest](artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/evidence_manifest_v1.json)、[dated note](docs/research/研究ノート/2026-10-06_track_b_ra_rte_mathematical_audit.md)。science/synthesis/solver/資源再採点0、共通API変更0。
既存科学分類・結果・STOPは不変。性能・新規性・algorithm採択・次実装／R2 authorizationは未確定。
**mandatory STOP。次の採択・実装・pilotの必要性／範囲はGPT判断。以下の既存本文を全文保持する。**

## Track B R1.5保存値帰属・mandatory STOP（2026-10-06）

input R1 commit `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b` の保存値だけを用いたPOSTHOC attribution / design input。
[帰属報告](docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md)と[GPT handoff](docs/tracks/algorithm_codesign/r1p5_gpt_handoff_20261006.md)：新science/synthesis/compile/候補追加0、R1科学分類は不変。
primaryは2-qubit finite P₃、distinct-basis controlled、x={1/8,1/4}、登録native三precision。
Aは登録(G_T,G_CX,G_1Q) frontにx=1/8の1e-4、x=1/4の1e-3/1e-4で残る。
normalizationだけでなくnative費用とbias/shotの関係を整理し、固定合成列への依存も保存した。
[stdlib保存値解析](scripts/tracks/algorithm_codesign/analyze_r1p5_saved_attribution.py)、[全summary](artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)、[provenance manifest](artifacts/track_b_r1p5_saved_attribution/2026-10-06/evidence_manifest_v1.json)、
[日付note](docs/research/研究ノート/2026-10-06_track_b_r1p5_saved_attribution.md)。共通library変更・独立validation・新algorithm採択はない。
限定診断SUPPORTS_RA_RTE_DESIGNは設計入力のみ。eta探索/R2/DF接続/追加scienceは未認可。
**mandatory STOP。次の数学設計・研究方針判断はGPT側。以下の既存本文を全文保持する。**

## Track B R1一回結果・mandatory STOP（2026-10-06）

固定S `d43d64a821a0249a0dfab12a2472bd3a72fdee74` →直接子authorization-only A
`411f08f768244fe87b600d82308c3851847fe9e4`からrun1/retry0。
[結果照合](docs/tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md)：126 keys / 264 rows / 132 controlled tasks完了、全task適格。
terminal R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW、[保存field専用監査](scripts/tracks/algorithm_codesign/audit_r1_saved_result.py) PASS。
primary distinct-basis controlledではB²改善とnative/shot資源のtrade-offを保存し、自動研究GOはない。
[全264 rows CSV](artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)、[evidence manifest](artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)、
[GPT判断への入口](docs/tracks/algorithm_codesign/r1_post_run_gpt_review_request_20261006.md)。原result/marker/source、既存証拠・共通API・Track A保持。
science終了後mandatory STOP、追加合成/target/grid/分子/DF/trajectory/GPUは行わない。
研究方針・RQ・新規性・着地点・追加検証の必要性/範囲はGPT側。
以下のpending/最新記述は当時の履歴として本文をそのまま保持する。

# プロジェクト案内

## Track B R1 source preparation — source review / mandatory STOP

2026-10-06の[preregistration v2](docs/tracks/algorithm_codesign/rte_reallocation_r1_preregistration_v2.md)、
[native semantics](docs/tracks/algorithm_codesign/rte_reallocation_r1_native_semantics_v1.md)、
[GPT source review](docs/tracks/algorithm_codesign/rte_reallocation_r1_source_review_request_20261006.md)。
ordinary/PTSC-K0/A、PauliだけCTS control、distinct-basis controlled primary。canonical/resource vector。
[B専用source](src/trottertracks/algorithm_codesign/rte_reallocation/)、
[runner](scripts/tracks/algorithm_codesign/run_r1_rte_reallocation.py)、
[semantic tests](tests/tracks/algorithm_codesign/test_r1_rte_reallocation_semantics.py)、
[accounting/launch tests](tests/tracks/algorithm_codesign/test_r1_rte_reallocation_accounting_launch.py)、
[source manifest](artifacts/track_b_rte_reallocation_r1_source/2026-10-06/source_manifest_v1.json)。
27 focused tests PASS、42 static angles/126 keys、264予定rows。登録synthesis/resource取得0、R1未認可。
旧R0/R0.5/科学証拠・共通API・Track A保持。以下の本文は履歴を含めそのまま保存。


## Track B R0.5: scoped equivalence audit / GPT review

2026-10-06のdocs/symbolic-only [監査](docs/tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md)と
[GPT handoff](docs/tracks/algorithm_codesign/rte_reallocation_r05_gpt_handoff_20261006.md)を追加。
指定P0–P3内の限定classification `METHOD_DELTA_CANDIDATE`、gate `CONDITIONAL-R1`。
R1の必要性はGPT判断、実行は未認可。固定9条件/18 signのtechnical比較のみ一回。
[checker](scripts/tracks/algorithm_codesign/check_rte_reallocation_r05_symbolic.py)、
[manifest](artifacts/track_b_rte_reallocation_r05/2026-10-06/audit_manifest_v1.json)、
[dated note](docs/research/研究ノート/2026-10-06_track_b_rte_reallocation_r05.md)を参照。
旧R0/科学証拠・Track Aは保持し、下記の旧本文はそのまま残す。mandatory STOP。


## Track B R0: finite-mean RTE representation audit (2026-10-06)

- [Review packet](docs/tracks/algorithm_codesign/rte_reallocation_r0_review_packet_20261006.md), independent all-odd proof, scoped prior art and conditional R1 proposal.
- [Technical checker](scripts/tracks/algorithm_codesign/check_rte_reallocation_symbolic.py) and [R0 provenance](artifacts/track_b_rte_reallocation_r0/2026-10-06/audit_manifest_v1.json). A80/B9 exact fixtures; one technical run, science 0.
- R0 math checked; new-method priority/resource value unresolved. Existing STOPs remain; RUN_READY=false; GPT review required.


Track B最新は[BS-0.5 method/target設計監査](docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md)。
現candidateは同じsparse/operator LCUと別delta未定義のため専用arm/new-method claimを外す。
ordinary finite-RTEの形式仕様、O/C別比較、多資源Pareto、Wada grouped LCUをreviewへ戻す。
12 targetはcommuting control4＋performance8。新実装/science/tests0、RUN_READY=false、mandatory STOP。
[監査manifest](artifacts/track_b_bs05_design_audit/2026-10-06/audit_manifest_v1.json)。application採否・route closureはGPT判断。
旧design v1、SP results/markers、BF/BM closure、A証拠は維持。以下の「最新」は判断履歴。

Track B最新は[SP-1後GPT reviewのblock合成仕様化](docs/tracks/algorithm_codesign/block_synthesis_design_review_20261006.md)。
数学/API、claim/強い対照、小型pilot未承認案を一つの入口へ整理したdocs-only準備。
有限精度coherent対象・block単位・誤差配分を比較するRQ案。finite-RTE平均Mの直接合成は未実証候補。
[input identity／設計manifest](artifacts/track_b_block_synthesis_design/2026-10-06/)。実装/新science/tests0、RUN_READY=false。
既存SP-1 result/marker・BF/BM closure・A証拠を保持。資料公開後STOPし、具体domain/辞書/対照/会計の採否はGPTへ戻す。
以下の「最新」は当時の履歴。

Track B最新は[SP-1一回結果・保存値監査](docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)。
固定S `0d01ed9a`→direct authorization-only A `9630e061`でrun1、SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW。
48 rows／96 axes完了、42 rows適格／6 rowsモデルshot cap。technical failureなし、保存値監査PASS。
CのDRはn=8/16でgain、32でloss、64でcap。CのD-only/R-only material gainは0。
[結果・marker・evidence manifest](artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/)、
[保存field専用監査](scripts/tracks/algorithm_codesign/audit_sp1_saved_result.py)、
[GPT側の研究再評価依頼](docs/tracks/algorithm_codesign/sp1_post_run_gpt_review_request_20261006.md)。
mandatory STOP、retry0、追加science未認可。以下のsource準備・未実行statusは当時の履歴として保持する。

Track B最新は[SP-1結果前契約・source最終review](docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)。
GPTがdomainを採用し、必須修正を反映した。role/mask非依存pre-placement fusion、加法的primitive T費用。
16登録pathの静的監査はfusion candidate/actual/cross-roleすべて0。実adapter/runnerと59 focused testsを追加。
[source manifest](artifacts/track_b_sp1_wrapper_source/2026-10-06/source_manifest_v1.json)。
science sweep/合成/trajectory/GPUは0、RUN_READY=false、authorizationはpending。
新Sを固定してGitHub公開後、最終source reviewへSTOP。以下の「未完了」は受領前の履歴。

Track Bの最新準備は[SP-1 wrapper累積・placement契約案](docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_proposal_v1.md)。
GPT reviewを具体化した12 synthetic wrapper／4 mask案と、独立namespaceのFraction会計kernel／17 focused tests。
science sweep・合成・trajectoryは0、RUN_READY=false。common fusion／実adapter／source freeze／別authorizationは未完了。
[準備manifest](artifacts/track_b_sp1_wrapper_preparation/2026-10-06/preparation_manifest_v1.json)。
SP-0.5結果・marker、BF/BM closures、A証拠を保持し、公開後STOPして契約案reviewへ戻す。
以下は受領前の履歴。

Track B最新は[SP-0.5一回実行・保存値監査・GPT handoff](docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md)。
source `65f6fcdb`、直下authorization `9477cd2f`のHEADで一回だけ実行し、PRIMITIVE_TRADEOFF_EXISTS。
23/23 keys・16/16 rows、strict14、zero-cost1、J=1 control1、error/cap/runtime/numeric failureなし。
[元resultとmarker／監査manifest](artifacts/track_b_sp05_economics_result/2026-10-06/v1/)、
[保存field専用監査script](scripts/tracks/algorithm_codesign/audit_sp05_saved_result.py)。J再評価・science rerunなし。
mandatory STOP、retry0、wrapper pilot未認可。cheap exact notchのprimitive確認のみ、研究方針はGPTへ戻す。
以下の準備・未実行の記述は結果前履歴として保持する。


Track Bの現段階は[SP-0.5経済性gate・結果前source review](docs/tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md)。
GPT reviewに従い16-cell wrapper pilotの前に、合成器一つ・exact π/4 catalogue一つ・8 target／23 keysを固定する。
B専用実装と34 focused technical testsは準備済み。登録target合成・J採点0、RUN_READY=false、別実行承認待ち。
[preparation manifest](artifacts/track_b_sp05_economics_preparation/2026-10-06/preparation_manifest_v1.json)。
B-F限定closure／BM現new-method closure／旧BM-1未実行／過去STOPを保持する。以下の「最新」は受領前の履歴。


Track B最新は[BM-0.5後の合成・placement設計仕様案](docs/tracks/algorithm_codesign/synthesis_placement_design_review_20261006.md)。
GPT reviewに従いB-F限定仮説とB-M現adapterのnew-method路線を閉じ、旧BM-1は未実行のまま保持する。
新候補の四mask、ancilla込みPAI、有限bias／weight／shot／T会計、原始pilot案を一文書と小さいJSONへ整理した。
一般原理は既知、利益・新規性は未実証。angle inventoryは確認した保存記録では未確立。
科学実行・実装・testsは0、次段未認可、利用者/GPT reviewへSTOP。以下のBM-0.5以前は履歴として読む。

Track B最新は[BM-0.5同値性監査](docs/tracks/algorithm_codesign/bm05_review_packet_20261005.md)。
固定nested列のcompact BCHとBM分解は三次で同値。同backend／同集約ならscoreも一致、現new-method gateは不通過。
BM-1未実行・未認可。applicationの必要性／別deltaはGPTへ返してmandatory STOP。以下はBM-0までの履歴。

Track Bの最新文書は[BF-A後review・BM-0 packet](docs/tracks/algorithm_codesign/bm0_review_packet_20261005.md)。
GPT reviewのB-F限定closureを記録し、B-M native列・DF評価・小型pilotを具体化した設計案。
新scienceは0、B-M新規性／BM-1 scopeはreview待ち、実行未認可、mandatory STOP。以下は既存履歴を保持する。

GPTによる研究Bの全面再設計は[固定資料handoff index](docs/tracks/algorithm_codesign/research_redesign_handoff_20261005.md)から読む。
BF-A・過去STOP・最新Aの別commit参照・未公開既存文書snapshotをまとめたdocs-only入口。新科学計算・方針採択なし、mandatory STOP。

Track B worktreeの現在の入口は[Algorithm Co-design](docs/tracks/algorithm_codesign/README.md)。
最新は[BF1-R0復元結果](docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md)：
read-only recovery complete、事後にpreregistered primary BF-Aを復元。全結果後mandatory STOP、研究判断はGPT側。
新たな利用者指示は[BF1-R0 read-only recovery](docs/tracks/algorithm_codesign/bf1_read_only_recovery_contract_v1.md)。
既存cellからの一回の復元だけを認可し、science rerunは認可しない。module/runner/tests/contractはB専用namespace。
原science実行は[BF-1一回実行の結果照合](docs/tracks/algorithm_codesign/bf1_one_shot_result_validation_20261005.md)。
別branchの[JSON保存修正と限定検証](docs/tracks/algorithm_codesign/bf1_serialization_repair_review_20261005.md)は
技術作業のみ。研究方針全体の修正はGPT側で扱い、科学再実行は認可しない。
JSON保存例外により`INCONCLUSIVE`、mandatory STOP、retryなし。B code/runner/test/preparation/result artifactは
同READMEから辿る。以下のA記録はこのbranchのM2 result snapshotであり、並行Aの最新状態へは更新しない。

最終更新：2026-10-04

このファイルは、人またはGPTがリポジトリ全体を読むときの入口である。研究内容の正本、
実装、検証コード、結果データ、発表資料を区別し、古い研究経路を現在の結論として読まない
ための案内をまとめる。

PR-2の最新状態は、旧S0 STOPとdevelopment-only S2結果を保持し、matched-accuracy resource-map研究の
compile-free M1-Aを`SELECTION_LIMITED`で停止した後、`PROCEED_BOUNDED_COMPILE_EXPANSION`として
M1-B1を実行・検証した段階である。旧16-cell selectorは「proxyではactual frontierを完全に保持できなかった」
監査結果として保存する。M1-Aでaccuracy適格だったB2/B3 194 cellを32 trajectory・Re/Im二軸、B0/B1 16 cellを
二軸で測る12,448-wrapper mapは全件完了した。actual six-metric ParetoはB2 rank 3、q=1の2件で、
primary RZ point minimumは`B2-rank3-q1-r4-K2`、状態準備cost感度のlower envelopeも全てB2だった。
研究判断は`CONTINUE_RESOURCE_STUDY`。その後、development actual Pareto 2件とB0/B1/B3代表を合わせた
5構成、primary RZ、6指標Pareto、10% materiality、重大cost underestimate、4 terminal statusを
M2 held-out transfer契約へzero-compute固定した。入口は
`docs/research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md`と
`docs/research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md`と
`docs/research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md`と
`docs/pr2_matched_accuracy_m1_b1_result_validation.md`と
`docs/research/pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md`。追加96 trajectory、held-out H4 1.30 Å、
transfer、S3は当時未実行・未承認だった。M2外部reviewの修正要求は
`docs/research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md`へ反映し、
usable B2だけをsupportとratioに使うv2 source/planをcommit固定した。science実装の入口は
`docs/research/pr2_matched_accuracy_m2_transfer_execution_implementation.md`である。
actual source `2978e2f`、別authorization `90a9f24`、最終review承認と利用者指示を経て、M2を一度だけ実行した。
最新は[結果照合](docs/pr2_matched_accuracy_m2_transfer_result_validation.md)：H4 1.30 Å、STO-3G、DF rank12、
`T=0.8`の固定5構成、196 wrapperが完了し、B2二件がParetoに残って`TRANSFER_SUPPORTED`。
result commitへ収録するlocal execution evidenceで、固定構成のtransfer以外へ一般化しない。現在はmandatory STOP、研究方針全面review待ち。
追加96、held-out再探索、S3、別geometry/分子へ進まない。

## 最初に読む順序

1. [`docs/research/研究概要・現状.md`](docs/research/研究概要・現状.md)
   研究目的、採用中の前提、現在の実装・検証段階、主要結果、未解決事項の最新要約。
2. [`docs/research/研究目的・研究課題.md`](docs/research/研究目的・研究課題.md) と
   [`docs/research/研究方法・解析手順.md`](docs/research/研究方法・解析手順.md)
   研究設計と解析手順の正本。
3. [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md)
   各結果を再利用できるか、欠落データや再計算が必要かを示す状態表。
4. [`artifacts/validation_manifest.json`](artifacts/validation_manifest.json)
   検証結果、生成コード、テスト、成果物を結ぶ機械可読な証拠台帳。
5. [`docs/research/prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)
   事前検証カタログのうち、実施したID・work packageと文書、artifact、runner、testの対応表。
6. 数値を使う場合だけ、対応する `docs/*_validation.md` と `artifacts/` のJSONを確認する。

`docs/research/研究ノート/` は意思決定の時系列記録であり、現在の仕様ではない。過去の暫定値が
後日のノートや研究概要で変更されている場合は、最新の研究概要と規範文書を優先する。

## 現在の研究段階

2026-09-25現在、A0後のP-B/P-C/P-Aテーマ選定と、各候補の停止点まで完了した。P-Bは現H4 gridで
実用的signal差がなく停止した。P-A v1はblind transferを通過したが、形式化とforced-support
Taylor-order-2検証でinterval分割・一区間baselineとの差が全30 taskで0となり、interval claimを停止した。
P-Cは0.80--1.20 Åの局所pilotを通過したが、事前登録した0.70--1.60 Å tracking・breakdown検証では
追跡prefixが独立prefixと全点同一で、1.40/1.60 Åのcoefficient予測誤差が33.653%/123.245%となった。
pair予測、continuity診断、mechanism discriminationも固定gateを通らず、
`stop_pc_current_h4_family_as_primary`となった。

従って、A/B/Cに確認済みの主研究候補はない。P-Dのpilotと現実化gateは通過したが、S0/S1の
公平再最適化ではB1b/B2/B4が全scopeで同じnew fourth、`delta=0.2,R=16`を選び、B2のB4 regretは0、
限定K4でも選択は変わらなかった。finite補正が選択を変えるCase C/Dの証拠は得られていない。
outer-stageだけのB1aは一段延長後も`m_D=128`上限へ達したため、一次分類Case Bに
`undetermined_boundary`を付け、GO判定を出さず停止した。P-Dの研究方針とbaseline設計を再検討し、
H12、長RPE、compiled総costへはまだ広げない。
その後、固定S1 artifactの事後再解析で主baselineのB1b/B2/B4一致を確認し、P-D S2を停止した。
R3も一般multi-fidelity法との差分を固定できず`STOP_R3_NO_METHOD_DELTA`となった。2026-09-26には、
有限RTE打切り誤差をHadamard複素信号の位相方向と半径方向へ分ける新候補についてFR-0を完了し、
事前登録済みFR-1を固定2×2 toyで実行した。補正後演算子$A_{\rm corr}$と実際の平均
$A_{\rm mean}=A_{\rm corr}/\mathcal B$を区別した境界は495適用recordで違反0、semantic・負時間・K4・
非対称配置も通過した。一方、利用可能な$\underline\rho=0.8$でのG2は不通過で、真の$\rho$を使う
場合だけ改善した。その後のFR-R1aでは正scalar処理後の片側差が0、FR-R1bでは同情報のstrict gainが
8件あったが固定予算の片側認証差が0となり、現行判断は`MECHANISM_ONLY_NO_PRACTICAL_GO`である。
FR-R2は開始せず、2026-09-27に研究を候補探索から理論成果の完成へ切り替えた。C1/C2を中核、C3を
条件付き応用とし、先行研究との定理単位の照合と証明義務T1--T4を終えるまで新しい数値計算を行わない。

それ以前の中心課題は、DF Hamiltonianを決定論部分とランダム部分へ分けたpartial-$S_2$について、有限RTE、
RPEの信号半径・測定回数、1 shot当たりのコンパイル後回路コストを接続することだった。PF係数、有限RTE、
ランダム回路コスト、短いRPE段の接続は限定条件で検証済みだが、最終総cost評価には達していない。
既存設定`CA/10`を暫定目標にすると$q_{\max}=32768$が必要で、従来の固定
$\delta=0.1,r=4,K=2$は長roundへ単純外挿できないことを確認した。その後、H4の
実行済み$\delta$窓でround別$(r_m,K_m)$を再探索し、$\delta=0.01,0.0125,0.02$に
行列検査を通るscheduleを構成した。さらに、短時間幅0.02--0.000390625で局所回路指標が
変わらないことと、イベント列長8、16、32への移送を検査し、$r\leq16$では1--3イベント、
$r=32$では4イベント補正を用いる中央RTEブロックcost proxyを接続した。この限定proxyでは
$\delta=0.02$が全6指標で最小となった。

2026-09-21に、広い検証backlogを一括実行せず、研究方向を選ぶGate S1を先に置く方針を採用し、
最初の`WP00 -> WP02 -> WP01-S`を実行した。H4 rank-12固定snapshotへPF入力を結び直し、
CA/CA/10/CA/100のround horizonを監査した。CA/10は既存3 schedule・56点行列検査を再利用できるが、
CA/100は既存3つの$\delta$がすべて経験的PF予算を超える。CA/10の条件付き比較では
$L_D=0$を拡張した解析的成分作用数proxyでscreen outし、$L_D=12$の点推定は$L_D=3$より
21.0%低かった。
ただし保守的な長$q$移送scenarioでは区間が重なるため、結論は未決定である。続くWP04では、
両候補に公平な$\beta$・$\alpha$再配分を与えると決定論endpointの点推定差は4.96%へ縮み、
5%・25%区間がともに重なることを確認した。共通の主要因は$\beta$、次いで$\alpha$再配分で、
$L_D=3$のcompiled-cost整合round schedule単独の利得は固定schedule比1.93%だった。WP03では
$C_D$、論文D6、支配固有位相係数だけを差し替えた18条件の選択が全て
$L_D=12,\delta=0.02$で変わらず、係数選択も区間重なりを解消しなかった。Gate S1では
これらを統合し、区間判定を「同点」でなく`undetermined`、最大の残存不確かさをfull controlled
interrogationの回路scope・構造とした。T4とT7を主軸、T1/T2/T5/T6を限定継続、T3を保留とし、
次の一件をWP06-aとした。WP06-aではsupport限定Gaussian completionが単一Z/ZZ eventのRZを
39--62%減らした一方、異なるsupportの長さ3列ではfull basis共有より15.0%増え、事前の5% triggerが
発火した。control・relative-phaseの現行方針は同値性を通過した。WP06-bでは独立trainingから、同一
元basisのsingleton runだけsupport限定へ置換するpolicyを固定した。未使用列長3, 6でRZ -10.67%、
CX -8.13%、total depth -2.25%、最大operator残差$1.34\times10^{-15}$だった。中央RTE差だけを既存
proxyの$q$ slopeへ加えたbridgeでは$L_D=3/12$の点順位が反転した。続くWP05-aでは選択policyを
complete controlled partial-$S_2$／Hadamard wrapperへ接続し、$q=1,2$較正から未使用$q=4$をRZ最大
2.29%で予測した。中央additive bridgeのfull-wrapper RZ残差も最大2.63%で5%基準を通過した。
固定WP04条件の点順位は$L_D=3$となったが区間は重なった。WP05-bでは$q=8$と比較対照
$\delta=0.01$へ拡張し、$\delta=0.02,r=32,q=8$の初回5%逸脱を独立32 trajectoryで再検証した。
再検証では選択policyのRZ誤差0.52%、全metric最大0.54%、full basisのRZ誤差0.83%となり、
5%基準を通過した。続くWP01-D/C07では$\alpha$・shot数を候補ごとに再最適化し、点推定で
$L_D=3$が$L_D=12$より13.92%低かった。5% local model区間は僅かに分離した一方、25%移送区間は
重なるため、頑健な方向判断は未確定である。G08で後半3 roundへのcost集中を確認し、M08の
$q=16,32$直接holdoutはselected RZ 2.466%、観測RZ最大3.286%で通過した。これらの実測幅による
再集計ではlocal区間が分離するが、直接domainは$q\leq32$で25%移送区間は重なるため、頑健判定は
変わらず、現比較は最終的な科学的優位性評価ではない。
2026-09-23のM06/L08では、同一trajectoryをoptimization level 2で再compileした。q=16,32 proxyは5%基準を通過したが、固定plan focused再集計の点推定差は8.42%、区間分離上限は1.881%となり、実測selected RZ discrepancy 2.340%で区間が重なった。従ってcompilerをまたぐlocal分離は未確立で、頑健判定は`undetermined_under_compiler_and_transfer_sensitivity`である。
続くN07/P03では不確かさをsampling、model bias、compiler、長q移送、状態準備、外部移送に分離し、状態準備をRZ相当/shotのパラメータとして再集計した。`L_D=3`は2,376 shot多く、共通準備costは常に点推定利得を縮める。点推定break-evenはopt1で約9,905万、opt2 focusedで約4,707万RZ相当/shotだが、opt2 focusedはP=0ですでに区間が重なり、compiler-robustな区間優位性は確立しない。WP11では11個のartifactをT1--T7へ統合し、T4/T7を主軸、T1を範囲変更、T3を保留とした。次の一件は`L_D=3`のopt2未測定`r=1,2,4,8,16`を埋めるall-r coherent opt2再最適化であり、外部instance pilotは棄却せずその後へ延期する。
固定条件、数値、成果物は
[`research_direction_prevalidation.md`](docs/research_direction_prevalidation.md)と
[`research_direction_ablation.md`](docs/research_direction_ablation.md)、
[`research_direction_pf_sensitivity.md`](docs/research_direction_pf_sensitivity.md)、
[`research_direction_gate_s1.md`](docs/research_direction_gate_s1.md)、
[`research_direction_structure_pilot.md`](docs/research_direction_structure_pilot.md)、
[`research_direction_sequence_policy.md`](docs/research_direction_sequence_policy.md)、
[`research_direction_full_scope.md`](docs/research_direction_full_scope.md)、
[`research_direction_full_scope_extension.md`](docs/research_direction_full_scope_extension.md)、
[`research_direction_decision_cost.md`](docs/research_direction_decision_cost.md)、
[`research_direction_late_round_proxy.md`](docs/research_direction_late_round_proxy.md)、
[`research_direction_compiler_transfer.md`](docs/research_direction_compiler_transfer.md)、
[`research_direction_uncertainty_break_even.md`](docs/research_direction_uncertainty_break_even.md)、
[`research_direction_wp11_synthesis.md`](docs/research_direction_wp11_synthesis.md)に記録する。

最終的な全RPE段の総コスト最適化と、決定論PFに対する最終的な優位性評価はまだ行っていない。
最新の到達点と次の検証は、必ず
[`研究概要・現状.md`](docs/research/研究概要・現状.md)で確認する。

## ディレクトリの役割

| 場所 | 役割 | 読み方 |
|---|---|---|
| `src/trotterlib/` | ライブラリ本体 | 現行実装と検証ロジック。分類は[`src/trotterlib/README.md`](src/trotterlib/README.md) |
| `scripts/` | 実行入口 | 検証runner、batch、診断、旧経路。分類は[`scripts/README.md`](scripts/README.md) |
| `tests/` | 自動テスト | API・数値恒等式・成果物schemaの回帰検査。科学的結論そのものではない |
| `docs/research/` | 研究方針の正本 | 概要、目的、方法、評価計画、研究ノート |
| `docs/` | 実装・検証の説明 | 各検証の条件、結果、限界、実装規約。索引は[`docs/README.md`](docs/README.md) |
| `artifacts/` | 計算結果と入力snapshot | JSON等の証拠、キャッシュ、途中状態。利用規則は[`artifacts/README.md`](artifacts/README.md) |
| [`partial_randomized_trotter_prevalidation_catalog.md`](partial_randomized_trotter_prevalidation_catalog.md) | 研究方向を選ぶための事前検証backlog | 実行層・Gate S1・条件付きbranchの索引。実施済み範囲との対応は[`prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)を参照 |
| `BentoSlide構成案*.md` | 発表資料生成用の指示 | 研究の正本ではない。位置づけは[`docs/presentations/README.md`](docs/presentations/README.md) |
| ルートのPDF | 参考論文または発表資料 | 種別は[`docs/references/README.md`](docs/references/README.md)で確認 |

## 実装と検証の関係

```text
研究方針・条件
  docs/research/
        ↓
ライブラリ実装
  src/trotterlib/
        ↓
実行入口                 回帰テスト
  scripts/run_*.py   ↔    tests/test_*.py
        ↓
結果・入力snapshot
  artifacts/
        ↓
結果の説明と状態
  docs/*_validation.md
  VALIDATION_STATUS.md
  artifacts/validation_manifest.json
```

`scripts/run_*.py` はメインロジックの置き場所ではなく、引数を受け取り
`src/trotterlib/` の処理を呼び出して成果物を保存する入口である。検証名と同名の
ライブラリ、runner、test、文書、artifactを一組として読む。

## 共有サーバー向けexecution infrastructure

長時間検証を独立taskへ分割して安全に実行・再開する経路は、
`src/trotterlib/parallel_validation_executor.py`、`scripts/run_parallel_validation_batch.py`、
`tests/test_parallel_validation_executor.py`、
[`docs/server_parallel_validation_execution.md`](docs/server_parallel_validation_execution.md)を一組として読む。
これは実装・運用基盤であり、新しい科学的検証結果や最終総cost評価ではない。

WP11が選択したM06-F計算経路は、`src/trotterlib/research_direction_full_opt2.py`、
`src/trotterlib/research_direction_full_opt2_completion.py`、
`src/trotterlib/research_direction_full_opt2_extension_analysis.py`、
`src/trotterlib/research_direction_proxy_lineage_reconciliation.py`、
`scripts/run_research_direction_full_opt2_compute.py`、
`scripts/run_research_direction_full_opt2_analysis.py`、
`scripts/run_research_direction_full_opt2_completion.py`、
`scripts/run_research_direction_full_opt2_extension_analysis.py`、
`scripts/run_research_direction_proxy_lineage_reconciliation.py`、
`tests/test_research_direction_full_opt2.py`、
`tests/test_research_direction_full_opt2_completion.py`、
`tests/test_research_direction_full_opt2_extension_analysis.py`、
`tests/test_research_direction_proxy_lineage_reconciliation.py`、
[`docs/research_direction_full_opt2.md`](docs/research_direction_full_opt2.md)を一組として読む。
初期36 taskとfresh-32拡張15 taskは51/51で完了し、両gate通過後のcoherent再最適化も完了した。
A0は新規compileなしで最新fresh `q=1,2` proxyを旧`q=16,32`固定holdoutへ再照合した。
compute resultは`artifacts/research_direction_full_opt2/2026-09-24/`、最終監査・解析は
`artifacts/research_direction_full_opt2/2026-09-25/`に置く。

2026-09-25からはA0完了後の研究テーマ選定を`P-B -> P-C -> P-A`で行う。P-Bは
`src/trotterlib/research_direction_signal_weight_pilot.py`、
`scripts/run_research_direction_signal_weight_pilot.py`、
`tests/test_research_direction_signal_weight_pilot.py`、
[`docs/research_direction_signal_weight_pilot.md`](docs/research_direction_signal_weight_pilot.md)を一組として読む。
現H4 gridでは案Bを進めるsignal差が得られず停止した。P-Cは
`src/trotterlib/research_direction_geometry_energy_difference_pilot.py`、
`scripts/run_research_direction_geometry_energy_difference_pilot.py`、
`tests/test_research_direction_geometry_energy_difference_pilot.py`、
[`docs/research_direction_geometry_energy_difference_pilot.md`](docs/research_direction_geometry_energy_difference_pilot.md)、
`artifacts/research_direction_geometry_energy_difference_pilot/2026-09-25/`を一組として読む。
固定H4 5 geometryの未使用geometry/delta差分bias予測gateを通過して案Cを候補として残し、
P-Aは`src/trotterlib/research_direction_joint_synthesis_pilot.py`、
`scripts/run_research_direction_joint_synthesis_pilot.py`、
`tests/test_research_direction_joint_synthesis_pilot.py`、
[`docs/research_direction_joint_synthesis_pilot.md`](docs/research_direction_joint_synthesis_pilot.md)を一組として読む。
未使用列長3、5、8でinterval-union DPが現行policy比pooled RZを7.19%減らし、全gateを通過した。
3 pilotの比較は`research_direction_theme_selection.py`と同名runner/test、
[`docs/research_direction_theme_selection.md`](docs/research_direction_theme_selection.md)へ固定した。
P-Aを暫定主題、P-Cを副候補、P-Bを現範囲で停止とした。続くscoped prior-art auditは
[`docs/research/pa_joint_synthesis_prior_art_audit.md`](docs/research/pa_joint_synthesis_prior_art_audit.md)に、
計算前のblind条件は
[`docs/research/pa_joint_synthesis_blind_validation_preregistration.md`](docs/research/pa_joint_synthesis_blind_validation_preregistration.md)に固定した。
実装は`src/trotterlib/research_direction_joint_synthesis_blind_validation.py`、
`scripts/run_research_direction_joint_synthesis_blind_validation.py`、
`tests/test_research_direction_joint_synthesis_blind_validation.py`を一組として読む。compile前dry-runで
48 holdout taskと6 operator probeのevent digestを
`artifacts/research_direction_joint_synthesis_blind_validation/2026-09-25/`へ固定した。未使用H5 snapshotと
H4 opt2 compiler contextの2 stratumは48/48 holdout、6/6 operator probeまで完了し、両方で全6 gateが
通過した。結果とscopeは
[`docs/research_direction_joint_synthesis_blind_validation.md`](docs/research_direction_joint_synthesis_blind_validation.md)に記録する。
blind gate時点ではP-A v1を正式候補へ進めた。続く形式化は
`src/trotterlib/research_direction_joint_synthesis_formalization.py`、
`scripts/run_research_direction_joint_synthesis_formalization.py`、
`tests/test_research_direction_joint_synthesis_formalization.py`、
[`docs/research/pa_joint_synthesis_v1_formalization.md`](docs/research/pa_joint_synthesis_v1_formalization.md)を
一組として読む。DP最適性は有限proxy候補内で形式化できたが、全54 recordが1 run 1 segmentで、
interval分割と非零Taylor-order構造は未検証だった。P-Aを条件付き候補へ狭め、次はこの2点を明示的な
one-segment baselineに対して判別する。H12、長RPE総cost、coupling/noiseは次の必須作業ではない。

続く非退化mechanism validationは
`src/trotterlib/research_direction_joint_synthesis_mechanism_validation.py`、
`scripts/run_research_direction_joint_synthesis_mechanism_validation.py`、
`tests/test_research_direction_joint_synthesis_mechanism_validation.py`、
[事前登録](docs/research/pa_joint_synthesis_mechanism_validation_preregistration.md)、
[結果文書](docs/research_direction_joint_synthesis_mechanism_validation.md)を一組として読む。
training/blindをDF fragmentで分離したforced-support order-2全30 taskでcandidateと一区間baselineの
plan・compiled metricが完全一致し、run内分割は0件だった。P-Aのinterval claimを停止してP-Cへ戻る。
H12、長RPE総cost、coupling/noiseは次の必須作業ではない。


続くP-C tracking・breakdown validationは
`src/trotterlib/research_direction_geometry_tracking_breakdown.py`、
`scripts/run_research_direction_geometry_tracking_breakdown.py`、
`tests/test_research_direction_geometry_tracking_breakdown.py`、
[事前登録](docs/research/pc_geometry_tracking_breakdown_preregistration.md)、
[結果文書](docs/research_direction_geometry_tracking_breakdown.md)、
`artifacts/research_direction_geometry_tracking_breakdown/2026-09-25/`を一組として読む。
固定8 geometry・2 policyの16/16 taskを完了した。追跡prefixは全点で独立先頭3 fragmentと同一、
blind coefficientは3/5、pairは1/4だけが固定誤差基準内で、診断正解率は50%だった。
7 gate中3 gate通過でcurrent H4 familyのP-Cを主研究候補から停止した。
先行P-C pilotの訂正版exact-energy artifactも同時に参照する。

P-D energy係数・random-tail負担Pareto監査は
`src/trotterlib/research_direction_energy_tail_pareto.py`、
`scripts/run_research_direction_energy_tail_pareto.py`、
`tests/test_research_direction_energy_tail_pareto.py`、
[事前登録](docs/research/pd_energy_tail_pareto_preregistration.md)、
[結果文書](docs/research_direction_energy_tail_pareto.md)、
`artifacts/research_direction_energy_tail_pareto/2026-09-25/`を一組として読む。
development `L_D=3`とblind `L_D=4`で同じselection reversalを確認し、7 gate全てを通過した。
ただしexact two-block pilotであり、負時間RTEと内部`H_D`誤差の次gateを通るまでは
`P-D-conditional-candidate`として扱う。

P-D現実化Go/No-Goは`research_direction_pd_realization.py`と同名runner/test、
[事前登録](docs/research/pd_realization_go_no_go_preregistration.md)、
[結果文書](docs/research_direction_pd_realization.md)、
`artifacts/research_direction_pd_realization/2026-09-25/`を一組として読む。

続くS0/S1公平再最適化は
`src/trotterlib/research_direction_pd_fair_comparison.py`、
`scripts/run_research_direction_pd_fair_comparison.py`、
`tests/test_research_direction_pd_fair_comparison.py`、
[S0契約](docs/research/pd_primary_research_contract.md)、
[既知baseline](docs/research/pd_prior_art_and_baselines.md)、
[S1事前登録](docs/research/pd_s1_fair_comparison_preregistration.md)、
[結果文書](docs/research_direction_pd_fair_comparison.md)、
`artifacts/research_direction_pd_fair_comparison/2026-09-26/`を一組として読む。
B1b/B2/B4は同じ選択となりCase C/Dの証拠は得られず、B1aの`m_D`上限依存により
`stop_s1_undetermined_boundary_no_go_decision`で停止した。

その固定artifactの事後再解析は
`src/trotterlib/research_direction_pd_s1_posthoc.py`、
`scripts/run_research_direction_pd_s1_posthoc.py`、
`tests/test_research_direction_pd_s1_posthoc.py`、
[外部review](pd_s1_review_5c331f0.md)、
[事後計画](docs/research/pd_s1_posthoc_reanalysis_plan.md)、
[結果文書](docs/research_direction_pd_s1_posthoc.md)、
`pd_s1_posthoc_reanalysis_v1.json`を一組として読む。一次Case Bは保存し、B1b/B2/B4だけを
固定候補集合でCase A相当と事後解釈する。P-D S2は開始せず、R3も未採用である。

R3の次段判断は
[R3先行研究監査と条件付き最小研究契約](docs/research/r3_prior_art_and_minimal_contract.md)を読む。
広いsplit/error/cost最適化は既存研究との重複が強いためNo-Goである。R3-Sはselectiveな認証・棄却へ狭めて監査したが、一般certified multi-fidelity法との差分と
quantum-specific保証を固定できなかった。`STOP_R3_NO_METHOD_DELTA`でR3を停止し、数値pilotを開始しない。

finite-RTE phase/radius分離FR-1は
`src/trotterlib/finite_rte_phase_amplitude.py`、
`scripts/run_finite_rte_phase_amplitude.py`、
`tests/test_finite_rte_phase_amplitude.py`、
[FR-0契約](docs/research/finite_rte_phase_amplitude_contract.md)、
[scoped先行研究監査](docs/research/finite_rte_phase_amplitude_prior_art.md)、
[FR-1事前登録](docs/research/finite_rte_phase_amplitude_fr1_preregistration.md)、
[結果文書](docs/finite_rte_phase_amplitude_validation.md)、
`artifacts/finite_rte_phase_amplitude/2026-09-26/`を一組として読む。
33条件・99状態でG0/G1/G3/G4を通過し、G2は不通過だった。判定は
`GO_FR2_MECHANISM_ONLY`であり、利用可能情報による実用的GOではないため旧FR-2は開始しない。
その後のFR-R0で正scalar分離、情報層、強いbaseline、非一様系のdecision gateを正式契約へ固定した。
FR-R1a事後解析は完了し、正scalar処理後にFR固有の片側認証差が残らない
`POSTHOC_SCALAR_EXPLAINS_OLD_GAIN`となった。続くFR-R1b非一様4×4検証では、同情報FR境界が
norm境界より厳しい8 witnessを得た一方、固定予算の片側認証差は0だった。現行判断は
`MECHANISM_ONLY_NO_PRACTICAL_GO`で、強制停止中である。

FR-1後の再設計は、[提案文書](fr1_revised_research_plan_20260926.md)と
[FR-R0正式契約](docs/research/fr_revision_scalar_structure_contract.md)を読む。後者が現行の規範で、
正scalar分離、I0/I1/I2、共通$\gamma$と最適化比較、FR-R1の事前登録要件を固定する。
statusは`FR_R1B_COMPLETE_MECHANISM_ONLY_NO_PRACTICAL_GO_MANDATORY_STOP`である。
[FR-R1a実装](src/trotterlib/fr_revision_fr1a_posthoc.py)、
[runner](scripts/run_fr_revision_fr1a_posthoc.py)、
[test](tests/test_fr_revision_fr1a_posthoc.py)、
[FR-R1a事後計画](docs/research/fr_revision_fr1a_posthoc_plan.md)と
[FR-R1a結果](docs/fr_revision_fr1a_posthoc.md)、
[artifact](artifacts/fr_revision_fr1a_posthoc/2026-09-26/)、
[FR-R1b事前登録](docs/research/fr_revision_nonuniform_preregistration.md)、
[FR-R1b実装](src/trotterlib/fr_revision_nonuniform.py)、
[runner](scripts/run_fr_revision_nonuniform.py)、
[test](tests/test_fr_revision_nonuniform.py)、
[FR-R1b結果](docs/fr_revision_nonuniform.md)、
[artifact](artifacts/fr_revision_nonuniform/2026-09-27/)を一組として読む。
FR-R2は開始していない。

FR-R1b後の現行方針は、[研究主張・証明義務・完成原稿契約](docs/research/fr_research_claim_and_manuscript.md)
を読む。正scalar分離と方向依存補正をC1、同じ情報層でのstrict improvementと改善不能条件をC2、
resource designを条件付きC3として分離する。新規性とT1--T4の証明監査が終わるまでは、既存artifact
だけを用いて原稿を完成させ、FR-R2、H4/H12、長RPE、新しいgridへ進まない。

## ファイルの状態区分

### 現行の正本

- `docs/research/研究概要・現状.md`
- `docs/research/研究目的・研究課題.md`
- `docs/research/研究方法・解析手順.md`
- `docs/research/数値実験・評価計画.md`
- `VALIDATION_STATUS.md`
- `artifacts/validation_manifest.json`
- `docs/research/fr_revision_scalar_structure_contract.md`（FR-R0の現行比較契約）
- `docs/research/fr_research_claim_and_manuscript.md`（FR-R1b後の研究完成フェーズ契約）

### 現行実装・検証

- `docs/research/pr2_s0_s1_execution_amendment_v3.md`、
  `artifacts/pr2_s1_s3_preregistration/2026-09-28/pr2_s0_s1_authorization_manifest_v3.json`
  （S0実行、条件付きS1 correctness、S1後mandatory STOPの結果前許可）
- `src/trotterlib/pr2_s0_s1_validation.py`、`scripts/run_pr2_s0_s1_validation.py`、
  `tests/test_pr2_s0_s1_validation.py`（snapshot/identity/corrected-estimator gateとS1 correctness-only経路。
  S2/S3 commandは実装しない）
- `src/trotterlib/pr2_v4_s2_development_validation.py`、`scripts/run_pr2_v4_correctness.py`、
  `scripts/run_pr2_s2_development.py`、`tests/test_pr2_v4_s2_development_validation.py`
  （別snapshot系列のV4 correctnessとdevelopment-only S2比較のserial経路）
- `src/trotterlib/pr2_v4_s2_parallel_execution.py`、
  `scripts/run_pr2_s2_development_parallel.py`、
  `tests/test_pr2_v4_s2_parallel_execution.py`、
  `docs/pr2_s2_parallel_execution.md`（同じS2 cellと段階barrierを保つbounded CPU並列実行層。
  実H4 S2結果は`docs/pr2_v4_s2_development_validation.md`と
  `artifacts/pr2_v4_s2_development/2026-09-29/`で追跡する）
- `docs/research/pr2_matched_accuracy_prior_art_gate_v1.md`、
  `docs/research/pr2_matched_accuracy_resource_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_implementation_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md`、
  `docs/research/pr2_matched_accuracy_m1_execution_authorization_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_execution_authorization_v1_1.md`、
  `docs/research/pr2_m1_a_selection_limited_external_review_request_3c1831e.md`、
  `docs/research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md`、
  `docs/research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md`、
  `docs/research/pr2_m1_b1_execution_authorization_external_review_request_8fc2400.md`、
  `docs/research/pr2_m1_b1_preexecution_external_review_request_1228168.md`
  （S2後の新規性gate、M1前研究契約、zero-compute実装契約。候補identity、16-cell selector、schema、
  seed規則、追加prior-art gate、M1-A/M1-B hard barrier、compile-free M1-A予算を固定済み）
- `src/trotterlib/pr2_matched_accuracy_m1_contract.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_precompile_barrier.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_execution.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_b1_contract.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_b1_execution.py`、
  `scripts/run_pr2_matched_accuracy_m1_contract.py`、
  `scripts/run_pr2_matched_accuracy_m1_precompile_barrier.py`、
  `scripts/run_pr2_matched_accuracy_m1_a.py`、
  `scripts/run_pr2_matched_accuracy_m1_b1_contract.py`、
  `scripts/run_pr2_matched_accuracy_m1_b1.py`、
  `tests/test_pr2_matched_accuracy_m1_contract.py`、
  `tests/test_pr2_matched_accuracy_m1_precompile_barrier.py`、
  `tests/test_pr2_matched_accuracy_m1_execution.py`、
  `tests/test_pr2_matched_accuracy_m1_b1_contract.py`、
  `tests/test_pr2_matched_accuracy_m1_b1_execution.py`、
  `docs/pr2_matched_accuracy_m1_a_validation.md`、
  `docs/pr2_matched_accuracy_m1_b1_result_validation.md`、
  `artifacts/pr2_matched_accuracy_m1_contract/2026-09-29/`、
  `artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/`、
  `artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/`、
  `artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/`、
  `artifacts/pr2_matched_accuracy_m1_b1_result_validation/2026-10-03/`
  （候補列挙、synthetic barrier、development-only dense signal M1-A、194+16 cellのbounded compile
  plan/source/result/validation。science runnerはcompile map完成後に研究四分岐を自動判定せずreview待ちで
  停止し、別validatorがcheckpoint/cache再集計と研究reviewを行う。このstageではheld-out未承認）
- `src/trotterlib/pr2_matched_accuracy_m1_b1_result_validation.py`、
  `scripts/run_pr2_matched_accuracy_m1_b1_result_validation.py`、
  `tests/test_pr2_matched_accuracy_m1_b1_result_validation.py`
  （保存済みM1-A/M1-B1 artifact、全checkpoint、candidate別SQLite cacheをread-onlyで検査し、
  actual Pareto、旧selector、fixed-q=8、proxy、状態準備感度を再集計する。分子snapshot/held-outは読まない）
- `docs/research/pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md`、
  `docs/research/pr2_m2_transfer_contract_external_review_request_06b2c32.md`、
  `src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`、
  `scripts/run_pr2_matched_accuracy_m2_transfer_contract.py`、
  `tests/test_pr2_matched_accuracy_m2_transfer_contract.py`、
  `artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`
  （M1-B1の5構成、判定量、重大underestimate、4 status、196-wrapper上限をzero-compute固定する。
  v1/draftは履歴として保存し、正式v2はusable B2にPareto support/ratioを統一してcommit固定した。
  zero-compute contract runner自身はheld-outを開かず科学実行を認可しない。後続M2結果は下記の専用入口）
- `docs/research/pr2_s0_reproduction_stop_c644925.md`、
  `docs/research/pr2_s0_external_review_request_c644925.md`、
  `artifacts/pr2_s0_s1_validation/2026-09-28/`（development byte-level hash不一致による
  `STOP_INPUT_REPRODUCTION_MISMATCH`、S1未実行、held-out signal/cost/ranking未開封、および外部レビュー依頼）
- `src/trotterlib/` のDF、RTE、RPE、compiled-cost関連モジュール
- 対応する `scripts/run_*.py`、`tests/test_*.py`、`docs/*_validation.md`
- manifestに登録され、statusと限界が明示されたartifact

### 歴史的・補助的資料

- `abe_trotter_project.ipynb`：旧来の高次PF解析ノートブック
- `Partial Randomized Study Protocol.md`：初期計画と意思決定の履歴
- `codex-inst.md`：過去の実装依頼メモ
- `README_partial_randomized_pf.md`：旧screeningを含む実装経路の説明
- `docs/main_audit_20260801.md`：特定時点の監査記録
- `partial_randomized_trotter_validation_review_898da848.md`：commit `898da848`を対象にした外部GPTレビュー。
  現在のP-A/P-C/P-B判断より前の評価であり、一次証拠または現行仕様ではない
- `partial_randomized_trotter_research_redesign_20260925.md`：上記レビューを受けたテーマ再設計入力。
  pilotの着想と停止条件を確認する補助資料で、実施結果と現在の判断は正本文書を優先する
- `fr1_revised_research_plan_20260926.md`：FR-1後の再設計入力。採択済み部分の正本は
  `docs/research/fr_revision_scalar_structure_contract.md`を優先する
- `research_focus_and_completion_plan_16d4482_20260927.md`：FR-R1b後の完成方針を検討した入力資料。
  採択済みの主張階層、証明義務、停止条件は`docs/research/fr_research_claim_and_manuscript.md`を優先する

これらは削除していないが、現在の研究方針や最新結果を確定する根拠には使わない。

## GPTが回答・資料作成するときの確認事項

- 事前検証カタログの実施状況は、[`prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)から専用文書・artifact・testまで追跡する。
- 「理論上の関係」「ローカル検証済み」「実装のみ」「未検証」「最終結論」を分ける。
- 数値には、分子、距離、basis、DF rank、$L_D$、時間幅、Taylor cutoff、検証範囲を添える。
- `C_use`を厳密上界と呼ばない。実行したdelta窓上の経験的包絡である。
- H4/H6の係数をH12へ外挿しない。
- PFの決定論的biasとQPE/RPEの統計誤差を混同しない。
- dirty worktreeの結果やローカルテストを、公開済み・CI固定済みの証拠と表現しない。
- スライド構成案や研究ノートの単独記述より、研究概要、検証文書、manifestを優先する。

## 保守規則

研究上の決定や検証結果が変わった場合は、次を同じ変更で更新する。

1. `docs/research/研究概要・現状.md`
2. 対応する研究方針文書または検証文書
3. 当日の研究ノート
4. 証拠構成が変わる場合は `artifacts/validation_manifest.json`

新しい検証を追加するときは、可能な限り同じ語幹で
`src/trotterlib/`、`scripts/`、`tests/`、`docs/`、`artifacts/`を対応させる。

## M2 usable B2契約修正 v2（2026-10-04）

外部reviewの修正要求を[amendment v2](docs/research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)へ反映した。
Pareto supportとprimary ratioは共にaccuracy-eligibleかつprimary重大underestimateのないB2だけを使う。
v1証拠・固定5構成・seed・196-wrapper上限を維持した。この契約修正時点では科学実行とheld-out accessは未認可だった。
moduleは`src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`、runnerは
`scripts/run_pr2_matched_accuracy_m2_transfer_contract.py`、testは
`tests/test_pr2_matched_accuracy_m2_transfer_contract.py`、schema/planは
`artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`から辿れる。

## PR-2 M2科学実行コードの入口

[実装資料](docs/research/pr2_matched_accuracy_m2_transfer_execution_implementation.md)と
`src/trotterlib/pr2_matched_accuracy_m2_transfer_execution.py`、
`scripts/run_pr2_matched_accuracy_m2_transfer.py`、
`tests/test_pr2_matched_accuracy_m2_transfer_execution.py`、
`artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/`を対応させる。
契約v2と正式planはcommit固定済み。source/authorization/環境gateはheld-out読み込みより前に置き、
旧draftは履歴だけであり、正式planはCOMMIT_BOUND。source/plan固定後に別authorizationと最終reviewを経て
一回のM2を実行した。科学sourceは変更せず、結果後のmandatory STOPを維持する。

## PR-2 M2最終実行前レビューの入口

[実行authorization](docs/research/pr2_matched_accuracy_m2_transfer_execution_authorization_v1.md)と
[最終review依頼](docs/research/pr2_m2_execution_authorization_external_review_request_90a9f24.md)、
`artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/authorization_audit_v1.json`を一組として読む。
actual science sourceとplanを変更せず別authorizationをcommitし、review待ちで一旦停止した。
review承認と利用者指示を得た後に一度実行し、現在は`TRANSFER_SUPPORTED`後の研究方針review待ちである。

## PR-2 M2 held-out結果の入口

[結果照合](docs/pr2_matched_accuracy_m2_transfer_result_validation.md)と
`artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/`のresult、runner manifest、complete marker、
launch/post-execution auditを対応させる。196 wrapper、5構成、source128、pre/post84＋134 testsを検査した
result commitへ収録するlocal execution evidenceであり、immutable CIではない。`.runtime`とone-shot registryはcommit対象ではない。
固定5構成のtransfer支持だけを解釈し、全status後STOP・追加科学計算未認可を維持する。

## Track B G1限定source準備（2026-10-09）

[G1 source review](docs/tracks/algorithm_codesign/g1_source_review_20261009.md)から、
[独立構造監査・controller](scripts/tracks/algorithm_codesign/g1_decision_packet/README.md)、
[入口](scripts/tracks/algorithm_codesign/run_g1_decision_packet.py)、
[focused tests](tests/tracks/algorithm_codesign/test_g1_source_preparation.py)、
[準備artifact](artifacts/track_b_g1_source_preparation/2026-10-09/evidence_manifest_v1.json)へ辿る。
59件のlocal off-domain testsが通過。本構造監査・固定8人工LPは未実行で、別のsource-bound明示指示を待つ。
既存source/contract/result/marker/STOPと共有APIを保持する。全outcomeでSTOPしGPT G1へ戻す。

## Track B G2保存値診断（2026-10-09）

[GPT G2 handoff](docs/tracks/algorithm_codesign/g2_saved_diagnostic_handoff_20261009.md)と
[独立数学監査](docs/tracks/algorithm_codesign/g2_independent_math_audit_20261009.md)、
`scripts/tracks/algorithm_codesign/g2_saved_diagnostic.py`、
`tests/tracks/algorithm_codesign/test_g2_saved_diagnostic.py`、
`artifacts/track_b_g2_saved_diagnostic/2026-10-09/evidence_manifest_v1.json`を対応させる。
固定保存表504 profileのpost-hoc/development診断。旧G1/RA-D0結果・markerは保持し、
新registered LP/science/synthesisは0。完了後STOPし研究価値・次scopeをGPT G2へ返す。


## Track B G3有限law・known return比較（2026-10-09）

[GPT G3 handoff](docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md)、
[GPT G2 review](docs/research/track_b_G2_scientific_review_20261009.md)、
`scripts/tracks/algorithm_codesign/g3_finite_law.py`、`g3_return_comparator.py`、`audit_g3_saved_outputs.py`、
`tests/tracks/algorithm_codesign/test_g3_finite_law.py`、`test_g3_return_comparator.py`、
`artifacts/track_b_g3_finite_law/2026-10-09/evidence_manifest_v1.json`を対応させる。
承認されたdevelopment保存値288 profile/6,912 lawsと、条件付きreturn12合成/12 native event/162 lawsを完了。
限定poolでT/1Q差を記録、全B2任意混合の最適性・旧RA-D0 witnessではない。mandatory STOP、次判断はGPT G3。


## Track B G4条件付き分離・matched CTS（2026-10-09）

[G4結果/GPT handoff](docs/tracks/algorithm_codesign/g4_results_and_gpt_handoff_20261009.md)、
[独立証明](docs/tracks/algorithm_codesign/g4_A_independent_proof_20261009.md)、
[CTS結果前仕様](docs/tracks/algorithm_codesign/g4_B_matched_CTS_contract_20261009.md)、
[G4-C設計のみ](docs/tracks/algorithm_codesign/g4_C_next_validation_design_20261009.md)、
`scripts/tracks/algorithm_codesign/g4_independent_certificate.py`、`g4_cts_specialization.py`、
`g4_matched_cts.py`、`audit_g4_saved_outputs.py`、2 focused test files、
`artifacts/track_b_g4_conditional_separation/2026-10-09/evidence_manifest_v1.json`を対応させる。
固定class分離とI1 CTS対照を分け、全工程STOP・研究採否はGPT。共有API変更なし。


## 2026-10-10 Track B G5：固定辞書の終了認証（最新追記）

GPT G4 reviewを利用者が採用し、現固定toy・同辞書の実用優位実験主線を区切る。
[R0/G1/G4/G5 claim map](docs/tracks/algorithm_codesign/g5_claim_evidence_map_20261010.md)、
[G5 handoff](docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md)、
[結果前scope](docs/tracks/algorithm_codesign/g5_fixed_dictionary_closure_scope_20261010.md)、
[static access inventory](docs/tracks/algorithm_codesign/g5_static_access_inventory_20261010.md)。
[stdlib saved-only verifier](scripts/tracks/algorithm_codesign/g5_fixed_dictionary_closure.py)、
[off-domain tests](tests/tracks/algorithm_codesign/test_g5_fixed_dictionary_closure.py)、
[manifest](artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/evidence_manifest_v1.json)、
[研究note](docs/research/研究ノート/2026-10-10_track_b_g5_fixed_dictionary_closure.md)を入口とする。
x1/4・7 prototypes/3 precision・6頂点252 profiles、同Bernstein/L1-charge digital classへの保守的三下界を同CTS lawが下回る。
新LP/synthesis/circuit/matrix/DF/分子/NPZ/GPUは0。限定理論成果は保持、Track B全体の終了・次method採択ではない。
mandatory STOP。上記より前のpending/最新記述は当時の履歴としてそのまま保持する。


## 2026-10-10 Track B G6：return generatorの独立数学監査（最新追記）

利用者採用の[GPT G5研究方針](docs/research/track_b_G5_research_direction_review_20261010.md) §19に基づく新しい探索的数理監査。
[G6結果/GPT引継ぎ](docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md)、
[一般証明](docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)、
[一次文献・同値性](docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md)、
[finite-bit/access](docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)。
[局所prototype](src/trottertracks/algorithm_codesign/return_aggregation.py)、
[technical checker](scripts/tracks/algorithm_codesign/g6_return_generator_audit.py)、
[16 focused tests](tests/tracks/algorithm_codesign/test_g6_return_aggregation.py)、
[記録](artifacts/track_b_g6_return_generator_audit/2026-10-10/)。
ideal数式を確認、母関数/zero-fillは既知components。native費用・独立新規性は未確定。科学実行0、mandatory STOP。


## Track B G7（2026-10-10、source preparation）

GPT G6 review §12の限定予算/control/implementation economics。
[数学・実行契約](docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md)、
[source](src/trottertracks/algorithm_codesign/g7_generator.py)、
[runner](scripts/tracks/algorithm_codesign/g7_budget_control_economics.py)、
[tests](tests/tracks/algorithm_codesign/test_g7_budget_and_control.py)、
[contract](artifacts/track_b_g7_budget_control_preparation/2026-10-10/contract_v1.json)。
2 known development inputs/4 arms/24 fixed keys、conditional controlled-Q/Rz/CPU vector、18 focused tests。
この追記はsource preparation。結果は別manifestへ追加し、完了/失敗後mandatory STOP。


## 2026-10-10 Track B G7：取得完了・mandatory STOP（最新追記）

[G7結果/GPT引継ぎ](docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md)。
`G7_LIMITED_IMPLEMENTATION_ECONOMICS_COMPLETE_AWAITING_GPT_REVIEW`。source ab2549f41b3546fb3940342a2162dd9ee93699c4、24 keys/8 rows、strict error PASS、retry0。
固定P5では登録3対照後にもconditional期待T減少、P3ではclosed-form対照が小さい。
hard shot cap・classical generation/angle acquisition・未指定provider costを併記。
新規性/主method/次stage未採択、G5閉鎖/G6原証拠とmarker保持、GPT判断へ戻す。


## Track B G8（2026-10-10、source preparation）

[G8 proof/contract](docs/tracks/algorithm_codesign/g8_proof_contract_and_on_demand_scope_20261010.md)：採用GPT G7 review §11に基づく限定確認。
2 known development inputs/4 same production laws、finite-provider parameter、分離failure配分、
on-demand strict Rzと対称bounded cache。G7のstatus/point comparison/consumed markerを保持。
17 off-domain focused tests、source preparation時点のnew native acquisition0。一束後mandatory STOP。
source：`src/trottertracks/algorithm_codesign/g8_bounds.py`、`g8_pipeline.py`、`g8_reference_audit.py`。
runner：`scripts/tracks/algorithm_codesign/g8_on_demand_provider_budget.py`、tests：`tests/tracks/algorithm_codesign/test_g8_on_demand_and_bounds.py`。
artifacts：`artifacts/track_b_g8_on_demand_preparation/2026-10-10/` と新result directory。


## Track B G8 completed evidence (2026-10-10)

- [Results and GPT handoff](docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md): on-demand20/8rows, hypothetical provider budget, saved-only14checks.
- [G8 result inventory](artifacts/track_b_g8_on_demand_result/2026-10-10/v1/evidence_manifest_v1.json); mandatory STOP, no native-method adoption.


## Track B G9 source preparation（2026-10-10）

[Fixed proof/scope](docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md)：GPT G8 review §14を採用。known P5/指定3-qubit provider、6 direct+5 helper診断、19 keys/新CTS1 key、23 focused tests。source固定後一束のみ、終了後mandatory STOP。

## G9 one-shot STOP（2026-10-10）

`G9_TECHNICAL_INCONCLUSIVE`：epsilon引数のFraction→mp.mpf変換で停止、native比較0 row。
新規helper attempts1 / 新規sequence取得0 / retry0。source・contract・過去941pathは不変。
23 focused testsと23 saved-output checksは準備/整合証拠で、native資源の科学結果ではない。
[G9 failure・GPT handoff](docs/tracks/algorithm_codesign/g9_results_and_gpt_handoff_20261010.md)。mandatory STOP、次の研究判断はGPTへ返す。

## G9 v2 API-boundary source preparation（2026-10-10）

[G9 v2準備/入口](docs/tracks/algorithm_codesign/g9_v2_api_boundary_source_review_20261010.md)。精度値の型接続を最小修正、19 stub-only/launch tests PASS。
46科学条件・同19-key inventory・旧v1 source/result/markerは保持。
新実合成・登録matrix/予算/科学実行0、v2 marker absent、別authorization pending。
独立branchで資料公開後STOPし、新source-bound明示認可を待つ。

## G9 v2 one-shot completed / STOP（2026-10-10）

[G9 v2 results/GPT handoff](docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md)。`G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`、11 rows/22 axes、19 keys（new1/reuse18）、retry0。
known P5/指定3-qubit providerの登録direct6方式でclosed P5のT intercept/Kが小さく、CTSのCXは小さい。
1,866 event accounting、saved-only25 checks PASS。旧v1結果/marker、critical80/protected982は不変。
実量子shots/trajectory/DF/分子/NPZ/GPU/LPは0、source-bound local evidence。次の科学実行・採択はGPT判断、mandatory STOP。


## Track B G9 v2科学review受領（2026-10-10）

[採用review](docs/research/track_b_G9_v2_scientific_review_20261010.md) / [G10 bundle範囲](docs/tracks/algorithm_codesign/g9_v2_review_intake_and_g10_scope_20261010.md)。
return集約family限定継続、closed P5を低次数の標準実装へ。
同p/x/providerのm=3/5/7比較が次のscope、m5は保存anchor。
今回はdocs-only受領、CTS下界独立検証/G10実装・契約・実行は未実施。旧G9 result/marker/STOP不変。
[受領manifest](artifacts/track_b_g9_v2_review_intake/2026-10-10/intake_manifest_v1.json)。G10一束後mandatory STOP、科学判断はGPT。


## G10 saved-policy audit / degree-comparison source preparation（2026-10-10）

[G10 source review / GPT入口](docs/tracks/algorithm_codesign/g10_source_review_20261010.md)。保存P5 direct945 bindingsのT再計数と任意proposal固定policy下界を独立確認。
ordinary/partial/closed P3/CTSは保存closed P5と分離、general full/closed P5はこの下界では未分離。原G9分類は不変。
同p/x/3-qubit providerのm3/5/7（17 rows/34 axes）を固定。m5はsaved-only共通policy再会計、
登録P3/P7は未取得。41 off-domain focused tests PASS、固定runtime/未認可拒否を確認。
実synthesis0、G10 marker absent、authorization pending。旧source/result/auth/marker/STOP・Track Aを保持。
[evidence manifest](artifacts/track_b_g10_degree_preparation/2026-10-10/evidence_manifest_v1.json) / [contract](artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json) / [future runner](scripts/tracks/algorithm_codesign/g10_degree_matched_native.py) / [focused tests](tests/tracks/algorithm_codesign/test_g10_degree_preparation.py)。
次は固定source review→別authorization-only child→新one-shot指示。全結果STOP、研究判断はGPT。


## G10 one-shot technical STOP（2026-10-10、最新追記）

[結果/GPT引継ぎ](docs/tracks/algorithm_codesign/g10_results_and_gpt_handoff_20261010.md)。`G10_TECHNICAL_INCONCLUSIVE`：固定RSS cap512 MiB超過、guard peak539.546875 MiB。
source `05c5ef23` / authorization-only child `f5cd0755`、一回のみ、retry0。new synthesis27 / reuse19、17行保存。
technical prefixは科学判断に使用せず、原結果/marker/STOPと旧1241 pathを保持。
保存hash/sequence/会計45,605条件の監査PASS。critical113のfull source prefix保持、追記のみ。
追加science/matrix/synthesis/sampler/budget/lower処理なし。mandatory STOP、次の研究/実行scopeはGPT判断。


## G10 RSS技術調査・STOP（2026-10-10、最新追記）

[技術報告/GPT引継ぎ](docs/tracks/algorithm_codesign/g10_rss_failure_technical_investigation_20261010.md)。原分類 `G10_TECHNICAL_INCONCLUSIVE` は不変。
保存JSONのI/O検算で全serial copy＋JSON chunk list/joinのメモリ増幅を確認。
同原bytes/hashのmaterialized peak508.90625 MiB / stream253.25 MiB。元G10 heap/例外位置の完全再現ではない。
独立調査branch、既存source/result/audit/marker/STOP保持。production修正・cap変更・science再実行0。
typed streaming/lifetime/failure保存の限定scopeをGPTへ返し、mandatory STOP。


## 2026-10-10 Track B G10 RSS修正source S2準備

新sourceは `src/trottertracks/algorithm_codesign/g10_io.py`、future runnerは `scripts/tracks/algorithm_codesign/g10_degree_matched_native_v2.py`。

報告: `docs/tracks/algorithm_codesign/g10_rss_repair_source_and_gpt_review_20261010.md`。契約・根拠: `artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/`。
旧source・結果・marker・STOPを保護する。科学実行/A2/fresh production markerは未認可・未実施。mandatory STOP。


## G10 v2 one-shot technical STOP（2026-10-10、最新追記）

[結果/GPT引継ぎ](docs/tracks/algorithm_codesign/g10_v2_results_and_gpt_handoff_20261010.md)：`G10_TECHNICAL_INCONCLUSIVE`、JSON string-key validation failure、RSS237.33984375 MiB。S2 a139b91→authorization-only A2 1a2cd26、一回/retry0。正式payload/tokenなし、usable rows0。旧結果/marker/STOP・critical143/protected1300は保持。
[保存失敗監査](scripts/tracks/algorithm_codesign/audit_g10_v2_saved_failure.py) / [新資料](artifacts/track_b_g10_degree_result/2026-10-10/v2/)。mandatory STOP、追加science/repair未認可、科学判断はGPT。


## G10 S3 JSON-key compatibility preparation（2026-10-10、最新追記）

[S3 source review/GPT](docs/tracks/algorithm_codesign/g10_v3_key_compatibility_source_and_gpt_review_20261010.md) / [payload type/key audit](docs/tracks/algorithm_codesign/g10_v3_payload_type_and_key_audit_20261010.md)。旧str(key)・衝突/順序を局所正規化で保存、科学event不変。76 focused tests、17人工bytes/hash、saved/typed IO低メモリを確認。本番完了保証なし。
[future v3 runner](scripts/tracks/algorithm_codesign/g10_degree_matched_native_v3.py) / [static verifier](scripts/tracks/algorithm_codesign/verify_g10_key_compatibility_preparation_v3.py) / [tests](tests/tracks/algorithm_codesign/test_g10_key_compatibility_v3.py) / [artifact](artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/)。pending authorization、A3/本番markerなし、science0、旧v1/v2証拠保護、mandatory STOP。


## Track B G10 v3 completed / mandatory STOP（2026-10-10、最新追記）

[結果/GPT handoff](docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md)：`G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`。
S3 b9ed014→直接authorization-only A3 53a7bc4、一回/retry0。17rows/34axes、46keys（new27/reuse19）、10,936bindings。
stream/token/STOPとouter正常COMPLETE status一致。outer wall22.2261s/CPU20.0291s/peak240.015625MiB、512MiB cap不変。
保存監査のfield仮定/report組立の訂正と初回記録を保存、本番再実行0。旧結果/marker/STOP、critical180/protected1352保持。
`artifacts/track_b_g10_degree_result/2026-10-10/v3/evidence_manifest_v3.json` とsaved-only `audit_g10_v3_saved_outputs.py` を参照。
known synthetic providerの次数内有限operator比較のみ、science完了後mandatory STOP、追加科学/採択/新規性判断はGPTへ返す。

## Track B G10 v3科学レビュー採用（2026-10-11）

[GPT G10 v3レビュー](docs/research/track_b_G10_v3_scientific_review_20261010.md)を採用し、固定入力の一般full-return性能探索をG10で区切る。
低次数構成・一般構成・native限界を[構成ノート初稿](docs/manuscripts/track_b_return_aggregation_native_resource_note_v0_1.md)と
[claim/evidence表](docs/tracks/algorithm_codesign/g10_v3_claim_evidence_map_20261010.md)へ整理した。
保存値照合と図のscriptは`scripts/tracks/algorithm_codesign/{summarize_g10_v3_review_saved_values,plot_g10_v3_saved_review_figures}.py`。
新資料は`artifacts/track_b_g10_v3_scientific_review_intake/2026-10-10/`。原COMPLETEとmandatory STOPを保持し、m9/新入力/G11は未認可。

## Track B：return集約技術ノート完成・現系列の区切り（2026-10-11）

[採用した論文化/着地点レビュー](docs/research/track_b_G10_v3_publication_and_endpoint_review_20261011.md)に沿い、
[G6/G9/G10の自己完結的技術ノートv1.0](docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md)を完成。
[完了対応表](docs/tracks/algorithm_codesign/g10_publication_endpoint_completion_20261011.md)、
[新資料manifest](artifacts/track_b_return_aggregation_note_completion/2026-10-11/evidence_manifest.json)を参照。
一般構成/有限bit/P5少数群/固定native限界を統合し、一般full性能探索と同一着地点review反復は区切る。
原COMPLETE・科学source・旧証拠保持、新science0、投稿承認/新規性採択/次実行なし、mandatory STOP。

## Track B DF希薄三次補正：限定独立監査（2026-10-11）

[監査](docs/tracks/algorithm_codesign/df_sparse_correction_independent_audit_20261011.md)、[一次文献比較](docs/tracks/algorithm_codesign/df_sparse_correction_prior_art_comparison_20261011.md)、[GPT handoff](docs/tracks/algorithm_codesign/df_sparse_correction_gpt_handoff_20261011.md)。
原本MD/ZIPを固定・remote取得後、77 audit checksと15 focused controls、5 representative compileを実施。production DF/digital/native T・独立新規性は未解決、科学的GOではない、mandatory STOP。
新scriptsは`scripts/tracks/algorithm_codesign/audit_df_sparse_correction{,_native}.py`、testは`tests/test_df_sparse_correction_audit.py`、新artifactは`artifacts/track_b_df_sparse_correction_audit/2026-10-11/`。共有library/API・旧成果不変。

## Track B DF補正 whole / local Pauli：比較事前契約（2026-10-11）

[契約](docs/tracks/algorithm_codesign/df_correction_comparison_preregistration_20261011.md)、[実行案](docs/tracks/algorithm_codesign/df_correction_comparison_execution_plan_20261011.md)、[利用者/GPT handoff](docs/tracks/algorithm_codesign/df_correction_comparison_gpt_handoff_20261011.md)。最新GPT MD/ZIP原本を固定・remote照合。CD/Eの別domain、共通誤差/shot規則、最大22compile/64T-target案を結果前固定しただけで、science/compile/T合成0、runs0、source未実装、mandatory STOP。artifactは`artifacts/track_b_df_correction_comparison_preparation/2026-10-11/`。旧監査・共有API・結果/marker/STOP不変。


## 2026-10-11 Track B DF補正比較 v2 source preparation

[v2比較契約](docs/tracks/algorithm_codesign/df_correction_comparison_preregistration_v2_20261011.md) / [source計画](docs/tracks/algorithm_codesign/df_correction_comparison_execution_plan_v2_20261011.md)。GPTレビューM1–M7を実装。新namespace `src/trottertracks/algorithm_codesign/df_correction_comparison/`、runner `scripts/tracks/algorithm_codesign/run_df_correction_comparison_v2.py`、focused tests `tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py`、source/evidence `artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/`。登録CD/E/Tは0、別途source整合確認・authorizationが必要。旧source/results/markers/STOP保持、mandatory STOP。


## DF correction comparison integrity source v3 (2026-10-11)

[限定修正・GPT引き継ぎ](docs/tracks/algorithm_codesign/df_correction_comparison_integrity_repairs_v3_20261011.md)：科学契約v2不変、512 KiB分割とinventory SHA結合、88 focused PASS。登録比較/T合成0、authorization=false、mandatory STOP。


## Track B DF補正A/B比較 one-shot technical STOP（2026-10-12）

[結果/GPT handoff](docs/tracks/algorithm_codesign/df_correction_comparison_results_and_gpt_handoff_20261012.md)：`TECHNICAL_INCONCLUSIVE`。固定S' 41349b7→直接authorization-only A a1f537b、一回/retry0。CD6task保存、E-r3 whole fingerprint guardで停止、E-r6はparallel termination、T0。prefixを科学的勝敗へ使わず、source216/contract/auth/marker保持。原結果は`artifacts/track_b_df_correction_comparison_result/2026-10-11/v2/`、[保存値監査](artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/saved_output_audit_v1.json)と[evidence manifest](artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/evidence_manifest_v1.json)。mandatory STOP、次scopeはGPT/利用者判断。


## Track B DF補正比較 fingerprint原因監査（2026-10-12）

[限定技術監査/GPT handoff](docs/tracks/algorithm_codesign/df_correction_fingerprint_cause_audit_and_gpt_handoff_20261012.md)：保存source/ASTのみ。input/recursive-definition hash guardを確認、float DAG/frame受渡しに構成差、actual hash/None未保存のため因果未確定。旧`TECHNICAL_INCONCLUSIVE`/source216/result28/auth/contract/marker不変。新matrix/compile/T合成/runner0、修正未採択、mandatory STOP。[監査manifest](artifacts/track_b_df_correction_fingerprint_cause_audit/2026-10-12/evidence_manifest_v1.json)。


## Track B fingerprint診断source準備（2026-10-12）

[GPT handoff](docs/tracks/algorithm_codesign/df_correction_fingerprint_diagnostics_preparation_and_gpt_handoff_20261012.md)：B-local観測adapter、lossless matrix bytes・recursive payload・None/exception保存、登録外2-mode旧/new controls。
最終focused tests104 PASS。controlledで同じ行列でもrecursive自動名によるhash差を保存した。登録r3停止の原因は未確定。
既存moduleの数式・guard・expected・contract不変。旧marker/result/auth/STOP保持、登録science/T合成0、RUN_READY=false。
新source snapshotと全失敗test記録を公開し、mandatory STOP。次scopeの判断はGPT/利用者側。


## Track B E-r3 identity-only診断準備（2026-10-12）

[scope/source handoff](docs/tracks/algorithm_codesign/e_r3_identity_diagnostic_source_review_and_gpt_handoff_20261012.md)：保存r3 uncontrolledのみ、current→legacy各一回の診断専用runner・新launch/manifest/result namespaceを準備。
native transpile/CD/r6/Tなし。definition内部計算はobserverとCPU/RSS/output guardで会計。新off-domain tests33 PASS。
registered診断0、旧source/authorization/marker/result/STOP保持。新authorization pending・runs0。
source/scope確認と別明示one-shot承認までmandatory STOP。旧科学分類TECHNICAL_INCONCLUSIVE不変。

source固定前の最終確認：profiler context違反のtechnical STOPを追加し、new focused testsは34 PASS。初回33 PASSとdraft checkpointは保持。


## E-r3 identity-only observer修正 v2（2026-10-12）

[限定修正・技術検証](docs/tracks/algorithm_codesign/e_r3_identity_observer_repair_preparation_20261012.md)：observer設置／復元失敗をtechnical STOPへ伝播。77 focused PASS、contract全bytes・expected・capsは不変。新sourceとv2 namespaceを固定し、READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION。旧S/科学結果/marker/authorization/STOP保持、登録診断/A作成0。追加GPT詳細研究レビューは不要、別明示one-shot承認までmandatory STOP。


## Track B E/T resumption v4 preparation（2026-10-12）

[Source/contract/validation report](docs/tracks/algorithm_codesign/df_correction_et_resumption_preparation_20261012.md)：accepted layered input/strict-target/native-IR/provenance gates; CD saved-only; native22/T64 unchanged; 141 focused PASS. New namespace `src/trottertracks/algorithm_codesign/df_correction_et_resumption/`、runner `scripts/tracks/algorithm_codesign/run_df_correction_et_resumption_v4.py`、artifact `artifacts/track_b_df_correction_et_resumption_preparation/2026-10-12/`。新authorization pending、登録E/T0、mandatory STOP。旧source/results/markers/STOP保持。


## Track B E/T storage v5 preparation（2026-10-12）

[保存方式限定修正・validation・承認待ち](docs/tracks/algorithm_codesign/df_correction_et_storage_v5_preparation_and_handoff_20261012.md)。
namespace `src/trottertracks/algorithm_codesign/df_correction_et_storage_v5/`、runner `run_df_correction_et_resumption_v5.py`、
storage/integration focused tests、I/O replay/static provenance auditor、[新artifact](artifacts/track_b_df_correction_et_storage_preparation/2026-10-12/v5/)を追加。
164 local focused PASS、保存15recordのbytes/hash復元、旧284result/363非index source不変。
科学条件・native22/T64・64 MiB cap不変、登録E/T0・new A0、READY_FOR_SEPARATE_E_T_V5_ONE_SHOT_AUTHORIZATION、mandatory STOP。


## Track B E/T v5 one-shot結果（2026-10-12）

[結果・技術監査・GPT handoff](docs/tracks/algorithm_codesign/df_correction_et_v5_results_and_gpt_handoff_20261012.md)：S5 d0a12bae→authorization-only A5 0e147e27、一回/retry0。
TECHNICAL_INCONCLUSIVE、r6 controlled wholeのnumerical upper 2e-7認証失敗。CD再計算0、native10、T0、partial勝敗なし。
原219file/24,075,195bytes、saved-only audit22,821 PASS。[新audit](artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v2/)。旧source/results/markers/STOP保持、mandatory STOP。


## Track B E/T numerical observer v6（2026-10-12）

[準備・technical continuation](docs/tracks/algorithm_codesign/df_correction_et_numeric_v6_preparation_20261012.md)：same numerical guard/caps、分岐前upper/ledger/IR保存、134 local focused PASS。namespace `df_correction_et_numeric_v6`、runner `run_df_correction_et_resumption_v6.py`、new result v3。旧source/結果/marker保持。利用者の追加認可により技術修正・新versionを継続し、science変更時はGPTへ戻す。


## Track B E/T phase v7

[raw phase会計修正](docs/tracks/algorithm_codesign/df_correction_et_phase_v7_preparation_20261012.md)：非対応Wへの未実施置換誤差を除きraw位相/strict保証を保持、unsupported Tはunknown。91 focused PASS、新namespace df_correction_et_phase_v7、新result v4。旧marker/結果不変、利用者の技術継続認可内。


## E/T saved accounting recovery

[準備](docs/tracks/algorithm_codesign/df_correction_et_saved_recovery_preparation_20261012.md)：v7全E/T完了後の保存修復、large unknown dictionaryをlossless blob化、元ASTのsaved-only会計を復元する。追加science/native/T0、原630file/marker保持、新classificationはSAVED_E_T_ACCOUNTING_COMPLETE。


## Track B E/T 限定計算・保存値集計の完了（2026-10-12）

[最終技術監査・GPT handoff](docs/tracks/algorithm_codesign/df_correction_et_completed_saved_accounting_and_gpt_handoff_20261012.md)：`SAVED_E_T_ACCOUNTING_COMPLETE`。
元v7でnative22／T key39（backend33、Clifford fast path6）を取得。大きい辞書の保存停止後、同じpost-T ASTをsaved-onlyで復元した。
元v7のTECHNICAL_INCONCLUSIVE／markerは不変。復元で新science0、815 saved-output checks PASS、全Aのactual Tはunknownを維持する。
CDは保存参照のみ、別E fixtureとの混合で全PR勝敗を作らない。研究継続・新規性・資源的意義はGPT判断、mandatory STOP。
復元結果：`artifacts/track_b_df_correction_et_saved_recovery_result/2026-10-12/v2/`、監査／manifest：`artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/`。


## Track B DF三次補正 E/T：GPT科学判断・現系列の区切り（2026-10-12）

[採択済みGPTレビュー](docs/research/track_b_df_correction_et_scientific_review_20261012.md)／[系列の区切り・証拠索引](docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md)。
現generic whole実装を新規PRアルゴリズムの主線として拡張せず、この検証系列を区切る。
3条件の局所固定加法planでA completion下界>B完全event T上界を保持。A全T、4行目、全PR優劣、一般whole優劣は未確定。
有限stepのA指数平均とB weighted線形平均は区別する。Bの自動採択・Track B全体の終了を意味しない。
元v7 TECHNICAL_INCONCLUSIVE／復元SAVED_E_T_ACCOUNTING_COMPLETE／旧結果・marker・sourceは不変。
今回のCodex作業は文書・索引・provenance・公開確認のみ、新科学計算0、mandatory STOP。
同じ科学レビューは反復しない。新候補の研究判断・scope採択はGPT側、新実行は別途明示認可が必要。


## 2026-10-13 Track B：DF-RTE積生成の限定技術監査

[受領GPT方針](docs/research/track_b_post_closure_research_direction_review_20261013.md)に基づく
[方法差・source・費用監査](docs/tracks/algorithm_codesign/df_rte_product_generation_audit_v1_20261013.md)。
ideal Gaussian productは前提付きで成立。指定pipelineは既知部品へ還元され、独立した追加手順は未定義。
公開保存値だけではconditional population cost上限を認証できない。新行列/回路/compile/synthesis/NPZ/science0、
STOP_FOR_GPT_SCOPE_DECISION。元whole補正closure、Track A/G10/RA-D0結果・source・marker・STOPは維持。
[証拠manifest](artifacts/track_b_df_rte_product_generation_audit/2026-10-13/evidence_manifest_v1.json)。


## 2026-10-13 Track B：Gaussian積生成候補の区切り

[最新GPTレビュー原本](docs/research/track_b_gaussian_product_generation_scope_review_20261013.md)に従い、
[終了記録・証拠索引](docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md)を保存。
現手順を独立新規アルゴリズムとして採択せず、候補監査を区切る。期待資源の改善/非改善は未確定、性能negativeではない。
応用研究・新候補・Track B全体の終了は未採択。新science0、旧source/result/marker/STOP保持、mandatory STOP。


## Track B：時間方向相関の限定照合（2026-10-13）

[GPT方向レビュー原本](docs/research/track_b_interstep_correlated_randomization_direction_review_20261013.md)／[一次資料・式・会計の技術報告](docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md)。PRHS本文取得、qSHIFT v1/v3の表示差と実測shotの会計を確認。既知手順への対応、追加の具体的方法差は未確立。新実装・科学実行なし、既存休止・closure保持、mandatory STOP。
