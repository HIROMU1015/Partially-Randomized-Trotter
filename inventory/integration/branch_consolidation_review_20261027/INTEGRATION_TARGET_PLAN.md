# Branch consolidation: proposed targets and eight priority groups

観測: 2026-10-27T16:11:07.128871+09:00。`codex/pr-environment-inventory-20261026` input `2fb002f96b704955a20f72c29ca5277f4abbac2f`。現存484 branchを版固定して分類した。今回作成したのは計画と監査資料だけであり、研究code/test/docs、branch履歴、worktreeは変更していない。以下の配置・refactorは次回の明示指示後の提案である。

## 方針と検証の境界

全branch一括mergeではなく、既存入口の索引整理→係数/validator等の小さな共通化→必要な検証、の順とする。代表branchは実装参照先の候補であり、全研究の科学的正本や採択済み結論を意味しない。新しい統合branchは提案・作成していない。TYPEはbranch全体の格付けではなく、表に示した機能/研究関係に対する判断である。

後続検証は、(1) API互換性、(2) synthetic unit/regression、(3) 数式・仕様整合、(4) 既存成果物の参照/byte比較、(5) 数値的再検証、(6) SOURCE/input/cohort同一性確認を分ける。今回は全て静的source/metadata確認で、テスト実行0、科学計算0、科学結果の再解析0。既存文書のPASS/STOPは過去の報告であり本作業の検証PASSではない。unitという名前でも科学compile/optimizerを呼ぶsuite全体は実行対象に含めない。

|順|ID|TYPE|具体的対象|
|---:|---|---|---|
|1|INT-06|TYPE_1|DF/RTE共通ライブラリの既継承と入口整理|
|2|INT-07|TYPE_1|B2-D evaluatorの既統合と独立拡張|
|3|INT-09|TYPE_3|原稿/CEM/図表・boundary証拠の入口統合|
|4|INT-02|TYPE_2|S2/Yoshida4係数の重複とformula allowlist|
|5|INT-01|TYPE_2|整数count検証の共通coreと呼出し側policy|
|6|INT-03|TYPE_2|canonical JSONとdigestのpolicyを保つ共通化|
|7|INT-05|TYPE_2|Track B lossless storageの3完全clone|
|8|INT-08|TYPE_3|DF/Pauli/RPEの評価scope・N/G条件の証拠統合|

## 1. INT-06 — DF/RTE共通ライブラリの既継承と入口整理

**対象branchと固定tip**

- `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`
- `codex/h4-tol-p-seal-20261019` @ `821cc0f6ced57168293ea166644024867b064ffb`

**分類・代表・配置候補**

TYPE_1。代表候補 `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`。既存src/trotterlib/とPROJECT_MAPのcode入口。分子別のevidence入口は各固定commitへリンク。

**比較根拠となる実ファイル/API**

- [src/trotterlib/df_hamiltonian.py::df_hamiltonian_from_integrals](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/df_hamiltonian.py#L591) — `f29eafbf4412193e8e8660d9f051889c2cba684b`, line 591; `df_hamiltonian_from_integrals(*, constant: float, one_body: np.ndarray, two_body: np.ndarray, df_rank: int | None=None, df_tol: float | None=None, metadata: dict[str, Any] | None=None)`.
- [src/trotterlib/df_hamiltonian.py::df_hamiltonian_from_integrals](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/src/trotterlib/df_hamiltonian.py#L591) — `821cc0f6ced57168293ea166644024867b064ffb`, line 591; `df_hamiltonian_from_integrals(*, constant: float, one_body: np.ndarray, two_body: np.ndarray, df_rank: int | None=None, df_tol: float | None=None, metadata: dict[str, Any] | None=None)`.
- [src/trotterlib/df_rte_tail.py::extract_df_diagonal_tail](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/df_rte_tail.py#L563) — `f29eafbf4412193e8e8660d9f051889c2cba684b`, line 563; `extract_df_diagonal_tail(tail_id: str, blocks: Sequence[DFBlock], *, fragment_ids: Sequence[str] | None=None, basis_ids: Sequence[str] | None=None, identity_policy: IdentityPolicy='faithful_identity_in_tail', coefficient_atol: float=0.0, ranking_proxy_lambda_r: float | None=None, basis_registry: DFBasisRegistry | None=None)`.
- [src/trotterlib/df_rte_tail.py::extract_df_diagonal_tail](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/src/trotterlib/df_rte_tail.py#L563) — `821cc0f6ced57168293ea166644024867b064ffb`, line 563; `extract_df_diagonal_tail(tail_id: str, blocks: Sequence[DFBlock], *, fragment_ids: Sequence[str] | None=None, basis_ids: Sequence[str] | None=None, identity_policy: IdentityPolicy='faithful_identity_in_tail', coefficient_atol: float=0.0, ranking_proxy_lambda_r: float | None=None, basis_registry: DFBasisRegistry | None=None)`.
- [src/trotterlib/rte.py::finite_rte_distribution](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/rte.py#L1460) — `f29eafbf4412193e8e8660d9f051889c2cba684b`, line 1460; `finite_rte_distribution(dimensionless_step_time: float, finite_taylor_order: int)`.
- [src/trotterlib/rte.py::finite_rte_distribution](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/src/trotterlib/rte.py#L1460) — `821cc0f6ced57168293ea166644024867b064ffb`, line 1460; `finite_rte_distribution(dimensionless_step_time: float, finite_taylor_order: int)`.

**共通点と固有差分**

all-r→H4 sealの祖先関係を確認し、選定11 core modulesはH4/H6/Pauli各branchでも同一blob。既存共通コードを新たにコピーする必要がない。 共通moduleが同じでもHamiltonian snapshot、rank/tol、fragment順位、identity/one-body policy、L_D、cohort・cache契約は別。

**選択的な採用・不採用**

採用案: code入口の代表moduleと使用branch/固定versionの索引を追加する案。分子別adapter・validation packetへの導線を整理する。

不採用: 新merge、DFの再生成、rank選択変更、同一HEAD/同一moduleを同一科学結果と扱うこと。

**後続の検証設計と既存testの入口**

- 静的: module blob/importと指定commitの追跡
- 将来のentry docs/schemaのリンク検証のみ
- 別条件への数値移送は別科学許可

- [tests/test_df_hamiltonian.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/tests/test_df_hamiltonian.py) @ `f29eafbf4412193e8e8660d9f051889c2cba684b`: `test_one_body_number_operator_in_number_sector` (L23), `test_df_square_block_is_applied_matrix_free` (L41), `test_df_truncation_value_is_not_added_to_hamiltonian_constant` (L59), `test_h3_selected_rank_ground_state_matches_interaction_operator` (L86), `test_solve_df_ground_state_returns_residual_and_full_state` (L130), `test_numba_backend_matches_python_backend_when_available` (L152), `test_configured_df_rank_defaults_match_reference_project` (L175)。今回は未実行。
- [tests/test_rte_integer_counts.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/tests/test_rte_integer_counts.py) @ `f29eafbf4412193e8e8660d9f051889c2cba684b`: `test_integer_count_rejects_bool_float_and_string` (L28), `test_integer_count_accepts_python_and_numpy_integers` (L33), `test_positive_rte_step_apis_reject_invalid_counts` (L44), `test_occurrence_multiplicity_requires_a_positive_integer` (L69), `test_numpy_counts_are_normalized_and_zero_is_allowed_where_valid` (L82), `test_composed_bound_counts_never_silently_truncate` (L102), `test_finite_taylor_and_maximum_orders_require_integers` (L111), `test_sample_count_requires_an_integer_but_allows_zero` (L124), `test_rpe_and_truncation_dataclasses_validate_all_counts` (L148), `test_occurrence_truncation_rejects_invalid_positive_counts` (L224), `test_rpe_round_count_fields_reject_noninteger_or_negative` (L246)。今回は未実行。

**独立保持・未解決判断・順序**

H2/H4/H6/H8の独立入力・PF次数・実行delta窓・cohort/STOPを別保持。B3 code/SOURCEは除外。

未解決: 代表moduleと各研究の正本を混同しない見出し設計。

順序: 最初。コードを触らず再利用入口を明確化。担当案: ユーザーが対象groupと互換policyを選び、Codexは次回の明示指示内で実装する。科学実行は別途明示許可が必要。


## 2. INT-07 — B2-D evaluatorの既統合と独立拡張

**対象branchと固定tip**

- `codex/research-b-b2d-static-cost-outer-20261025` @ `3f5d60122f158b3320fadab86498963950f00734`
- `codex/research-b-final-pause-handoff-20261026` @ `e87375fa9a998dce80792be0f15597c747f00d83`

**分類・代表・配置候補**

TYPE_1。代表候補 `codex/research-b-final-pause-handoff-20261026` @ `e87375fa9a998dce80792be0f15597c747f00d83`。既存src/trottertracks/b2d_evaluator/README.mdとfinal-pause handoffを研究B入口候補にする。

**比較根拠となる実ファイル/API**

- [src/trottertracks/b2d_evaluator/evaluation.py::evaluate](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f5d60122f158b3320fadab86498963950f00734/src/trottertracks/b2d_evaluator/evaluation.py#L204) — `3f5d60122f158b3320fadab86498963950f00734`, line 204; `evaluate(graph: Graph, backend_factory, *, execution=None, scope=None, mock_only=False)`.
- [src/trottertracks/b2d_evaluator/evaluation.py::evaluate](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/src/trottertracks/b2d_evaluator/evaluation.py#L204) — `e87375fa9a998dce80792be0f15597c747f00d83`, line 204; `evaluate(graph: Graph, backend_factory, *, execution=None, scope=None, mock_only=False)`.
- [src/trottertracks/b2d_evaluator/input_package.py::connect_prepared](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/src/trottertracks/b2d_evaluator/input_package.py#L80) — `e87375fa9a998dce80792be0f15597c747f00d83`, line 80; `connect_prepared(preparation, *, blocks, physical_hamiltonian, normalized_tail, state, sector_basis_indices, recipe=InputRecipe())`.
- [src/trottertracks/b2d_evaluator/a0_reference.py::execute](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/src/trottertracks/b2d_evaluator/a0_reference.py#L177) — `e87375fa9a998dce80792be0f15597c747f00d83`, line 177; `execute(run, ledger, evidence, *, reference_factory=Reference, snapshot_loader=None, previous_rows=None)`.

**共通点と固有差分**

static-cost-outer→final-pauseは祖先。evaluation/dense_backend/input_packageは同blobで継承され、final側にouter/exact-R/full-DF/q4等が追加済み。 A0 math reference、B0 native/correctness、B1 IID、Exact-R、Full-DF、q4は異なるscope/入力/call budget。READMEには旧準備と失敗/再資格の履歴も残る。

**選択的な採用・不採用**

採用案: 既存READMEの履歴と現行入口を索引上で区別し、固定SOURCE/STOP/runner/testを研究単位別に辿れるようにする。新merge不要。

不採用: 旧Track Bへの科学的統合、未資格u_num/gamma/rare-event costを埋めた扱い、失敗B0を成功版で上書き。

**後続の検証設計と既存testの入口**

- 静的: ancestor・同blob・新增adapter/runner/test対応の確認
- 将来: mock backendのAPI契約回帰のみ
- 実operator/compile/shot/expected costの再評価は別科学許可

- [tests/management/test_b2d_full_df.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/tests/management/test_b2d_full_df.py) @ `e87375fa9a998dce80792be0f15597c747f00d83`: `test_fixed_cases_all_blocks_signed_stage_order_and_no_split_parameters` (L59), `test_out_of_scope_candidate_q_T_control_and_extra_split_rejected` (L76), `test_full_opaque_exact_scope_and_all_gates_before_comparisons_costs` (L86), `test_failure_partial_and_stop_no_metrics_or_automatic_retry` (L102), `test_readonly_preflight_source_fresh_output_and_versions_without_science` (L116), `test_cli_metadata_and_no_execution_without_explicit_flag` (L129)。今回は未実行。
- [tests/management/test_b2d_full_df_q4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/tests/management/test_b2d_full_df_q4.py) @ `e87375fa9a998dce80792be0f15597c747f00d83`: `test_scope_derived_from_signed_four_block_q4_schedule_and_q2_preserved` (L22), `test_opaque_one_shot_exact_scope_correctness_first_and_fresh_output` (L49), `test_failure_preserves_partial_stop_and_no_signal_cost_or_retry` (L68), `test_readonly_source_snapshot_and_profile_preflight_no_science` (L81), `test_observed_budget_only_with_unknown_nonfinite_and_outside_not_failure` (L95), `test_saved_comparisons_quote_fixed_cells_without_recomputing_statistics` (L108), `test_cli_requires_explicit_execution_flag_and_metadata_imports_only` (L118)。今回は未実行。

**独立保持・未解決判断・順序**

旧B0失敗・途中raw不足記録、修正後SOURCE、A0/B0/B1/Exact-R/Full-DF/Y4-q4の全cohort/STOPを別保持。

未解決: READMEの旧「最新」表示は履歴扱いとし現行採択を自動決定しない。

順序: INT-06と並行して入口整理。担当案: ユーザーが対象groupと互換policyを選び、Codexは次回の明示指示内で実装する。科学実行は別途明示許可が必要。


## 3. INT-09 — 原稿/CEM/図表・boundary証拠の入口統合

**対象branchと固定tip**

- `codex/df-pauli-evidence-paper-plan-20261026` @ `6e3436a16b13e539b0faa2a0a3286f6ab3549543`
- `codex/pauli-jw-research-summary-20261026` @ `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`
- `codex/pr-boundary-evidence-integration-20261027` @ `fbf701356cbc4da6ca8e3b7ef53533e1763243f9`
- `codex/pr-boundary-evidence-integration-local-full-20261027` @ `174ea14964e85f3c2fd623ce2972a4c221b01a73`

**分類・代表・配置候補**

TYPE_3。代表候補 `codex/pr-boundary-evidence-integration-20261027` @ `fbf701356cbc4da6ca8e3b7ef53533e1763243f9`。既存公開boundary integrationの50行CEMを追加入口候補とし、従来43行CEMも固定版として併記。

**比較根拠となる実ファイル/schema**

- [codex/df-pauli-evidence-paper-plan-20261026:EVIDENCE_MATRIX.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6e3436a16b13e539b0faa2a0a3286f6ab3549543/EVIDENCE_MATRIX.csv) @ `6e3436a16b13e539b0faa2a0a3286f6ab3549543` — 43版別row。
- [codex/df-pauli-evidence-paper-plan-20261026:RESEARCH_SYNTHESIS.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6e3436a16b13e539b0faa2a0a3286f6ab3549543/RESEARCH_SYNTHESIS.md) @ `6e3436a16b13e539b0faa2a0a3286f6ab3549543`。
- [codex/df-pauli-evidence-paper-plan-20261026:PAPER_OUTLINE.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6e3436a16b13e539b0faa2a0a3286f6ab3549543/PAPER_OUTLINE.md) @ `6e3436a16b13e539b0faa2a0a3286f6ab3549543`。
- [codex/pr-boundary-evidence-integration-20261027:EVIDENCE_MATRIX.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fbf701356cbc4da6ca8e3b7ef53533e1763243f9/EVIDENCE_MATRIX.csv) @ `fbf701356cbc4da6ca8e3b7ef53533e1763243f9` — 50版別row。
- [codex/pr-boundary-evidence-integration-20261027:RESEARCH_SYNTHESIS.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fbf701356cbc4da6ca8e3b7ef53533e1763243f9/RESEARCH_SYNTHESIS.md) @ `fbf701356cbc4da6ca8e3b7ef53533e1763243f9`。
- [codex/pr-boundary-evidence-integration-20261027:PAPER_OUTLINE.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fbf701356cbc4da6ca8e3b7ef53533e1763243f9/PAPER_OUTLINE.md) @ `fbf701356cbc4da6ca8e3b7ef53533e1763243f9`。

結合キー案は `(CEM_version, claim_id, study_id, SOURCE_commit, result_commit, cohort)`。既存 `document_path/result_locator/paper_placement` を辿り、図表と作成script/inputの元versionへの導線を追加する。cohortや入力同一性が不明なら欠損を残し、同名claimの上書きをしない。

**共通点と固有差分**

旧DF/Pauli paper planとboundary公開統合が同じCEM schema・研究索引を使う。 両branch tipは祖先関係でなく、公開scientific subsetとlocal-full保全を区別する。新50行を旧43行の科学結論の自動上書きにしない。

**選択的な採用・不採用**

採用案: claim_id/study_id＋SOURCE/result commit＋cohort＋適用条件の版別キーで索引・図表入口を結ぶ。追加row/locatorの対応を確認し、既存paper構成を採択し直さない。

不採用: 古いCEMの書換え、同名claimの無条件上書き、private local-full資料をpublicへ混入、raw再集計。

**後続の検証設計と既存testの入口**

- schema/link: claim/SOURCE/result/document locatorの双方向追跡
- public/private subset差分はmanifestの役割だけ照合
- 原稿の科学的採択・新しい詳細レビューは別指示

- 文書/schema/linkの検証設計のみ。今回はテスト未実行。

**独立保持・未解決判断・順序**

旧43行・新50行・図表source・独立cohort・negative result・最新採用/保留の元記録を版別保持。

未解決: 代表入口を置く位置と現在の科学採択はユーザー/研究担当が決める。

順序: INT-06/07の後、コード統合より先に資料入口を整備。担当案: ユーザーが対象groupと互換policyを選び、Codexは次回の明示指示内で実装する。科学実行は別途明示許可が必要。


## 4. INT-02 — S2/Yoshida4係数の重複とformula allowlist

**対象branchと固定tip**

- `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`
- `codex/research-b-final-pause-handoff-20261026` @ `e87375fa9a998dce80792be0f15597c747f00d83`

**分類・代表・配置候補**

TYPE_2。代表候補 `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`。src/trotterlib/product_formula.pyを係数の代表とし、B2-D側はS2/Y4専用入口を維持。

**比較根拠となる実ファイル/API**

- [src/trotterlib/product_formula.py::trotter_2nd_list](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/product_formula.py#L109) — `f29eafbf4412193e8e8660d9f051889c2cba684b`, line 109; `trotter_2nd_list()`.
- [src/trotterlib/product_formula.py::yoshida_4th_list](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/product_formula.py#L102) — `f29eafbf4412193e8e8660d9f051889c2cba684b`, line 102; `yoshida_4th_list()`.
- [src/trotterlib/product_formula.py::_get_w_list](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/product_formula.py#L134) — `f29eafbf4412193e8e8660d9f051889c2cba684b`, line 134; `_get_w_list(num_w: PFLabel)`.
- [src/trottertracks/b2d_evaluator/_baselines.py::trotter_2nd_list](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/src/trottertracks/b2d_evaluator/_baselines.py#L9) — `e87375fa9a998dce80792be0f15597c747f00d83`, line 9; `trotter_2nd_list()`.
- [src/trottertracks/b2d_evaluator/_baselines.py::yoshida_4th_list](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/src/trottertracks/b2d_evaluator/_baselines.py#L16) — `e87375fa9a998dce80792be0f15597c747f00d83`, line 16; `yoshida_4th_list()`.
- [src/trottertracks/b2d_evaluator/_full_df_schedule.py::_get_w_list](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/src/trottertracks/b2d_evaluator/_full_df_schedule.py#L20) — `e87375fa9a998dce80792be0f15597c747f00d83`, line 20; `_get_w_list(formula)`.

**共通点と固有差分**

trotter_2nd_list/yoshida_4th_listの関数body ASTが完全一致する2組。係数生成の重複を具体的に確認した。 trotterlib dispatcherは高次/新4次も扱うが、B2-DはS2/Yoshida4のみ。数学的積/append順・processor/actual adjointはadapter独立。

**選択的な採用・不採用**

採用案: 新しい非凍結呼出しから係数providerだけを共通化。ローカルallowlistで2nd/4th以外を拒否し、既存scheduleの順序・float評価順を変えない。

不採用: 他次数の自動有効化、係数和の修復、印刷小数/Fractionへの再解釈、B2-D processorの向き変更。

**後続の検証設計と既存testの入口**

- API/unit: 係数listのliteral/順序・コピー独立性・unsupported formula拒否
- regression: q2/q4のsigned stage orderをmock記述だけで照合
- operator/compiled数値同値性が必要なら科学実行許可を別途取得

- [tests/management/test_b2d_full_df.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/tests/management/test_b2d_full_df.py) @ `e87375fa9a998dce80792be0f15597c747f00d83`: `test_fixed_cases_all_blocks_signed_stage_order_and_no_split_parameters` (L59)。今回は未実行。
- [tests/management/test_b2d_full_df_q4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/tests/management/test_b2d_full_df_q4.py) @ `e87375fa9a998dce80792be0f15597c747f00d83`: `test_scope_derived_from_signed_four_block_q4_schedule_and_q2_preserved` (L22)。今回は未実行。

**独立保持・未解決判断・順序**

Full DF-PF q2とY4 q4、過去S2/Y4/高次PFの独立cohort・入力・SOURCE/STOPを保持。

未解決: 既存公開APIを追加するかprivate dispatcherのadapter参照を維持するか。

順序: 最初のコード候補。科学SOURCEは差し替えず新規利用から。担当案: ユーザーが対象groupと互換policyを選び、Codexは次回の明示指示内で実装する。科学実行は別途明示許可が必要。


## 5. INT-01 — 整数count検証の共通coreと呼出し側policy

**対象branchと固定tip**

- `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`
- `codex/research-b-final-pause-handoff-20261026` @ `e87375fa9a998dce80792be0f15597c747f00d83`

**分類・代表・配置候補**

TYPE_2。代表候補 `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`。既存src/trotterlib/rte.pyのcount正規化を再利用候補とし、B2-Dのstrict Python-int wrapperを維持。

**比較根拠となる実ファイル/API**

- [src/trotterlib/rte.py::require_integer_count](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/rte.py#L38) — `f29eafbf4412193e8e8660d9f051889c2cba684b`, line 38; `require_integer_count(value: object, *, name: str, minimum: int=0)`.
- [src/trottertracks/b2d_evaluator/_full_df_schedule.py::require_integer_count](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/src/trottertracks/b2d_evaluator/_full_df_schedule.py#L12) — `e87375fa9a998dce80792be0f15597c747f00d83`, line 12; `require_integer_count(value, *, name, minimum=0)`.

**共通点と固有差分**

両APIはbool/floatを黙って整数化せずminimumを検査する。同名APIを別実装で保持している。 trotterlibはoperator.indexとNumPy整数を許可しPython intへ正規化。B2-Dはtype(value) is intだけ許可。例外messageも異なる。

**選択的な採用・不採用**

採用案: 新規呼出しだけ、既存core前のstrict adapterを明示して共通minimum検査へ委譲。旧SOURCEのimport先・例外規約は固定したまま。

不採用: B2-DのNumPy整数受理への拡張、bool受理、科学runner全体merge、過去SOURCE/hash更新。

**後続の検証設計と既存testの入口**

- API/unit: bool/float/string/np.int64/negative/zero/minimum=1の受理表を別policyで固定
- regression: 既存integer-count testのうち純粋validator部分のみを選定
- 科学的countを変更する修正は別許可;今回はテスト未実行

- [tests/test_rte_integer_counts.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/tests/test_rte_integer_counts.py) @ `f29eafbf4412193e8e8660d9f051889c2cba684b`: `test_integer_count_rejects_bool_float_and_string` (L28), `test_integer_count_accepts_python_and_numpy_integers` (L33)。今回は未実行。
- [tests/management/test_b2d_full_df.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/tests/management/test_b2d_full_df.py) @ `e87375fa9a998dce80792be0f15597c747f00d83`: `test_out_of_scope_candidate_q_T_control_and_extra_split_rejected` (L76)。今回は未実行。

**独立保持・未解決判断・順序**

B2-D固定q2/q4・旧call budget/STOP、RTE側countsの全元SOURCEを別保持。

未解決: strict wrapperを残す設計でよいか。例外文を変えるかは後続仕様判断。

順序: 入口整理後、PF係数共通化の次に実装。担当案: ユーザーが対象groupと互換policyを選び、Codexは次回の明示指示内で実装する。科学実行は別途明示許可が必要。


## 6. INT-03 — canonical JSONとdigestのpolicyを保つ共通化

**対象branchと固定tip**

- `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`
- `codex/non-df-h4-pr-mean-cost-scalar-execution-20261025` @ `07d4f6ff7f1df085f292d36c36e460354df8a740`
- `n3-achievements-closure-20261013` @ `ccc575ee339be091fd4031272c38bee13866f097`

**分類・代表・配置候補**

TYPE_2。代表候補 `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`。既存parallel_validation_executorのcanonical_json/fingerprint_payloadを小さな再利用入口候補にする。

**比較根拠となる実ファイル/API**

- [src/trotterlib/parallel_validation_executor.py::canonical_json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/parallel_validation_executor.py#L77) — `f29eafbf4412193e8e8660d9f051889c2cba684b`, line 77; `canonical_json(payload: Any)`.
- [src/trotterlib/parallel_validation_executor.py::fingerprint_payload](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/parallel_validation_executor.py#L88) — `f29eafbf4412193e8e8660d9f051889c2cba684b`, line 88; `fingerprint_payload(payload: Any)`.
- [src/trotterlib/pauli_signal_adapter.py::digest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d4f6ff7f1df085f292d36c36e460354df8a740/src/trotterlib/pauli_signal_adapter.py#L24) — `07d4f6ff7f1df085f292d36c36e460354df8a740`, line 24; `digest(payload)`.
- [src/trottertracks/representation_exploration/n3_saved_evidence_v1.py::canonical](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ccc575ee339be091fd4031272c38bee13866f097/src/trottertracks/representation_exploration/n3_saved_evidence_v1.py#L17) — `ccc575ee339be091fd4031272c38bee13866f097`, line 17; `canonical(x)`.
- [src/trottertracks/representation_exploration/n3_saved_evidence_v1.py::sha](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ccc575ee339be091fd4031272c38bee13866f097/src/trottertracks/representation_exploration/n3_saved_evidence_v1.py#L18) — `ccc575ee339be091fd4031272c38bee13866f097`, line 18; `sha(data)`.

**共通点と固有差分**

sort_keys/compact separators/allow_nan=False/SHA256という処理構造を共有。別名の同等目的APIを確認した。 N3はensure_ascii=Falseでbytesを返し、executor/PauliはASCII escapeを使う。返却型str/bytesとfingerprint入力域が異なり、現行hashは同一とは限らない。

**選択的な採用・不採用**

採用案: UTF-8 bytes化までのpure helperを明示policy付きで設計し、既存呼出しの出力byteを固定fixtureで保証する。旧fingerprint生成は残す。

不採用: 既存manifest/ SOURCE/hashの再計算・Unicode正規化・科学proof/承認証明システムの新設。

**後続の検証設計と既存testの入口**

- API/unit: Unicode/ASCII/key順/NaN/Infinity/bytes返却をfixtureで照合
- regression: source/task fingerprint不変と旧checkpoint読取をmockで確認
- 科学入力や過去結果の再解析なし

- [tests/test_parallel_validation_executor.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/tests/test_parallel_validation_executor.py) @ `f29eafbf4412193e8e8660d9f051889c2cba684b`: `test_task_fingerprint_is_deterministic_and_binds_inputs` (L156)。今回は未実行。

**独立保持・未解決判断・順序**

N3の旧SOURCE3版/55宣言の未解決をhash helper共通化で解消した扱いにしない。

未解決: 新helperの公開場所とpolicy引数名。現行payload規約を既定値で混ぜない。

順序: countの次。旧namespaceからの移行は新規利用だけ。担当案: ユーザーが対象groupと互換policyを選び、Codexは次回の明示指示内で実装する。科学実行は別途明示許可が必要。


## 7. INT-05 — Track B lossless storageの3完全clone

**対象branchと固定tip**

- `track-b-g6-construction-paper-evidence-20261018` @ `d4163acd7b577afaa7d6db8c29db64f76ec03ae7`

**分類・代表・配置候補**

TYPE_2。代表候補 `track-b-g6-construction-paper-evidence-20261018` @ `d4163acd7b577afaa7d6db8c29db64f76ec03ae7`。既存df_correction_et_storage_v5/storage.pyを共有実装候補としv6/v7のimport wrapperを残す案。

**比較根拠となる実ファイル/API**

- [src/trottertracks/algorithm_codesign/df_correction_et_numeric_v6/storage.py::dump_json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/src/trottertracks/algorithm_codesign/df_correction_et_numeric_v6/storage.py#L214) — `d4163acd7b577afaa7d6db8c29db64f76ec03ae7`, line 214; `dump_json(path, data, *, replace_existing=False, ensure_ascii=True)`.
- [src/trottertracks/algorithm_codesign/df_correction_et_phase_v7/storage.py::load_json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/src/trottertracks/algorithm_codesign/df_correction_et_phase_v7/storage.py#L245) — `d4163acd7b577afaa7d6db8c29db64f76ec03ae7`, line 245; `load_json(path)`.
- [src/trottertracks/algorithm_codesign/df_correction_et_storage_v5/storage.py::iter_verified_bytes](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/src/trottertracks/algorithm_codesign/df_correction_et_storage_v5/storage.py#L257) — `d4163acd7b577afaa7d6db8c29db64f76ec03ae7`, line 257; `iter_verified_bytes(path)`.
- [src/trottertracks/algorithm_codesign/df_correction_comparison/storage.py::dump_json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/src/trottertracks/algorithm_codesign/df_correction_comparison/storage.py#L39) — `d4163acd7b577afaa7d6db8c29db64f76ec03ae7`, line 39; `dump_json(path, data, *, replace_existing=False, ensure_ascii=True)`.

**共通点と固有差分**

v5/v6/v7 storage.pyは同じGit blob 97408648f8224503c2955fe7c1a314660771563c。別相対pathに同一実装が3箇所ある。 旧comparison/storageはlogical JSON+chunk/index、V5-familyはshared immutable record/history・locking・legacy readerを持つ。schema/更新履歴/出力guardが違う。

**選択的な採用・不採用**

採用案: 非凍結の新呼出しから同一実装に委譲し、既存import path・MAGIC・logical byte identity・legacy読取を保つ。producer/元archiveを移行しない。

不採用: 旧schemaの無断一括変換、output guard変更、既存archive/manifestの書換え、resume/retry条件の共通化。

**後続の検証設計と既存testの入口**

- unit/regression: exact旧bytes/Unicode/chunk・corruption・partial publish・legacy読取・lockingを一時fixtureで照合
- API: dump_json/load_json/iter_verified_bytesのsignatureと例外維持
- 既存科学archiveへの書込・全件再hashなし

- [tests/tracks/algorithm_codesign/test_df_correction_et_v5_storage.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/tests/tracks/algorithm_codesign/test_df_correction_et_v5_storage.py) @ `d4163acd7b577afaa7d6db8c29db64f76ec03ae7`: `test_lossless_value_and_exact_original_bytes` (L25), `test_unicode_and_large_shared_blob_chunks_individually_UTF8` (L34), `test_large_payload_state_updates_and_native_prefix_final_share_immutable_bytes` (L42), `test_corrupted_graph_is_rejected_before_returning_data` (L57), `test_interruption_preserves_published_pointer_and_never_adopts_partial_blob` (L79), `test_failure_at_latest_pointer_publish_preserves_old_snapshot` (L96), `test_legacy_reader_compatibility_and_legacy_replacement_prohibited` (L103), `test_writer_enforces_fixed_output_budget_without_destroying_checkpoint` (L114), `test_two_processes_publish_shared_blob_once_and_restore_both` (L127), `test_new_guard_reads_decoded_child_CPU_receipt` (L135), `test_mutable_input_cannot_reuse_stale_blob_or_erase_prior_values` (L148), `test_logical_digest_and_physical_history_both_bound_to_native_inventory` (L158)。今回は未実行。

**独立保持・未解決判断・順序**

V5/V6/V7の独立SOURCE・failure/STOP・logical identity・native inventory bindingと元private履歴を保持。

未解決: 既存package内shared pathにするか、汎用利用が本当に必要ならtrotterlibへ切出すか。後者を既定にしない。

順序: digest policyの確定後。最初は新規読取呼出しだけ。担当案: ユーザーが対象groupと互換policyを選び、Codexは次回の明示指示内で実装する。科学実行は別途明示許可が必要。


## 8. INT-08 — DF/Pauli/RPEの評価scope・N/G条件の証拠統合

**対象branchと固定tip**

- `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`
- `codex/non-df-h4-pr-mean-cost-scalar-execution-20261025` @ `07d4f6ff7f1df085f292d36c36e460354df8a740`
- `codex/pauli-jw-research-summary-20261026` @ `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`

**分類・代表・配置候補**

TYPE_3。代表候補 `codex/pauli-jw-research-summary-20261026` @ `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`。既存EVIDENCE_MATRIX/CEMへ評価条件列と適用domainを相互参照する案。

**比較根拠となる実ファイル/API**

- [src/trotterlib/rpe_resource_accounting.py::evaluate_rpe_round_candidate](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/rpe_resource_accounting.py#L564) — `f29eafbf4412193e8e8660d9f051889c2cba684b`, line 564; `evaluate_rpe_round_candidate(preparation: DFPartialS2Preparation, specification: RPERoundSpecification, allocation: RPEErrorAllocation, pf_error_model: RPEPFErrorModel, *, beta_rpe: float, rte_steps_per_occurrence: int, finite_taylor_order: int, cost_metric: str, cost_provider: RPERoundCostProvider | None=None, rte_seed: int=0, hadamard_sampling_policy: RPEHadamardSamplingPolicy | None=None)`.
- [src/trotterlib/fixed_signal_accounting.py::fixed_signal_shots](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d4f6ff7f1df085f292d36c36e460354df8a740/src/trotterlib/fixed_signal_accounting.py#L24) — `07d4f6ff7f1df085f292d36c36e460354df8a740`, line 24; `fixed_signal_shots(*, epsilon_sig: float, bias: Mapping, allowance: Mapping, log_B_upper: float | None, evidence_kind='UNKNOWN', evidence_ref=None, epsilon_axes=None, alpha_axes=None)`.
- [src/trotterlib/pauli_rte_event_extension.py::account_target](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/src/trotterlib/pauli_rte_event_extension.py#L78) — `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`, line 78; `account_target(target, prices, epsilon, law)`.

**共通点と固有差分**

signal誤差・normalization・shot・costを接続する同じ研究課題を扱う。 RPE round phase誤差とfixed-time complex signal誤差は別。rank/representation、q/r/K、one-body/identity、iid/grid law、u_num/allowance、compiler/cost scopeが違う。

**選択的な採用・不採用**

採用案: 条件ごとの入力/ SOURCE/評価metric/未資格項目/claim locatorの比較schemaのみ整備。保存数値の再集計・順位付けはしない。

不採用: PF係数Cを厳密上界と呼ぶこと、未取得u_numやlogBを既定値で埋めてN/Gを算出、別cohortの単一化。

**後続の検証設計と既存testの入口**

- schema: SOURCE/input/geometry/basis/rank/L_D/delta/time/law/precision/cost scopeの欠損を明示
- 保存索引の引用整合だけ検証
- 数値再比較と有用性判定には別許可

- [tests/test_fixed_signal_evaluation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/tests/test_fixed_signal_evaluation.py) @ `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`: `test_missing_numerical_or_normalization_evidence_never_invents_N_G` (L326)。今回は未実行。

**独立保持・未解決判断・順序**

DF CEM・Pauli確認/独立cohort・rare event未観測・STOP・source/result別commitを維持。

未解決: どの共通taskで比較するかは後続ユーザー/研究担当が採択。

順序: index入口とAPI小規模整理後。科学計算なし。担当案: ユーザーが対象groupと互換policyを選び、Codexは次回の明示指示内で実装する。科学実行は別途明示許可が必要。


## 他の6グループ

上位8件以外の具体的API/commit/差分/保持条件はINTEGRATION_CANDIDATES.csvとIMPLEMENTATION_OVERLAP.csvを参照。

- **INT-04 / TYPE_3** Hadamard wrapperの類似gate列と相違するscope。codeは別保持。wrapper scope対照表を既存研究索引に提案。Pauli入口はstate preparationなし・cosine/sine軸。C4はoccupied-state preparationとsource/time/fingerprint検査を含む。nativeはsymbolic Circuitでmeasurementなし。 各SOURCE・prepared state・cohort・phase/measurementの元記録を別保持。
- **INT-10 / TYPE_3** Pauli SDK演算子とnative IRの対応。backend別実装は維持し、label/符号/control/global phaseの技術対照表を提案。SDK PauliEvolutionGateとnative rz/sx/cx IRは同一APIではない。IRはidentityを拒否しsigned_alpha/global_phase_piを明示、bit orderとcontrolled relative phaseが重要。 各compiler profile・合成精度・旧失敗/STOPを独立保持。
- **INT-11 / TYPE_4** Research C C1/C4/PR-aware DFの独立契約・現行作業。現在は関連moduleと失敗/修正の索引のみ。凍結SOURCEや現行branchを統合しない。C1のsaved-XDF exact x0、ffsim0.0.83/JAX-x64/dedicated-process hookと、PR-aware DFの新objective/scales・資格・bound/law contractは独立。C4 dictionary選択も別問題。 旧complex codec失敗・x0修正・各SOURCE/STOP・C1/C4/PR-awareの別cohortを保持。
- **INT-12 / TYPE_4** N1/N3旧SOURCE・失敗履歴。既存closure記録を履歴入口にし、旧SOURCE欠落を独立保持。現在ある同名moduleは、旧契約の3版/55宣言のhash一致回収を意味しない。独立cohort・失敗・source-fingerprintを混ぜない。 旧SOURCE未解決11件・3版55宣言、独立confirmationと失敗STOPを維持。
- **INT-13 / TYPE_4** DF-native B3と保留7 ref。保護・延期。公開metadataを記録するだけ。ユーザー申告の実行中を前提に放置。SERVER_B-WT-0003/0123/0124/0128/0130と別枠7 refを保護。 すべての科学SOURCE/input/result/STOP/ref/復旧契約を保持。
- **INT-14 / TYPE_3** 旧Track B co-design・DF correctionの研究比較。既存研究packet/比較表からD/E/M1/M1S/G系列の依存を索引化。Point/context、allocation、construction、scalar、native合成・interval精度・比較基準が系列ごとに違う。Research B B2-Dは別系列として維持。 D/E/M1/M1S/G別の独立結果・one-shot/STOP、旧B3名称の歴史文脈を維持。

## 保持する検証・研究導線

後続でcode入口を変更する場合に限り、影響するsrc/trotterlib/README.md、scripts/README.md、tests/README.md、PROJECT_MAP.md、関連docs/artifactsへのリンクを最小限更新する。旧SOURCEのimport、固定runner、入力snapshot、結果、negative result、STOP、grant/ledger/marker、one-shot/retry/resume条件は独立に保持する。新しい共通helperの導入で既存契約を緩めない。

DF-native B3と保留7 refはこの実装順序から除外する。Research Cは現行作業・別objective/初期化契約を保ち、N1/N3旧SOURCE欠落は現行同名ファイルによって解消した扱いにしない。
