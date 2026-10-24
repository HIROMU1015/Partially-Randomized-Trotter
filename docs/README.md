# 文書索引

このディレクトリには、研究方針の正本、実装規約、検証報告、発表資料の案内が共存する。
研究全体を初めて読む場合は、先に[`../PROJECT_MAP.md`](../PROJECT_MAP.md)と
[`research/研究概要・現状.md`](research/研究概要・現状.md)を読む。

## 研究方針と現在地

- [`research/研究概要・現状.md`](research/研究概要・現状.md)：最新の短い全体要約
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

## 実行・運用

- [`server_parallel_validation_execution.md`](server_parallel_validation_execution.md)：共有CPU/GPUサーバー向けのbounded実行、checkpoint、resume、dry-run
- [`examples/parallel_validation_h4_q1_manifest.json`](examples/parallel_validation_h4_q1_manifest.json)：H4 q=1のdry-run用manifest例

## 発表資料と参考文献

- [`presentations/README.md`](presentations/README.md)：発表資料・構成案の位置づけ
- [`references/README.md`](references/README.md)：同梱した論文PDFの位置づけ
- [`rte_source_versions.md`](rte_source_versions.md)：RTE一次資料の版管理

## 状態の読み方

個別文書に数値があっても、それだけで現在利用可能とは判断しない。
再現可能性、失効、成果物の有無は[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)と
[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)で確認する。


## PR混在辞書 F1実行前準備

[Phase 2D-F1-PRE](research/pr_algorithm_improvement_phase2d_f1_preparation_20261023/README.md)：保存B入力、専用環境、既存prototype adapter、case計画、承認境界。管理testのみ実行し、科学検証は未実行。既存の研究判断・validation結果を更新するものではない。


## PR混在辞書F1実行結果（2026-10-24）

[実行結果とSTOP](research/pr_algorithm_improvement_phase2d_f1_execution_20261024/README.md)：H-ID4armとSSのmean等は部分PASS、metadata TypeErrorでF1未完了。要約・残差・実績件数とprivate rawのhash台帳を公開。F2や再実行は開始していない。


## PR混在辞書F1：修正版の登録検証完了（2026-10-24）

[新結果とSTOP](research/pr_algorithm_improvement_phase2d_f1_rerun_20261024/README.md)。既存saved runner／adapter／reference／pricingを再利用し、SOURCE `8192530` と保存B入力で16familyを全件完了。全4armの登録正しさ検査とcatalog価格-DP照合はPASS、未資格intervalはfallbackのみ。認証score／shot／資源優位性は未評価。artifactは `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/`。旧停止記録は保全し、F2へ進まない。


## PR混在辞書 F2実行前準備（2026-10-24）

[準備・SOURCE再利用・case・統計資格・契約・STOP](research/pr_algorithm_improvement_phase2d_f2_preparation_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_saved.py`、SOURCE `trottertracks/pr_dictionary_validation/f2_*.py`、管理test `tests/management/test_pr_dictionary_f2_preparation.py`。planned outputは `artifacts/pr_algorithm_improvement_phase2d_f2/<execution-date>/<fresh-run-id>/`。finite-target主比較を採択し、mock/管理確認のみ完了。統計条件とinterval資格は未解決で、科学実行・新grantは0件。


## PR混在辞書 F2統計資格確認（2026-10-24）

[資格確認と実行前STOP](research/pr_algorithm_improvement_phase2d_f2_qualification_20261024/README.md)。既存interval／log／ceilの技術test `tests/management/test_pr_dictionary_f2_scalar_intervals.py` は5件PASS。科学SOURCEはF2-PREのまま、共通finite targetに対する有限ビット誤差予算γの保証と統計条件の採択が未解決で、F2科学実行・新grantは0件。機械可読記録は同資料の `qualification.json`。入口は既存 `scripts/run_pr_dictionary_f2_saved.py` を維持する。


## PR混在辞書 F2限定資源計測（2026-10-24）

[限定契約・再利用SOURCE・停止条件](research/pr_algorithm_improvement_phase2d_f2_resources_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_resources_saved.py`、接続 `trottertracks/pr_dictionary_validation/f2_resources.py`、管理test `tests/management/test_pr_dictionary_f2_resources.py`。出力は `artifacts/pr_algorithm_improvement_phase2d_f2_resources/`。旧統計資格付き契約を維持し、認証Nを生成しない。

限定F2の[実行結果・部分資源・STOP](research/pr_algorithm_improvement_phase2d_f2_resources_20261024/EXECUTION_REPORT.md)：raw全件PASS、compiled operator3件FAIL、mean／4arm期待資源は未完了。SOURCE `9b1e103` と一回の実行原本を保全し、retryなし。


## 限定F2・位相保持修正と新規実行（2026-10-24）

[修正・実行契約・停止条件](research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/README.md)。既存runnerとf2_resources／f2_executionを再利用し、gates.compiled_wrapperの公式definition展開を修正。回帰testは `tests/management/test_pr_dictionary_compile_phase.py`、planned artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/`。

[修正版F2の全件結果・資源表・STOP](research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md)：SOURCE `9468c70`、raw／compiled各13,676・reference27,356・mean4arm・weighted集計を完了。1-shotの指標順はPP<SP<PS<SS、認証Nは対象外。旧失敗は保持し、追加実験へ進まない。

[限定F2の資源差の原因調査](research/pr_algorithm_improvement_phase2d_f2_resource_causes_20261024.md)：SOURCE・保存CSVの静的照合のみ。PPのGaussian往復不要とK0/r2の抽出構造を説明し、compiler・共有・sampling weight別の寄与は未測定と明示した。新しい数値集計や回路生成は行わずmandatory STOP。

[限定F2の保存case寄与分析](research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)：8分類表・490行のP／条件付き費用／寄与を公開Φへexact照合。SOURCEは `src/trottertracks/pr_dictionary_validation/f2_attribution.py`、runnerは `scripts/run_pr_dictionary_f2_attribution_saved.py`、人工testは `tests/management/test_pr_dictionary_f2_attribution.py`、artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_attribution/2026-10-24/`。新build／compileなし、因果効果・Nは未確立。mandatory STOP。

[Gaussian singleton合成の改善候補](research/pr_gaussian_singleton_synthesis_design_20261024.md)：既存SOURCE・保存資料の静的調査。support1／support2の選択部分空間Givens案を推奨し、既存Nature helper・operator等価条件・未承認の48-wrapper最小対照案を整理。新しい実装・科学計算・build／compileは0件、削減量未測定。mandatory STOP。

[Rectangular Givens候補Aの限定実装・PS48準備](research/pr_rectangular_givens_preparation_20261024.md)：singleton専用adapter、内部T、actual adjointを接続。SOURCEは `src/trottertracks/pr_dictionary_validation/rectangular*`、runnerは `scripts/run_pr_rectangular_givens_saved.py`、mock testは `tests/management/test_pr_rectangular_givens_preparation.py`、artifactは `artifacts/pr_rectangular_givens_preparation/2026-10-24/`。実装・技術確認のみで、科学実行は別承認。mandatory STOP。

[Rectangular Givens候補AのPS48科学実行結果](research/pr_rectangular_givens_ps48_execution_20261024.md)：6基底・raw／compiled各48の固定gateが全件PASS。index21/25では改善、全48等重みRZ/CXは増加。case CSV・残差・古典費用・provenanceは `artifacts/pr_rectangular_givens/2026-10-24/`。全F2期待値や一般的優位の認定はせずmandatory STOP。

[Rectangular Givens候補Aの構造選択規則調査](research/pr_rectangular_selection_design_20261024.md)：SOURCEと保存結果の静的調査だけ。support／frame／境界の説明と予測の限界、両方式compileの強い対照、未承認の独立検証案を整理した。一般的な規則は未確立で、一旦終了を技術的に推奨。選択器実装・再集計・科学実行なし、mandatory STOP。

## B2-D Input package・preflight準備（2026-10-25）

[技術報告](research/research_b_b2d_input_preflight_technical_report_20261025.md)／[未採択の第一段階契約案](research/research_b_b2d_stage1_science_execution_contract_proposal_20261025.md)。実装 `src/trottertracks/b2d_evaluator/input_package.py`・`preflight.py`、人工test `tests/management/test_b2d_input_preflight.py`。science runner未起動、予定artifact `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/` は未生成。科学計算・新grantなし、実運用scope確定前にSTOP。

## B2-D Stage 1最終実行準備（2026-10-25）

[最終提案](research/research_b_b2d_stage1_final_execution_proposal_20261025.md)・[技術報告](research/research_b_b2d_stage1_final_preparation_technical_report_20261025.md)・[scope JSON](research/research_b_b2d_stage1_scope_20261025.json)。科学的処理境界の実績照合と失敗保存launcherを準備。入口 `scripts/run_b2d_stage1.py`、実装 `src/trottertracks/b2d_evaluator/`、人工test `tests/management/test_b2d_stage1_launcher.py`。62人工test PASS、production authority未接続、科学実行・grant操作なし。予定artifact `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/` は未生成。

## B2-D Stage 1一括準備完了（2026-10-25）

[最終契約・承認手順](research/b2d_stage1_execution_ready_20261025/CONTRACT.md)と[技術報告](research/b2d_stage1_execution_ready_20261025/TECHNICAL_REPORT.md)にSOURCE、実環境、scope、未承認packet、ユーザー操作を集約。science入口 `scripts/run_b2d_stage1.py`、metadata準備／ユーザー発行入口も同scripts索引を参照。実装 `src/trottertracks/b2d_evaluator/`、人工test `tests/management/test_b2d_execution_ready.py`。74人工test PASS、科学計算・正式grant操作なし。予定artifact `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/` は未生成。


## B2-D Stage 1通常ユーザー承認方式（2026-10-25）

最新の[未承認契約・通常権限の具体command](research/b2d_stage1_user_approval_20261025/CONTRACT.md)と[技術報告](research/b2d_stage1_user_approval_20261025/TECHNICAL_REPORT.md)。新科学SOURCE `91f45aeb50275bb6d22050a304d0a9b1ff025d70`、通常ユーザーのprivate手動記録へ限定修正し、root／外部APIを要求しない。同じOSアカウントの本人識別は運用規則であり独立認証ではない。入口 `scripts/{prepare_b2d_stage1,issue_b2d_stage1_approval,run_b2d_stage1}.py`、実装 `src/trottertracks/b2d_evaluator/`、人工test `tests/management/test_b2d_execution_ready.py`。既存74件を適合更新＋追加6件＝80件PASS。科学条件・scope・環境は不変、正式grant／claim／科学実行なし。予定artifactは固定checkout内 `artifacts/research_b_b2d_stage1/20261025/user-approval-once/`、今回未生成。旧root方式は上記過去段階の履歴として保全する。
