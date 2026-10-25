# 文書索引

研究B [Full DF-Yoshida4 q4：固定scope・検証・実行記録](research/research_b_full_df_yoshida4_q4_execution_20261026.md)。既存Full DF-PFを再利用、FD5/6・T±.18だけ。

研究B [Full deterministic DF-PF4 caseの科学実行結果](research/research_b_full_deterministic_df_pf_execution_20261026.md)。全correctness後にFull S2／Yoshida4のsignal・native単価を取得。保存Exact-R／finite-RTEとの固定点対応、未資格u_num／γ、総資源比較の未実施を区別してmandatory STOP。

研究B [Full deterministic DF-PFの再利用・実装・実行準備](research/research_b_full_deterministic_df_pf_readiness_20261026.md)。元c0／全DF因子・ordinary control・独立correctnessを接続。旧packageは指定commitから取得して必要定義だけを再利用、科学SOURCE8811ef089、人工39件PASS。新科学結果・費用比較は未取得。

研究B [Exact-R4 case：結果・原本・限界・STOP](research/research_b_bnext3_exact_r_execution_20261026.md)。固定SOURCE e76c04748、correctness全PASS、8 prepared X/Y wrapperの単価取得。D／outer誤差・未資格u_num／γ・総資源比較の未実施を区別する。

研究B [B-Next 3：Exact-R最小pilot実装・人工検証・実行準備](research/research_b_bnext3_exact_r_execution_readiness_20261026.md)。新Exact-R4 caseと保存finite-RTE4 caseを対応付け、phase／control／独立correctness・単価測定のscopeを固定。人工30件とread-only preflight PASS。科学実行・総資源優位の認定は行っていない。

研究B [固定24条件shot加重会計：全結果・限界・STOP](research/research_b_outer_shot_weighted_execution_20261025.md)。SOURCE9bebdc46、48 shot要求・144資源行・24比較を一回完了、scope一致・保存失敗0。科学状態INCONCLUSIVE_QUALIFICATIONS_AND_RARE_COSTS、u_num／γ／rare費用は未取得、無条件G・同精度優位は未認定。旧準備・設計と全原本を保全。

研究B [Shot加重評価：実装・人工検証・実行準備](research/research_b_outer_shot_weighted_execution_ready_20261025.md)。入口 `scripts/run_b2d_outer_shot_weighted.py`、保存adapter `b2d_evaluator/shot_weighted.py`、人工test `test_b2d_shot_weighted.py`。固定24条件・48 shot要求・144資源行・24比較を登録、人工45件PASS。実候補のnormalization／shot数／Gは未評価、明示的実行許可後のみ一回起動。

研究B [単一複素signalのshot加重資源評価：技術設計・将来scope](research/research_b_outer_shot_weighted_resource_design_20261025.md)。設計のみ、実shot数／G／新順位の評価なし。新人工test `tests/management/test_b2d_shot_design.py` と既存scalar testの計10件PASS。将来案は6候補×両T×2精度、48 shot要求・144条件付き資源行。計算SOURCE／launcherは未作成、実行は別途明示許可後のみ。

研究B [高次数RTEの最小費用感度：固定2比較・保存値解析](research/research_b_outer_rare_event_minimal_sensitivity_20261025.md)。入口 `scripts/run_b2d_outer_rare_sensitivity.py`、接続 `b2d_evaluator/rare_sensitivity.py`、人工test `test_b2d_rare_sensitivity.py`。8確率／16境界／80 SE cornerを一回完了、人工8件PASS。B4/B3は±2SEでS0差を維持、D1/R4は±1SEで符号未確定。新sampling／build／compileなし、B1のINCONCLUSIVEを維持し、新結果はCONDITIONAL_SENSITIVITY_ONLY・mandatory STOP。

研究B [B1固定IID期待native費用：契約・実装・結果](research/research_b_outer_b1_expected_native_cost_20261025.md)。新入口 `scripts/run_b2d_outer_b1.py`、接続 `b2d_evaluator/b1_native.py`、人工test `test_b2d_b1_native.py`。全3,072 trajectory・6,144費用要求・18,432 operator／12,288 expectationを完了、compiled最大4.55e-13。32 process、wall50:46.95、actual compile6078／hit66。全eventはorder0で科学状態INCONCLUSIVE_RARE_EVENT_COVERAGE、gamma／shots／総費用は未資格。Retry/resume／追加標本なし、mandatory STOP。以下のB0／A0記録は履歴として保全する。

## 研究B B0完了・native費用観測（2026-10-25）

[最新結果・SOURCE・独立診断・全費用・STOP](research/research_b_outer_b0_recovery_20261025.md)。SOURCE b190fe28b、全288 operator／192 expectation PASS後に96 wrapperを測定。Compiled最大4.43e-13、fixed tol1e-12、人工33件PASS。決定論的代表の費用観測で、期待費用・shots・γ資格は未評価。旧失敗記録を保全。以下は停止・復旧の履歴。

## 研究B B0復旧履歴・未解決のD1 correctness（2026-10-25）

[全修正・固定SOURCE・attempt・診断・STOP](research/research_b_outer_b0_recovery_20261025.md)。初期位相集計誤差と失敗保存不足を改善したが、D1の最終診断差は約1.39e-12で固定tol1e-12を超え、全B0は未完了・費用0。Focused人工30件PASS。旧原本・契約は履歴として保全し、追加計算を開始しない。

## 研究B B0：最初のcompiled gateでSTOP（2026-10-25）

[B0停止報告・call実績・保存の限界](research/research_b_outer_b0_execution_20261025.md)。SOURCE8d0cb96df、B3／T+.18／代表0／Xのcompiled correctness FAIL、費用計測0件。数値差・回路のFAIL時保存がなく原因未確定。Retry／追加診断なし、A0結果は保全。以下はA0と準備段階の履歴。

## 研究B A0実行完了、B0はSTOP（2026-10-25）

[A0実行報告・全8比較・原本・STOP](research/research_b_outer_a0_execution_20261025.md)。SOURCE1e3921ba1、8行×90/128桁を変更・中断・retryなしに完了。全8比較で経験的精度安定性とnominal bound整合を確認したが、certified u_numはnull。B0 native build／compileは未実行。以下は準備段階の履歴。

## 研究B A0高精度・B0 full-wrapper pilot準備（2026-10-25）

[技術報告・固定SOURCE・個別科学契約／command](research/research_b_outer_a0_b0_execution_preparation_20261025.md)。A0 1e3921ba1、B0 8d0cb96df、人工138件・両read-only preflight PASS。実高精度／回路build／compileは未実行。u_num・γ資格と科学的correctnessは未確認。A0とB0の実行許可は別々に必要。

## 研究B outer/RTE固定22行の完了結果（2026-10-25）

[全結果・r/K・normalization・数値的限界・STOP](research/research_b_outer_rte_22row_execution_20261025.md)。SOURCE26e8734afを変更せず、全gate・22行・1239 dispatch・274 keyを一回で完了した。原本とCSVは `artifacts/research_b_outer_rte22/20261025/`。u_numとdeterministic qualificationは未確定、精度／総資源の認定なし。追加科学計算へ進まない。

## 研究B outer/RTE固定22行の最終準備（2026-10-25）

[固定22行・候補別r/K・新scope・人工119件・read-only preflight・実行command](research/research_b_outer_rte_22row_execution_ready_20261025.md)。科学SOURCE `26e8734aff3969dc8077cb058826d12394be28e1`。実装準備のみ完了、科学計算未実行。u_num・γ・shotsは未取得、同精度／総資源優位を認定しない。別途明示許可後のみ実行する。

## 研究B outer Stage 1の完了結果（2026-10-25）

[実行報告・比較・normalization・限界・STOP](research/research_b_outer_splitting_stage1_execution_20261025.md)。SOURCE b85dd3284、16行・906 dispatch・490 key、全correctness PASS。原本と全行CSVは `artifacts/research_b_outer_splitting/20261025/`。技術的中断・修正・再実行なし。追加条件・native費用・Stage2は開始しない。

## 研究B outer 16行最終実行準備（2026-10-25）

[未承認の最終科学実行契約・技術報告](research/research_b_outer_splitting_stage1_execution_ready_20261025.md)。SOURCE候補b85dd3284、16行・906 dispatch・490 key・80 expectation。既存launcher／Input package／scope／保存へ接続し、105人工件PASS。科学処理0件。実行入口は `scripts/run_b2d_outer_stage1.py --execute …`、ユーザーの別途明示許可が必要。

## 研究B outer splitting準備（2026-10-25）

[実装・人工検証・未承認科学契約案](research/research_b_outer_splitting_stage1_preparation_20261025.md)。SOURCEは `src/trottertracks/b2d_evaluator/{outer,outer_backend}.py`、人工testは `tests/management/test_b2d_outer_preparation.py`。4方式を準備し、追加19＋既存72件PASS。科学計算・回路生成なし。比較条件と新scope／実行bindingは科学実行前に確定する。

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


## B2-D Stage 1承認管理撤去・実行簡素化（2026-10-25）

最新の[科学実行契約・単一command・技術報告](research/b2d_stage1_simple_execution_20261025.md)。新科学SOURCE `54008c697e686a42ec48941d25219d523bafd454`。入口は `scripts/run_b2d_stage1.py --execute --source-commit … --output …`。承認packet／grant／receipt・発行／準備scriptを撤去し、科学SOURCE・input identity・scope・correctness・fresh output・失敗保存・STOPを維持した。実装 `src/trottertracks/b2d_evaluator/`、人工test `tests/management/test_b2d_simple_execution.py` と既存5ファイル、計70件PASS。科学計算は未実行。予定artifact `artifacts/research_b_b2d_stage1/20261025/simple-once/` は未生成。旧方式は過去の未採択提案として保全し、今後のB2-D実行要件にはしない。


## B2-D Stage 1限定科学検証の完了（2026-10-25）

[実行結果・技術復旧・比較表・STOP](research/b2d_stage1_execution_20261025.md)。科学SOURCE758a05ed、入力／A4数学／published列gate全PASS、36行・941 dispatch・448 key完了。A4のA0／D-S2からのfull signal改善は観測したが、A3との差は丸め誤差水準でINCONCLUSIVE。原本2attempt・全比較・環境・CSV・保存値集計は `artifacts/research_b_b2d_stage1/20261025/`。ローカル結果でCI／外部再現ではない。追加科学計算・Stage2へ進まずmandatory STOP。
