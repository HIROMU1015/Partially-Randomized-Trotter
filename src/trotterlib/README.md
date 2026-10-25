# trotterlib モジュール索引

研究B [高次数RTEの最小費用感度：固定2比較・保存値解析](../../docs/research/research_b_outer_rare_event_minimal_sensitivity_20261025.md)。入口 `scripts/run_b2d_outer_rare_sensitivity.py`、接続 `b2d_evaluator/rare_sensitivity.py`、人工test `test_b2d_rare_sensitivity.py`。新sampling／build／compileなし、B1のINCONCLUSIVEを維持し、別のCONDITIONAL_SENSITIVITY_ONLYとして扱う。

研究B [B1固定IID期待native費用：契約・実装・結果](../../docs/research/research_b_outer_b1_expected_native_cost_20261025.md)。新入口 `scripts/run_b2d_outer_b1.py`、接続 `b2d_evaluator/b1_native.py`、人工test `test_b2d_b1_native.py`。全3,072 trajectory・6,144費用要求・18,432 operator／12,288 expectationを完了、compiled最大4.55e-13。32 process、wall50:46.95、actual compile6078／hit66。全eventはorder0で科学状態INCONCLUSIVE_RARE_EVENT_COVERAGE、gamma／shots／総費用は未資格。Retry/resume／追加標本なし、mandatory STOP。以下のB0／A0記録は履歴として保全する。

研究Bの[B0最新結果](../../docs/research/research_b_outer_b0_recovery_20261025.md)：SOURCE b190fe28bでfixed48代表・全correctnessと96 full-wrapper費用を完了。既存DF/RTE/Hadamard部品を維持し、B0専用compilerの微小角・離散phaseを修正した。Expected cost／shots／γ資格は未評価。旧記録を保全しmandatory STOP。

研究Bの[B0技術復旧](../../docs/research/research_b_outer_b0_recovery_20261025.md)はtrotterlibの科学条件を変更せず、B0専用compiler側のphase集計・失敗保存を修正した。D1の残差は固定tol未達のままで、全pilot・費用比較は未完了。旧研究の既定compilerと共有optimizer classは変更していない。

研究Bの[B0実行結果](../../docs/research/research_b_outer_b0_execution_20261025.md)：固定SOURCE8d0cb96dfの最初のcompiled operator検査でFAILし、費用計測0件でmandatory STOP。原因未確定、SOURCE修正・retryなし。以下の未実行記述は準備時点の履歴。

研究Bの[A0／B0準備](../../docs/research/research_b_outer_a0_b0_execution_preparation_20261025.md)では、`df_partial_s2.QiskitDFPartialS2CircuitBuilder.append_block_evolution` と `rpe_hadamard_interrogation.build_prepared_hadamard_wrapper` をouter専用adapterから利用する。旧partial-S2／RPE契約は維持。SOURCE B0 8d0cb96df、人工138件PASS、実回路build／compileは未実行。実行にはA0とは別のB0明示許可が必要。

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

- `df_partial_s2.py`：決定論half sweepとRTE中央部を持つpartial-$S_2$
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


## PR混在辞書の保存入力adapter（F1-PRE）

既存 `rte.py`／`df_trotter/ops.py` を変更せず、`trottertracks/pr_dictionary_validation/f1_saved_execution.py` と `f1_saved_plan.py` から再利用する準備を追加した。[準備記録](../../docs/research/pr_algorithm_improvement_phase2d_f1_preparation_20261023/README.md)がrunner、管理test、planned artifactの入口。科学実行は未承認・未実行。


## PR混在辞書F1の部分検証（2026-10-24）

既存RTE／Gaussian helperを再利用したone-shotの[結果](../../docs/research/pr_algorithm_improvement_phase2d_f1_execution_20261024/README.md)。4arm H-ID等は部分的にPASSしたが、compile cache用metadata設定で停止。`trottertracks/pr_dictionary_validation` の固定SOURCEは保全し、retry／resumeなし。要約／hash台帳は `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24/`。


## PR混在辞書F1：修正版の登録検証完了（2026-10-24）

[新結果とSTOP](../../docs/research/pr_algorithm_improvement_phase2d_f1_rerun_20261024/README.md)。既存saved runner／adapter／reference／pricingを再利用し、SOURCE `8192530` と保存B入力で16familyを全件完了。全4armの登録正しさ検査とcatalog価格-DP照合はPASS、未資格intervalはfallbackのみ。認証score／shot／資源優位性は未評価。artifactは `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/`。旧停止記録は保全し、F2へ進まない。


## PR混在辞書 F2実行前準備（2026-10-24）

[準備・SOURCE再利用・case・統計資格・契約・STOP](../../docs/research/pr_algorithm_improvement_phase2d_f2_preparation_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_saved.py`、SOURCE `trottertracks/pr_dictionary_validation/f2_*.py`、管理test `tests/management/test_pr_dictionary_f2_preparation.py`。planned outputは `artifacts/pr_algorithm_improvement_phase2d_f2/<execution-date>/<fresh-run-id>/`。finite-target主比較を採択し、mock/管理確認のみ完了。統計条件とinterval資格は未解決で、科学実行・新grantは0件。


## PR混在辞書 F2統計資格確認（2026-10-24）

[資格確認と実行前STOP](../../docs/research/pr_algorithm_improvement_phase2d_f2_qualification_20261024/README.md)。既存interval／log／ceilの技術test `tests/management/test_pr_dictionary_f2_scalar_intervals.py` は5件PASS。科学SOURCEはF2-PREのまま、共通finite targetに対する有限ビット誤差予算γの保証と統計条件の採択が未解決で、F2科学実行・新grantは0件。機械可読記録は同資料の `qualification.json`。入口は既存 `scripts/run_pr_dictionary_f2_saved.py` を維持する。


## PR混在辞書 F2限定資源計測（2026-10-24）

[限定契約・再利用SOURCE・停止条件](../../docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_resources_saved.py`、接続 `trottertracks/pr_dictionary_validation/f2_resources.py`、管理test `tests/management/test_pr_dictionary_f2_resources.py`。出力は `artifacts/pr_algorithm_improvement_phase2d_f2_resources/`。旧統計資格付き契約を維持し、認証Nを生成しない。

限定F2の[実行結果・部分資源・STOP](../../docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/EXECUTION_REPORT.md)：raw全件PASS、compiled operator3件FAIL、mean／4arm期待資源は未完了。SOURCE `9b1e103` と一回の実行原本を保全し、retryなし。


## 限定F2・位相保持修正と新規実行（2026-10-24）

[修正・実行契約・停止条件](../../docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/README.md)。既存runnerとf2_resources／f2_executionを再利用し、gates.compiled_wrapperの公式definition展開を修正。回帰testは `tests/management/test_pr_dictionary_compile_phase.py`、planned artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/`。

[修正版F2の全件結果・資源表・STOP](../../docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md)：SOURCE `9468c70`、raw／compiled各13,676・reference27,356・mean4arm・weighted集計を完了。1-shotの指標順はPP<SP<PS<SS、認証Nは対象外。旧失敗は保持し、追加実験へ進まない。

関連trackの `src/trottertracks/pr_dictionary_validation/f2_attribution.py` は[保存F2寄与分析](../../docs/research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)。既存 `f2_execution.aggregate_metrics` を変更せず再利用し、分類だけを追加した。runnerは `scripts/run_pr_dictionary_f2_attribution_saved.py`、testは `tests/management/test_pr_dictionary_f2_attribution.py`、artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_attribution/2026-10-24/`。回路・reference処理は呼ばない。

## 関連track：Rectangular GivensのPS48準備（2026-10-24）

`src/trottertracks/pr_dictionary_validation/rectangular.py` は既存Nature helperのGaussian suffixをsingletonにだけ接続する。[条件・SOURCE・未検証事項](../../docs/research/pr_rectangular_givens_preparation_20261024.md)。`rectangular_preflight.py`／`rectangular_execution.py`、runner `scripts/run_pr_rectangular_givens_saved.py`、mock test `tests/management/test_pr_rectangular_givens_preparation.py`、artifact `artifacts/pr_rectangular_givens_preparation/2026-10-24/`。共有runの既存合成を維持し、実回路・数値同値性・削減量は次回承認後の検証対象。

固定SOURCE `770e53a` の関連track `trottertracks.pr_dictionary_validation.rectangular*` による[PS48科学結果](../../docs/research/pr_rectangular_givens_ps48_execution_20261024.md)を取得した。6基底と全48 wrapperの同値性gateはPASS、資源差はcase依存。科学SOURCEの変更なし。結果は `artifacts/pr_rectangular_givens/2026-10-24/`、保存集計だけのutilityも同所。全F2や別候補へ進まずmandatory STOP。

## B2-D限定準備の既存関数再利用（2026-10-25）

`trottertracks/b2d_evaluator/input_package.py` は本ライブラリのDF分割・preparation・number sector・sector matrix／retained tail関数をlazyに再利用する。[技術報告](../../docs/research/research_b_b2d_input_preflight_technical_report_20261025.md)、[契約案](../../docs/research/research_b_b2d_stage1_science_execution_contract_proposal_20261025.md)、`tests/management/test_b2d_input_preflight.py` を参照。science runner未起動、将来artifact `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/` は今回作成しない。既存trotterlib科学SOURCEの変更なし。

## B2-D Stage 1 scope・launcher準備（2026-10-25）

既存trotterlibのGaussian preparation／sector行列化／RTEを固定科学境界へ含め、`trottertracks/b2d_evaluator/{scope,launcher,errors}.py` と `scripts/run_b2d_stage1.py` で実績照合・失敗保存を接続した。[最終提案](../../docs/research/research_b_b2d_stage1_final_execution_proposal_20261025.md)・[技術報告](../../docs/research/research_b_b2d_stage1_final_preparation_technical_report_20261025.md)、`tests/management/test_b2d_stage1_launcher.py` を参照。予定artifact `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/` は未生成。実科学SOURCEは未起動、production authority未接続、既存trotterlib科学コード変更なし。

## B2-D Stage 1一括準備完了（2026-10-25）

既存科学関数の変更なしで、`trottertracks/b2d_evaluator/{manual_authority,environment,_published_reference}.py` を追加。[最終契約・承認操作](../../docs/research/b2d_stage1_execution_ready_20261025/CONTRACT.md)・[技術報告](../../docs/research/b2d_stage1_execution_ready_20261025/TECHNICAL_REPORT.md)、`scripts/run_b2d_stage1.py`、`tests/management/test_b2d_execution_ready.py` を参照。科学SOURCE348e08、74人工test PASS。実科学計算・正式grant発行は未実施で、予定artifact `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/` は未生成。


## B2-D Stage 1通常ユーザー承認方式（2026-10-25）

最新の[未承認契約・通常権限の具体command](../../docs/research/b2d_stage1_user_approval_20261025/CONTRACT.md)と[技術報告](../../docs/research/b2d_stage1_user_approval_20261025/TECHNICAL_REPORT.md)。新科学SOURCE `91f45aeb50275bb6d22050a304d0a9b1ff025d70`、通常ユーザーのprivate手動記録へ限定修正し、root／外部APIを要求しない。同じOSアカウントの本人識別は運用規則であり独立認証ではない。入口 `scripts/{prepare_b2d_stage1,issue_b2d_stage1_approval,run_b2d_stage1}.py`、実装 `src/trottertracks/b2d_evaluator/`、人工test `tests/management/test_b2d_execution_ready.py`。既存74件を適合更新＋追加6件＝80件PASS。科学条件・scope・環境は不変、正式grant／claim／科学実行なし。予定artifactは固定checkout内 `artifacts/research_b_b2d_stage1/20261025/user-approval-once/`、今回未生成。旧root方式は上記過去段階の履歴として保全する。


## B2-D Stage 1承認管理撤去・実行簡素化（2026-10-25）

最新の[科学実行契約・単一command・技術報告](../../docs/research/b2d_stage1_simple_execution_20261025.md)。新科学SOURCE `54008c697e686a42ec48941d25219d523bafd454`。入口は `scripts/run_b2d_stage1.py --execute --source-commit … --output …`。承認packet／grant／receipt・発行／準備scriptを撤去し、科学SOURCE・input identity・scope・correctness・fresh output・失敗保存・STOPを維持した。実装 `src/trottertracks/b2d_evaluator/`、人工test `tests/management/test_b2d_simple_execution.py` と既存5ファイル、計70件PASS。科学計算は未実行。予定artifact `artifacts/research_b_b2d_stage1/20261025/simple-once/` は未生成。旧方式は過去の未採択提案として保全し、今後のB2-D実行要件にはしない。
