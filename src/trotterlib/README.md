# trotterlib モジュール索引

C1 Stage3の限定数値modeは`../trottertracks/c1_stage3_feasibility.py`。旧CAR／保存loader／ground・alias helper／PF・RTE reference／候補順序／call・資源計画を再利用し、QPY／回路／sampling／shot・費用経路へ接続しない。entry `scripts/run_research_c_c1_stage3_feasibility.py`、test `tests/management/test_research_c_c1_stage3_feasibility.py`、[技術報告](../../docs/research/research_c_c1_stage3_feasibility_preparation_20261025.md)、証拠 `artifacts/research_c_c1_stage3_feasibility_technical/2026-10-25/`。科学上限639、実H4未実行・STOP。


C1 Stage3の統合実行は`../trottertracks/c1_stage3_integrated.py`と`c1_stage3_molecular.py`、全law資格は`c1_stage3_qualification.py`、負係数RTE検査の新versionは`c1_stage3_native_bridge.py`。既存一般K／CAR／native／PF・RTE／RPE／compiler／費用評価を再利用し、旧SOURCEは保全する。entry `scripts/run_research_c_c1_stage3_integrated.py`、test `tests/management/test_research_c_c1_stage3_integrated.py`、[統合技術報告](../../docs/research/research_c_c1_stage3_integrated_preparation_20261025.md)、証拠 `artifacts/research_c_c1_stage3_integrated_technical/2026-10-25/`。実H4科学計算0、全lawの有用な合成誤差上界未資格、STOP。


C1 Stage3の新しい最小接続は`../trottertracks/c1_stage3_connection.py`（一般K tail／RTE／native直列接続）、`c1_stage3_reference.py`（独立generator／PF／RTE／virtual RPE）、`c1_stage3_accuracy.py`（CA／signal／shots／費用）にある。既存RTE・horizon・Hoeffding・phase復元・compilerを再利用。旧square-specific cost providerと短q proxyはCDFへ直接転用しない。[技術報告](../../docs/research/research_c_c1_stage3_preparation_20261025.md)、entry `scripts/run_research_c_c1_stage3.py`、人工test `tests/management/test_research_c_c1_stage3.py`。実H4科学計算なし、条件未採択・STOP。


関連trackの[C1 Stage2限定科学結果](../../docs/research/research_c_c1_stage2_native_execution_20261025.md)は既存Givens／diagonal emitter／CAR／controlled／X/Y／compilerを無変更で再利用し、全60主検査とunitarity52件PASS。CDF保存rank8／XDF rank10、call243/243、fit／retry／resume0。証拠 `artifacts/research_c_c1_stage2_science/2026-10-25/`。物理DF精度差があり同精度総資源優位ではない。Stage3未承認・mandatory STOP。

関連trackの[C1 Stage2 native接続](../../docs/research/research_c_c1_stage2_native_preparation_20261025.md)は `trottertracks.c1_stage2_native` から既存 `df_trotter.ops/decompose` と独立CAR、controlled/X/Y/compiler/metric資産を再利用する。旧library変更なし。entry `scripts/run_research_c_c1_stage2_native.py`、test `tests/management/test_research_c_c1_stage2_native.py`、SOURCE／証拠 `artifacts/research_c_c1_stage2_technical/2026-10-25/`。人工80件＋29 subtests PASS、科学条件は[未承認案](../../docs/research/research_c_c1_stage2_native_contract_unapproved_20261025.json)、実回路・科学call0、mandatory STOP。

関連trackの[C1保存fit評価専用科学実行](../../docs/research/research_c_c1_stage1_eval_only_execution_20261025.md)は `trottertracks.c1_stage1_eval_only` と既存CAR／一般K／XDFを無変更で使用した。full256 base・全8 frame／sector36を含む37 gate PASS、CDF/XDF物理誤差は別測定。entry `scripts/run_research_c_c1_stage1_eval_only.py`、人工19件PASS、科学call141/141、新fit／optimizer／retry／resume0。証拠 `artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/`。Stage2は別判断・mandatory STOP。以下の未承認・未再検証表示は当時の履歴。

関連trackの [C1保存fit評価専用準備](../../docs/research/research_c_c1_stage1_eval_only_preparation_20261025.md)は `trottertracks.c1_stage1_eval_only` から既存修正済みCAR／一般K／保存XDF評価を再利用し、数値libraryは無変更。entry `scripts/run_research_c_c1_stage1_eval_only.py`、新test `tests/management/test_research_c_c1_stage1_eval_only.py`、SOURCE／証拠 `artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/`。旧fit／新評価SOURCEを分離、人工82件＋29 subtests PASS、[未承認契約](../../docs/research/research_c_c1_stage1_eval_only_contract_unapproved_20261025.json)。実H4評価0、mandatory STOP。

関連trackの [C1 raw CAR frame修正](../../docs/research/research_c_c1_frame_repair_20261025.md)は `trottertracks.c1_stage1.raw_frame` の縮約だけをexplicit batched matmulへ接続。一般K／独立CAR代数／保存XDF／allowanceを保持し、新test `tests/management/test_research_c_c1_frame_repair.py` と既存回帰63件＋29 subtests、別プロセス64人工frame比較PASS。SOURCE／証拠 `artifacts/research_c_c1_frame_repair/2026-10-25/`、分子再解析・新fit0、H4の旧FAILは未再検証のままmandatory STOP。

関連trackの [C1 Stage 1互換性修正版実行](../../docs/research/research_c_c1_stage1_compat_execution_20261025.md)：SOURCE無変更で通常C-DF fit1回が収束（R8）。full256 frame0のraw CAR／K同値性がFAILし科学的STOP、既存保存XDF sector経路は構築したが物理CDF/XDF normは未到達。科学call107/147、retry/resume0、原本 `artifacts/research_c_c1_stage1_compat_science/2026-10-25/`、追加科学実行なし。

関連trackの [C1 Stage 1互換性修正](../../docs/research/research_c_c1_stage1_compatibility_20261025.md)は `trottertracks.c1_ffsim_compat` のtask専用batch logm bridgeだけを追加し、既存一般K／独立CAR／保存XDFの数値経路を保持する。runner `scripts/run_research_c_c1_stage1.py`、人工tests `tests/management/test_research_c_c1_ffsim_compat.py` と既存回帰59件＋29 subtests PASS、SOURCE／lock／証拠 `artifacts/research_c_c1_stage1_compatibility/2026-10-25/`。[新未承認契約](../../docs/research/research_c_c1_stage1_compatibility_contract_unapproved_20261025.json)、分子科学call0、別承認待ち・mandatory STOP。

関連track `trottertracks.c1_stage1*` の[Stage 1限定科学実行](../../docs/research/research_c_c1_stage1_execution_20261025.md)はffsim/SciPy batch logm非互換でfit初期化中に停止。保存XDFの既存sector演算・CAR/K/物理normへは未到達。artifact `artifacts/research_c_c1_stage1_science/2026-10-25/`、科学call2・fit1 attempt、追加実行なし、mandatory STOP。

関連trackの `trottertracks.c1_stage1` / `c1_stage1_execution` は、既存 `df_hamiltonian` のsector演算を保存XDFに再利用する[C1 Stage 1技術準備](../../docs/research/research_c_c1_stage1_technical_preparation_20261025.md)。runner `scripts/run_research_c_c1_stage1.py`、人工tests `tests/management/test_research_c_c1_stage1*.py`、SOURCE/lock/証拠 `artifacts/research_c_c1_stage1_technical/2026-10-24/`。実分子科学実行・回路は未実行・未承認。

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


## 研究C・C4：Q13限定科学検証（2026-10-24）

[実行契約・履歴・結果](../../docs/research/research_c_rustiq_execution_20261024.md)。SOURCEは `src/trottertracks/pr_dictionary_validation/rustiq_adapter.py` と `rustiq_execution.py`、入口は `scripts/run_research_c_rustiq_limited.py`、管理testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_limited/2026-10-24/`。保存F1の4case・B0/B1/R1だけを対象とし、旧SOURCE・grant・STOPを保持する。


## 研究C・C4：B2角度結合baseline限定科学比較（2026-10-24）

[結果・固定SOURCE・契約・証拠](../../docs/research/research_c_angle_fusion_b2_execution_20261024.md)。既存 `rustiq_adapter.py`／`rustiq_execution.py` と `scripts/run_research_c_rustiq_limited.py --scope C4_B2_SAVED_COMPARISON_V1` を使用。testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/`。全登録gateがPASSし、全8wrapper・全8compiled指標でB2=保存R1。単純融合がこのfixtureの改善を再現した。旧route再実行・新reference・Rustiq/R2・retryは0。独立研究を区切るかはユーザー／研究C／統括の次判断であり未採択、mandatory STOP。
