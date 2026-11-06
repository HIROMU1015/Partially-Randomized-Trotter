# trotterlib モジュール索引

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

## main集約の版差と原SOURCE（2026-11-03）

[索引の統合・SOURCEの版差・保存原版](../../docs/research/main_collection_versions_20261103.md)から、同名ファイルの各公開版とmainの配置を確認できる。
過去のSOURCE・test・結果は元snapshotの条件で読む。科学実行と元の研究testの再実行は行っていない。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](../../docs/integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](../../docs/research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。

## Track C H4の再利用adapter

DF/RTE/decoderを再利用する `../trottertracks/track_c_h4*.py` と `track_c_y8.py`、runner `scripts/run_track_c_h4_pilots.py`、人工test `tests/test_track_c_h4_pilots.py` は[報告](../../docs/research/track_c_h4_pilots_20261106/REPORT.md)から辿る。coreは変更せず、native compiler資格不一致で比較STOP。

## Track C compiler A

`../trottertracks/track_c_compiler_a.py`と`track_c_compiler_a_validation.py`、runner `scripts/run_track_c_compiler_a_correctness.py`、人工test `synthetic_checks()`、rawは[限定correctness報告](../../docs/research/track_c_compiler_a_correctness_20261106/REPORT.md)から辿る。H4 q1数値PASS、全RPE未資格。

## Track C compiler A 18round

`../trottertracks/track_c_compiler_a_18round.py`、runner `scripts/run_track_c_compiler_a_18round.py`、同moduleの`arithmetic_tests`/`verify_saved_models`、rawへの入口は[報告](../../docs/research/track_c_compiler_a_18round_qualification_20261106/REPORT.md)。全supportの構造上界を会計し、保存mean下で予算内、全RPEは未資格。

## Track C SOURCE保証とA価格

`../trottertracks/track_c_source_guarantee.py`、`track_c_cn_a_price.py`、runner `scripts/run_track_c_guarantee_cn_price.py`、技術検査`check_track_c_guarantee_cn_price.py`、[報告とraw](../../docs/research/track_c_guarantee_cn_price_20261106/REPORT.md)を参照。Aは固定行列モデル条件成立、Bはcohort内full/強対照差0・長q未判定。

## Track C2-B adapter

`../trottertracks/track_c2_b_pilot.py`と[限定pilot報告](../../docs/research/track_c2_b_pilot_20261106/REPORT.md)。既存core/過去の結果は変更せず、専用SOURCE・出力でSTOP。

## C2-B強baseline

`../trottertracks/track_c2_b_strong_baseline.py`は既存配分・誤差会計を再利用し、保存候補のMILP/alpha比較を行う。[報告](../../docs/research/track_c2_b_strong_baseline_20261106/REPORT.md)。旧SOURCE・raw不変更。

## DF接口の新A限定pilotでの再利用（2026-11-06）

[REPORT](../../docs/research/pr_reference_interference_pilot_20261106/REPORT.md)。`rte`、DF builder、repeated-S2 boundary fusionを変更せず、`trottertracks/pr_reference_interference_*.py` から再利用した。新2-register gadgetの検査と短q資源差は限定結果で、全RPE・T価格は未確定。旧SOURCE・STOPを維持する。

## 新A：長round移送での既存DF/RTE再利用（2026-11-07）

[REPORT](../../docs/research/pr_reference_transfer_qualification_20261106/REPORT.md)。既存DF/RTE/repeated-S2とcompiler Aを変更せず `../trottertracks/pr_reference_transfer_*.py` から再利用。`adapter_checks`、局所context、区間上界、価格包含検査を実施。固定モデルで総CX/RZの条件付き改善、depth/size/Tは未資格、mandatory STOP。
