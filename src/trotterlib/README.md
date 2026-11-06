# trotterlib モジュール索引

A1 Q1/Q2-Lは既存DF/RTE処理を再利用する[研究別接続](../trottertracks/algorithm_codesign/a1_law_validation.py)。
[runner・人工test・identity不一致STOP](../../docs/research/o2_a1_q1_q2l_20261106/REPORT.md)を参照する。core採択・CDF/bias取得成功を意味しない。

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


A1 v2の[payload保存修正・限定診断とSTOP](../../docs/research/o2_a1_q1_q2l_20261106/REPORT_V2.md)（2026-11-06）：旧target同値は未閉鎖、CDF以降は未実行。元runnerの`--diagnose`、完全payload、人工29件と原v1保全を記録する。


A1の[旧opt2来歴・完全payload・環境の限定静的調査](../../docs/research/o2_a1_opt2_provenance_20261106/REPORT.md)（2026-11-06）：記録SOURCEの回収先と未回収payload、元checkpoint抜粋、stdlib-only collector。科学再実行・library変更は0件、STOP維持。


A1の[新finite-target科学契約設計案](../../docs/research/o2_a1_new_finite_target_20261106/REPORT.md)（2026-11-06）：actual-Q/selected full複素signalの定義、Raw/2層/5層/CV、公平なconfidence/全費用と後日の科学call。設計のみ、未採択・科学実行0件。


A1新targetの[N1限定実行](../../docs/research/o2_a1_new_target_n1_20261106/EXECUTION.md)：既存DF/CDF/CV/policyを再利用するmodule・runner・人工test、SOURCE/環境lock。N1のみ、N2/N3は起動しない。


A1新targetの[N1結果・完全payload・correctness](../../docs/research/o2_a1_new_target_n1_20261106/REPORT.md)（2026-11-06）：同じ一回の準備からCDF/Q/νと層質量・basis対応を取得。N1成功後STOP、N2/N3・測定・価格/費用は未実行。


A1の[N2限定SOURCE・state/engine検証](../../docs/research/o2_a1_new_target_n2_20261106/EXECUTION.md)：N1保存payloadを再利用するmodule/runner/人工test、state inventoryと環境lock。N3なし、公開後STOP。


A1の[N2結果・state/engine保証範囲・STOP](../../docs/research/o2_a1_new_target_n2_20261106/REPORT.md)：一意ground/state誤差を確認。総b_engineはUNKNOWN、N3未準備。SOURCE/payload/人工test/保全へ辿れる。


A1の[保存engine接続・一様誤差評価plan・人工correctness](../../docs/research/o2_a1_engine_connection_20261106/REPORT.md)：最小adapter/test、接続/誤差計画/次回の限定科学scope。実上界未取得、N1/N2再計算・N3なし、公開後STOP。


A1の[一回の総engine誤差認証・全gate証明・STOP](../../docs/research/o2_a1_engine_qualification_20261106/REPORT.md)：固定exact engine/理想モデルの条件付き上界取得。N3・費用評価は未実行。


A1 N3の[技術接続・科学条件草案](../../docs/research/o2_a1_n3_connection_20261106/REPORT.md)：`a1_n3_connection.py`、人工専用runner/test、保存証拠への入口。人工46件PASS、実科学0・未採択・公開後STOP。


A1 N3の[採択済み一campaign・SOURCE/実行仕様](../../docs/research/o2_a1_n3_science_20261106/EXECUTION.md)：追加driver/test、固定exact engine・実call境界・保存とmandatory STOP。

A1 N3の[一campaign中断結果](../../docs/research/o2_a1_n3_science_20261106/REPORT.md)：取得済397 bit/価格、未完了77 trial、精度/CV比較未判定、journal保全とmandatory STOP。

A1 N3の[中断診断・再発防止・復旧2案](../../docs/research/o2_a1_n3_interruption_20261106/REPORT.md)：journal専用runner `scripts/diagnostics/a1_n3_interruption_audit.py`、`a1_n3_interruption_diagnostics`/`a1_n3_lifecycle_observer`（src/trottertracks/algorithm_codesign）、人工test `tests/test_a1_n3_interruption_diagnostics.py`。直接原因UNKNOWN、新科学0、mandatory STOP。

A1 N3の[持続manager接続と案B草案](../../docs/research/o2_a1_n3_manager_preparation_20261106/REPORT.md)：`a1_n3_cron_manager.py`/`a1_n3_managed_runtime.py`（src/trottertracks/algorithm_codesign）、人工入口 `scripts/diagnostics/run_a1_n3_manager_qualification.py`、unit `tests/test_a1_n3_managed_runtime.py`。旧数値SOURCE不変、科学0。


A1 N3案Bの[新しい一campaignの実行仕様](../../docs/research/o2_a1_n3_B_science_20261106/REPORT.md)（2026-11-06）：seed20261107、旧397件を分離、旧数値SOURCEとqualified managerを再利用する薄い入口。SOURCE公開照合後に一回開始し、成功・中断でmandatory STOP。


2026-11-06 A1 N3案B：[修正入口・manager_run02・限定再実行](../../docs/research/o2_a1_n3_B_retry_20261106/REPORT.md)。ユーザーは技術的な意図しない中断の修復/再投入も許可。科学条件・取得済bit・旧SOURCE/STOPは維持する。
