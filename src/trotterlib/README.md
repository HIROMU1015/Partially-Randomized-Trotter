# trotterlib モジュール索引

## 2026-10-10 Track A：H4-P runner・実行前固定 v1

最新入口は[H4-P準備契約](../../docs/research/track_a_ax2b_h4_native_receipt_preparation_v1.md)。専用source・runner・48合成testsを追加した。
CPU3・900秒・AS8GiB・output16MiB・load1/prepare8を未来の計画へ指定する。
今回の実分子load/native準備/signal/sampling/wrapper build/compileは0。
H4-P取得planのsealは認可ではなく、H4 science manifestは未sealのまま。
`H4_NATIVE_RECEIPT_NOT_AUTHORIZED` / `H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED`。
`DRAFT_NOT_AUTHORIZATION`、mandatory STOP。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4実行前契約・metadata固定 v3

最新入口は[H4契約・metadata preflight](../../docs/research/track_a_ax2b_h4_prelaunch_contract_v3.md)。
保存入力/source/旧8 cellを照合し、179 primitive-time組/537 probesを固定した。
専用metadata tests12 passed。新科学計算/array load/sampling/circuit/compile0。
native instruction receipt、CPU実割当、別grantは未固定。manifestは未sealを維持。
`H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
以下は各stage当時の履歴。旧結果・sourceと既存dirty差分を保全する。

## 2026-10-10 Track A：H4/H6 backend接続準備 v2

最新入口は[接続・実行gate・合成検証の報告](../../docs/research/track_a_ax2b_bound_ports_preparation_v2.md)。
H4独立MP/stage/event port、専用H6 sector/native backendと別grant必須launcherを追加した。
新39＋前回49の88 local synthetic/mock tests pass。分子の正しさ・総u・CI証拠ではない。
source固定のみ。actual input/coverage、CPU/別認可は未seal。新科学計算/sampling/circuit/compile0。
`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。旧証拠・既存dirty差分を保全。
以下は各stage当時の履歴。

## 2026-10-10 Track A：独立レビュー後の準備

[GPT独立レビュー](../../docs/research/track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md)を受け、[準備追補](../../docs/research/track_a_ax2b_post_independent_review_amendment_v1.md)と[H6準備契約 v2](../../docs/research/track_a_ax2b_h6_pilot_preparation_contract_v2.md)を追加。
H4-N/A/E/Mの限定計画、独立small reference・u-aware会計・tol-only adapter、別H6 controller/caps/watchdogを準備した。
専用49 local synthetic tests pass。分子H4/H6検証の新結果・総u認定ではない。
H6 molecular backend/science launcher、H4全stage検証port、input/CPU/別認可は未完了。
旧source/results/freeze/manifestと既存dirty差分を保全。`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
以下は各stage当時の履歴。

固定PM-2保存値解析sourceで一回の解析を完了し、[結果照合](../../docs/pr2_pm2_precision_resource_result_validation.md)へ記録した。
共有science/sourceは変更していない。POSTHOC local evidence、研究方針review待ちSTOP、次段未認可。
以下の未実行記述は実装・source固定時点の履歴である。

PM-2保存値解析は共有science sourceを変更せず、
`../trottertracks/resource_applicability/pm2_precision_analysis.py`に実装した。
[実装資料](../../docs/research/pr2_pm2_precision_analysis_implementation.md)と専用synthetic tests、
`artifacts/resource_applicability/pr2_pm2_precision_implementation/2026-10-05/`を一組にする。
source `324435d77b6642dbd44e8d1f178420daf62e77ed`、本解析未実行・明示指示待ち。

PM-2契約・保存JSON入力inventoryは共有science sourceを変更せず、
`../trottertracks/resource_applicability/pm2_precision_contract.py`へ分離する。
[契約](../../docs/research/pr2_pm2_precision_resource_contract_v1.md)にmodule/runner/test/artifactをまとめた。
準備だけで、精度別shot/work解析やscience実行の認可ではない。

PM-1 source `fd7552e`と134 hashは不変のまま、最終review・authorization確定・明示launchを経て一回実行した。
[結果照合](../../docs/pr2_pm1_discard_result_validation.md)：H4 developmentの8 signals/16 wrappersが完了、
pre/post201 local tests passed、mandatory STOP、次段未認可。科学module/共有helperは変更していない。
下の準備・未認可記述は各milestone当時の履歴として保持する。

Track A PM-0の純stdlib事後解析は[別namespaceのmodule](../trottertracks/resource_applicability/pm0_evidence_attribution.py)
と[報告](../../docs/research/pr2_post_m2_evidence_attribution.md)に置く。
既存M1/M2 collectorに追加sourceを混ぜないため`trotterlib`の科学コードは不変。
PM-1の[contract module](../trottertracks/resource_applicability/pm1_discard_contract.py)と
[future science adapter](../trottertracks/resource_applicability/pm1_discard_execution.py)も同namespaceへ分離した。
既存DF/PF/compile helperは変更せず再利用する。source固定・限定testsだけで、科学実行は未認可。
条件とSTOPは[PM-1契約](../../docs/research/pr2_pm1_nearby_discard_contract_v1.md)を参照する。

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
