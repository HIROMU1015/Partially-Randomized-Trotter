# scripts 索引

`scripts/` はコマンドラインから実行する入口を置く。研究ロジックの本体は原則として
`src/trotterlib/` にあり、runnerは条件の読込み、呼出し、成果物保存を担当する。

最新の研究段階は[`../docs/research/研究概要・現状.md`](../docs/research/研究概要・現状.md)、
各runnerと成果物の対応は[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)を
参照する。下記の「現行」は、最終科学結論を意味せず、現在の近似検証で使う経路を表す。

## 現行の検証runner

### PF誤差係数

| runner | 用途 |
|---|---|
| `run_pf_delta_validation.py` | 小規模系でPF係数の定義・delta依存・参照値を比較 |
| `run_pf_c_system_size_validation.py` | H-chainの系サイズ方向とstate-action経路を検証 |
| `run_df_ground_state.py` | DF Hamiltonianの基底状態snapshotを作成 |
| `benchmark_df_ground_state.py` | 基底状態計算経路の計測 |
| `validate_pauli_partial_cgs_all_prefixes.py` | Pauli partial回路係数のprefix検査 |

### finite RTE

| runner | 用途 |
|---|---|
| `run_finite_rte_signal_validation.py` | 演算子誤差、状態上の複素期待値、半径、位相上界を検証 |

### コンパイル後回路コスト

| runner | 用途 |
|---|---|
| `run_random_circuit_cost_validation.py` | イベント単純加算と一体コンパイルを比較 |
| `run_rte_boundary_cost_validation.py` | 2・3イベントの境界補正を検証 |
| `run_rte_boundary_cost_replication.py` | 境界分類を別乱数seedで再検証 |
| `run_hierarchical_cost_validation.py` | 局所補正、長さ、制御付き反復をまとめて検証 |
| `run_rte_order_stratified_cost_validation.py` | 低確率Taylor次数を条件付き抽出して検証 |
| `run_rte_connected_cluster_cost_validation.py` | 1--3イベント境界補正の運用推定を検証 |
| `run_rte_connected_cluster_transfer_validation.py` | $L_D$・短時間幅が異なる条件へ移した場合を検証 |
| `run_rte_connected_cluster_holdout_supplement.py` | 独立検証の標本を補充 |
| `run_rte_connected_cluster_k4_calibration.py` | 不通過条件の4イベント係数を較正 |
| `run_rte_connected_cluster_k4_extrapolation_diagnostic.py` | 4イベント補正の外挿診断 |
| `run_rte_paired_k4_l8_validation.py` | 同一イベント列で4イベント補正の構造残差を検証 |
| `run_controlled_repetition_holdout.py` | 制御付き反復回路の反復数依存を検証 |
| `run_rte_cost_angle_invariance_validation.py` | 回転角をまたぐmetric cache再利用可否を検証 |
| `run_h5_system_size_cost_validation.py` | H5の対応あり構造検証 |
| `run_h5_independent_cost_validation_batch.py` | H5の独立係数較正・未使用回路検証 |

### RPE信号・測定回数・1 shotコストへの接続

| runner | 用途 |
|---|---|
| `run_rpe_round_cost_connection_validation.py` | 短いRPE段で信号半径、shot数、回路costを接続 |
| `run_rpe_hadamard_failure_validation.py` | 仮想Hadamard測定と失敗確率を検証 |
| `run_rpe_hadamard_proxy_resource_validation.py` | $q=8$の1 shot cost proxyと資源集計を検証 |
| `run_rpe_allocation_sensitivity_validation.py` | 位相誤差・失敗確率配分の感度を比較 |
| `run_rpe_four_round_accounting_validation.py` | $q=1,2,4,8$の限定4段を集計 |
| `run_rpe_four_round_phase_validation.py` | $q=8$物理信号と4段の分枝付き位相復元を検証 |
| `run_rpe_target_round_horizon_validation.py` | 目標精度から必要round数を決め、固定設定の長$q$可否を行列診断 |
| `run_rpe_delta_round_schedule_validation.py` | 実行済み$\delta$をPFでscreeningし、round別$(r_m,K_m)$ scheduleを行列検証 |
| `run_rpe_delta_compiled_cost_validation.py` | 検証済み局所境界係数をround scheduleへ接続し、中央RTEブロックのcompiled-cost proxyを比較 |

## 長時間・複数条件のbatch runner

次のrunnerは、上記の検証を複数jobで実行・再開するためのもの。単独の数値だけで結論を
判断せず、対応する検証文書とmanifestを読む。

- `run_rte_cost_data_batch.py`
- `run_rte_cost_followup_batch.py`
- `run_df_anchor_refinement_pipeline.py`
- `run_df_ld_window_cgs.py`
- `run_df_screening_anchor_cgs_h3_h14.py`
- `run_h5_df_cgs_gpu_slopes.py`
- `run_h9_ld5_df_cgs_ground_cache_slope_check.py`

## 診断・変換・可視化

| script | 用途 |
|---|---|
| `check_validation_manifest.py` | manifestのschemaと参照パスを検査 |
| `check_partial_randomized_diagnostics.py` | partial-randomized計算の診断値を確認 |
| `analyze_df_ld_direction_trend.py` | $L_D$方向の傾向を解析 |
| `compare_df_error_budget_rules.py` | 誤差予算規則を比較 |
| `compare_df_error_budget_rules_from_summaries.py` | 保存済み要約から誤差予算規則を比較 |
| `cache_grouping_step_costs.py` | grouping step costをキャッシュ |
| `export_df_cgs_cost_table.py` | CGS cost表を書き出す |
| `plot_df_optimized_costs.py` | DF最適化costを可視化 |
| `plot_df_partial_grouping_cost_comparison.py` | partial grouping間のcostを比較 |
| `profile_h5_8th_cgs.py` | 旧高次PF経路のH5 profiling |

## 旧経路・現在の結論に直接使わないrunner

次は履歴・比較・再現のため残している。出力を現行結果として使う前に
[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)を確認する。

- `run_partial_randomized_pf.py`
- `run_df_screening_cost_minimization.py`
- `run_grouped_uwc_pf_qpe.py`
- `run_grouped_uwc_theta_sweep.py`
- `run_uwc_grouped_alpha_h2_h6.py`

特に旧DF screeningの入力表は基底状態不一致のため失効しており、UWC数値は参照raw artifactが
未登録である。runnerが存在することと、研究結果として利用可能であることを混同しない。

## runnerを追加するとき

新しい検証runnerは、可能なら次の同名セットを用意する。

```text
src/trotterlib/<validation_name>.py
scripts/run_<validation_name>.py
tests/test_<validation_name>.py
docs/<validation_name>.md
artifacts/<validation_name>/
```

結果を研究上の証拠へ加える場合は、`VALIDATION_STATUS.md`、研究概要、研究ノート、
`artifacts/validation_manifest.json`も更新する。
