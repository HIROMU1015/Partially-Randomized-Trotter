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

- `run_parallel_validation_batch.py`：共有サーバー向けtask manifest生成、dry-run、bounded実行、resume、status
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
| `retire_archived_remote_branches.py prepare / tags / delete` | 固定済みの承認済みbranch整理計画を段階実行する。先に完全バックアップと保存tagを検証し、公開台帳とSHAが一致する旧branchだけを削除する。[引き継ぎ](../docs/research/branch_archive_20261102/HANDOFF.md)を読む |
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

## 過去の研究SOURCEの保存

[G6 technical runner SOURCE](tracks/algorithm_codesign/g6_return_generator_audit.py)を元の内容で収録した。
[収録範囲と元snapshot](../docs/research/main_collection_20261103.md)を参照。元の一回実行・SOURCE/runtime binding・marker・STOPを継承し、今回の収録では実行していない。

## main集約の版差と原SOURCE（2026-11-03）

[索引の統合・SOURCEの版差・保存原版](../docs/research/main_collection_versions_20261103.md)から、同名ファイルの各公開版とmainの配置を確認できる。
過去のSOURCE・test・結果は元snapshotの条件で読む。科学実行と元の研究testの再実行は行っていない。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](../docs/integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](../docs/research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。

## ローカルSOURCE入口の収録（2026-11-03）

[AX-2A metadata-only writer](resource_applicability/prepare_track_a_ax2a.py)と[AX-2B metadata-only writer](resource_applicability/prepare_track_a_ax2b_preflight.py)の原版を追加した。[原本対応と未検証範囲](../docs/research/main_local_results_collection_20261103.md)を参照する。既存moduleの存在と構文だけを確認し、runner・研究test・科学計算は実行していない。


## 保存済み公開ファイルの固定commit照合

[`verify_archived_publication_bytes.py`](verify_archived_publication_bytes.py)は、E-R3準備版とB3 API preflightの
既存公開リストを使い、固定commitの文書・SOURCE・小さなmetadataのbytesを読み取り専用で確認する。
使い方・対象外ファイル・原検証との違いは[固定commit参照の案内](../docs/research/branch_reference_navigation_20261103.md#削除したbranchへ問い合わせない公開bytes確認)を参照する。
科学runnerや原結果の再解析は実行しない。

## B2 RPE接続の人工検証（2026-11-06）

[REPORTとpytest入口](../docs/research/track_a_b2_rpe_implementation_20261106/REPORT.md)を参照。新しい科学runnerは追加していない。固定M1版から取り込んだ`run_o1_m1_h4_operator_correctness.py`は既存人工テストの参照helperとしてのみ使用し、実H4 mainは実行していない。

## B2 P1の限定信号検証（2026-11-07）

`run_b2_p1_signal_validation.py`は明示private入力でm0/m17だけを実行する。[SOURCE・結果・STOP](../docs/research/track_a_b2_p1_two_round_20261107/REPORT.md)。他round・shot・価格へ進まない。

`run_b2_p1_remaining_rounds.py`は別承認されたm1..16を固定順で評価し、初回不適格で停止する。[全18点の条件付き信号結果](../docs/research/track_a_b2_p1_remaining_rounds_20261107/REPORT.md)。回路build・endpoint再計算・branch・shot・価格なし、公開後STOP。

`run_b2_p1_branch_validation.py`は固定保存raw平均を既存nearest-branch decoderへ渡す。[branch/energy結果とSTOP](../docs/research/track_a_b2_p1_branch_20261107/REPORT.md)。新信号・private読込・shot・価格なし。

`run_b2_p1_shot_accounting.py`は固定保存18roundのboundsを既存Hoeffding会計へ接続する。[shot結果とSTOP](../docs/research/track_a_b2_p1_shots_20261107/REPORT.md)。価格・費用は算出しない。

## B2 P1 m0限定full no-prep compile（2026-11-07）

`run_b2_p1_compiled_cost.py`：固定済みcompiler・入力・代表trajectory一つのm0 X/Y接続検証。
[実行条件とSTOP](../docs/research/track_a_b2_p1_compiled_cost_20261107/REPORT.md)。保存実行の再実行や価格MCを許可する入口ではない。

## B2人工価格方式比較（2026-11-07）

`run_b2_cost_method_validation.py`は人工二mode回路だけでwhole/stream/fusion/split/cache/parallelを比較する。
[固定SOURCE・結果・STOP](../docs/research/track_a_b2_p1_long_cost_method_20261107/REPORT.md)。実H4や本格MCの実行入口ではない。

## B2 m0実H4 streaming限定照合（2026-11-07）

`run_b2_m0_streaming_h4.py`は同じ保存trajectoryのnative2経路を照合後、stream X/Y各1回だけcompile。
[保存結果・SOURCE・STOP](../docs/research/track_a_b2_p1_m0_streaming_h4_20261107/REPORT.md)。Q2以降や再実行の許可ではない。

## B2 m1/Q2実H4 streaming限定照合（2026-11-07）

`run_b2_q2_streaming_h4.py`は既存m0 helperとnative/stream/compilerを再利用し、固定8 eventの境界を比較する。
[SOURCE・全6価格・STOP](../docs/research/track_a_b2_p1_q2_streaming_h4_20261107/REPORT.md)。両方式X/Y計4回を実行済み。再実行・m2以降・MCの許可ではない。

## Fingerprintの人工benchmark（2026-11-07）

`benchmark_streamed_circuit_fingerprint.py`は人工回路のみで旧実装と逐次hashの同一SHA・時間・メモリを測定。
[SOURCE・測定protocol・限界](../docs/research/streamed_circuit_fingerprint_20261107/REPORT.md)。分子入力・compiler・solverは呼ばない。

## B2人工pipelineの段階計測（2026-11-07）

`benchmark_b2_fingerprint_pipeline.py`は既存人工B2・stream builder・wrapper・pair compilerを再利用する。
[固定SOURCE・Q1/Q2/Q8比較・段階別memory](../docs/research/b2_fingerprint_pipeline_20261107/REPORT.md)。実H4やMCは実行しない。

## 実H4 Q2のopt-in hash限定照合（2026-11-07）

`run_b2_q2_fingerprint_h4.py`は旧Q2の保存入力・event・価格と既存のopt-in pipelineを再利用する。
[SOURCE・実行条件・結果・STOP](../docs/research/b2_q2_fingerprint_h4_20261107/REPORT.md)。native1回・compile2回に限定。再実行や他roundの許可ではない。

## 大きい人工flat回路のstage RSS（2026-11-07）

`benchmark_large_artificial_memory.py`は既存人工unit・wrapper・compiler/hashを再利用し、
parentが自workerのRSSを5ms間隔で観測する。[JSON/CSV・条件・測定限界・STOP](../docs/research/large_artificial_memory_20261107/REPORT.md)。
