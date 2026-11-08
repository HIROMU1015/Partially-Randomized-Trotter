# scripts 索引

`run_track_d_c8_1_completion.py`：保存sample00の最終pairedχを入口とし、PASS後だけ保存残7raw/最大28P0・post、全8適格時だけ価格統計。

`run_track_d_c8_1_nofusion.py`：旧診断完了後、保存q64の4 postcheckだけをsimulation fusion無効で再検証しSTOP。

`run_track_d_c8_1_diagnostics.py`：承認済み保存q64 postcheckの失敗診断、数値条件不変。

[C8-1停止済み結果](../docs/research/track_d_c8_1_q64_20261108/REPORT.md)：旧SOURCE6b1d613d sampling/raw/P0と新SOURCE8a6bbdac postの来歴を保全。sample00 post全4が固定observable許容値で不成立、残7未開始。両runnerは再実行・resumeしない。次の数値診断はユーザー/GPTの新しい明示許可後のみ。

[C8-1高速化引継ぎrunner](run_track_d_c8_1_accelerated.py)：[承認済みPLAN](../docs/research/track_d_c8_1_q64_20261108/ACCELERATED_COMPLETION_PLAN.md)。旧controllerのactive task drain/STOPを待って未着手taskだけ継続する。compile予約1、raw/post native並列、sampling0/完了済み再実行0。旧SOURCE/STOPを保持、sample00 gate・固定精度・総32 compile上限は不変。

[B/C固定scalar runner](run_track_d_bc_total.py)：[PLAN](../docs/research/track_d_bc_total_20261108/PLAN.md)の保存N/直接平均/固定A・F・Sによる一campaign。`--preflight`はmetadata照合、`--execute`は全13-round条件付き会計。新回路/fit/Nなし、失敗後retry/resume禁止、公開照合後mandatory STOP。

[B6b明示runner](run_track_d_b6b.py)：[固定PLAN](../docs/research/track_d_b6b_bc_integration_20261108/PLAN.md)の固定Git metadata/coverageと4人工case・最大16 P0 wrapperだけ。実H4のsampling/build/compile・価格再解析・N算定・全RPE集計を実行しない。人工検証後は結果公開とmandatory STOP。

`run_track_d_c6b_accelerated.py` は[利用者承認済みSOURCE引継ぎ](../docs/research/track_d_c6b_q8_20261107/ACCELERATED_COMPLETION_PLAN.md)の残task専用入口。固定済みeventを読み、完了済みraw/compile/postcheckを再実行しない。新sampling0、旧新128 compile上限。

`run_track_d_c6b.py` is the explicit q8/n32 exploratory runner, separate from the frozen C6a q2/q4 entry. See [fixed scope](../docs/research/track_d_c6b_q8_20261107/PLAN.md). Canonical sampling, C3/C5 builders and the existing P0/checker are reused. Raw PASS gates each path's maximum four compiles; no automatic additional stage after STOP.

C6a結果のC6b提案整理は `summarize_track_d_c6a.py`（完全pilotの既存統計・モデルだけ、q8実行なし）。[結果](../docs/research/track_d_c6a_paired_price_20261106/RESULT.md)。

C6a専用入口：`run_track_d_c6a.py`。[固定PLAN](../docs/research/track_d_c6a_paired_price_20261106/PLAN.md)。既存canonical sampler/CAR/C3/C5/P0を再利用し、明示実行だけ。

[C5結果](../docs/research/track_d_c5_q2_20261106/RESULT.md)：共通request入口・人工16件・固定q2 raw/4postcheck PASS。strong手動境界融合/位相集約を統一し、P0後7価格は旧C2と一致。有限fixture限定、C6未実施、mandatory STOP。

[run_track_d_c5_q2.py](run_track_d_c5_q2.py)は[C5固定q2 PLAN](../docs/research/track_d_c5_q2_20261106/PLAN.md)専用。共通ordered requestでrawを検査し、PASS時だけ最大4 wrapper。完全byte一致のC2 compiled保存回路は再利用し、新samplingやq4/q8再実行はしない。

[C3 q4/q8補完結果](../docs/research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md)：raw両q・全8 postcheck・最終paired/χ照合PASS。強いstep境界融合を維持して両fixtureでCX/depth減、qubits+8。中断後は保存回路の最後の検証のみ、新compile0。固定fixture限定・mandatory STOP。 補完入口は[`complete_track_d_c3_run02.py`](complete_track_d_c3_run02.py)、保全回帰testは`tests/check_track_d_c3_completion.py`。

Track D C3は[`run_track_d_c3_q4_q8.py`](run_track_d_c3_q4_q8.py)。[固定PLAN](../docs/research/track_d_c3_q4_q8_20261106/PLAN.md)でq4/q8だけを扱い、既存compile/postcheckを再利用する。通常testから科学実行しない。

[C2固定q2結果](../docs/research/track_d_c2_q2_compile_20261106/RESULT.md)：4 wrapper compile/postcheck PASS。ReferenceはCX4100→3860・depth6096→3014、RZ/qubits増。固定fixture限定、mandatory STOP。

[Track D C2固定q2 compile](../docs/research/track_d_c2_q2_compile_20261106/PLAN.md)：C1の保存rawを使う2方式×X/Yの4 wrapper限定。P0・cut・phase固定、公開後mandatory STOP。

[B1/C1結果](../docs/research/track_d_b1_c1_validation_20261106/RESULT.md)：元H向け13-round条件付きoperator十分条件成立、固定q2 raw correctness PASS。compile・新sampling・N評価0、B2/C2未実施、mandatory STOP。

[Track D B1/C1固定検証](../docs/research/track_d_b1_c1_validation_20261106/PLAN.md)：保存13-roundの元H向けoperator誤差接続と、sample00→01・q2のwhole-path raw correctnessだけ。B2/C2・compile・新samplingは対象外。

[Stage A結果](../docs/research/track_d_h4_paired32_reference_20261106/RESULT.md)：保存32本すべてでCX/depth減、RZ増、qubits9→17。32 precompile／128 postcheck PASS、有限cohort限定、mandatory STOP。

[run_track_d_h4_paired_reference.py](run_track_d_h4_paired_reference.py) は[Stage A保存32 path](../docs/research/track_d_h4_paired32_reference_20261106/PLAN.md)専用。元SOURCE・保存events/reference/rawを必要とし、全precompile PASS後だけ最大128 wrapper compile。新sampling・RPE統計／全費用へ進まない。

[run_track_d_h4_reference.py](run_track_d_h4_reference.py) は[実H4固定word02](../docs/research/track_d_h4_reference_validation_20261106/PLAN.md)のcorrectnessとPASS時6 full-wrapper compileに限定する。元SOURCEと保存結果の隔離入力が必要で、終了後mandatory STOP。

[run_track_d_minimal_reference.py](run_track_d_minimal_reference.py) は[固定人工path](../docs/research/track_d_minimal_reference_validation_20261106/PLAN.md)のcorrectnessとPASS時の6 wrapper compileだけを実行し、mandatory STOPする。通常の回帰testではない。

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

[Track D C7-1 q16](run_track_d_c7_1_q16.py)：[固定PLAN](../docs/research/track_d_c7_1_q16_20261107/PLAN.md)の16 fresh whole-path・raw PASS後最大64 P0 wrapperだけ。凍結C6aモデル/CX残差、既存高速化再利用、公開後mandatory STOP。通常testから分子実行しない。

[C7-2人工runner](run_track_d_c7_2_toy.py)／[結果projection](summarize_track_d_c7_2_toy.py)：[固定54case](../docs/research/track_d_c7_2_cx_structure_20261107/PLAN.md)だけを実行。実H4 input/イベントは読み込まず、旧研究runnerのmainを呼ばない。post/source/phase照合後STOP、q32実行なし。

[C7-3 q32明示runner](run_track_d_c7_3_q32.py)：[PLAN](../docs/research/track_d_c7_3_q32_20261107/PLAN.md)の8 fresh whole-path/cut16、raw PASS後最大32 P0。旧scheduler/compile/state-action再利用、構造feature/凍結モデル診断、通常testから分子実行しない。

[C7-3高速化completion](run_track_d_c7_3_accelerated.py)：旧raw8再利用、未着手P0/postを別taskで実行。旧SOURCE/STOPは不変、[承認済み切替PLAN](../docs/research/track_d_c7_3_q32_20261107/ACCELERATED_COMPLETION_PLAN.md)と引継ぎhashを固定して明示起動する。

[C7-3停止済み結果の収録](collect_track_d_c7_3_q32.py)／[scalar報告書](summarize_track_d_c7_3.py)：SOURCE94e94のSTOP後だけ、旧sampling/raw35bfdと新compile/post94e94の来歴・160 payload member hashを整理。保存check・統計の読取りのみ、回路/統計の再計算なし。
