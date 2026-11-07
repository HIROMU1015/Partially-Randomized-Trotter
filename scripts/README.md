# scripts 索引

A1の非sampling限定runnerは [run_a1_q1_q2l.py](tracks/algorithm_codesign/run_a1_q1_q2l.py)。
[固定H4 m0のidentity不一致STOP](../docs/research/o2_a1_q1_q2l_20261106/REPORT.md)：一回実行、CDF前に停止。自動再実行・測定への進行はしない。

O2-S2の固定pair-table runnerは [run_s2_multi_event_validation.py](tracks/algorithm_codesign/run_s2_multi_event_validation.py)。
[v2六cell報告](../docs/research/o2_s2_multi_event_validation_v2_20261106/REPORT.md)：許可された転送修正再実行で完了、T主指標でqB/BはP0/Bを改善せず。
runnerの出力先はv2。v1のSOURCE・途中結果は保全し、新しい科学再実行・条件探索は自動開始しない。

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


A1 v2の[payload保存修正・限定診断とSTOP](../docs/research/o2_a1_q1_q2l_20261106/REPORT_V2.md)（2026-11-06）：旧target同値は未閉鎖、CDF以降は未実行。元runnerの`--diagnose`、完全payload、人工29件と原v1保全を記録する。


A1の[旧opt2来歴・完全payload・環境の限定静的調査](../docs/research/o2_a1_opt2_provenance_20261106/REPORT.md)（2026-11-06）：記録SOURCEの回収先と未回収payload、元checkpoint抜粋、stdlib-only collector。科学再実行・library変更は0件、STOP維持。


A1の[新finite-target科学契約設計案](../docs/research/o2_a1_new_finite_target_20261106/REPORT.md)（2026-11-06）：actual-Q/selected full複素signalの定義、Raw/2層/5層/CV、公平なconfidence/全費用と後日の科学call。設計のみ、未採択・科学実行0件。


A1新targetの[N1限定実行](../docs/research/o2_a1_new_target_n1_20261106/EXECUTION.md)：既存DF/CDF/CV/policyを再利用するmodule・runner・人工test、SOURCE/環境lock。N1のみ、N2/N3は起動しない。


A1新targetの[N1結果・完全payload・correctness](../docs/research/o2_a1_new_target_n1_20261106/REPORT.md)（2026-11-06）：同じ一回の準備からCDF/Q/νと層質量・basis対応を取得。N1成功後STOP、N2/N3・測定・価格/費用は未実行。


A1の[N2限定SOURCE・state/engine検証](../docs/research/o2_a1_new_target_n2_20261106/EXECUTION.md)：N1保存payloadを再利用するmodule/runner/人工test、state inventoryと環境lock。N3なし、公開後STOP。


A1の[N2結果・state/engine保証範囲・STOP](../docs/research/o2_a1_new_target_n2_20261106/REPORT.md)：一意ground/state誤差を確認。総b_engineはUNKNOWN、N3未準備。SOURCE/payload/人工test/保全へ辿れる。


A1の[保存engine接続・一様誤差評価plan・人工correctness](../docs/research/o2_a1_engine_connection_20261106/REPORT.md)：最小adapter/test、接続/誤差計画/次回の限定科学scope。実上界未取得、N1/N2再計算・N3なし、公開後STOP。


A1の[一回のengine数値認証・runner仕様](../docs/research/o2_a1_engine_qualification_20261106/EXECUTION.md)：固定kernel、厳密state上界による最終会計、旧SOURCE保全とSTOP。


A1の[一回の総engine誤差認証・全gate証明・STOP](../docs/research/o2_a1_engine_qualification_20261106/REPORT.md)：固定exact engine/理想モデルの条件付き上界取得。N3・費用評価は未実行。


A1 N3の[技術接続・科学条件草案](../docs/research/o2_a1_n3_connection_20261106/REPORT.md)：`a1_n3_connection.py`、人工専用runner/test、保存証拠への入口。人工46件PASS、実科学0・未採択・公開後STOP。


A1 N3の[採択済み一campaign・SOURCE/実行仕様](../docs/research/o2_a1_n3_science_20261106/EXECUTION.md)：追加driver/test、固定exact engine・実call境界・保存とmandatory STOP。

A1 N3の[一campaign中断結果](../docs/research/o2_a1_n3_science_20261106/REPORT.md)：取得済397 bit/価格、未完了77 trial、精度/CV比較未判定、journal保全とmandatory STOP。

A1 N3の[中断診断・再発防止・復旧2案](../docs/research/o2_a1_n3_interruption_20261106/REPORT.md)：journal専用runner `scripts/diagnostics/a1_n3_interruption_audit.py`、`a1_n3_interruption_diagnostics`/`a1_n3_lifecycle_observer`（src/trottertracks/algorithm_codesign）、人工test `tests/test_a1_n3_interruption_diagnostics.py`。直接原因UNKNOWN、新科学0、mandatory STOP。

A1 N3の[持続manager接続と案B草案](../docs/research/o2_a1_n3_manager_preparation_20261106/REPORT.md)：`a1_n3_cron_manager.py`/`a1_n3_managed_runtime.py`（src/trottertracks/algorithm_codesign）、人工入口 `scripts/diagnostics/run_a1_n3_manager_qualification.py`、unit `tests/test_a1_n3_managed_runtime.py`。旧数値SOURCE不変、科学0。


A1 N3案Bの[新しい一campaignの実行仕様](../docs/research/o2_a1_n3_B_science_20261106/REPORT.md)（2026-11-06）：seed20261107、旧397件を分離、旧数値SOURCEとqualified managerを再利用する薄い入口。SOURCE公開照合後に一回開始し、成功・中断でmandatory STOP。


2026-11-06 A1 N3案B：[修正入口・manager_run02・限定再実行](../docs/research/o2_a1_n3_B_retry_20261106/REPORT.md)。ユーザーは技術的な意図しない中断の修復/再投入も許可。科学条件・取得済bit・旧SOURCE/STOPは維持する。


2026-11-06 A1 N3：[範囲見直し・752-shot予備契約案・現行状態と停止手順](../docs/research/o2_a1_n3_scope_redesign_20261106/REPORT.md)。A1対B1-OLの段階的予備検証を優先するユーザー指示に沿う設計。具体shot/seedは未採択・科学実行0、人工11件PASS。現行jobの停止/変更/signal送信は0、現行jobは観測時RUNNING。


2026-11-07 A1 N3：[新Bの正規停止・保全と752-shot実行準備](../docs/research/o2_a1_n3_pilot_preparation_20261107/REPORT.md)。保存H4/1.00 Å/STO-3G/legacy DF12/L_D3/m0/q1/δ.02/r1/K2/atol0の新Bを親TERM一回で停止し、1339 bit/1318価格、既知RZ6,435,609・総費用UNKNOWNを保全。新pilotは準備版、人工34件PASS、科学実行0。seed/規則/科学実行承認待ちでSTOP。


2026-11-07 A1対B1-OL：[採択済み752-shot予備検証](../docs/research/o2_a1_n3_pilot_science_20261107/REPORT.md)。保存H4/1 Å/STO-3G/DF12/L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y、seed20261108。既存pilot driverとqualified managerを新activationで接続。人工テスト後SOURCE公開照合して一campaignだけ実行し、完了/中断でmandatory STOP。精度・期待費用保証は対象外。


2026-11-07 A1対B1-OL予備検証：科学SOURCE a139d7dからconsume-once managerで一回起動したが、資源判定で最初のtrial前に正規STOP。取得0/752、Born/compile/RZ0、bundle decode1。比較は未評価、科学retry0。停止原本と全会計を[結果報告](../docs/research/o2_a1_n3_pilot_science_20261107/REPORT.md)へ保全。次attemptは空き資源確認と別途明示承認が必要。C1凍結維持。


2026-11-07 A1/B1-OL：[752-shot pilotのCPU判定改善とattempt02](../docs/research/o2_a1_n3_pilot_admission_20261107/REPORT.md)。前attemptの0件STOPを保全し、同じ科学条件・seed/streamで一回再投入する。実入力前にCPU tickを二窓観測し、最大30分の有限資源待機を行う。科学的STOP条件・exact kernel・C1凍結を維持。


2026-11-07 A1/B1-OL：[752-shot予備検証attempt02の完了結果](../docs/research/o2_a1_n3_pilot_admission_20261107/REPORT.md)。保存H4/1 Å/STO-3G/legacy DF12/L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y、actual Q/保証ground/exact engineを維持。752 bit/価格を完備、A1 2,009,673 RZ、B1-OL 1,622,442 RZ、全3,632,115 RZ。両軸zero-variance保留規則により全体INCONCLUSIVE。人工52 PASS、旧SOURCE/STOPとC1凍結を保全、公開照合後mandatory STOP。本検証の精度・期待費用保証ではない。


## 2026-11-07 SCV scalar / paired pilot preparation

Artificial-only SCV implementation; real 224-shot execution remains unadopted. [module](../src/trottertracks/algorithm_codesign/scv_scalar.py) [driver](../src/trottertracks/algorithm_codesign/scv224_pilot.py) [plan-only CLI](../scripts/tracks/algorithm_codesign/prepare_scv224_pilot.py) [control tests](../tests/tracks/algorithm_codesign/test_scv_scalar.py) [driver tests](../tests/tracks/algorithm_codesign/test_scv224_pilot.py) [report](../docs/research/scv224_preparation_20261107/REPORT.md) [evidence](../artifacts/scv224_preparation/2026-11-07/SOURCE.json)


O2-SCV224 adopted activation: [module](../src/trottertracks/algorithm_codesign/scv224_activation.py) [runner](../scripts/tracks/algorithm_codesign/run_scv224_science.py) [test](../tests/tracks/algorithm_codesign/test_scv224_activation.py) [report](../docs/research/o2_scv224_science_20261107/REPORT.md) [artifact directory](../artifacts/o2_scv224/2026-11-07/). One campaign, max224 physical shots, retry0, exact engine unchanged.


2026-11-07：[Signed support実H4 metadata-only限定検証](../docs/research/scv_signed_support_h4_metadata_science_20261107/REPORT.md)。
既存signed kernelを再利用する薄い入口、対応人工テスト、固定設定・成果物への案内。
新SOURCEは `src/trottertracks/algorithm_codesign/scv_signed_h4_metadata.py`、
CLIは `scripts/tracks/algorithm_codesign/run_scv_signed_h4_metadata.py`、
testは `tests/tracks/algorithm_codesign/test_scv_signed_h4_metadata.py`。
metadata上の構造的適格性のみを対象とし、公開後mandatory STOP。


## 2026-11-07 Signed Support 448-shot SOURCE preparation

2026-11-07：Signed Support 448-shotの実装仕様を採択し、保存control復元・薄いactivation・人工correctness・SOURCE固定を準備した。実H4科学計算は未許可・未実行、公開照合後mandatory STOP。既存科学SOURCE/結果/STOPは変更しない。
[report](../docs/research/scv_signed_support_448_preparation_20261107/REPORT.md)、[activation](../src/trottertracks/algorithm_codesign/scv_signed_activation.py)、[saved restore](../src/trottertracks/algorithm_codesign/scv_signed_restore.py)、[orchestrator](../src/trottertracks/algorithm_codesign/scv_signed_pilot.py)、[runtime adapter](../src/trottertracks/algorithm_codesign/scv_signed_runtime.py)、[deferred runner](../scripts/tracks/algorithm_codesign/run_scv_signed_pilot.py)、[artificial tests](../tests/tracks/algorithm_codesign/test_scv_signed_pilot.py)、[technical evidence](../docs/research/scv_signed_support_448_preparation_20261107/TEST_RESULTS.json)。将来の科学outputは `artifacts/scv_signed_support_fresh_pilot/2026-11-07/campaign_run01/`（今回は未作成）。


## 2026-11-07 Signed Support 448-shot complete result

2026-11-07：Signed Support 448-shot一campaignをSOURCE f70e555で完了。保存linear H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子sector/L_D3/m0/q1/δ=.02/r1/K2/atol0、N1 actual Q/selected full・N2保証ground・認証済みexact Fraction engine、X/Y。train128＋共通eval320、448 bit/価格、2,156,805 RZを保全。rank/guard合格だが両軸の追加分散point改善・学習費回収の兆候なし。精度/期待費用保証・正式GO/NO-GOなし、科学retry0、公開照合後mandatory STOP。
[report](../docs/research/scv_signed_support_448_science_20261107/REPORT.md)、[saved-record integrity](../docs/research/scv_signed_support_448_science_20261107/SAVED_RECORD_INTEGRITY.json)、[complete raw/manifest](../artifacts/scv_signed_support_448_results/2026-11-07/run01/README.md)。実SOURCE f70e555のcode/既存STOPは変更しない。


## C-S same-position residual preparation (2026-11-08)

[Technical report](../docs/research/pr_o1_m1_delta_cs_preparation_20261108/REPORT.md). New library-only placement/splice API and targeted artificial tests; no scientific runner or activation entry was added. Do not restart old M1/F2/Parent/SP runners for this preparation.

## C-S V0 static-only preflight（2026-11-08）

`tracks/algorithm_codesign/cs_v0_preflight.py` はSOURCE/保存bytes/runtimeだけを照合し、科学activation optionを持たない。[実行条件・境界](../docs/research/cs_v0_technical_preparation_20261108/V0_SCIENCE_EXECUTION_DRAFT.md)。本番V0未許可、終了後STOP。

## C-S V0 one-campaign activation

`tracks/algorithm_codesign/run_cs_v0.py` は既定ではscienceを起動しない。固定SOURCEと`--execute-one-campaign`で明示実行する経路は[契約](../docs/research/cs_v0_activation_20261108/V0_EXECUTION_CONTRACT.md)の別途科学実行許可後だけ使用。旧cs_v0_preflightを再利用し、output/cacheは一回限り。

C-S V0の固定run_cs_v0.pyをSOURCE0bd18bbから一回実行した[結果](../docs/research/cs_v0_science_20261108/REPORT.md)：保存basis文字列parameterのbinding TypeErrorでSTOP。実行prefixを再利用して再試行しない。

C-S V0 run_cs_v0.pyの[修復SOURCE](../docs/research/cs_v0_repair_20261108/REPORT.md)は新lockとcampaign_run02を使い、旧outputを再使用しない。

C-S V0の修復SOURCE e69d1dfによるrun02は[正常完了](../docs/research/cs_v0_repair_results_20261108/REPORT.md)。旧run01とsourceを保全し、追加実行へ進めない。

O2-A2の人工検査は `PYTHONPATH=src python -m unittest discover -s tests/tracks/algorithm_codesign -p test_a2_lcu_access.py -v`。新しいscience runnerは設けず、実入力の読込/RNG/compileは起動しない。[仕様・結果](../docs/research/o2_a2_executable_access_20261108/REPORT.md)。

O2案Mの人工入口は `scripts/tracks/algorithm_codesign/run_m_label_average_artificial.py --output <fresh-directory>`。新しい `test_m_label_average.py` だけを実行し、SOURCE/結果/log/STOPを保存する。実bundle/RNG/compile入口なし。[技術報告](../docs/research/o2_m_instrument_technical_validation_20261108/REPORT.md)。

O2案Mの保存header人工入口は `scripts/tracks/algorithm_codesign/run_m_n0_header_artificial.py --output <fresh-directory>`。科学入口 `run_m_n0_metadata.py --source-commit <fixed-SHA> --output <fresh-directory>` は別途許可後だけ使用する。実N1 passは今回0。[仕様・証拠・STOP](../docs/research/o2_m_n0_header_binding_preparation_20261108/REPORT.md)。
