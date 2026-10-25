# scripts 索引

## C1：H6 native finite-probe／kernel費用（科学未承認）

`run_research_c_c1_h6_native.py`：defaultはpassive preflight。30 local factors→3 shells→4候補×literal/sharedの8 kernel routes、actual adjoint/X-Y/P0 metricsへ接続。[報告](../docs/research/research_c_c1_h6_native_preparation_20261025.md)／[未承認298-call契約](../docs/research/research_c_c1_h6_native_contract_unapproved_20261025.json)。新fit/PF再計算/shots/総費用dispatchなし。

## C1：保存H6 rank14のsector PF実行結果

`run_research_c_c1_h6_saved_pf.py`をSOURCE87ea924ae…で1回実行。72/72 units、4候補・52roundが登録十分条件PASS。[報告](../docs/research/research_c_c1_h6_saved_pf_execution_20261025.md)。fit/optimizer/retry/resume0、旧runは再起動せずfresh run01を保全、native/shots/costへ進まずmandatory STOP。

## C1：保存H6 rank14のsector PF（科学未承認）

`run_research_c_c1_h6_saved_pf.py`：defaultはpassive preflight。保存CDF14／XDF14、一般K同値性とground/aliasの全PASS後、決定論S2 s1/s4×13roundを評価する専用entry。[技術報告](../docs/research/research_c_c1_h6_saved_pf_preparation_20261025.md)／[未承認72-call契約](../docs/research/research_c_c1_h6_saved_pf_contract_unapproved_20261025.json)。fit・rank探索・回路・shots・費用dispatchなし。

## C1：H6 rank14修正codec runnerの限定実行結果

`run_research_c_c1_h6_corrected_fit.py`をSOURCE33432e786…で1回実行し20/20 units・optimizer成功／DF予算PASS。[結果](../docs/research/research_c_c1_h6_corrected_fit_execution_20261025.md)。fresh run01とSTOPを保全、retry/resume0。新fit・後続評価は自動実行しない。

## C1：H6修正codecのrank14 1 fit（科学未承認）

`run_research_c_c1_h6_corrected_fit.py`：defaultはpassive byte preflight。保存XDF14→修正codec→sector400初期gate→最大1 ffsim fit→終了sector／予算の専用経路。[技術報告](../docs/research/research_c_c1_h6_corrected_fit_preparation_20261025.md)／[未承認20-call契約](../docs/research/research_c_c1_h6_corrected_fit_contract_unapproved_20261025.json)。新科学は別途許可後のみ。

## C1：修正codecの保存H6 round-trip

`run_research_c_c1_h6_codec_roundtrip.py`は既定byte preflightのみ。将来、保存rank14/18の初期J/Uを再分解せず、修正codec→4sector状態→同値性・DF予算保持だけを最大20callsで確認する。[未承認契約](../docs/research/research_c_c1_h6_codec_roundtrip_contract_unapproved_20261025.json)／[人工結果](../docs/research/research_c_c1_h6_codec_roundtrip_preparation_20261025.md)。科学許可後のみ実行し、fit／PF／native／shots／費用は対象外。

## C1：保存H6初期化診断

`run_research_c_c1_h6_initialization.py`は保存H6のrank14／18、変換前／直後／保存CDF終了後だけを評価する専用entry。別fit/eval環境、既存ffsim codec・SectorCAR・rank_assessmentを再利用し、optimizer／PF／nativeを呼ばない。[固定35-call条件](../docs/research/research_c_c1_h6_initialization_contract_20261025.json)／[実行結果](../docs/research/research_c_c1_h6_initialization_execution_20261025.md)。今回の許可は消化済み、fresh再実行には別の範囲・許可が必要。

`run_research_c_c1_h2_h6.py --molecule 2|6 --contract <contract>` は[H2・H6準備](../docs/research/research_c_c1_h2_h6_preparation_20261025.md)。既定はmetadata／byte preflightだけ。新科学条件を採択して別途許可後だけ `--execute --fit-python <isolated-fit-python> --output <fresh>` を使う。H2上限196 unitsはrank→native→数値CA/shots/cost、H6上限129はsector400 DF/PFまで。1コマンド1分子、retry/resume0、H6 nativeや次分子へ自動進行しない。[未承認案](../docs/research/research_c_c1_h2_h6_contract_unapproved_20261025.json)。人工14＋既存166 tests・29 subtests PASS、今回実科学0・STOP。

`run_research_c_c1_stage3_deterministic_accuracy.py` は[決定論4候補の精度準備](../docs/research/research_c_c1_stage3_deterministic_accuracy_preparation_20261025.md)。既定はbyte preflight。別科学許可後にだけ `--execute --output <fresh>` で保存PF・DF・shots・費用、s1保存Stage2残差／s4局所reference2／共通shellを接続する。[未承認契約](../docs/research/research_c_c1_stage3_deterministic_accuracy_contract_unapproved_20261025.json)最大110 units、fit／optimizer／retry／resume／新build・compile／shot-count solving0。厳密CA未資格、数値pilot PASSだけ費用へ接続。人工12＋独立process回帰140 PASS／29 subtests、実H4科学call0・STOP。

`run_research_c_c1_stage3_cost_compat.py` は[固定Qiskitのcompile競合修正](../docs/research/research_c_c1_stage3_cost_compat_repair_20261025.md)。旧P0を変更せずcompile callbackだけ排他化、他candidate workは旧並列計画。既定はbyte preflight、今回の修正再実行許可に従って`--execute --output <fresh>`で同8候補cost-onlyを実行する。[936-call契約](../docs/research/research_c_c1_stage3_cost_compat_contract_20261025.json)。旧run/STOPは再利用せず、人工100 PASS。

`run_research_c_c1_stage3_cost_only.py --mode cost-only` は[8候補native費用準備](../docs/research/research_c_c1_stage3_cost_preparation_20261025.md)。既定はbyte preflightのみ。別明示科学許可後だけ`--execute --output <fresh>`でD s1/s4・PR s4/r16/K0/K2各armの費用を測り、保存13round X/Y楽観Nで条件付き総RZ等を積算する。[未承認契約](../docs/research/research_c_c1_stage3_cost_contract_unapproved_20261025.json)最大936 units、PR各32校正、fit／retry／resume0。PF/RTE／shots再算定なし。support未資格は点推定のみ、CA／実shots保証へ昇格しない。人工94 PASS、今回実H4科学call0・STOP。

`run_research_c_c1_stage3_shots_screen.py` は[楽観的shotsスクリーニング](../docs/research/research_c_c1_stage3_shots_preparation_20261025.md)。既定はbyte preflightのみ。別明示許可後だけ`--execute --output <fresh>`で保存20候補×13roundのPF/RTE/Aから仮定付きX/Y算定値・合計・減衰診断を作る。[未承認契約](../docs/research/research_c_c1_stage3_shots_contract_unapproved_20261025.json)は最大282 logical units、fit／retry／resume0。未知誤差は実際の0と認定せず、CA・回路・総費用・資源順位なし。人工81 PASS、今回再解析0・STOP。

`run_research_c_c1_stage3_feasibility.py --mode feasibility-only` は[C1限定数値経路](../docs/research/research_c_c1_stage3_feasibility_preparation_20261025.md)。既定はbyte preflightだけ。別明示科学許可後だけ`--execute --output <fresh>`で260セルのPF/RTE誤差・減衰と20候補の十分条件分類を保存する。full modeへのdispatchなし、CA達成／shots／費用／順位は評価しない。[639-call未承認契約](../docs/research/research_c_c1_stage3_feasibility_contract_unapproved_20261025.json)、SOURCE・人工結果 `artifacts/research_c_c1_stage3_feasibility_technical/2026-10-25/`。今回科学call0、STOP。


`run_research_c_c1_stage3_integrated.py` は[C1 Stage3統合runner](../docs/research/research_c_c1_stage3_integrated_preparation_20261025.md)。既定はbyte preflightのみ、別途チャットでの科学許可後だけ`--execute --output <fresh>`を使う。保存入力からevidence生成・shots／費用比較までの一体経路で、旧packet評価SOURCEは無変更。[未承認契約](../docs/research/research_c_c1_stage3_integrated_contract_unapproved_20261025.json)は固定20候補／13round・最大5,637 logical units・fit／retry／resume0。人工19＋既存65 tests／29 subtests PASS、今回実H4科学call0、全law合成誤差未資格・mandatory STOP。


`run_research_c_c1_stage3.py` は[C1 Stage3技術準備](../docs/research/research_c_c1_stage3_preparation_20261025.md)のSOURCE／保存入力byte確認を既定とする。科学条件を採択して別途チャットで許可した後だけ`--evaluate-evidence <packet> --output <fresh.json>`で精度／shot／費用を評価する。分子evidence生成は既存loader／native／RTEと新しい最小接続関数を組み合わせる。[未承認契約](../docs/research/research_c_c1_stage3_contract_unapproved_20261025.json)、人工test `tests/management/test_research_c_c1_stage3.py`、証拠 `artifacts/research_c_c1_stage3_technical/2026-10-25/`。今回科学call0・mandatory STOP。


`run_research_c_c1_stage2_native.py` の[C1 Stage2限定実行](../docs/research/research_c_c1_stage2_native_execution_20261025.md)は採択されたT1/10・S2・P0 profileで全60主検査＋unitarity52件PASS。SOURCE／runner無変更、call243/243、fit／retry／resume0、68 QPY・12 caseのnative metric記録。原本 `artifacts/research_c_c1_stage2_science/2026-10-25/`。Stage3未承認・mandatory STOP。

`run_research_c_c1_stage2_native.py` は[C1 Stage2接続](../docs/research/research_c_c1_stage2_native_preparation_20261025.md)のbyte preflightを既定とする。実行前にチャットで条件採択・別明示科学許可を受け、`--execute --output <fresh>` を使用する。approval記録や証明照合は不要。処理 `trottertracks.c1_stage2_native/validation/execution`、test `tests/management/test_research_c_c1_stage2_native.py`、SOURCE／52 pins／人工証拠 `artifacts/research_c_c1_stage2_technical/2026-10-25/`、[未承認契約](../docs/research/research_c_c1_stage2_native_contract_unapproved_20261025.json)。人工80件＋29 subtests PASS、今回実回路build/compile・科学call0、mandatory STOP。

`run_research_c_c1_stage1_eval_only.py` の[保存fit評価専用科学実行](../docs/research/research_c_c1_stage1_eval_only_execution_20261025.md)は、公開SOURCE `bec41ac7dc7e034bd664817acfdcfed858ddadd0` を無変更で使用し37 gate PASS。旧fit来歴を独立に保持、新fit0・call141/141・retry/resume0。結果 `artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/`、直前の人工19件PASS。Stage2適格性は別判断、旧FAIL／STOP保全・mandatory STOP。以下の準備段階の未承認表示は履歴。

`run_research_c_c1_stage1_eval_only.py` は[C1評価専用入口](../docs/research/research_c_c1_stage1_eval_only_preparation_20261025.md)。既定はbyte/JSON preflight、実行は新scopeの別科学承認が必要。`--worker-fit`／fit Python引数はなく、旧one-fit runnerを再起動しない。処理 `trottertracks.c1_stage1_eval_only`、人工test `test_research_c_c1_stage1_eval_only.py`、SOURCE／証拠 `artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/`、[未承認契約](../docs/research/research_c_c1_stage1_eval_only_contract_unapproved_20261025.json)。人工82件＋29 subtests PASS、新fit0／最大141、H4未評価・mandatory STOP。

[C1 raw frame修正](../docs/research/research_c_c1_frame_repair_20261025.md)の人工監査は `artifacts/research_c_c1_frame_repair/2026-10-25/artificial_parallel_audit.py` と `preloaded_backend_audit.py`。新test `tests/management/test_research_c_c1_frame_repair.py`、修正 `trottertracks.c1_stage1.raw_frame`、SOURCE／証拠 `artifacts/research_c_c1_frame_repair/2026-10-25/`。人工63件＋29 subtests PASS、分子科学処理0。`run_research_c_c1_stage1.py` はone-fit入口のままで、保存exportの評価専用runnerではない。別準備・科学承認待ち、mandatory STOP。

`run_research_c_c1_stage1.py` の[互換性修正版Stage 1科学実行](../docs/research/research_c_c1_stage1_compat_execution_20261025.md)は新approvalで1 fit収束後、frame0 raw CAR／K同値性FAILによりSTOP。科学call107/147、retry/resume0、証拠 `artifacts/research_c_c1_stage1_compat_science/2026-10-25/`。SOURCEと既存runnerは変更せず、物理sector norm未到達、追加実行・Stage2/3なし。

`run_research_c_c1_stage1.py` は[ffsim／SciPy互換性修正版](../docs/research/research_c_c1_stage1_compatibility_20261025.md)でvenv Python symlinkを保持し、`trottertracks.c1_ffsim_compat` 経由の初期化へ接続する。人工tests `test_research_c_c1_ffsim_compat.py` を含む59件＋29 subtests PASS、証拠・SOURCE／lock `artifacts/research_c_c1_stage1_compatibility/2026-10-25/`。[新未承認契約](../docs/research/research_c_c1_stage1_compatibility_contract_unapproved_20261025.json)は新scope、fresh outputへの別明示承認が必要。科学実行0、mandatory STOP。

`run_research_c_c1_stage1.py` の[限定科学実行・技術的STOP](../docs/research/research_c_c1_stage1_execution_20261025.md)：固定SOURCE `de76aee72`、1 fitはSciPy1.14.1 batch logm非互換でoptimizer前に停止。call2/147、科学retry/resume0。artifact `artifacts/research_c_c1_stage1_science/2026-10-25/`。SOURCEを修正せず証拠保存・mandatory STOP。

`run_research_c_c1_stage1.py` は[C1 Stage 1技術準備](../docs/research/research_c_c1_stage1_technical_preparation_20261025.md)の限定入口。既定は数値decodeなしpreflight、科学runは別の明示承認が必要。処理 `trottertracks.c1_stage1*`、人工test `tests/management/test_research_c_c1_stage1*.py`、SOURCE/lock/証拠 `artifacts/research_c_c1_stage1_technical/2026-10-24/`。今回は人工検証のみ、mandatory STOP。

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


## PR混在辞書 F1保存入力経路（準備のみ）

`run_pr_dictionary_f1_saved.py` は旧 `run_pr_dictionary_f1.py` の認可gateを再利用する将来F1専用入口。新規の明示承認・固定packetが必要で、F1-PREでは実行しない。SOURCE、管理test、planned artifactは[準備記録](../docs/research/pr_algorithm_improvement_phase2d_f1_preparation_20261023/README.md)を参照。


## PR混在辞書F1の実行結果（2026-10-24）

`run_pr_dictionary_f1_saved.py` の承認済みone-shotは最初のcompile requestで技術的停止。SOURCE修正や再実行は行っていない。[実行結果](../docs/research/pr_algorithm_improvement_phase2d_f1_execution_20261024/README.md)と `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24/` に完了／未完了、残差、provenanceを記録した。


## PR混在辞書F1：修正版の登録検証完了（2026-10-24）

[新結果とSTOP](../docs/research/pr_algorithm_improvement_phase2d_f1_rerun_20261024/README.md)。既存saved runner／adapter／reference／pricingを再利用し、SOURCE `8192530` と保存B入力で16familyを全件完了。全4armの登録正しさ検査とcatalog価格-DP照合はPASS、未資格intervalはfallbackのみ。認証score／shot／資源優位性は未評価。artifactは `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/`。旧停止記録は保全し、F2へ進まない。


## PR混在辞書 F2実行前準備（2026-10-24）

[準備・SOURCE再利用・case・統計資格・契約・STOP](../docs/research/pr_algorithm_improvement_phase2d_f2_preparation_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_saved.py`、SOURCE `trottertracks/pr_dictionary_validation/f2_*.py`、管理test `tests/management/test_pr_dictionary_f2_preparation.py`。planned outputは `artifacts/pr_algorithm_improvement_phase2d_f2/<execution-date>/<fresh-run-id>/`。finite-target主比較を採択し、mock/管理確認のみ完了。統計条件とinterval資格は未解決で、科学実行・新grantは0件。


## PR混在辞書 F2統計資格確認（2026-10-24）

[資格確認と実行前STOP](../docs/research/pr_algorithm_improvement_phase2d_f2_qualification_20261024/README.md)。既存interval／log／ceilの技術test `tests/management/test_pr_dictionary_f2_scalar_intervals.py` は5件PASS。科学SOURCEはF2-PREのまま、共通finite targetに対する有限ビット誤差予算γの保証と統計条件の採択が未解決で、F2科学実行・新grantは0件。機械可読記録は同資料の `qualification.json`。入口は既存 `scripts/run_pr_dictionary_f2_saved.py` を維持する。


## PR混在辞書 F2限定資源計測（2026-10-24）

[限定契約・再利用SOURCE・停止条件](../docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_resources_saved.py`、接続 `trottertracks/pr_dictionary_validation/f2_resources.py`、管理test `tests/management/test_pr_dictionary_f2_resources.py`。出力は `artifacts/pr_algorithm_improvement_phase2d_f2_resources/`。旧統計資格付き契約を維持し、認証Nを生成しない。

限定F2の[実行結果・部分資源・STOP](../docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/EXECUTION_REPORT.md)：raw全件PASS、compiled operator3件FAIL、mean／4arm期待資源は未完了。SOURCE `9b1e103` と一回の実行原本を保全し、retryなし。


## 限定F2・位相保持修正と新規実行（2026-10-24）

[修正・実行契約・停止条件](../docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/README.md)。既存runnerとf2_resources／f2_executionを再利用し、gates.compiled_wrapperの公式definition展開を修正。回帰testは `tests/management/test_pr_dictionary_compile_phase.py`、planned artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/`。

[修正版F2の全件結果・資源表・STOP](../docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md)：SOURCE `9468c70`、raw／compiled各13,676・reference27,356・mean4arm・weighted集計を完了。1-shotの指標順はPP<SP<PS<SS、認証Nは対象外。旧失敗は保持し、追加実験へ進まない。

`run_pr_dictionary_f2_attribution_saved.py` は[保存F2寄与分析](../docs/research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)の入口。既存CSV・law・B atomまたは公開派生featuresだけを使い、回路・lawを生成しない。処理は `trottertracks.pr_dictionary_validation.f2_attribution`、人工testは `tests/management/test_pr_dictionary_f2_attribution.py`、artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_attribution/2026-10-24/`。

## Rectangular Givens PS48の実行前準備（2026-10-24）

`run_pr_rectangular_givens_saved.py` は[候補Aの限定実装](../docs/research/pr_rectangular_givens_preparation_20261024.md)の入口。`--check-only` は保存hash・case identityだけを確認する。`--execute` は別のPS48科学承認・新one-shotを要求する。処理は `trottertracks.pr_dictionary_validation.rectangular*`、mock testは `tests/management/test_pr_rectangular_givens_preparation.py`、planと技術証拠は `artifacts/pr_rectangular_givens_preparation/2026-10-24/`。今回は科学実行なし。

PS48の[科学実行結果](../docs/research/pr_rectangular_givens_ps48_execution_20261024.md)は固定runner `run_pr_rectangular_givens_saved.py` で一回取得した。SOURCE変更・retryなし。保存結果だけの集計utilityは `artifacts/pr_rectangular_givens/2026-10-24/summarize_saved_ps48.py`、CSV／summaryも同所。これは新build／compileや全F2への展開の許可を意味しない。


## 研究C・C4：Q13限定科学検証（2026-10-24）

[実行契約・履歴・結果](../docs/research/research_c_rustiq_execution_20261024.md)。SOURCEは `src/trottertracks/pr_dictionary_validation/rustiq_adapter.py` と `rustiq_execution.py`、入口は `scripts/run_research_c_rustiq_limited.py`、管理testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_limited/2026-10-24/`。保存F1の4case・B0/B1/R1だけを対象とし、旧SOURCE・grant・STOPを保持する。


## 研究C・C4：B2角度結合baseline限定科学比較（2026-10-24）

[結果・固定SOURCE・契約・証拠](../docs/research/research_c_angle_fusion_b2_execution_20261024.md)。既存 `rustiq_adapter.py`／`rustiq_execution.py` と `scripts/run_research_c_rustiq_limited.py --scope C4_B2_SAVED_COMPARISON_V1` を使用。testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/`。全登録gateがPASSし、全8wrapper・全8compiled指標でB2=保存R1。単純融合がこのfixtureの改善を再現した。旧route再実行・新reference・Rustiq/R2・retryは0。独立研究を区切るかはユーザー／研究C／統括の次判断であり未採択、mandatory STOP。
