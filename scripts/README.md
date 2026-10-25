# scripts 索引

研究B [`run_b2d_full_df_pf.py`](run_b2d_full_df_pf.py)をSOURCE8811ef089で一回完了。[4 caseの結果・単価・STOP](../docs/research/research_b_full_deterministic_df_pf_execution_20261026.md)。全72 operator／16 expectation PASS後に8単価、scope224一致・中断0。使用済みoutputを再起動せず、追加科学段階へ自動移行しない。以下の準備記録を保全。

研究B [`run_b2d_full_df_pf.py`](run_b2d_full_df_pf.py)の[実行準備・scope・単一command](../docs/research/research_b_full_deterministic_df_pf_readiness_20261026.md)。`--inspect`／`--preflight`は科学処理・出力作成なし。Full S2／Yoshida4の4 case・8 wrapperは別途実行許可後に限り起動しmandatory STOP。人工39件PASS、実候補は未実行。

研究B [`run_b2d_exact_r.py`](run_b2d_exact_r.py)をSOURCE e76c04748で一回完了。[4 caseの結果・単価・STOP](../docs/research/research_b_bnext3_exact_r_execution_20261026.md)。全36 operator／16 expectation PASS後に8費用、scope一致・中断0。使用済みoutputを再起動せず、追加科学段階へ自動移行しない。以下の準備記録を保全する。

研究B [Exact-R4 caseの実行準備](../docs/research/research_b_bnext3_exact_r_execution_readiness_20261026.md)。[`run_b2d_exact_r.py`](run_b2d_exact_r.py)の`--inspect`／`--preflight`は科学処理・出力作成なし。`--execute`は固定SOURCE・fresh outputで一回実行しmandatory STOP。人工30件PASS、実候補は未実行。

研究B [固定24条件shot加重会計：全結果・限界・STOP](../docs/research/research_b_outer_shot_weighted_execution_20261025.md)。SOURCE9bebdc46、48 shot要求・144資源行・24比較を一回完了、scope一致・保存失敗0。科学状態INCONCLUSIVE_QUALIFICATIONS_AND_RARE_COSTS、u_num／γ／rare費用は未取得、無条件G・同精度優位は未認定。旧準備・設計と全原本を保全。

研究B [Shot加重評価：実装・人工検証・実行準備](../docs/research/research_b_outer_shot_weighted_execution_ready_20261025.md)。入口 `scripts/run_b2d_outer_shot_weighted.py`、保存adapter `b2d_evaluator/shot_weighted.py`、人工test `test_b2d_shot_weighted.py`。固定24条件・48 shot要求・144資源行・24比較を登録、人工45件PASS。実候補のnormalization／shot数／Gは未評価、明示的実行許可後のみ一回起動。

研究B [高次数RTEの最小費用感度：固定2比較・保存値解析](../docs/research/research_b_outer_rare_event_minimal_sensitivity_20261025.md)。入口 `scripts/run_b2d_outer_rare_sensitivity.py`、接続 `b2d_evaluator/rare_sensitivity.py`、人工test `test_b2d_rare_sensitivity.py`。8確率／16境界／80 SE cornerを一回完了、人工8件PASS。B4/B3は±2SEでS0差を維持、D1/R4は±1SEで符号未確定。新sampling／build／compileなし、B1のINCONCLUSIVEを維持し、新結果はCONDITIONAL_SENSITIVITY_ONLY・mandatory STOP。

研究B [B1固定IID期待native費用：契約・実装・結果](../docs/research/research_b_outer_b1_expected_native_cost_20261025.md)。新入口 `scripts/run_b2d_outer_b1.py`、接続 `b2d_evaluator/b1_native.py`、人工test `test_b2d_b1_native.py`。全3,072 trajectory・6,144費用要求・18,432 operator／12,288 expectationを完了、compiled最大4.55e-13。32 process、wall50:46.95、actual compile6078／hit66。全eventはorder0で科学状態INCONCLUSIVE_RARE_EVENT_COVERAGE、gamma／shots／総費用は未資格。Retry/resume／追加標本なし、mandatory STOP。以下のB0／A0記録は履歴として保全する。

## 研究B B0固定pilot完了

[`run_b2d_outer_b0.py`](run_b2d_outer_b0.py) をSOURCE b190fe28bで実行し、48代表・全288 operator／192 expectation・96費用測定を完了。[結果・費用・修正・STOP](../docs/research/research_b_outer_b0_recovery_20261025.md)。[`diagnose_b2d_b0_compiler.py`](diagnose_b2d_b0_compiler.py) の `--independent-operator` は保存D1同条件のbalanced complex128比較とQPY保存に使用した。使用済みoutputは再起動せず、MC／shotsへ自動進行しない。以下は履歴。

## 研究B B0復旧・保存回路のcompiler診断

[復旧報告・SOURCE・新attempt・STOP](../docs/research/research_b_outer_b0_recovery_20261025.md)。`run_b2d_outer_b0.py --diagnose-first` は旧B3/+T/代表0/Xのみ、費用なし。`diagnose_b2d_b0_compiler.py` は保存されたB3またはD1/+T/代表0/X回路に限定する技術診断であり、通常の全pilot runnerではない。最終SOURCE5c1fc5bc3でもD1残差は固定tol未達、追加実行しない。

## 研究B B0実行：compiled correctness FAIL・mandatory STOP

[`run_b2d_outer_b0.py`](run_b2d_outer_b0.py) をSOURCE8d0cb96dfで一回実行し、最初のcompiled operator照合で停止した。[実行報告・原本・call数](../docs/research/research_b_outer_b0_execution_20261025.md)。全48 trajectory・96 compile要求は未完了、費用0件。使用済みoutputは再起動せず、原因を確定したことにしない。以下はA0結果と準備履歴。

## 研究B A0実行完了・B0 STOP

[`run_b2d_outer_a0.py`](run_b2d_outer_a0.py) をSOURCE1e3921ba1で一回実行し、8行×90/128桁・3400 DAG node・80 target／signal各要求を完了した。[結果・入力前提・精度の限界・STOP](../docs/research/research_b_outer_a0_execution_20261025.md)。使用済みoutputを再起動しない。B0は未実行・STOPのまま。以下は準備履歴。

## 研究B A0／B0（準備完了・科学実行未実施）

[`run_b2d_outer_a0.py`](run_b2d_outer_a0.py) と [`run_b2d_outer_b0.py`](run_b2d_outer_b0.py) はそれぞれ `--inspect` がmetadataのみ、個別明示許可後だけ `--execute …`。[SOURCE・scope・人工138件・個別command](../docs/research/research_b_outer_a0_b0_execution_preparation_20261025.md)。A0は8行×90/128桁、B0は48代表・96 compile要求。出力は各 `artifacts/research_b_outer_{a0,b0}/` で今回未生成。次はA0実行を先に判断し、B0／MC／shotsへ自動進行しない。

## 研究B outer/RTE固定22行の科学実行完了（2026-10-25）

[`run_b2d_outer_rte22.py`](run_b2d_outer_rte22.py) をSOURCE26e8734afで一回実行し、全correctnessと22行／1239 dispatch／274 keyを完了した。[結果・未確定事項・STOP](../docs/research/research_b_outer_rte_22row_execution_20261025.md)。原本は `artifacts/research_b_outer_rte22/20261025/analytic-once/`。Retry/resume・追加条件・native／shot段階へ自動進行しない。以下は準備履歴。

## 研究B outer/RTE固定22行（準備完了・科学計算未実行）

[`run_b2d_outer_rte22.py`](run_b2d_outer_rte22.py) は `--inspect` でmetadataのみ表示する。別途明示許可後だけ `--execute --source-commit … --output …`。[固定22行・SOURCE 26e8734af・単一command・人工119件](../docs/research/research_b_outer_rte_22row_execution_ready_20261025.md)。候補別r/K、1239 dispatch／274 key／RTE各96／110 expectations、専用output `artifacts/research_b_outer_rte22/`。Fresh作成・correctness先行・失敗保存・retry/resume禁止・STOPを既存launcherで維持する。

## 研究B outer Stage 1固定16行（準備完了・科学実行未承認）

[`run_b2d_outer_stage1.py`](run_b2d_outer_stage1.py) は `--inspect` でmetadataだけを表示する。科学実行はユーザーの明示許可後に `--execute --source-commit … --output …`。Source・固定条件・scope・fresh outputを検査し、既存Input package／保存／STOPを再利用する。[契約・単一command・人工105件](../docs/research/research_b_outer_splitting_stage1_execution_ready_20261025.md)。Source候補b85dd3284、16行・906 dispatch・490 key・80 expectation、科学処理未起動。

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

## B2-D Input package・preflight準備（2026-10-25）

この準備はCLI science runnerを追加しない。呼出し経路 `trottertracks.b2d_evaluator.input_package.load_candidate`／read-only `preflight.verify_execution`、人工test `tests/management/test_b2d_input_preflight.py` と[技術報告](../docs/research/research_b_b2d_input_preflight_technical_report_20261025.md)を一組として読む。予定outputは `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/`、今回は未作成。全scope・承認binding・失敗保存launcherは実行前の残事項。

## B2-D Stage 1 launcher準備（2026-10-25）

`run_b2d_stage1.py --inspect` は固定recipe／scopeのmetadataのみを表示する。`--execute` は現SOURCEではproduction authority未接続のためclaim前に拒否。[最終提案](../docs/research/research_b_b2d_stage1_final_execution_proposal_20261025.md)・[技術報告](../docs/research/research_b_b2d_stage1_final_preparation_technical_report_20261025.md)、`src/trottertracks/b2d_evaluator/{scope,launcher,errors}.py`、`tests/management/test_b2d_stage1_launcher.py` を一組として読む。予定outputは `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/`、今回未作成。実行承認・grant発行／claimはない。

## B2-D Stage 1一括準備完了（2026-10-25）

`prepare_b2d_stage1.py` は固定SOURCE・実環境から未承認packetを作るmetadata入口。`issue_b2d_stage1_approval.py` は**実承認後にユーザー／管理者自身が行う正式発行操作**で、今回未実行。`run_b2d_stage1.py` は管理者所有receiptと既存packet／markerを照合した後だけscienceへ進む。[契約・具体command](../docs/research/b2d_stage1_execution_ready_20261025/CONTRACT.md)・[技術報告](../docs/research/b2d_stage1_execution_ready_20261025/TECHNICAL_REPORT.md)、`tests/management/test_b2d_execution_ready.py` を参照。科学SOURCE候補348e08、74人工test PASS、科学計算・正式grantなし。予定artifactは `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/`、今回未生成。


## B2-D Stage 1通常ユーザー承認方式（2026-10-25）

最新の[未承認契約・通常権限の具体command](../docs/research/b2d_stage1_user_approval_20261025/CONTRACT.md)と[技術報告](../docs/research/b2d_stage1_user_approval_20261025/TECHNICAL_REPORT.md)。新科学SOURCE `91f45aeb50275bb6d22050a304d0a9b1ff025d70`、通常ユーザーのprivate手動記録へ限定修正し、root／外部APIを要求しない。同じOSアカウントの本人識別は運用規則であり独立認証ではない。入口 `scripts/{prepare_b2d_stage1,issue_b2d_stage1_approval,run_b2d_stage1}.py`、実装 `src/trottertracks/b2d_evaluator/`、人工test `tests/management/test_b2d_execution_ready.py`。既存74件を適合更新＋追加6件＝80件PASS。科学条件・scope・環境は不変、正式grant／claim／科学実行なし。予定artifactは固定checkout内 `artifacts/research_b_b2d_stage1/20261025/user-approval-once/`、今回未生成。旧root方式は上記過去段階の履歴として保全する。


## B2-D Stage 1承認管理撤去・実行簡素化（2026-10-25）

最新の[科学実行契約・単一command・技術報告](../docs/research/b2d_stage1_simple_execution_20261025.md)。新科学SOURCE `54008c697e686a42ec48941d25219d523bafd454`。入口は `scripts/run_b2d_stage1.py --execute --source-commit … --output …`。承認packet／grant／receipt・発行／準備scriptを撤去し、科学SOURCE・input identity・scope・correctness・fresh output・失敗保存・STOPを維持した。実装 `src/trottertracks/b2d_evaluator/`、人工test `tests/management/test_b2d_simple_execution.py` と既存5ファイル、計70件PASS。科学計算は未実行。予定artifact `artifacts/research_b_b2d_stage1/20261025/simple-once/` は未生成。旧方式は過去の未採択提案として保全し、今後のB2-D実行要件にはしない。


## B2-D Stage 1実行結果と保存値集計（2026-10-25）

[結果・2attemptの技術履歴・STOP](../docs/research/b2d_stage1_execution_20261025.md)。既存 `run_b2d_stage1.py` の科学実行は完了し、追加起動しない。`artifacts/research_b_b2d_stage1/20261025/summarize_saved.py --output-dir <fresh-dir>` は保存JSONだけから36行CSVと要約を作るstdlib補助で、新しいoperator／signal／科学backendを実行しない。
