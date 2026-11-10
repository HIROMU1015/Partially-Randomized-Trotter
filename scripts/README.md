# scripts 索引

`plan_pr_r_dependent_n.py`：[R依存Nのmetadata専用入口](../docs/research/pr_r_dependent_n_model_20261111/REPORT.md)。system/round-R mapから既存taskを生成し、科学計算を起動しない。承認後のB接続は`pr_r_dependent_n.RBatch`を使う。

`run_h4_df10_rpe_batch.py`の[全13round N完了](../docs/research/h4_df10_rpe_batch_results_20261110/REPORT.md)：固定24候補を一回評価し旧m3を保持。`verify_h4_df10_rpe_shot_integers.py`は同じ保存ε/αの巨大N整数化を検証し、上方向だけ補正した受渡しrecordを別保存。signal再計算・C/G0、再実行しない。

`run_h4_df10_rpe_batch.py`：[H4 N一括入口](../docs/research/h4_df10_rpe_batch_preparation_20261110/REPORT.md)。既定はmetadata計画、別途許可後の`--execute`で不足24候補をN専用評価する。保存m3を再計算せず、C/Gを呼ばない。

[H4価格providerのB接続](../docs/research/h4_price_pipeline_integration_20261110/REPORT.md)：科学runnerは追加せず、Bのh4_saved_price_providerをrun_missingのC callbackへ渡す。lookupのみ、null/資格不足の自動再呼出しなし。

`run_h6_df19_rpe_m8_12.py`の[完了結果](../docs/research/h6_df19_rpe_m8_12_results_20261110/REPORT.md)：SOURCE fca50d36fの一括10候補全ideal条件付きPASS。保存m0..7を再計算せず26候補/13roundのNを取得。価格/G/最適化へ自動進行せず、再実行しない。

`run_h6_df19_rpe_m8_12.py`：[残5roundのN専用一括準備](../docs/research/h6_df19_rpe_m8_12_preparation_20261110/REPORT.md)。既存`run_missing/H6ActionBatch`を1回呼び、保存m0..7を再計算せず13round coverageへ接続する。追加人工7件PASS。実行は明示許可後だけ、C/G・retry/resumeなし。

`plan_pf_pr_rpe_cost.py`：[統合REPORT](../docs/research/pf_pr_rpe_cost_integration_20261110/REPORT.md)。固定Git JSONからH4/H6 schedule・candidate・保存N/C・不足batchを静的に出力する。実signal/N/価格予測/G実行modeなし。数値入口は`pf_pr_rpe_pipeline.run_fixed_evaluation`で、実データ使用には別途科学scopeが必要。

`run_h6_df19_rpe_m7.py`の[完了結果](../docs/research/h6_df19_rpe_m7_results_20261110/REPORT.md)：SOURCE dcd6a7c87の固定2候補を一回評価、ideal条件付きNは各軸PF705/PR846。旧roundは再計算せず保存値へ接続。m8以降・価格へ自動進行しない。

`run_h6_df19_rpe_m7.py`：[m7準備](../docs/research/h6_df19_rpe_m7_preparation_20261110/REPORT.md)。T12.8/qPF8の固定PF/PR各1候補だけを旧D2 actionへ接続する。`check_h6_df19_rpe_m7.py`の人工3件PASS、旧m6/D2 guard不変。実H6は明示許可後にfresh outputで実行し、m8以降・価格へ進まない。

`run_h6_df19_rpe_m6.py`の[完了結果](../docs/research/h6_df19_rpe_m6_results_20261110/REPORT.md)：SOURCE c9aa89548で固定2候補を一回評価し両方ideal条件付きPASS、PF/PR各軸N705/772。旧結果とrun01を保全、m7以降・価格へ自動進行しない。

`run_h6_df19_rpe_m6.py`：[m6準備](../docs/research/h6_df19_rpe_m6_preparation_20261110/REPORT.md)。既存D2のaction/geometryを変更せず、T6.4・PF/PR各1候補だけへ接続する。`check_h6_df19_rpe_m6.py`の人工3件PASS。実H6は別途明示許可後にfresh outputで一回実行し、価格・m7以降へ進まない。

`run_h6_df19_rpe_d2.py`の[完了結果](../docs/research/h6_df19_rpe_d2_results_20261110/REPORT.md)：SOURCE e0503b769、固定8候補を一回評価し全てideal条件付きPASS。run01と旧D1/計画を保全、m6以降・価格・全Gへ自動進行しない。

`run_h6_df19_rpe_d2.py`：[固定D2計画](../docs/research/h6_df19_rpe_d2_preparation_20261110/REPORT.md)の4round・8候補だけを実行する入口。保存H/stateとtail定数、T=.4 reference、D1政策を再利用し、対角化なしの行列指数/Taylor作用と既存D1 geometryでNを評価する。科学実行は利用者の明示許可範囲に限り、fresh outputを指定する。sampling/build/compile/fit/Gなし。人工7件の記録は[こちら](../docs/research/h6_df19_rpe_d2_results_20261110/synthetic_tests.txt)。

`run_h6_df19_rpe_d1.py`：[係数prior・D1一括計画](../docs/research/h6_df19_rpe_prior_execution_preparation_20261110/REPORT.md)。保存係数prior→h=.8/.1→固定4点N→条件付き部分費用の明示入口。今回実行したのは`--help`と人工testのみ。実H6入力を渡す実行は別途科学許可が必要。

[H6 RPE D1](../docs/research/h6_df19_rpe_d1_20261110/REPORT.md)：今回追加したのはAPIと人工test。`python -B tests/check_h4_df10_rpe_connection.py -v`／`python -B tests/check_h6_df19_rpe_connection.py -v`で人工25件。実H6の数値driverは起動していない。[別許可後の計画](../docs/research/h6_df19_rpe_d1_20261110/D1_PLAN.md)。

[Track D×元PR/RPE共通interface](../docs/research/track_d_resource_interface_20261110/REPORT.md)（2026-11-10）：Strong／whole-path Referenceの科学inputs・control価格を分離し、既存総集計へ薄く接続。固定H4/CDF8/LD2/S2/s1/r32/K0/h=.1・元H ground/準備費除外・13round/P0/C1 OFFは旧N/価格/総値の読取専用。人工＋read-only 24 test method PASS、実H4の新N・再集計・回路・fit0。他algorithm統合なし。次候補は同条件r16（未実行）、公開確認後STOP。 実装：`trottertracks.track_d_resource_interface`、読取CLI：`scripts/inspect_track_d_resource_interface.py`、test：`tests/check_track_d_resource_interface.py`。

[Track D run内部再走査削減](../docs/research/track_d_bridge_run_reuse_20261110/REPORT.md)（2026-11-10）：Linear H4／1Å／STO-3G／Nα=Nβ2・元H／number-CDF8／元順LD2／S2/s1/r32/K0／h=.1、保存q4/cut2・q8/cut4各sample00／C1 OFF、元H ground仮定・準備費除外。run内部再走査削減で旧新の各反復word／位相／全56資源一致、70人工回帰＋保存54case PASS。前回高速版に比べnative段階q4 1.15〜1.18倍・q8 1.32〜1.33倍、q8 RSSは4〜8%増。全unit epochとCX frontier再構築／最悪O(RL)/O(L)は残る。A/F/S不変、新P0／sampling／q16以上0、公開照合後STOP。 実装：`src/trottertracks/track_d_bridge_run_reuse.py`、`scripts/run_track_d_bridge_run_reuse.py`、`tests/check_track_d_bridge_run_reuse.py`。

[Track D stream bridge性能結果](../docs/research/track_d_stream_bridge_fast_20261110/REPORT.md)：`run_track_d_stream_bridge_fast.py`の固定入力準備・人工54比較、旧keyword不備を保全した`complete_track_d_stream_bridge_benchmark.py`、同時主比較`measure_track_d_stream_bridge_paired.py`。最初のfrozen runnerのbenchmark keyword不備は履歴として保持し、完成済みCLIを再開しない。新しい実行ではmodule APIと互換adapterをscopeに合わせて再利用する。原本／run01／run02／STOP保持、新P0=0。

`run_track_d_phase_q8_transfer.py`：[Phase＋q8結果](../docs/research/track_d_phase_q8_transfer_20261110/REPORT.md)。selectで価格非依存にq8先頭IDを固定、executeでq4 phase4＋q8 raw予測4、予測commit後のみcompare。旧q4観測／counterを再利用し、新P0は呼ばない。run01の初回preflight失敗／修正／結果／STOPを保持し、追加pathやq16以上へ自動進行しない。

`run_track_d_h4_native_bridge.py`：[保存H4 q4結果](../docs/research/track_d_h4_native_bridge_20261110/REPORT.md)。predictはraw4のみ、比較前にcommitで固定し、compareが既存compiled4と照合。`check_track_d_h4_native_phase.py`は固定予測word／compiledのread-only順序・角・phase診断だけ。run01／旧STOPを保全し、再開／追加pathへ自動進行しない。

`run_track_d_native_emitter_development.py`：[技術結果](../docs/research/track_d_native_emitter_development_20261109/REPORT.md)。固定54保存人工raw QPYをhash照合し、最大5qubitのnative lowering／部分bridge／位相付き作用／保存価格を確認した完了済みrunner。新P0・H4・sampling・fit/N0。run01／STOPを保全し、追加実行へ自動進行しない。

`run_track_d_t1_h0.py`：[固定PLAN](../docs/research/track_d_t1_h0_p0_bridge_20261109/PLAN.md)の保存54人工case、指定3人工P0、保存H4 q4_sample00/q64_sample00だけを扱う完了済み一括runner。新sampling/H4 compile/H4 simulationは行わず、原本はread-only。run01/STOPを保持し、再開・追加実行しない。[結果](../docs/research/track_d_t1_h0_p0_bridge_20261109/REPORT.md)。

[Track D T0](../docs/research/track_d_t0_native_resources_20261109/REPORT.md)：新しい科学runnerは追加せず、`python3 -B tests/check_track_d_native_resources.py NativeTests`と既存環境の`python -B tests/check_track_d_native_resources.py QiskitNativeTests`で人工計数／readerを検査する。新P0・H4・simulationなし。既存設計mockも回帰実行し合計29件PASS。

[O5 C1実H4/q4結果](../docs/research/track_d_o5_c1_head_tail_h4_q4_20261108/REPORT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/Number-CDF8/元順LD2/S2/s1/r32/K0/δ=.1、保存q4_sample00/cut2・元H ground仮定/準備費除外。全256 raw/新4 post PASS、両方式CX−30、depthはStrong−178/Reference−81、相対depth差縮小97・CX差不変。旧4適格再利用、新P0合計4。χ要約の配列形状エラーSTOPを保全し保存結果集約のみ完了、科学retry0。全law/全RPE未資格、原本非公開・公開照合後STOP。

`run_track_d_head_tail_artificial.py`：固定q2_mixedの一回限定C1比較。局所/全入力PASS後だけP0、旧C7-2 baseline適格再利用、公開後STOP。

`run_track_d_o5_q4_metadata.py`：保存C6a q4全16本のevent-only読取、固定selector/featureへの薄い接続と集計。

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
[run_track_d_b4_2.py](run_track_d_b4_2.py)は[B4-2固定PLAN](../docs/research/track_d_b4_2_rpe_shots_20261106/PLAN.md)の保存B1→元H ideal統計専用。明示`--execute`、fresh run01、retry/resume0、mandatory STOP。一般testや旧runnerから自動実行しない。

[Track D B4-1](../docs/research/track_d_b4_1_rpe_statistics_implementation_20261106/REPORT.md)は科学runnerを追加しない。人工検証入口は `PYTHONPATH=src python -B -m pytest -p no:cacheprovider -q tests/check_track_d_original_h_accuracy.py`。実H4評価は別許可。

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

## Track D O5 cut：人工専用入口

`run_track_d_cut_artificial.py --output <新directory>`：事前固定2ケースのdescriptor選択→既存builder/checker→P0 full wrapper→STOP。旧H4 loaderやsamplingは呼ばない。実行済み結果は[報告](../docs/research/track_d_cut_artificial_20261108/REPORT.md)。

[O5 q4移送preflight STOP](../docs/research/track_d_cut_h4_q4_20261108/REPORT.md)：C6a保存event/QPY原本が未取得のため、既存runner再実行・新runner追加は行っていない。原本回収が先。

## O5 q4保存原本の限定接続

`run_track_d_cut_h4_select.py`：hash一致したq4_sample00だけから不変ruleで一回decisionを価格参照前に保存。`run_track_d_cut_h4_reuse.py`：中央cut選択時だけ、旧QPY/PASS/全7価格を照合して新compileなしでSTOP。[完了済み結果](../docs/research/track_d_cut_h4_q4_completion_20261108/REPORT.md)。再実行しない。
[C7-4保存再解析runner](run_track_d_c7_4.py)／[結果projection](summarize_track_d_c7_4.py)：保存88 pathのみの探索fit/CV/bootstrapと固定人工6系列/短P0 2本。新H4 sampling/build/compileなし。既存runのmain/guardを変更しない。
