# trotterlib モジュール索引

`trottertracks.h6_df19_rpe_d1`：[H6係数prior・D1準備](../../docs/research/h6_df19_rpe_prior_execution_preparation_20261110/REPORT.md)。`h4_df10_rpe_d1.coefficient_prior/analyze_bound_point`を再利用し、H6 schema・二つのh候補・固定4点・部分価格表を接続。人工12件、実H6は別許可待ち。

`trottertracks.h6_df19_rpe_connection`：[H6 D1](../../docs/research/h6_df19_rpe_d1_20261110/REPORT.md)。固定binding・外部prior・uniform α・保存cohortの薄い接続。`h4_df10_rpe_connection`の算術、既存geometry/Hoeffding/horizonを再利用。人工25件PASS、実H6数値評価は別許可待ち。

[Track D×元PR/RPE共通interface](../../docs/research/track_d_resource_interface_20261110/REPORT.md)（2026-11-10）：Strong／whole-path Referenceの科学inputs・control価格を分離し、既存総集計へ薄く接続。固定H4/CDF8/LD2/S2/s1/r32/K0/h=.1・元H ground/準備費除外・13round/P0/C1 OFFは旧N/価格/総値の読取専用。人工＋read-only 24 test method PASS、実H4の新N・再集計・回路・fit0。他algorithm統合なし。次候補は同条件r16（未実行）、公開確認後STOP。 実装：`trottertracks.track_d_resource_interface`、読取CLI：`scripts/inspect_track_d_resource_interface.py`、test：`tests/check_track_d_resource_interface.py`。

[Track D run内部再走査削減](../../docs/research/track_d_bridge_run_reuse_20261110/REPORT.md)（2026-11-10）：Linear H4／1Å／STO-3G／Nα=Nβ2・元H／number-CDF8／元順LD2／S2/s1/r32/K0／h=.1、保存q4/cut2・q8/cut4各sample00／C1 OFF、元H ground仮定・準備費除外。run内部再走査削減で旧新の各反復word／位相／全56資源一致、70人工回帰＋保存54case PASS。前回高速版に比べnative段階q4 1.15〜1.18倍・q8 1.32〜1.33倍、q8 RSSは4〜8%増。全unit epochとCX frontier再構築／最悪O(RL)/O(L)は残る。A/F/S不変、新P0／sampling／q16以上0、公開照合後STOP。 実装：`src/trottertracks/track_d_bridge_run_reuse.py`、`scripts/run_track_d_bridge_run_reuse.py`、`tests/check_track_d_bridge_run_reuse.py`。

`trottertracks.track_d_stream_bridge_fast`／`track_d_stream_resources`：[同一rule高速化](../../docs/research/track_d_stream_bridge_fast_20261110/REPORT.md)。旧1Q関数／CX feedを再利用し、exact bit keyの二pass cacheとstream直結、late phase ledgerの接続を追加。旧counter／compiler／phase実装は不変。63人工回帰・保存54case・固定H4 q4/q8 exact差分PASS、一般P0／長qは未資格。

`trottertracks.track_d_native_phase160`：[Phase＋q8結果](../../docs/research/track_d_phase_q8_transfer_20261110/REPORT.md)。旧emitter出力のbinary64 phase項をexact比からMPFR160へ集約するside ledger。局所合成／1Q／CX ruleは無変更、旧binary64値も保存。q4命令列不変・q8全28価格一致、一般P0／長qは未資格。

[Track D H4 native価格移送](../../docs/research/track_d_h4_native_bridge_20261110/REPORT.md)：既存 `track_d_native_emitter`／T0／1Q／CX bridgeをbyte不変で再利用。固定raw4の全28資源が保存P0と一致。新最適化ruleなし、一般P0／長qは未資格。新runnerはI/O・観測・比較、原本非公開・公開照合後STOP。

`trottertracks.track_d_native_emitter`／`track_d_p0_1q_bridge`：[接続と技術結果](../../docs/research/track_d_native_emitter_development_20261109/REPORT.md)。既存emitterのconstructor局所binding、bounded chunk、標準definition lowering、公式ZSXXのstrict-shorter 1Q＋旧CX bridge。47人工回帰・保存54case全指標一致、一般P0_UNQUALIFIED。

`trottertracks.track_d_p0_cx_bridge`：[T1〜H0](../../docs/research/track_d_t1_h0_p0_bridge_20261109/REPORT.md)。既存T0を再利用し、両wireで隣接する同方向CXだけを相殺する。可変frontierを保持、1Q再合成や全P0互換は未実装／未資格。新8＋既存29回帰PASS、保存H4は指定2pathの資源照合だけ。

`trottertracks.track_d_native_resources`：[T0実装](../../docs/research/track_d_t0_native_resources_20261109/REPORT.md)。stdlibのnative stream counter、三指標max-plus、template連結／embedding。既存科学builder/counterは無変更、`NATIVE_EXACT`／`P0_UNQUALIFIED`を区別する。人工回帰29件PASS、新科学実行0。

[O5 C1実H4/q4結果](../../docs/research/track_d_o5_c1_head_tail_h4_q4_20261108/REPORT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/Number-CDF8/元順LD2/S2/s1/r32/K0/δ=.1、保存q4_sample00/cut2・元H ground仮定/準備費除外。全256 raw/新4 post PASS、両方式CX−30、depthはStrong−178/Reference−81、相対depth差縮小97・CX差不変。旧4適格再利用、新P0合計4。χ要約の配列形状エラーSTOPを保全し保存結果集約のみ完了、科学retry0。全law/全RPE未資格、原本非公開・公開照合後STOP。

`trottertracks.track_d_head_tail_artificial`：既存C3/StrongBuilderを用いるopt-inの人工head→tail橋。local_checks/raw_checksは既存checkerを再利用し、分子入力を扱わない。

O5 metadata cohort入口は`scripts/run_track_d_o5_q4_metadata.py`。固定`track_d_cut_selection`/`track_d_cut_h4_saved`とC7-4 `event_features`を再利用し、回路や価格を評価しない。

`trottertracks.track_d_c8_1_completion`：旧raw/P0/修正版postの継承bindingと、既存unfused Aerの完了batch観測。

`trottertracks.track_d_c8_1_nofusion`：公式Aer doubleを再利用しsimulation fusionを無効化。P0/精度/全列を維持。

`trottertracks.track_d_c8_1_diagnostics`：旧checkerの例外前診断保存。

[C8-1の科学結果](../../docs/research/track_d_c8_1_q64_20261108/REPORT.md)：CPU予約修正と19人工/回帰PASSは維持したが、q64 sample00のcompiled CAR-observable検査は両方式X/YでFAIL。失敗値保存の不足も記録し、verifier/閾値の変更・H4再診断は未実施。既存判定・SOURCE・STOPを保持する。

[C8-1 handoff helper](../trottertracks/track_d_c8_1_handoff.py)：完了済みprior hash、drain後の継承・不整合拒否、stage別CPU予約を扱う薄い補助。[PLAN](../../docs/research/track_d_c8_1_q64_20261108/ACCELERATED_COMPLETION_PLAN.md)／[人工tests](../../tests/check_track_d_c8_1_handoff.py)。sampler/builder/compiler/verifierは既存を再利用し、科学条件・閾値を変えない。

[B/C全13-round scalar adapter](../trottertracks/track_d_bc_total.py)：既存B6b guard/aggregate_totalsを再利用し、事前固定A/F/Sの52価格展開と保存共分散の線形伝播だけ。[結果](../../docs/research/track_d_bc_total_20261108/REPORT.md)／[人工テスト](../../tests/check_track_d_bc_total.py)。sampling/circuit/fit/shot solverなし。通常libraryの既定動作を変更しない。

[B6b B/C adapter](../trottertracks/track_d_b6b_bc.py)／[人工検証](../trottertracks/track_d_b6b_toy.py)：B4-2固定handoff/13-round coverageとideal共通N条件、既存aggregate_totalsへのguard付き接続。[報告](../../docs/research/track_d_b6b_bc_integration_20261108/REPORT.md)。会計検証は人工値だけ、H4 totalなし。既存gadget/phase compilerと[tests/check_track_d_b6b.py](../../tests/check_track_d_b6b.py)を使用。通常libraryの既定動作を変えない。

`trottertracks/track_d_state_action_acceleration.py` は既存Aer/CAR検査用のopt-in高速化。cost回路・P0・precisionを変更しない。[C6b限定引継ぎPLAN](../../docs/research/track_d_c6b_q8_20261107/ACCELERATED_COMPLETION_PLAN.md)の新workerだけで選択する。

The opt-in `../trottertracks/track_d_c6b_holdout.py`, `../../scripts/run_track_d_c6b.py` and `../../tests/check_track_d_c6b.py` share the existing canonical/CAR/whole-path/P0 primitives. The [C6b PLAN](../../docs/research/track_d_c6b_q8_20261107/PLAN.md) limits new execution to q8/n32 exploratory paired pricing; import does not execute a scientific run.

Track D C6a：`trottertracks/track_d_c6a_pilot.py` はfresh canonical event接続・paired統計、既存C3/C5 builder/価格検査を共用する。[PLAN](../../docs/research/track_d_c6a_paired_price_20261106/PLAN.md)。旧ライブラリ入口からは起動しない。

[C5結果](../../docs/research/track_d_c5_q2_20261106/RESULT.md)：共通request入口・人工16件・固定q2 raw/4postcheck PASS。strong手動境界融合/位相集約を統一し、P0後7価格は旧C2と一致。有限fixture限定、C6未実施、mandatory STOP。

[C5共通request/有限fixture価格](../trottertracks/track_d_whole_path_price.py)は、C3の共通化した演算を再利用するopt-in入口。method/axis、SOURCE/law/cut、full-wrapper scopeと資格を記録し、未検証q/母平均を誤受理しない。[固定PLAN](../../docs/research/track_d_c5_q2_20261106/PLAN.md)、人工testは`tests/check_track_d_c5.py`。

[C3 q4/q8補完結果](../../docs/research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md)：raw両q・全8 postcheck・最終paired/χ照合PASS。強いstep境界融合を維持して両fixtureでCX/depth減、qubits+8。中断後は保存回路の最後の検証のみ、新compile0。固定fixture限定・mandatory STOP。

Track D C3のmode混在head境界の最小拡張は[`track_d_c3_whole_path.py`](../trottertracks/track_d_c3_whole_path.py)。StrongBuilder・既存emitterを再利用し、[PLAN](../../docs/research/track_d_c3_q4_q8_20261106/PLAN.md)のq4/q8限定fixtureを検証する。人工testは`tests/check_track_d_c3.py`。
[Track D B4-2結果](../../docs/research/track_d_b4_2_rpe_shots_20261106/RESULT.md)：既存元Hadapter・Hoeffding・horizonを変更せず再利用し、保存13-roundの条件付き十分N=14738。library追加変更なし、actual／価格未資格、mandatory STOP。

[元H用RPE統計adapter](../trottertracks/track_d_original_h_accuracy.py)は[Track D B4-1](../../docs/research/track_d_b4_1_rpe_statistics_implementation_20261106/REPORT.md)用。既存accuracyの共通幾何・energy本体とHoeffding／horizon／decoderを再利用。実H4のN算定は未実施。

[C2固定q2結果](../../docs/research/track_d_c2_q2_compile_20261106/RESULT.md)：4 wrapper compile/postcheck PASS。ReferenceはCX4100→3860・depth6096→3014、RZ/qubits増。固定fixture限定、mandatory STOP。

[Track D C2固定q2 compile](../../docs/research/track_d_c2_q2_compile_20261106/PLAN.md)：C1の保存rawを使う2方式×X/Yの4 wrapper限定。P0・cut・phase固定、公開後mandatory STOP。

[B1/C1結果](../../docs/research/track_d_b1_c1_validation_20261106/RESULT.md)：元H向け13-round条件付きoperator十分条件成立、固定q2 raw correctness PASS。compile・新sampling・N評価0、B2/C2未実施、mandatory STOP。

[Track D B1/C1固定検証](../../docs/research/track_d_b1_c1_validation_20261106/PLAN.md)：保存13-roundの元H向けoperator誤差接続と、sample00→01・q2のwhole-path raw correctnessだけ。B2/C2・compile・新samplingは対象外。

[Stage A結果](../../docs/research/track_d_h4_paired32_reference_20261106/RESULT.md)：保存32本すべてでCX/depth減、RZ増、qubits9→17。32 precompile／128 postcheck PASS、有限cohort限定、mandatory STOP。

[保存32 path Reference拡張](../trottertracks/track_d_h4_paired_reference.py) は[Track D Stage A](../../docs/research/track_d_h4_paired32_reference_20261106/PLAN.md)に限定し、既存H4 adapter／check／P0 pipelineを再利用する。通常libraryの既定動作を変更しない。

[実H4 Reference adapter](../trottertracks/track_d_h4_reference.py) は[固定word02検証](../../docs/research/track_d_h4_reference_validation_20261106/PLAN.md)専用。元CのStrongBuilder・event/phase compiler、既存directional diagonalと人工Track D shellを再利用する。共通libraryの既定動作は変更しない。

[track_d_minimal_reference.py](track_d_minimal_reference.py) は[Track D固定人工検証](../../docs/research/track_d_minimal_reference_validation_20261106/PLAN.md)専用adapter。既存DF primitive/event/native controlとphase保持compileを再利用し、分子・sampling・全RPEの実行入口を持たない。

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

[Track D C7-1](../trottertracks/track_d_c7_1_q16.py)：[PLAN](../../docs/research/track_d_c7_1_q16_20261107/PLAN.md)のq16/cut8/n16専用request・価格統計/凍結モデル診断。旧q2/4・q8 guardを保全し、C3/C5/C6a/C6bの科学部品を再利用。明示runnerのみ、全law/長q未資格、mandatory STOP。

[C7-2人工adapter](../trottertracks/track_d_c7_2_toy.py)：実StrongBuilder/C3 emittersの2mode人工束縛、小規模独立exterior reference、max-plus/DAG feature。旧分子guardを変更せず、[固定PLAN](../../docs/research/track_d_c7_2_cx_structure_20261107/PLAN.md)の54人工回路のみ。

[C7-3 q32 adapter](../trottertracks/track_d_c7_3_q32.py)：新Q32Request/seed、旧q16 helperのisolated domain束縛、C7-2 max-plus feature再利用。[固定PLAN](../../docs/research/track_d_c7_3_q32_20261107/PLAN.md)専用で旧guardを変更しない。

[C7-3検証高速化adapter](../trottertracks/track_d_c7_3_acceleration.py)：exact標準gate cache、公式Aer double/unitaryは9qubit以下だけ、Reference17qubitは既存全column statevector。P0/cost QPY/precisionを変更せず、新SOURCE worker内でだけ選択。

## Track D O5 cut接続

`trottertracks.track_d_cut_selection` はC7-4 descriptorで中央±1を選び、C3 `prepare_steps` を呼ぶ人工cohort専用adapter。`track_d_cut_artificial` はC7-2の2-mode fixture/checkerと固定2ケースを接続する。新gadget/solver/compilerはない。[結果と制約](../../docs/research/track_d_cut_artificial_20261108/REPORT.md)。

[O5 q4移送の入力状態](../../docs/research/track_d_cut_h4_q4_20261108/INPUT_AND_REUSE.md)：既存cut選択器/C3/Strong/gadget/compilerのSOURCEを保全。原本不足で科学接続未実行、新adapter/gadget/テスト重複実装0。

## O5 q4保存payload接続

`trottertracks.track_d_cut_h4_saved` は既存archive/recordのlossless hash binding、`track_d_cut_h4_reuse` は中央cutの同request/P0/旧PASS資格を確認する薄いguard。既存selector/C3/Strong/gadget/compilerは変更なし。人工guardは`tests/check_track_d_cut_h4_saved.py`と`check_track_d_cut_h4_reuse.py`。[資料](../../docs/research/track_d_cut_h4_q4_completion_20261108/IMPLEMENTATION_AND_REUSE.md)。
C7-4の[保存モデル再解析](../trottertracks/track_d_c7_4_models.py)と[人工native比較](../trottertracks/track_d_c7_4_stability.py)は明示runnerだけから利用する。q≥64分子計算を含まず、[技術報告](../../docs/research/track_d_c7_4_model_stability_20261107/REPORT.md)を参照する。
