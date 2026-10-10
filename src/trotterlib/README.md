> 2026-10-07 Track B RA-D0 v3：**READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION**（実行承認ではない）。
> [source review](../../docs/tracks/algorithm_codesign/ra_d0_source_review_v3_20261007.md) / [GPT handoff](../../docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_v3_20261007.md) / [manifest](../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/evidence_manifest_v3.json)。
> [focused verifier](../../scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v3.py) / [v3 tests](../../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v3.py)。
> exact-certified B2 minimum infeasibilityを正常outcomeに修正。pointをfreezeへ記録しbudget/queryを空にして次nへ進む。
> uncertified failureはtechnical STOP。数値・candidate・grid・call/resource capは維持。登録最適化0、authorizationなし、mandatory STOP。

> 2026-10-07 Track B RA-D0 v2：**READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW**。
> [source review](../../docs/tracks/algorithm_codesign/ra_d0_source_review_v2_20261007.md) / [GPT handoff](../../docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_20261007.md) / [evidence manifest](../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/evidence_manifest_v2.json)。
> [future runner](../../scripts/tracks/algorithm_codesign/run_ra_d0_one_shot.py) / [focused verifier](../../scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v2.py) / [v2 tests](../../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v2.py)。
> B0_saved/ideal分離、数値B1⊂B2⊂B3、profile-paired budget、batch freeze-before-B3、
> anchor-first、main LP 55,275 / auxiliary込み110,550、resource/launch guardを固定。
> 登録最適化・実budget/minimum/witness取得0、authorizationなし。旧本文・旧STOP・Track Aは保持。mandatory STOP。

## Track B RA-D0 static preparation / mandatory STOP（2026-10-06）

[source review](../../docs/tracks/algorithm_codesign/ra_d0_source_preparation_review_v1.md)：21 columns/x、18 sign pairs一致、35 focused tests PASS。
B専用namespace `src/trottertracks/algorithm_codesign/ra_d0/`、static generator／verifier、
`artifacts/track_b_ra_d0_preparation/2026-10-06/`に候補・grid・semantic audit・provenanceを保存。
ideal nestingと数値membershipを区別し、query実行scope／資源上限をGPTへ返す。
登録最適化／新合成／新science0、RUN_READY=false、既存分類・Track A・旧STOP保持。
**mandatory STOP。以下の既存本文を全文保持する。**

## Track B RA-RTE統合数学監査・mandatory STOP（2026-10-06）

R1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942` 基点、GPT設計案へのDOCS_SYMBOLIC_ONLY_MATHEMATICAL_AUDIT。
[命題別監査](../../docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_v1.md)と[GPT handoff](../../docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_gpt_handoff_20261006.md)：一block／finite table／canonicalのfixed-n LPは仮定付きで成立。
shot-gridの固定total cap保存には反例。log/root・sampler認証、Delta=0、peak workspaceの規約を実行前修正へ返す。
[stdlib人工bookkeeping](../../scripts/tracks/algorithm_codesign/check_ra_rte_mathematical_bookkeeping.py)の[50 checks](../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_checks_v1.json)を一般証明と分離した。
[manifest](../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/evidence_manifest_v1.json)、[dated note](../../docs/research/研究ノート/2026-10-06_track_b_ra_rte_mathematical_audit.md)。science/synthesis/solver/資源再採点0、共通API変更0。
既存科学分類・結果・STOPは不変。性能・新規性・algorithm採択・次実装／R2 authorizationは未確定。
**mandatory STOP。次の採択・実装・pilotの必要性／範囲はGPT判断。以下の既存本文を全文保持する。**

## Track B R1.5保存値帰属・mandatory STOP（2026-10-06）

input R1 commit `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b` の保存値だけを用いたPOSTHOC attribution / design input。
[帰属報告](../../docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md)と[GPT handoff](../../docs/tracks/algorithm_codesign/r1p5_gpt_handoff_20261006.md)：新science/synthesis/compile/候補追加0、R1科学分類は不変。
primaryは2-qubit finite P₃、distinct-basis controlled、x={1/8,1/4}、登録native三precision。
Aは登録(G_T,G_CX,G_1Q) frontにx=1/8の1e-4、x=1/4の1e-3/1e-4で残る。
normalizationだけでなくnative費用とbias/shotの関係を整理し、固定合成列への依存も保存した。
[stdlib保存値解析](../../scripts/tracks/algorithm_codesign/analyze_r1p5_saved_attribution.py)、[全summary](../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)、[provenance manifest](../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/evidence_manifest_v1.json)、
[日付note](../../docs/research/研究ノート/2026-10-06_track_b_r1p5_saved_attribution.md)。共通library変更・独立validation・新algorithm採択はない。
限定診断SUPPORTS_RA_RTE_DESIGNは設計入力のみ。eta探索/R2/DF接続/追加scienceは未認可。
**mandatory STOP。次の数学設計・研究方針判断はGPT側。以下の既存本文を全文保持する。**

## Track B R1一回結果・mandatory STOP（2026-10-06）

固定S `d43d64a821a0249a0dfab12a2472bd3a72fdee74` →直接子authorization-only A
`411f08f768244fe87b600d82308c3851847fe9e4`からrun1/retry0。
[結果照合](../../docs/tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md)：126 keys / 264 rows / 132 controlled tasks完了、全task適格。
terminal R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW、[保存field専用監査](../../scripts/tracks/algorithm_codesign/audit_r1_saved_result.py) PASS。
primary distinct-basis controlledではB²改善とnative/shot資源のtrade-offを保存し、自動研究GOはない。
[全264 rows CSV](../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)、[evidence manifest](../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)、
[GPT判断への入口](../../docs/tracks/algorithm_codesign/r1_post_run_gpt_review_request_20261006.md)。原result/marker/source、既存証拠・共通API・Track A保持。
science終了後mandatory STOP、追加合成/target/grid/分子/DF/trajectory/GPUは行わない。
研究方針・RQ・新規性・着地点・追加検証の必要性/範囲はGPT側。
以下のpending/最新記述は当時の履歴として本文をそのまま保持する。

# trotterlib モジュール索引

## Track B R1 builder is a separate namespace

[R1専用実装](../trottertracks/algorithm_codesign/rte_reallocation/)を新設し、共有library/既存even RTE invariantを変更しない。
[preregistration/source review](../../docs/tracks/algorithm_codesign/rte_reallocation_r1_source_review_request_20261006.md)。
27 focused synthetic semantic/accounting/launch tests。DF controlled wrapperのvalidationではない。
登録synthesis/resource取得0、science未認可、mandatory STOP。


## Track B R0.5: shared API unchanged

[R0.5監査](../../docs/tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md)と
[GPT handoff](../../docs/tracks/algorithm_codesign/rte_reallocation_r05_gpt_handoff_20261006.md)を追加。
新checkerはdocs/symbolic技術監査専用。library/共通API/既存RTE実装は変更しない。
odd eventのnative/controlled実装は未検証。R1未認可、mandatory STOP。


## Track B R0 boundary (2026-10-06)

[Finite-mean reallocation audit](../../docs/tracks/algorithm_codesign/rte_reallocation_r0_review_packet_20261006.md) uses a standalone exact checker. Shared RTE/DF APIs are unchanged; RTEEvent remains even-order only. Odd-event algebra is not a validated builder. No track-specific copy of trotterlib is created.


Track B [BS-0.5 ordinary形式仕様](../../docs/tracks/algorithm_codesign/bs05_ordinary_finite_rte_baseline_v1.md)はdocs-only。
固定rte.pyをtext照合しただけで、共有implementation/APIを変更・実行していない。
operator/channelは別target、現candidateの独立delta未定義。実装/science/tests0、mandatory STOP。以下は履歴。

Track B [SP-1後block合成の数学/API仕様](../../docs/tracks/algorithm_codesign/block_synthesis_design_review_20261006.md)はdocs-only。
channel用Gateとoperator LCUを別型にする設計案。新namespace/コードは未作成、共有library/API変更なし。
current rte.pyは固定git blobのtext参照だけで呼び出さない。実装/science/tests0、mandatory STOP。以下は既存履歴。

Track B [SP-1一回結果・保存値監査](../../docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)は
固定`../trottertracks/algorithm_codesign/synthesis_placement/`の実行記録。
48 rows／96 axesを完了、mandatory STOP、retry0。共有trotterlib実装/APIは変更していない。
[保存field audit](../../scripts/tracks/algorithm_codesign/audit_sp1_saved_result.py)はscience moduleをimportせず照合する。
actual DF/RTE wrapperやcompiled費用の検証ではない。次の研究判断はGPT側。
以下のsource準備・未実行記述は結果前履歴として保持する。

Track B [SP-1結果前契約](../../docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)の
sequence/adapter/accounting/launch/result modulesは`../trottertracks/algorithm_codesign/synthesis_placement/`。
共有trotterlibをimport/変更せず、保存B primitiveを入力にしたsynthetic adapterと59 focused tests。
登録science sweep0、実行authorization pending。共通APIの採択やDF wrapper検証ではない。
以下の未完成記述は前段履歴として保持する。

Track B [SP-1準備](../../docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_proposal_v1.md)の
[wrapper_accounting.py](../trottertracks/algorithm_codesign/synthesis_placement/wrapper_accounting.py)は
共通libraryから独立したexact Fraction会計fixture kernel。17 focused testsはlocal pass。
共有API変更なし、science入力／合成呼出し0。actual wrapper adapter／interval guard／science runnerは未完成。
以下は元のSP-0.5準備履歴。現在のSP-0.5結果は別の固定result commitを参照する。

Track Bの[SP-0.5経済性gate](../../docs/tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md)は
`../trottertracks/algorithm_codesign/synthesis_placement/`の新namespaceに置く。
gate-string合成／interval guard／T×weight会計だけを扱い、共有library実装・DF wrapperをimportしない。
34 focused technical tests pass、登録target計測は未実行・未認可。


Track Bの[BM-0.5記号監査](../../docs/tracks/algorithm_codesign/bm05_review_packet_20261005.md)は
共有libraryをimportせず、[専用script](../../scripts/tracks/algorithm_codesign/audit_bm05_symbolic_equivalence.py)の
抽象words／Fractionだけで実施する。library module・API・science runnerの変更なし、BM-1未認可。

Track Bのexperimental codeは共有APIを変更せず、`../trottertracks/algorithm_codesign/`へ置く。
[BF-1 preparation](../../docs/tracks/algorithm_codesign/README.md)のCPU mean adapterであり、
DF controlled wrapperの検証・scientific run・algorithm採択は未実施。

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

## Track B G1限定技術sourceへの参照

G1のexact symbolic kernel・audit・controllerは
[専用script namespace](../../scripts/tracks/algorithm_codesign/g1_decision_packet/README.md)へ置いた。
共有library/APIを変更する機能ではない。
[source review](../../docs/tracks/algorithm_codesign/g1_source_review_20261009.md)、
[入口](../../scripts/tracks/algorithm_codesign/run_g1_decision_packet.py)、
[focused tests](../../tests/tracks/algorithm_codesign/test_g1_source_preparation.py)、
[準備artifact](../../artifacts/track_b_g1_source_preparation/2026-10-09/evidence_manifest_v1.json)を参照する。
本構造監査・固定8人工LPは別明示指示まで未実行。科学的B2/B3比較、backend採用は認可しない。

## Track B G2 evidence entry (2026-10-09)

[Saved-value diagnosis](../../docs/tracks/algorithm_codesign/g2_saved_diagnostic_handoff_20261009.md)
is isolated in `scripts/tracks/algorithm_codesign/g2_saved_diagnostic.py`; no shared library API was changed.
The saved-table diagnostic is complete and STOPped; it is not a new registered RA-D0 certificate.


## Track B G3への参照（2026-10-09）

共有library APIは変更していない。Track B限定scripts
`../../scripts/tracks/algorithm_codesign/g3_finite_law.py`、`g3_return_comparator.py`と
[結果/handoff](../../docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md)、
[manifest](../../artifacts/track_b_g3_finite_law/2026-10-09/evidence_manifest_v1.json)から辿る。
旧R1 native/numeric/accountingをidentity固定して再利用。288 finite profilesと固定return12合成を完了、mandatory STOP。


## Track B G4との境界

[G4 handoff](../../docs/tracks/algorithm_codesign/g4_results_and_gpt_handoff_20261009.md)。
独立certificateとCTS specializationは`scripts/tracks/algorithm_codesign/g4_*.py`。
CTS native/numeric/accountingは既存`src/trottertracks/algorithm_codesign/rte_reallocation/`をreadonly共用。
共有`src/trotterlib`実装/APIに変更なし。両one-shot完了・mandatory STOP。


## 2026-10-10 Track B G5：shared APIは変更しない

[G5保存値終了認証](../../docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md)と
[static access inventory](../../docs/tracks/algorithm_codesign/g5_static_access_inventory_20261010.md)。
Track B verifierはstdlibのみ、shared libraryのimport・変更・copyなし。
既存even paired DF wrapperの存在をodd/complement/reallocationの検証済み能力として転用しない。
[runner](../../scripts/tracks/algorithm_codesign/g5_fixed_dictionary_closure.py)、
[tests](../../tests/tracks/algorithm_codesign/test_g5_fixed_dictionary_closure.py)、
[manifest](../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/evidence_manifest_v1.json)。mandatory STOP。


## 2026-10-10 Track B G6との境界

shared library/API実装は変更しない。[新局所formal prototype](../trottertracks/algorithm_codesign/return_aggregation.py)はTrack B namespace。
[一般証明・access・停止記録](../../docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md)を参照。
controlled native実装、Hamiltonian、PR/QPE接続の検証ではない。


## Track B G7（2026-10-10）

共有API変更なし。実験実装は別namespaceの
[g7_generator.py](../trottertracks/algorithm_codesign/g7_generator.py)、
[g7_provider.py](../trottertracks/algorithm_codesign/g7_provider.py)、
[g7_reference.py](../trottertracks/algorithm_codesign/g7_reference.py)、
[g7_launch.py](../trottertracks/algorithm_codesign/g7_launch.py)。
productionと小support referenceを分離。conditional controlled-Q前提、一般分子実装ではない。


## 2026-10-10 Track B G7：取得完了・mandatory STOP（最新追記）

[G7結果/GPT引継ぎ](../../docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md)。
`G7_LIMITED_IMPLEMENTATION_ECONOMICS_COMPLETE_AWAITING_GPT_REVIEW`。source ab2549f41b3546fb3940342a2162dd9ee93699c4、24 keys/8 rows、strict error PASS、retry0。
固定P5では登録3対照後にもconditional期待T減少、P3ではclosed-form対照が小さい。
hard shot cap・classical generation/angle acquisition・未指定provider costを併記。
新規性/主method/次stage未採択、G5閉鎖/G6原証拠とmarker保持、GPT判断へ戻す。


## Track B G8（2026-10-10、source preparation）

[G8 proof/contract](../../docs/tracks/algorithm_codesign/g8_proof_contract_and_on_demand_scope_20261010.md)：採用GPT G7 review §11に基づく限定確認。
2 known development inputs/4 same production laws、finite-provider parameter、分離failure配分、
on-demand strict Rzと対称bounded cache。G7のstatus/point comparison/consumed markerを保持。
17 off-domain focused tests、source preparation時点のnew native acquisition0。一束後mandatory STOP。
共有API変更なし。実験namespaceは `src/trottertracks/algorithm_codesign/g8_*.py`。


- Track B G8 evidence: [conditional on-demand result](../../docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md). Library code unchanged; experimental modules remain in `trottertracks.algorithm_codesign`.


## Track B G9 source preparation（2026-10-10）

[Fixed proof/scope](../../docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md)：GPT G8 review §14を採用。known P5/指定3-qubit provider、6 direct+5 helper診断、19 keys/新CTS1 key、23 focused tests。source固定後一束のみ、終了後mandatory STOP。

## G9 one-shot STOP（2026-10-10）

`G9_TECHNICAL_INCONCLUSIVE`：epsilon引数のFraction→mp.mpf変換で停止、native比較0 row。
新規helper attempts1 / 新規sequence取得0 / retry0。source・contract・過去941pathは不変。
23 focused testsと23 saved-output checksは準備/整合証拠で、native資源の科学結果ではない。
[G9 failure・GPT handoff](../../docs/tracks/algorithm_codesign/g9_results_and_gpt_handoff_20261010.md)。mandatory STOP、次の研究判断はGPTへ返す。

## G9 v2 API-boundary source preparation（2026-10-10）

[G9 v2準備/入口](../../docs/tracks/algorithm_codesign/g9_v2_api_boundary_source_review_20261010.md)。精度値の型接続を最小修正、19 stub-only/launch tests PASS。
46科学条件・同19-key inventory・旧v1 source/result/markerは保持。
新実合成・登録matrix/予算/科学実行0、v2 marker absent、別authorization pending。
独立branchで資料公開後STOPし、新source-bound明示認可を待つ。

## G9 v2 one-shot completed / STOP（2026-10-10）

[G9 v2 results/GPT handoff](../../docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md)。`G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`、11 rows/22 axes、19 keys（new1/reuse18）、retry0。
known P5/指定3-qubit providerの登録direct6方式でclosed P5のT intercept/Kが小さく、CTSのCXは小さい。
1,866 event accounting、saved-only25 checks PASS。旧v1結果/marker、critical80/protected982は不変。
実量子shots/trajectory/DF/分子/NPZ/GPU/LPは0、source-bound local evidence。次の科学実行・採択はGPT判断、mandatory STOP。
