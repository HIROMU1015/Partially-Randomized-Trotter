> 2026-10-07 Track B RA-D0 v3：**READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION**（実行承認ではない）。
> [source review](../docs/tracks/algorithm_codesign/ra_d0_source_review_v3_20261007.md) / [GPT handoff](../docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_v3_20261007.md) / [manifest](../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/evidence_manifest_v3.json)。
> [focused verifier](tracks/algorithm_codesign/verify_ra_d0_source_review_v3.py) / [v3 tests](../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v3.py)。
> exact-certified B2 minimum infeasibilityを正常outcomeに修正。pointをfreezeへ記録しbudget/queryを空にして次nへ進む。
> uncertified failureはtechnical STOP。数値・candidate・grid・call/resource capは維持。登録最適化0、authorizationなし、mandatory STOP。

> 2026-10-07 Track B RA-D0 v2：**READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW**。
> [source review](../docs/tracks/algorithm_codesign/ra_d0_source_review_v2_20261007.md) / [GPT handoff](../docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_20261007.md) / [evidence manifest](../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/evidence_manifest_v2.json)。
> [future runner](tracks/algorithm_codesign/run_ra_d0_one_shot.py) / [focused verifier](tracks/algorithm_codesign/verify_ra_d0_source_review_v2.py) / [v2 tests](../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v2.py)。
> B0_saved/ideal分離、数値B1⊂B2⊂B3、profile-paired budget、batch freeze-before-B3、
> anchor-first、main LP 55,275 / auxiliary込み110,550、resource/launch guardを固定。
> 登録最適化・実budget/minimum/witness取得0、authorizationなし。旧本文・旧STOP・Track Aは保持。mandatory STOP。

## Track B RA-D0 static preparation / mandatory STOP（2026-10-06）

[source review](../docs/tracks/algorithm_codesign/ra_d0_source_preparation_review_v1.md)：21 columns/x、18 sign pairs一致、35 focused tests PASS。
B専用namespace `src/trottertracks/algorithm_codesign/ra_d0/`、static generator／verifier、
`artifacts/track_b_ra_d0_preparation/2026-10-06/`に候補・grid・semantic audit・provenanceを保存。
ideal nestingと数値membershipを区別し、query実行scope／資源上限をGPTへ返す。
登録最適化／新合成／新science0、RUN_READY=false、既存分類・Track A・旧STOP保持。
**mandatory STOP。以下の既存本文を全文保持する。**

## Track B RA-RTE統合数学監査・mandatory STOP（2026-10-06）

R1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942` 基点、GPT設計案へのDOCS_SYMBOLIC_ONLY_MATHEMATICAL_AUDIT。
[命題別監査](../docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_v1.md)と[GPT handoff](../docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_gpt_handoff_20261006.md)：一block／finite table／canonicalのfixed-n LPは仮定付きで成立。
shot-gridの固定total cap保存には反例。log/root・sampler認証、Delta=0、peak workspaceの規約を実行前修正へ返す。
[stdlib人工bookkeeping](tracks/algorithm_codesign/check_ra_rte_mathematical_bookkeeping.py)の[50 checks](../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_checks_v1.json)を一般証明と分離した。
[manifest](../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/evidence_manifest_v1.json)、[dated note](../docs/research/研究ノート/2026-10-06_track_b_ra_rte_mathematical_audit.md)。science/synthesis/solver/資源再採点0、共通API変更0。
既存科学分類・結果・STOPは不変。性能・新規性・algorithm採択・次実装／R2 authorizationは未確定。
**mandatory STOP。次の採択・実装・pilotの必要性／範囲はGPT判断。以下の既存本文を全文保持する。**

## Track B R1.5保存値帰属・mandatory STOP（2026-10-06）

input R1 commit `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b` の保存値だけを用いたPOSTHOC attribution / design input。
[帰属報告](../docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md)と[GPT handoff](../docs/tracks/algorithm_codesign/r1p5_gpt_handoff_20261006.md)：新science/synthesis/compile/候補追加0、R1科学分類は不変。
primaryは2-qubit finite P₃、distinct-basis controlled、x={1/8,1/4}、登録native三precision。
Aは登録(G_T,G_CX,G_1Q) frontにx=1/8の1e-4、x=1/4の1e-3/1e-4で残る。
normalizationだけでなくnative費用とbias/shotの関係を整理し、固定合成列への依存も保存した。
[stdlib保存値解析](tracks/algorithm_codesign/analyze_r1p5_saved_attribution.py)、[全summary](../artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)、[provenance manifest](../artifacts/track_b_r1p5_saved_attribution/2026-10-06/evidence_manifest_v1.json)、
[日付note](../docs/research/研究ノート/2026-10-06_track_b_r1p5_saved_attribution.md)。共通library変更・独立validation・新algorithm採択はない。
限定診断SUPPORTS_RA_RTE_DESIGNは設計入力のみ。eta探索/R2/DF接続/追加scienceは未認可。
**mandatory STOP。次の数学設計・研究方針判断はGPT側。以下の既存本文を全文保持する。**

## Track B R1一回結果・mandatory STOP（2026-10-06）

固定S `d43d64a821a0249a0dfab12a2472bd3a72fdee74` →直接子authorization-only A
`411f08f768244fe87b600d82308c3851847fe9e4`からrun1/retry0。
[結果照合](../docs/tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md)：126 keys / 264 rows / 132 controlled tasks完了、全task適格。
terminal R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW、[保存field専用監査](tracks/algorithm_codesign/audit_r1_saved_result.py) PASS。
primary distinct-basis controlledではB²改善とnative/shot資源のtrade-offを保存し、自動研究GOはない。
[全264 rows CSV](../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)、[evidence manifest](../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)、
[GPT判断への入口](../docs/tracks/algorithm_codesign/r1_post_run_gpt_review_request_20261006.md)。原result/marker/source、既存証拠・共通API・Track A保持。
science終了後mandatory STOP、追加合成/target/grid/分子/DF/trajectory/GPUは行わない。
研究方針・RQ・新規性・着地点・追加検証の必要性/範囲はGPT側。
以下のpending/最新記述は当時の履歴として本文をそのまま保持する。

# scripts 索引

## Track B R1 native-resource runner: source review only

[run_r1_rte_reallocation.py](tracks/algorithm_codesign/run_r1_rte_reallocation.py)のplanはstatic key inventory照合だけ。
runはsource Sのdirect authorization-only child Aと新明示指示を要求する。現在pending、科学実行なし。
[契約/source review](../docs/tracks/algorithm_codesign/rte_reallocation_r1_source_review_request_20261006.md)、
[source manifest](../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/source_manifest_v1.json)。
27 focused tests、126 planned keys、mandatory STOP。


## Track B R0.5 symbolic checker

[check_rte_reallocation_r05_symbolic.py](tracks/algorithm_codesign/check_rte_reallocation_r05_symbolic.py)：
固定9条件/18 signのexact Fraction/Pauli比較を一回実施。science runnerではない。
[監査](../docs/tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md)、
[保存artifact](../artifacts/track_b_rte_reallocation_r05/2026-10-06/exact_symbolic_comparison_v1.json)。
追加比較・再実行・科学実行の認可なし、mandatory STOP。


## Track B R0 technical checker

[check_rte_reallocation_symbolic.py](tracks/algorithm_codesign/check_rte_reallocation_symbolic.py) verifies fixed free-word/Fraction identities. It is not a sampler, solver or science runner; no trotterlib imports. [Review/provenance](../docs/tracks/algorithm_codesign/rte_reallocation_r0_review_packet_20261006.md) records one local technical execution and STOP.


Track B [BS-0.5設計監査](../docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md)はdocs-only。
runner/solver/synthesis/testsを追加・実行していない。ordinaryの式とledgerは実装済みsourceではない。
RUN_READY=false、mandatory STOP、次実装/検証scopeはGPT判断。以下は既存実行履歴。

Track B [block合成の次仕様案](../docs/tracks/algorithm_codesign/block_synthesis_design_review_20261006.md)はdocs-only。
新runner/solver/library生成/合成/testsは追加・実行していない。旧SP one-shotを再使用しない。
API/対照/pilot案の採否はGPT reviewへ戻し、RUN_READY=false／mandatory STOP。以下は既存実行履歴。

Track B [SP-1一回結果](../docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)は
SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW、mandatory STOP、retry0。science runnerのone-shotはconsumed。
[audit_sp1_saved_result.py](tracks/algorithm_codesign/audit_sp1_saved_result.py)はstdlibの保存field照合専用。
source/authorization/hash/inventory、保存interval算術／分類predicate／duplicate／capを監査する。
science moduleやrunnerをimportせず、matrix・guard・係数・Bernstein shots・resourceを再評価しない。
[audit／evidence manifest](../artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/)。保存値監査PASS、次stage未認可。
以下のrunner準備・pending記述は結果前履歴。

Track B [SP-1 runner](tracks/algorithm_codesign/run_sp1_wrapper_pilot.py)を追加した。
`plan --source-commit <full S>`は静的角度/key/fusionとsource identityだけを照合する。
資源/信号採点・合成・marker作成は0。`run`は別の明示authorizationとclean direct-child Aを要求する。
[採用契約](../docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)、
[source manifest](../artifacts/track_b_sp1_wrapper_source/2026-10-06/source_manifest_v1.json)。
focused testsは`test_sp1_wrapper_*.py`だけ。science未認可、最終source reviewへSTOP。以下は履歴。

Track B [SP-1 wrapper契約案](../docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_proposal_v1.md)は準備段階。
science runnerはまだ作成していない。共通会計kernelと17 artificial focused testsだけを追加し、
science sweep／SP-0.5 rerunを行わない。実adapter／fusion監査／source freeze／別authorization前にSTOP。
限定test入口：`PYTHONPATH=src python3 -m unittest discover -s tests/tracks/algorithm_codesign -p test_sp1_wrapper_accounting.py -v`。
以下のSP-0.5 runnerは別のconsumed one-shotで、SP-1の入口ではない。

Track Bの[SP-0.5一回結果](../docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md)は
mandatory STOP、retry0、wrapper pilot未認可。[audit_sp05_saved_result.py](tracks/algorithm_codesign/audit_sp05_saved_result.py)は
保存field／hash／counts／分類predicate／provenanceだけをstdlibで照合する。
synthesis／PAI／J／error-guard関数を再実行しない。旧runner runは一回consumed、追加runしない。


Track Bの[SP-0.5専用runner](tracks/algorithm_codesign/run_sp05_synthesis_economics.py)は
[結果前契約](../docs/tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md)に沿ったprimitive合成経済性gateの入口。
planは合成0、runはsource S→別authorization-only child A・明示指示・fresh one-shot markerを要求する。
準備tests34件pass、登録target合成0、現在run未認可。16-cell wrapper／DF／分子／trajectoryは扱わない。


Track B [BM-0.5](../docs/tracks/algorithm_codesign/bm05_review_packet_20261005.md)の
[形式word監査](tracks/algorithm_codesign/audit_bm05_symbolic_equivalence.py)は抽象非可換letters／Fractionだけの
degree3比較。physical inputやscience runnerを呼ばない。保存reportとscopeは[専用索引](tracks/algorithm_codesign/README.md)。
現adapterは同値でnew-method gate不通過、BM-1は実行しない。以下のBF記述は既存履歴。

Track Bの[BF1-R0一回replay結果](../docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md)は
recovery complete、事後primary BF-A。R0 markerもconsumed、retryなし。以下のR0準備説明は結果前履歴。

Track Bの`tracks/algorithm_codesign/run_bf1_read_only_recovery.py`は[BF1-R0契約](../docs/tracks/algorithm_codesign/bf1_read_only_recovery_contract_v1.md)
に沿った一回の保存値replay専用入口。科学runnerを呼ばず、cache不足で停止する。

Track Bの`tracks/algorithm_codesign/prepare_bf1.py`はformula-only domain列挙と限定synthetic testsの入口。
`run_bf1.py`は別authorization・commit-bound sourceを必要とするone-shot runner。
[BF-1規範](../docs/tracks/algorithm_codesign/bf1_preregistration_v1.md)。固定source/authorizationで一回実行後、
JSON保存例外により[INCONCLUSIVEで停止](../docs/tracks/algorithm_codesign/bf1_one_shot_result_validation_20261005.md)。
one-shot markerはconsumed、retryなし。
[別branchの保存型修正](../docs/tracks/algorithm_codesign/bf1_serialization_repair_review_20261005.md)では
runnerを変更・実行せず、限定synthetic testsのみを行う。
[v2 amendment](../docs/tracks/algorithm_codesign/bf1_execution_gate_revision_v2.md)はpost-search cross-scoreと
authorization-only child commit方式を固定する。
[v3 amendment](../docs/tracks/algorithm_codesign/bf1_assembly_guard_revision_v3.md)はassemblyの逐次誤差伝播を固定する。
prepareは既存v1 domainを参照して新規v3 packetを作り、旧v1/v2を保持する。

`scripts/` はコマンドラインから実行する入口を置く。研究ロジックの本体は原則として
`src/trotterlib/` にあり、runnerは条件の読込み、呼出し、成果物保存を担当する。

最新の研究段階は[`../docs/research/研究概要・現状.md`](../docs/research/研究概要・現状.md)、
各runnerと成果物の対応は[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)を
参照する。下記の「現行」は、最終科学結論を意味せず、現在の近似検証で使う経路を表す。

## 現行の検証runner

### PF誤差係数

| runner | 用途 |
|---|---|
| `run_pf_delta_validation.py` | 小規模系でPF係数の定義・delta依存・参照値を比較。cross-validation結合には`--snapshot --n-electrons`を使用可能 |
| `run_pf_c_system_size_validation.py` | H-chainの系サイズ方向とstate-action経路を検証 |
| `run_df_ground_state.py` | DF Hamiltonianの基底状態snapshotを作成 |
| `benchmark_df_ground_state.py` | 基底状態計算経路の計測 |
| `validate_pauli_partial_cgs_all_prefixes.py` | Pauli partial回路係数のprefix検査 |

### finite RTE

| runner | 用途 |
|---|---|
| `run_finite_rte_signal_validation.py` | 演算子誤差、状態上の複素期待値、半径、位相上界を検証 |
| `run_fr_revision_nonuniform.py` | FR-R1bの凍結expected生成と非一様4×4・負時間・K4 controlの実行を分離し、R0--R7を判定 |
| `run_pr2_s0_s1_validation.py` | 結果前v3に従い、非上書きの独立`S0`/`S1` commandだけを提供する。S2/S3 commandと自動進行は持たない |
| `run_pr2_new_series_v1_v3.py` | 旧STOPを保持した別snapshot系列のV1–V3だけを非上書き実行し、V4/S1′未承認で停止する |
| `run_pr2_v4_correctness.py` | 別snapshot系列のcorrected/raw signal、Re/Im Hadamard semantics、controlled full-wrapper compileを検査する |
| `run_pr2_s2_development.py` | V4 PASS後の固定development-only resource comparisonをserial実行し、mandatory STOPする |
| `run_pr2_s2_development_parallel.py` | 同じS2 cell、seed、32→96拡張barrierを維持し、full-wrapper compileを最大8 CPU workersで並列実行する。既定4 workers、persistent SQLite cache、別outputを使用する |
| `run_pr2_matched_accuracy_m1_contract.py` | M1の208候補identity、最大4境界候補、16-cell selectorをsynthetic fixtureで検査するzero-compute runner。NPZ、signal、trajectory、circuit、compileを開かずmandatory STOPする |
| `run_pr2_matched_accuracy_m1_precompile_barrier.py` | M1-A/M1-B hard barrierのlimited/clear synthetic分岐を検査するzero-compute runner。limited時のcompile job生成を拒否し、科学計算を承認しない |
| `run_pr2_matched_accuracy_m1_a.py` | 結果前authorizationに従い、development H4 1.00 Åだけで最大212候補のcompile-free signal/selectorを実行する。実結果は210候補、52未選択frontierにより`SELECTION_LIMITED`、compile 0で停止 |
| `run_pr2_matched_accuracy_m1_b1_contract.py` | byte-fixed M1-A結果からrandom 194＋baseline 16 cellと12,448 wrapperのcache identityを生成するstandard-library-only zero-compute runner。snapshot、trajectory、circuit、compiler、GPUを開かず停止する |
| `run_pr2_matched_accuracy_m1_b1.py` | actual execution source commitに結合したzero-compute planをfreezeし、別commitのresult-prior authorization後だけ12,448-wrapper compile mapを最大6 workersで実行する。研究四分岐を自動選択せずreview待ちで停止する |
| `run_pr2_matched_accuracy_m1_b1_result_validation.py` | 保存済みM1-A/M1-B1 result、210 checkpoint、210 candidate別SQLite cacheをread-only再検査し、actual frontier、fixed-q=8差、旧selector、proxy、状態準備感度と外部review判断をJSONへ出す。分子snapshotとheld-outは読まない |
| `run_pr2_matched_accuracy_m2_transfer_contract.py` | commit済みM1-B1 result/validationだけを読み、5構成、usable B2限定のtransfer判定、seed、196-wrapper上限を固定するstandard-library-only zero-compute runner。v2正式planはsource commit blob照合必須、未commitは `--draft` のみ。held-out pathをresolve/stat/hash/loadせず、科学実行を認可しない |
| `run_pr2_matched_accuracy_m2_transfer.py` | actual source commitに結合したzero-science `plan`と、別のcommit済みauthorizationを要求するone-shot `run`を分離する。最大5 workers、固定5構成、196 wrappers。一回の実行は`TRANSFER_SUPPORTED`で完了し、[結果照合](../docs/pr2_matched_accuracy_m2_transfer_result_validation.md)後mandatory STOP。再実行/resume/別output救済・追加科学計算は認可しない |

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
| `run_research_direction_prevalidation.py` | WP00比較契約、WP02 round-horizon coverage、WP01-Sの条件付き小系比較を順に生成 |
| `run_research_direction_ablation.py` | WP04のschedule、$\beta$、$\alpha$、cost-provider寄与分解と信号・統計bound診断を生成 |
| `run_research_direction_pf_sensitivity.py` | WP03でPF係数だけを差し替え、$L_D$・$\delta$選択とcandidate regretの感度を生成 |
| `run_research_direction_gate_s1.py` | WP00/WP02/WP01-S/WP04/WP03のfingerprint済み結果をGate S1の研究方向判断へ統合 |
| `run_research_direction_structure_pilot.py` | WP06-aでfull/support限定Gaussian basis、basis融合、control、relative phaseを代表Z/ZZと短列で比較 |
| `run_research_direction_sequence_policy.py` | WP06-bでsequence-aware basis policyを独立training/holdoutし、中央RTE差を既存$q$ slopeへ接続 |
| `run_research_direction_full_scope.py` | WP05-aで選択済みbasis policyをcomplete controlled partial-$S_2$／Hadamard wrapperへ接続し、$q=1,2$較正と未使用$q=4$を直接transpile |
| `run_research_direction_full_scope_extension.py` | WP05-bで$q=8$と比較対照$\delta=0.01$へfull-scope較正を拡張 |
| `run_research_direction_full_scope_replication.py` | WP05-bRで境界条件$\delta=0.02,r=32,q=8$を独立32 trajectoryで再検証 |
| `run_research_direction_decision_cost.py` | WP01-D/C07でfull-scope proxyを使い、候補ごとの誤差配分とshot数を再最適化 |
| `run_research_direction_decision_synthesis.py` | WP01-D/C07のlocal区間と移送感度区間を分離して方向判断を生成 |
| `run_research_direction_round_dominance.py` | G08でround別cost・proxy不確かさ・PF/RTE riskを分解しM08対象を固定 |
| `run_research_direction_proxy_precision.py` | M08で支配的$r=32$の未使用$q=16,32$ full-wrapper holdoutを直接transpile |
| `run_research_direction_m08_reaggregation.py` | M08実測幅、従来5%、移送25%でWP01-D/C07判断区間を再集計 |
| `run_research_direction_compiler_transfer_compute.py` | M06/L08用に同一trajectoryをoptimization level 2で再compileする計算専用runner |
| `run_research_direction_compiler_transfer_analysis.py` | M06/L08のoptimization level 2結果を解析し、固定planでfocused再集計するrunner |
| `run_research_direction_uncertainty_break_even.py` | N07不確かさ台帳とP03状態準備break-evenを既存artifactから再集計するrunner |
| `run_research_direction_wp11_synthesis.py` | WP11で実施済みartifactをT1--T7へ統合し、次の検証を一件だけ選ぶrunner |
| `run_research_direction_full_opt2_compute.py` | M06-Fのall-r opt2 cell manifest、dry-run、checkpoint/resume computeを担当するrunner |
| `run_research_direction_full_opt2_analysis.py` | 完了済みM06-F computeだけを解析し、coherent opt2再最適化と必要時のfresh 32-trajectory manifestを生成するrunner |
| `run_research_direction_full_opt2_completion.py` | M06-Fのtask/worker/checkpoint/aggregate完全性と解析gateを再監査し、既存結果を上書きせず日付付きartifactを生成するrunner |
| `run_research_direction_full_opt2_extension_analysis.py` | 初回36とfresh-32 15 taskを統合監査し、両gate通過時だけcoherent opt2再最適化とbreak-even再集計を行うrunner |
| `run_research_direction_proxy_lineage_reconciliation.py` | A0として最新fresh `q=1,2` RZ proxyを旧opt2 `q=16,32`固定holdoutへ再適用し、lineageと適用domainをfingerprint付きで記録 |
| `run_research_direction_signal_weight_pilot.py` | P-Bとして既存H4 PF artifactのenergy bias、target weight、q別signalを再集計し、signal-aware選択差の有無を記録 |
| `run_research_direction_geometry_energy_difference_pilot.py` | P-CとしてH4 5 geometryのsigned PF係数、geometry/delta holdout、差分energy biasを再集計 |
| `run_research_direction_geometry_tracking_breakdown.py` | 事前固定したH4 8 geometryでindependent/tracked prefix、signed PF bias、blind/pair予測、continuity診断を評価 |
| `run_research_direction_joint_synthesis_pilot.py` | P-Aとしてfull共有、event-support、現行policy、interval-union DPを独立event列で同一compile比較 |
| `run_research_direction_theme_selection.py` | fingerprint済みP-B/P-C/P-Aを4問で比較し、暫定主題と次の停止点を記録 |
| `run_research_direction_energy_tail_pareto.py` | P-Dとして固定PF familyの次数、exact two-block energy bias、`Gamma_R`、finite-RTE解析負担をdevelopment/blindで比較 |
| `run_research_direction_pd_realization.py` | P-D Go/No-Goとして負時間finite-RTE/control位相、fragment内部`H_D`誤差、fresh `L_D=5`選択差を固定条件で検証 |
| `run_research_direction_pd_fair_comparison.py` | P-D S1として共通時間・位相予算でB0/B1a/B1b/B2/B4をnested/native比較し、regret、限定K4、境界規則でCase A--Dを判定 |
| `run_research_direction_pd_s1_posthoc.py` | 固定S1 v2 artifactだけを再集計し、主baseline、5%近傍、B1aのm_D依存、nested/native内訳を事後解釈として保存 |
| `run_finite_rte_phase_amplitude.py` | FR-1の固定2×2 gridでphase/radius境界、semantic、G0--G4を評価しfingerprint付きartifactを保存 |
| `run_fr_revision_fr1a_posthoc.py` | 完了済みFR-1の33条件・99状態を再構成し、正scalarと強いnorm/FR baselineを事後再集計。旧判断とFR-R1b事前登録は変更しない |
| `run_research_direction_joint_synthesis_blind_validation.py` | P-A v1を未使用H5 snapshotとH4 opt2へ移す事前登録済みblind検証。compile前manifest、per-task checkpoint、resume、status対応 |
| `run_research_direction_joint_synthesis_formalization.py` | 完了済みpilot/blind artifactを再監査し、P-A v1の目的関数、DP、計算量、同値性条件と機構coverageをfingerprint付きで記録 |
| `run_research_direction_joint_synthesis_mechanism_validation.py` | P-A v1を明示的一区間baselineとforced-support order-2 streamで比較し、固定gateに従って継続／停止を判定 |

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

## Track B G1限定診断のsource

[run_g1_decision_packet.py](tracks/algorithm_codesign/run_g1_decision_packet.py)はread-only source確認と、
別明示指示後の一回だけのdecision packetを分離する。
[専用module](tracks/algorithm_codesign/g1_decision_packet/README.md)、
[source review](../docs/tracks/algorithm_codesign/g1_source_review_20261009.md)、
[focused tests](../tests/tracks/algorithm_codesign/test_g1_source_preparation.py)、
[証拠manifest](../artifacts/track_b_g1_source_preparation/2026-10-09/evidence_manifest_v1.json)。
旧guard/verifier/binaryは不変。本構造監査・固定8人工LPは未実行。全outcomeでSTOP、retry=0。

## Track B G2 saved-value diagnosis (2026-10-09)

`tracks/algorithm_codesign/g2_saved_diagnostic.py` is stdlib-only post-hoc arithmetic for the fixed
21-column tables, 252 profiles/x. See [handoff](../docs/tracks/algorithm_codesign/g2_saved_diagnostic_handoff_20261009.md).
Completed; exclusive diagnostic markers are retained. Do not rerun or infer production authorization.
It imports no old execution controller or shared library and performs no solver/synthesis/circuit work.


## Track B G3（2026-10-09）

`tracks/algorithm_codesign/g3_finite_law.py`（消費済みPhase A）、
`tracks/algorithm_codesign/g3_return_comparator.py`（消費済みPhase B）、
`tracks/algorithm_codesign/audit_g3_saved_outputs.py`（保存値のread-only照合）。
[仕様と結果](../docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md)、
[manifest](../artifacts/track_b_g3_finite_law/2026-10-09/evidence_manifest_v1.json)。
各Phase一回/retry0。full suite/旧run/registered LP/CTS/DFへの自動進行なし、mandatory STOP。


## Track B G4

`tracks/algorithm_codesign/g4_independent_certificate.py`（旧helperなしsaved-value認証）、
`g4_cts_specialization.py`（exact Pauli代数）、`g4_matched_cts.py`（条件付き12合成one-shot）、
`audit_g4_saved_outputs.py`（STOP後の保存値照合）。
[G4 handoff](../docs/tracks/algorithm_codesign/g4_results_and_gpt_handoff_20261009.md)を先に読む。
両marker consumed、retry不可。Cは設計のみ。次science stage未認可。


## Track B G5 fixed-dictionary closure（2026-10-10）

[Saved-only stdlib verifier](tracks/algorithm_codesign/g5_fixed_dictionary_closure.py)：fixed G5 sourceで一回完了。
x1/4・6頂点252 profiles、strict rational/dyadic CTS照合とdigital bridge。新LP/synthesis/matrix/circuitなし。
[scope](../docs/tracks/algorithm_codesign/g5_fixed_dictionary_closure_scope_20261010.md)、
[result](../docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md)、
[manifest](../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/evidence_manifest_v1.json)。marker消費済み、再実行しない。mandatory STOP。


## Track B G6 formal technical audit（2026-10-10）

[G6 checker](tracks/algorithm_codesign/g6_return_generator_audit.py)は独立off-domain形式testsのみ。
本回technical bundle完了、science runnerではない。新technical marker/STOPを保存し、再呼出ししない。
[結果・認可境界](../docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md)。


## Track B G7（2026-10-10）

[限定費用runner](tracks/algorithm_codesign/g7_budget_control_economics.py)、
[契約](../docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md)。
fixed source/full SHA/clean HEAD/new marker/24 keysのみ。再実行不可、bundle後STOP。


## 2026-10-10 Track B G7：取得完了・mandatory STOP（最新追記）

[G7結果/GPT引継ぎ](../docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md)。
`G7_LIMITED_IMPLEMENTATION_ECONOMICS_COMPLETE_AWAITING_GPT_REVIEW`。source ab2549f41b3546fb3940342a2162dd9ee93699c4、24 keys/8 rows、strict error PASS、retry0。
固定P5では登録3対照後にもconditional期待T減少、P3ではclosed-form対照が小さい。
hard shot cap・classical generation/angle acquisition・未指定provider costを併記。
新規性/主method/次stage未採択、G5閉鎖/G6原証拠とmarker保持、GPT判断へ戻す。


## Track B G8（2026-10-10、source preparation）

[G8 proof/contract](../docs/tracks/algorithm_codesign/g8_proof_contract_and_on_demand_scope_20261010.md)：採用GPT G7 review §11に基づく限定確認。
2 known development inputs/4 same production laws、finite-provider parameter、分離failure配分、
on-demand strict Rzと対称bounded cache。G7のstatus/point comparison/consumed markerを保持。
17 off-domain focused tests、source preparation時点のnew native acquisition0。一束後mandatory STOP。


- `scripts/tracks/algorithm_codesign/audit_g8_saved_outputs.py`: stdlib saved-output/hash audit; no generator/native reacquisition. [G8 handoff](../docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md).


## Track B G9 source preparation（2026-10-10）

[Fixed proof/scope](../docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md)：GPT G8 review §14を採用。known P5/指定3-qubit provider、6 direct+5 helper診断、19 keys/新CTS1 key、23 focused tests。source固定後一束のみ、終了後mandatory STOP。
