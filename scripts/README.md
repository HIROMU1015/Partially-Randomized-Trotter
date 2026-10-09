# scripts 索引

## 2026-10-06 H4 signal/compile INPUT_BOUND草案・容量不足STOP

run02の6凍結入力へplanを結合し、source19/218 templates/compiler/run ID/outputを不変に保った。
stage必要3.5 GiB/560000 inodesに対しavailable3.419376 GiB、約82.6 MiB不足。CPU候補6 core/memory/quota観測はPASS。
全72h監視・74784 records・149569 ledger deltas・全8KiB worker logs・1308 signal files・journal/temp/directory余裕を含む。
CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は次段の提案、review=false、利用者の別stage承認/明示launch未取得。
signal/seed/sampling/build/compile/transpile/taskset/worker/GPU/共有環境・他job変更0、入力再生成0。
[次段scope・容量](../docs/research/track_a_h4_signal_compile_plan_review.md)と[資料入口・承認対象](../artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06/README.md)を参照する。
`H4_SIGNAL_COMPILE_PLAN_PREPARED_STORAGE_BLOCKED_STOP`。容量と別認可成立後もfresh検査不合格なら起動せず、map後MAP_COMPLETE_STOP。
scientific runtimeはcommitしない。旧10GiB案/旧source/旧bundle/入力生成完了と以下の履歴は保持する。

## 2026-10-06 H4入力生成run02・6入力freeze完了STOP

利用者の明示再実行指示でworker bootstrapのstdlib signal shadowを-Pで修正し、run01を保存してrun02を別固定した。
CPU [3,5,6,7,8,9]・6 workers・own-run限定、fresh resource/容量/quota検査PASS。
H4 linear/STO-3G/DF12の追加6距離入力を一度生成しfreeze完了。NPZ6 bytes SHAを照合し、own driver/worker残存0。
sourceは049e69919af16ad29a67a217dc7a407d6b1754a6。科学/seed/compiler/resource条件は不変、source19変更はgates/worker起動だけ。
`INPUTS_FROZEN_STOP`、next_stage_authorized=false、mandatory_stop=true。signal/compile/GPU/追加transpile0、共有環境・他job変更0。
[修正・完了scope](../docs/research/track_a_h4_worker_bootstrap_run02.md)と[完了報告](../artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/COMPLETION_REPORT_v1.md)を参照する。scientific runtimeはcommitしない。以下は各時点の履歴。

## 2026-10-06 H4入力生成stage容量確認・最終承認待ち

容量準備の判定は「足りる」。6入力生成→freeze→STOPの必要量3GiB/260000 inodesに対し、
2026-10-06 16:58:53 JSTのnonroot available約3.615GiB、225817022 inodes、user/group/project quota非有効をread-only確認。
32保存配列/NPY・ZIP overhead/64MiB IPC上限/temp-final/72h監視259202 files/journal/metadata余裕を含む。
全campaign10GiBはcharge capとして維持し、旧全量空き確保案を履歴に保存した上でstage-specific補足を追加した。
source19/plan/auth/approved=false reviewはbyte-identical、CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は未承認。
CPU使用許可/独立最終review/明示launch/fresh CPU・memory・pressure/OOM・容量検査が残る。signal/compile容量は別認可。
[容量根拠](../docs/research/track_a_h4_input_generation_stage_storage_review.md)と[最終承認資料入口](../artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_STAGE_STORAGE_CONFIRMED_AWAITING_FINAL_APPROVAL`で公開後STOP。
新科学/追加transpile/taskset/worker/GPU/共有環境・他job変更0。旧10GiB案を含む以下は当時の履歴。

## 2026-10-06 H4入力生成 CPU/launch最終案・利用者承認未取得

提案CPU[3,5,6,7,8,9]、6 worker、異なる6 physical core・NUMA0。約3秒の受動負荷sampleで各core busy0%。
source19 pathsとplan v2 bytes/source_rootは不変。authは候補CPU集合だけ、reviewはauth digestだけ変更しapproved=falseを保持。
own新規runだけにtaskset maskを指定する未実行commandを用意した。CPU許可/専有予約/独立review/明示launchは未取得。
memory/context read-only確認は成功。filesystem空き約3.717GiBは総上限10GiB全量確保案に未達で、launch容量条件は未解決。
12 metadata gate tests PASS、fail/error/skip0。観測01/02の失敗logと03の容量未解決記録を保持し、追加transpile0/旧28件不変。
[提案資料](../docs/research/track_a_h4_input_generation_cpu_launch_proposal.md)と[bundle・一括承認判断](../artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL`で公開後STOP。taskset/科学/worker/GPU/共有環境・他job変更0。

## 2026-10-06 H4入力生成 resource observer修正・未承認草案再固定

真のv2 hierarchy rootをnamespace/mount/所属から判定し、rootの非root memory interface要求を修正した。
全可視非root祖先の制限・pressure/OOMは保持し、非root欠測/不明namespace/hidden mountはSTOP、host-only fallbackなし。
observer33 zero-science tests PASS、実read-only観測成功。準備観測available約981.826GiB、PSI/OOM0はlaunch成立ではない。
production変更はresources observerとgateのnew audit pathだけ。科学/並列/seed/compiler source15件は不変。
[実装資料](../docs/research/track_a_h4_geometry_resource_observer_fix.md)と[source bundle](../artifacts/resource_applicability/track_a_h4_geometry_resource_observer_fix/2026-10-06/README.md)、
[new認可草案v2](../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/README.md)を参照する。source固定→別草案commit、binding検査結果はv2へ記録。
requested workers6、allowed_cpus=[]、approved=false。CPU/launch context・独立最終review・明示launch未解決、実行準備完了とはしない。
`H4_INPUT_GENERATION_RESOURCE_FIX_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/追加transpile/GPU/本番起動/共有環境・他job変更0。
旧bundle/科学証拠/原稿/Track Bと旧監査履歴を保存し、系列transpile28/64を維持する。

## 2026-10-06 H4 geometry 入力生成専用認可草案・実行未承認

凍結science source6a121725（17 Python＋親2件）を変えず、入力生成source-bound planとresult-prior認可草案を追加した。
requested workers6、inputs/freeze digestはnull、218 templatesを機械転記。reviewはapproved=false、allowed_cpus=[]。
CPU許可は利用者指示で未確定のまま。現在process CPU0–255を許可とみなさず、launch contextとmemory観測は未解決。
既存observerはroot cgroup memory.max欠落で停止し、実行準備完了とはしない。source/共有設定を緩和しない。
新57 zero-science gate tests PASS、fail/error/skip0。合格経路はメモリ内模擬承認だけ、追加transpile0・旧累積28/64不変。
[実装・停止条件](../docs/research/track_a_h4_geometry_input_generation_authorization_draft.md)と[bundle・最終レビュー入口](../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_AUTHORIZATION_DRAFT_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/GPU/本番起動/共有環境・他job変更0。
有効execution authorization0、final review/利用者の明示launch未実施。入力生成・本計算・signal/compile認可・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry compile並列source再固定・科学未実行

compileの逐次waitをadmitted worker数以下のbounded投入・回収へ変更した。処理中ownerを追跡し、
COMPLETEとidentity/digest検査後だけ再利用する。trajectory/axis順・weight、科学scope・seed/compilerは不変。
[実装資料](../docs/research/track_a_h4_geometry_parallel_source_implementation.md)と[new bundle](../artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06/README.md)を現在の入口とする。
既存94＋並列回帰17＝111 synthetic tests PASS、fail/error/skip0。今回transpile3、旧25＋新3＝28/64。
fake futures/mock workersだけで制御を検査し、実worker/production性能は未検証。旧bundle/audit/契約・保存証拠は不変。
SOURCEとsourceを変更しないREVIEWの2 commitを分ける。分子アクセス/科学処理/GPU/本番起動/認可発行/共有環境・他job変更0。
`H4_GEOMETRY_PARALLEL_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。入力生成plan/auth作成・本計算・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry server-native source固定・科学未実行

利用者の新指示で契約v2 D1〜D4を実装条件へ採用し、旧未承認履歴を保存した。
新namespace `src/trottertracks/resource_applicability/h4_geometry/`、二つのfuture runner、専用synthetic testsの入口は
[実装資料](../docs/research/track_a_h4_geometry_server_native_source_implementation.md)と
[bundle](../artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06/README.md)。
最終94 tests pass、fail/error/skip0。失敗・再検査込みsynthetic transpile25/64、旧benchmark128再実行0。
SOURCE_COMMITとsourceを変えないREVIEW_BUNDLE_COMMITを分離し、actual blob/hashは別監査で固定する。
旧247 source・v1/v2 bundle・準備25 files・保存6 JSONは不変。分子入力/科学処理/本番runner launch/GPU/環境・他job変更/認可発行0。
`H4_GEOMETRY_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。別入力生成authorizationの作成へ進めるかをレビューし、今回は発行・実行しない。
以下は各milestone当時の履歴。


## 2026-10-06 H4 geometry契約v2・レビュー待ちSTOP

現在の入口は[契約v2 bundle](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/README.md)。v1 commit `7c1a3d43f61c5501a9e79206b7c60933f94b1077`を保存し、
D1〜D4を具体的な採用案、memory admissionを8+8w+16 GiB、認可を入力生成→freeze STOP→別signal/compile認可へ分離した。
H4 linear/STO-3G/DF rank12、6距離・218 template・32 paired trajectories・74,784上限は不変。8 system＋ancilla1(index8)、合計9 qubits。
[pure JSON validator](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/contract_validator_v2.py)と[専用合成検査](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/run_contract_tests_v2.py)はreview用で、science source/runnerではない。
新規320件pass（fail/skip0）、旧129件は保存・runner再実行0。旧v1 manifestはbase blobで照合し書き換えない。
D1〜D4レビュー承認は未解決、science/source port/input generation/next stage認可false、plan未seal、mandatory STOP。
今回の公開指示は軽量契約bundleと関連文書だけのcommit/non-force push。以下は各段階当時の履歴。

原稿用の表示・照合script：

- [build_track_a_manuscript_figures.py](resource_applicability/build_track_a_manuscript_figures.py)：
  固定9 CSV/JSONのcommit blob/SHAを照合して4主図・補足図1と表示CSVを新規outputへ生成する。
- [verify_track_a_manuscript_bundle.py](resource_applicability/verify_track_a_manuscript_bundle.py)：
  22assetのmanifest、表示値、原稿リンク・hashと専用21 synthetic/local testsを検査する。

[原稿付録の再表示手順](../docs/manuscripts/track_a_resource_study_supplement_v0_1.md)を参照。
科学module、分子snapshot、sampler、compiler、GPUは使わない。新しい科学runnerではない。

`resource_applicability/run_pr2_pm2_precision_analysis.py`は明示指示後に一回完了し、
[PM-2結果照合](../docs/pr2_pm2_precision_resource_result_validation.md)へ戻った。source不変、保存値解析のみ。
固定outputは作成済みで、上書き/resume/retryはしない。追加science/次段は未認可。
以下のsource固定・launch待ち記述は当時の履歴である。

PM-2解析sourceの入口は`resource_applicability/run_pr2_pm2_precision_analysis.py`と
`resource_applicability/run_pr2_pm2_implementation_tests.py`（synthetic-only、real saved evidenceも禁止）。
[実装と停止条件](../docs/research/pr2_pm2_precision_analysis_implementation.md)を参照する。
source `324435d77b6642dbd44e8d1f178420daf62e77ed`、62 synthetic tests passed。本解析は明示launch待ち、固定outputは未作成。
以下の契約準備・未実装記述は当時の履歴である。

PM-2の入口は`resource_applicability/run_pr2_pm2_precision_contract.py`（保存JSON4件の準備bundleをstdoutへ出すのみ）と
`resource_applicability/run_pr2_pm2_preparation_tests.py`（保護path guard付き専用contract tests）。
[契約/schema/input inventory](../docs/research/pr2_pm2_precision_resource_contract_v1.md)を参照する。
精度解析runnerは未実装、ε sweep・順位・P envelope・新signal/compileは未実行、解析は未認可。

PM-1は最終review・authorization一項目確定・利用者launch後に一回完了した。
[結果と照合](../docs/pr2_pm1_discard_result_validation.md)：8 signals/16 wrappers、pre/post201 local tests passed。
source不変、`PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`でmandatory STOP、retry/resume/次段なし。
下のdraft/future記述は実装・準備時点の履歴であり、現在の実行状態はこの節を優先する。

`resource_applicability/run_pr2_post_m2_evidence_attribution.py`はTrack A PM-0の保存JSON/source再集計専用。
stdlibのみ、artifact内pathを辿らず、bundleをstdoutへ出すだけでファイルを書き換えない。
[報告・対応source/test/artifact](../docs/research/pr2_post_m2_evidence_attribution.md)を参照する。
PM-1準備は`resource_applicability/run_pr2_pm1_discard_contract.py`（stdoutのみのsealed plan生成）、
`resource_applicability/run_pr2_pm1_preparation_tests.py`（科学データguard付き限定7-file tests）。
`resource_applicability/run_pr2_pm1_discard.py`は別committed authorization後のfuture科学入口であり、今回は呼ばない。
[契約・source・preparation audit](../docs/research/pr2_pm1_nearby_discard_contract_v1.md)を参照する。本実行未認可。

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


## H4 geometry 契約準備 v1（2026-10-06・local未commit）

[準備bundle](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06/README.md)は契約schema・zero-compute plan・pure JSON validatorと129合成検査の入口。
6距離、218 template/点、74,784 wrapper、最大12 workersを固定し、生成/seed/memory/wall/outputはreview待ち。
science source/runnerの追加ではなく、旧公開draft・source・科学結果は不変。本計算・port・commit/pushは未認可、STOP。


## H4候補間compile投入・12 worker明示再実行

利用者の増員再実行指示により[run03 source・認可・検査記録](../docs/research/track_a_h4_cross_candidate_run03.md)を追加した。候補内2回路の完了待ちで4 workerがidleとなる問題を、候補間bounded queueで修正。旧run02はworker failure STOPで全証跡を保持し、旧6入力を再生成せず利用する。49人工job/metadata testsはlocal PASSで科学的結果ではない。12 workerのfresh CPU/memory/容量/hash検査後だけ一度起動しMAP_COMPLETE_STOP。旧累積bytes/wall/actual invocationsを引継ぎ、科学条件・compiler・上限は不変。


## H4 run04：identity hash分割・5秒監視維持

[run04固定sourceと再実行binding](../docs/research/track_a_h4_streaming_monitor_run04.md)を追加した。run03はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。65 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/7 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## H4 run05：identity hash分割・5秒監視維持

[run05固定sourceと再実行binding](../docs/research/track_a_h4_lazy_identity_run05.md)を追加した。run04はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。75 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/12 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## 2026-10-07 H4新host A案 source/人工検証固定・本計算未認可

[監視修正・32人工tests](../docs/research/track_a_h4_new_server_monitor_fix_a_20261007.md)。既存private venv不変で準備用A案採用。
旧source19は3変更/16不変、新module込み25 closure。256×256人工matrix＋9-qubitの旧/new byte/digest一致、独立observerのGIL/GC観測・I/O delay/EOF/所有・資源境界を検証。
SOURCE `b2a5ad89e8b39d72716f7ddb17d263bd0cdedb45`、production/追加transpile0、旧28/64・benchmark128保持。環境18 version差、旧45 raw-reference RECORD差保持・normalized22差を明記。入力6/freeze/runtime/control未受領。
observer AS256MiB/RSS64MiB/admission120.25GiB、容量5.5625GiB案は未承認。allowed_cpus=[]/approved=false/runtime_authorization=false/launch=null、STOP。科学成果/原稿/Track Bと旧資料を保持。

人工runner `scripts/resource_applicability/run_h4_monitor_fix_a_tests.py`、auditor `scripts/resource_applicability/audit_h4_monitor_fix_a.py`。実runnerを起動しない。


## 2026-10-08 H4本計算前準備・最終承認待ちSTOP

[統合入口・最終承認案](../docs/research/track_a_h4_prelaunch_preparation_20261008.md)。新host schema/profile/observer/CPU/one-shot/累積budget bindingを整備。
SOURCE `ad57d1639133f7158cce58d767b8e0aa179bf044`、32 closure、57限定人工tests PASS。core quota read-only確認、worker12＋driver/observer各1別coreを提案。
carry20/165214360 bytes/5466.188392877579秒、残74764を保持。全74784 logicalを保証する最小actual cap+20→74804案は未承認。
候補環境はprivate venv不変、18 version/旧参照対raw45・normalized22差。追加output5GiB/301000 inodes、charge約8.29GiB、copy前は暫定6GiB。
入力6/freeze/native stop proof未受領、allowed_cpus=[]/approved=false/runtime_authorization=false/未seal、science/追加transpile/GPU/共有環境・他job変更0。明示承認・final review・launch前にSTOP。


人工runner `resource_applicability/run_h4_prelaunch_tests.py`、source-bound auditor `resource_applicability/audit_h4_prelaunch_preparation.py`。本番runnerを起動しない。

## 2026-10-09 H4 cleanup ESRCH修正・独立再review PASS

[新SOURCE・独立再review・残る承認条件](../docs/research/track_a_h4_cleanup_esrch_fix_20261009.md)。旧992c09d6から独立worktreeでP1を修正し、SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33を固定。
両pidfd送信経路はESRCHだけ既退出扱いで後続cleanupを継続し、他の送信error・所有検証を保持する。
限定人工39件PASS、別担当13純mock件PASSとbinding照合でP1_SCOPE_TECHNICAL_PASS。wait/reap/FD/pipe/first STOP理由保持まで確認。
新test `tests/tracks/resource_applicability/test_h4_cleanup_esrch.py` と既存人工runnerで追跡し、source/profile/plan/auth/reviewを新SHAへ再結合した。
入力0/6・freeze/native停止証拠未受領はNOT_EVALUABLE、未seal。環境/CPU/observer/74804案は未承認、carry20/165214360 bytes/5466.188392877579秒・現actual cap74784不変。
approved=false、runtime_authorization=false、allowed_cpus=[]。追加transpile/科学actual/本番起動0。旧資料を保存して本計算STOP。

## 2026-10-09 H4全byte受領・carry合格・native終端proof待ち

[受領・binding・最終承認案](../docs/research/track_a_h4_byte_receipt_binding_20261009.md)。packet86691840B/全2150files/既知9SHAをbyte-only照合しPASS、NPZ6/freeze受領完了。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変。ledger chain/cumulative journalからcarry20/165214360 bytes/5466.188392877579秒を保持、現cap74784・残74764。
run05 log/exact旧sourceからworker cleanup到達は推認可能だが、driver/12 workers停止後identity/残存0 native proofが不足。古いrun02停止監査をrun05proofへ流用しない。
input/profile/source/output/carryとv6 plan/auth/reviewを結合し、control82件のbasename・単一link・sender manifest mappingを照合。source条件は緩めていない。
sealed=false/approved=false/runtime_authorization=false/allowed_cpus=[]。環境/CPU/observer/累積74804案/一度のmap launchは未承認。科学array読込/新科学actual/追加transpile/共有設定変更0でSTOP。

## 2026-10-09 H4追加native停止proof合格・technical再seal

[再seal・独立最終整合review・一括承認案](../docs/research/track_a_h4_native_proof_seal_20261009.md)。追加JSON17521B/SHA一致、旧host/run05の13identity・2回残存0・元3証拠hashを照合して現在のnative停止条件PASS。
過去のexit code/正確な終了・reap時刻/原boot IDは未記録のままnull。今回の観測で補完せず、連続監視や歴史cleanup順の証明とも扱わない。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変、profile/input/carry/control83件の固定validator合格でplan再seal、sealed=true。
approved=false/runtime_authorization=false/allowed_cpus=[]。carry20/165214360 bytes/5466.188392877579秒、現cap74784・残74764を保持。
候補environment/compiler・CPU・observer・累積actual74804案・一度のmap launchは未承認。承認による最終artifact/digest再結合とfresh resource/CPU/fs/inode/quota gateをlaunch前に確認。
科学array読込/新science actual/追加transpile/source変更/共有設定変更/本計算0でSTOP。

## 2026-10-09 H4一度のmap実行を利用者承認・最終artifact固定

[実行認可・直前gate・起動報告入口](../docs/research/track_a_h4_authorized_launch_20261009.md)。利用者の明示認可で候補environment/compiler採用、worker12 CPUs2/4–6/8–15・driver16・observer18、observerAS256MiB/RSS64MiB/admission120.25GiBを認可。
carry20/165214360 bytes/5466.188392877579秒を保持し、累積actualだけ74804へ+20改定。SOURCE6bd1ba01・science/compiler/options/他caps不変。
sealed/approved/runtime_authorization=true、allowed_cpusはexact14role集合。独立v8 review PASS。artifact commit後fresh CPU/memory/PSI/OOM/FS/inode/quotaとSOURCE/profile/input/carry/unusedrootを確認しPASSなら追加承認なし一度起動。
既存proof/回帰/benchmark再実行0、追加準備campaign/transpile0。oldpartial/cache/GPU/共有環境・venv・他job変更なし、完了またはfail-closed STOP後終了・retry/次stageなし。
これは認可artifact固定時点のsnapshot。実起動/PID/状態は入口への追記・外部runtime receiptで別記録する。

## 2026-10-09 H4 library cache保存先修正・再実行予算不合格

[修正・再実行条件](../docs/research/track_a_h4_library_cache_fix_20261009.md)。前回のOpenFermion→Matplotlib mkdir拒否を、homeの新private library cacheへprocess限定MPLCONFIGDIRを結合して修正。
driver/workerとも既存directoryのEEXIST probe以外のcache writeを拒否、29816BのSHA固定。47限定回帰＋3 import case PASS、科学array/transpile/実worker/affinity/GPU0。
SOURCE `b8b3ce6e8c98f1ec0419a7af79c5d7c5f3a3b9bb`、36 closure、science/compiler/options不変。前回費用を返却せずcarry20/4428938712B/5472.345380863175sへ結合。
累積worst charge13165893832B=12.261694GiB>承認10GiBで未seal/approved=false/runtime_authorization=false、再起動0。13GiBは未承認proposalのみ。
既承認environment/CPU/observer/actual74804を保持。cap改定・新SOURCE/gate binding/review・fresh gate後の一度再実行が残る。
private homeは共有systemと区別し、旧run/one-shot/失敗証拠・全予約課金を保持する。

## 2026-10-09 H4軽量高速化・限定同等性確認

限定[runner](resource_applicability/run_h4_lightweight_speedup_tests.py)・[tests](../tests/tracks/resource_applicability/test_h4_lightweight_speedup.py)、実装[signal](../src/trottertracks/resource_applicability/h4_geometry/signal.py)・[ledger](../src/trottertracks/resource_applicability/h4_geometry/ledger.py)、[bundle](../artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/README.md)。

[変更・限定検証・binding](../docs/research/track_a_h4_lightweight_speedup_20261009.md)。driverの距離内共通準備を再利用し、one/DF block呼出を静的13×218→13、全prepareを218→10種類に削減。ledger deltaは変更entry/reservation各最大1だけを参照し、全件走査・saved-historyコピーを除いた。
SOURCE `4d2d1492fc23d0736c305533d78967cc1db8a7c8`、closure38。限定48人工PASS、全218prep/代表8wrapper+dense256case1/代表4signal/ledger13fileの旧new bytes・digest一致。12workersはmock、単一test process内部thread1、science array/transpile/GPU/affinity/production0。
実Gaussian/旧compiler output/実速度・H4本体成功は未検証。monitor/caps/compiler/science/carry不変、旧partial/cacheと混合しない。
carry20/4428938712B/5472.345380863175s、actual74804既承認、worst charge12.261694GiB>承認10GiBは残る。未seal/approved=false/runtime_authorization=false、absolute_launch_command=null、本計算0。

## 2026-10-09 H4利用者が13GiB累積charge・一度の再実行を明示認可

[認可・source・一度の起動入口](../docs/research/track_a_h4_approved_relaunch_20261009.md)。利用者の「これについては問題ないので再実行して」を、既存13GiB cumulative charge案と一度のmap再実行の承認として反映。
SOURCE `a7b617600cd7063f7870f2059d5694ef00283f0e`/closure39、output改定schema/gate/実OutputBudget capとmarginのみ変更。legacy10GiB default・科学/compiler/その他caps・prepare再利用/ledger保存は維持。
限定22pure gate PASS、旧48speedup/library/cleanup/native証拠campaign再実行なし。carry20/4428938712B/5472.345380863175s返却なし、actual74804・新残74784。
worst13165893832B <= 新cap13958643712B、余裕792749880B。sealed/approved/runtime_authorization=trueの認可artifactへ再結合。
独立review/artifact固定後fresh SOURCE/profile/input/carry・CPU/memory/PSI/OOM/fs/block/inode/quota/unusedroot/one-shot合格時にそのまま一度起動。実run状態はruntime証跡へ記録。
既承認12workers CPUs2/4–6/8–15、driver16/observer18、thread1・observerAS256MiB/RSS64MiB/admission120.25GiB保持。自動retry/入力再生成/旧partial/cache/次stage/GPU/共有設定変更なし。
