# 文書索引

[A1 Q1/Q2-L：固定H4 m0のidentity不一致STOP](research/o2_a1_q1_q2l_20261106/REPORT.md)（2026-11-06）：
公開実行SOURCE・原STOP・静的SOURCE差。実CDF/ν/bias未取得、科学retryなし。

[O2-S2複数event有限table評価v2・六cell完了（2026-11-06）](research/o2_s2_multi_event_validation_v2_20261106/REPORT.md)：
転送修正後の固定SOURCE、3,969 pair/三law、量子T主指標の不改善、古典費の制約とSTOP。v1途中結果も保全。

研究別の成果・SOURCE・raw・検証状態への入口は[研究結果への統一索引](research/RESEARCH_RESULTS_INDEX.md)。
このcheckoutにない後続資料も、固定commitから参照できる。

このディレクトリには、研究方針の正本、実装規約、検証報告、発表資料の案内が共存する。
研究全体を初めて読む場合は、先に[`../PROJECT_MAP.md`](../PROJECT_MAP.md)と
[`research/研究概要・現状.md`](research/研究概要・現状.md)を読む。

## 研究方針と現在地

- [`research/研究概要・現状.md`](research/研究概要・現状.md)：最新の短い全体要約
- [`research/README.md`](research/README.md)：研究文書内の索引
- [`research/研究目的・研究課題.md`](research/研究目的・研究課題.md)：目的と研究課題
- [`research/研究方法・解析手順.md`](research/研究方法・解析手順.md)：採用する解析手順
- [`research/数値実験・評価計画.md`](research/数値実験・評価計画.md)：検証と評価の計画
- `research/研究ノート/`：時系列の判断記録。現在の仕様ではない

## 現在の主な検証文書

### PF係数

- [`pf_delta_validation.md`](pf_delta_validation.md)
- [`pf_c_system_size_validation.md`](pf_c_system_size_validation.md)

### finite RTE

- [`rte_conventions.md`](rte_conventions.md)
- [`rte_truncation_budget.md`](rte_truncation_budget.md)
- [`finite_rte_signal_validation.md`](finite_rte_signal_validation.md)
- [`df_rte_tail_extraction.md`](df_rte_tail_extraction.md)
- [`df_rte_event_circuit_api.md`](df_rte_event_circuit_api.md)

### コンパイル後回路コスト

- [`random_circuit_cost_validation.md`](random_circuit_cost_validation.md)
- [`rte_boundary_cost_validation.md`](rte_boundary_cost_validation.md)
- [`rte_boundary_pair_validation.md`](rte_boundary_pair_validation.md)
- [`hierarchical_cost_validation.md`](hierarchical_cost_validation.md)
- [`rte_connected_cluster_cost_validation.md`](rte_connected_cluster_cost_validation.md)
- [`rte_compiled_cost_validation_summary.md`](rte_compiled_cost_validation_summary.md)
- [`rte_compiled_event_cost.md`](rte_compiled_event_cost.md)
- [`df_partial_s2_compiled_cost.md`](df_partial_s2_compiled_cost.md)
- [`df_partial_s2_repeated_compiled_cost.md`](df_partial_s2_repeated_compiled_cost.md)

### RPEへの接続

- [`rpe_resource_accounting.md`](rpe_resource_accounting.md)
- [`rpe_hadamard_interrogation.md`](rpe_hadamard_interrogation.md)
- [`rpe_round_cost_connection_validation.md`](rpe_round_cost_connection_validation.md)
- [`rpe_hadamard_failure_validation.md`](rpe_hadamard_failure_validation.md)
- [`rpe_hadamard_proxy_resource_validation.md`](rpe_hadamard_proxy_resource_validation.md)
- [`rpe_allocation_sensitivity_validation.md`](rpe_allocation_sensitivity_validation.md)
- [`rpe_four_round_accounting_validation.md`](rpe_four_round_accounting_validation.md)
- [`rpe_four_round_phase_validation.md`](rpe_four_round_phase_validation.md)
- [`rpe_target_round_horizon_validation.md`](rpe_target_round_horizon_validation.md)
- [`rpe_delta_round_schedule_validation.md`](rpe_delta_round_schedule_validation.md)
- [`rpe_delta_compiled_cost_validation.md`](rpe_delta_compiled_cost_validation.md)

## 実行・運用

- [`server_parallel_validation_execution.md`](server_parallel_validation_execution.md)：共有CPU/GPUサーバー向けのbounded実行、checkpoint、resume、dry-run
- [`examples/parallel_validation_h4_q1_manifest.json`](examples/parallel_validation_h4_q1_manifest.json)：H4 q=1のdry-run用manifest例

## 発表資料と参考文献

- [`presentations/README.md`](presentations/README.md)：発表資料・構成案の位置づけ
- [`references/README.md`](references/README.md)：同梱した論文PDFの位置づけ
- [`rte_source_versions.md`](rte_source_versions.md)：RTE一次資料の版管理

## 状態の読み方

個別文書に数値があっても、それだけで現在利用可能とは判断しない。
再現可能性、失効、成果物の有無は[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)と
[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)で確認する。

## mainに収録した過去の成果

[収録済み成果と原SOURCE](research/main_collection_20261103.md)：第1回はTrack B G6の既存結果・文書・再現用SOURCE。元の科学的区分とSTOPを保持する。

[枝ごとの収録範囲](research/main_collection_20261103_branches.tsv)と[大きな原データの既存所在](research/main_collection_20261103_raw_locations.tsv)から、main内の成果と原snapshotを辿れる。

## main集約の版差と原SOURCE（2026-11-03）

[索引の統合・SOURCEの版差・保存原版](research/main_collection_versions_20261103.md)から、同名ファイルの各公開版とmainの配置を確認できる。
過去のSOURCE・test・結果は元snapshotの条件で読む。科学実行と元の研究testの再実行は行っていない。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。

## ローカルの未コミット資料からの追加収録（2026-11-03）

[原稿・レビュー・研究ノートと保存版の対応](research/main_local_results_collection_20261103.md)。原文の科学的採用や新規実行の認定ではない。


A1 v2の[payload保存修正・限定診断とSTOP](research/o2_a1_q1_q2l_20261106/REPORT_V2.md)（2026-11-06）：旧target同値は未閉鎖、CDF以降は未実行。元runnerの`--diagnose`、完全payload、人工29件と原v1保全を記録する。


A1の[旧opt2来歴・完全payload・環境の限定静的調査](research/o2_a1_opt2_provenance_20261106/REPORT.md)（2026-11-06）：記録SOURCEの回収先と未回収payload、元checkpoint抜粋、stdlib-only collector。科学再実行・library変更は0件、STOP維持。


A1の[新finite-target科学契約設計案](research/o2_a1_new_finite_target_20261106/REPORT.md)（2026-11-06）：actual-Q/selected full複素signalの定義、Raw/2層/5層/CV、公平なconfidence/全費用と後日の科学call。設計のみ、未採択・科学実行0件。


A1新targetの[N1限定実行](research/o2_a1_new_target_n1_20261106/EXECUTION.md)：既存DF/CDF/CV/policyを再利用するmodule・runner・人工test、SOURCE/環境lock。N1のみ、N2/N3は起動しない。


A1新targetの[N1結果・完全payload・correctness](research/o2_a1_new_target_n1_20261106/REPORT.md)（2026-11-06）：同じ一回の準備からCDF/Q/νと層質量・basis対応を取得。N1成功後STOP、N2/N3・測定・価格/費用は未実行。


A1の[N2限定SOURCE・state/engine検証](research/o2_a1_new_target_n2_20261106/EXECUTION.md)：N1保存payloadを再利用するmodule/runner/人工test、state inventoryと環境lock。N3なし、公開後STOP。


A1の[N2結果・state/engine保証範囲・STOP](research/o2_a1_new_target_n2_20261106/REPORT.md)：一意ground/state誤差を確認。総b_engineはUNKNOWN、N3未準備。SOURCE/payload/人工test/保全へ辿れる。


A1の[保存engine接続・一様誤差評価plan・人工correctness](research/o2_a1_engine_connection_20261106/REPORT.md)：最小adapter/test、接続/誤差計画/次回の限定科学scope。実上界未取得、N1/N2再計算・N3なし、公開後STOP。


A1の[一回の総engine誤差認証・全gate証明・STOP](research/o2_a1_engine_qualification_20261106/REPORT.md)：固定exact engine/理想モデルの条件付き上界取得。N3・費用評価は未実行。


A1 N3の[技術接続・科学条件草案](../docs/research/o2_a1_n3_connection_20261106/REPORT.md)：`a1_n3_connection.py`、人工専用runner/test、保存証拠への入口。人工46件PASS、実科学0・未採択・公開後STOP。


A1 N3の[採択済み一campaign・SOURCE/実行仕様](../docs/research/o2_a1_n3_science_20261106/EXECUTION.md)：追加driver/test、固定exact engine・実call境界・保存とmandatory STOP。

A1 N3の[一campaign中断結果](research/o2_a1_n3_science_20261106/REPORT.md)：取得済397 bit/価格、未完了77 trial、精度/CV比較未判定、journal保全とmandatory STOP。

A1 N3の[中断診断・再発防止・復旧2案](research/o2_a1_n3_interruption_20261106/REPORT.md)：journal専用runner `scripts/diagnostics/a1_n3_interruption_audit.py`、`a1_n3_interruption_diagnostics`/`a1_n3_lifecycle_observer`（src/trottertracks/algorithm_codesign）、人工test `tests/test_a1_n3_interruption_diagnostics.py`。直接原因UNKNOWN、新科学0、mandatory STOP。

A1 N3の[持続manager接続と案B草案](research/o2_a1_n3_manager_preparation_20261106/REPORT.md)：`a1_n3_cron_manager.py`/`a1_n3_managed_runtime.py`（src/trottertracks/algorithm_codesign）、人工入口 `scripts/diagnostics/run_a1_n3_manager_qualification.py`、unit `tests/test_a1_n3_managed_runtime.py`。旧数値SOURCE不変、科学0。


A1 N3案Bの[新しい一campaignの実行仕様](research/o2_a1_n3_B_science_20261106/REPORT.md)（2026-11-06）：seed20261107、旧397件を分離、旧数値SOURCEとqualified managerを再利用する薄い入口。SOURCE公開照合後に一回開始し、成功・中断でmandatory STOP。


2026-11-06 A1 N3案B：[修正入口・manager_run02・限定再実行](research/o2_a1_n3_B_retry_20261106/REPORT.md)。ユーザーは技術的な意図しない中断の修復/再投入も許可。科学条件・取得済bit・旧SOURCE/STOPは維持する。


2026-11-06 A1 N3：[範囲見直し・752-shot予備契約案・現行状態と停止手順](research/o2_a1_n3_scope_redesign_20261106/REPORT.md)。A1対B1-OLの段階的予備検証を優先するユーザー指示に沿う設計。具体shot/seedは未採択・科学実行0、人工11件PASS。現行jobの停止/変更/signal送信は0、現行jobは観測時RUNNING。


2026-11-07 A1 N3：[新Bの正規停止・保全と752-shot実行準備](research/o2_a1_n3_pilot_preparation_20261107/REPORT.md)。保存H4/1.00 Å/STO-3G/legacy DF12/L_D3/m0/q1/δ.02/r1/K2/atol0の新Bを親TERM一回で停止し、1339 bit/1318価格、既知RZ6,435,609・総費用UNKNOWNを保全。新pilotは準備版、人工34件PASS、科学実行0。seed/規則/科学実行承認待ちでSTOP。


2026-11-07 A1対B1-OL：[採択済み752-shot予備検証](../docs/research/o2_a1_n3_pilot_science_20261107/REPORT.md)。保存H4/1 Å/STO-3G/DF12/L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y、seed20261108。既存pilot driverとqualified managerを新activationで接続。人工テスト後SOURCE公開照合して一campaignだけ実行し、完了/中断でmandatory STOP。精度・期待費用保証は対象外。


2026-11-07 A1対B1-OL予備検証：科学SOURCE a139d7dからconsume-once managerで一回起動したが、資源判定で最初のtrial前に正規STOP。取得0/752、Born/compile/RZ0、bundle decode1。比較は未評価、科学retry0。停止原本と全会計を[結果報告](../docs/research/o2_a1_n3_pilot_science_20261107/REPORT.md)へ保全。次attemptは空き資源確認と別途明示承認が必要。C1凍結維持。


2026-11-07 A1/B1-OL：[752-shot pilotのCPU判定改善とattempt02](../docs/research/o2_a1_n3_pilot_admission_20261107/REPORT.md)。前attemptの0件STOPを保全し、同じ科学条件・seed/streamで一回再投入する。実入力前にCPU tickを二窓観測し、最大30分の有限資源待機を行う。科学的STOP条件・exact kernel・C1凍結を維持。


2026-11-07 A1/B1-OL：[752-shot予備検証attempt02の完了結果](../docs/research/o2_a1_n3_pilot_admission_20261107/REPORT.md)。保存H4/1 Å/STO-3G/legacy DF12/L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y、actual Q/保証ground/exact engineを維持。752 bit/価格を完備、A1 2,009,673 RZ、B1-OL 1,622,442 RZ、全3,632,115 RZ。両軸zero-variance保留規則により全体INCONCLUSIVE。人工52 PASS、旧SOURCE/STOPとC1凍結を保全、公開照合後mandatory STOP。本検証の精度・期待費用保証ではない。


2026-11-07：[層内control・層別＋CVの未採択設計](../docs/research/intrastratum_cv_design_20261107/REPORT.md)。固定理論8d8a290/752結果3c8e55a/N1 law744e1a0を静的確認。rotation fragment群、signed support、product label一致を候補化し、一層一特徴・224-shot paired診断を提案。追加測定なしのmetadata特徴とexact条件付き平均の式を整理したが、相関・分散/費用改善は未検証。実装・新科学計算・保存結果再解析・test実行0、正式採択なし、C1凍結・旧STOP保全、公開照合後mandatory STOP。


2026-11-07：[SCV最小実装・224-shot準備](research/scv224_preparation_20261107/REPORT.md)。rotation fragment群indicator/条件付き平均/h1 scalar fit/freeze/全5層paired評価を独立SOURCEへ追加。新規人工34＋再利用72＝106件PASS。実H4群・平均/moment・科学RNG/Born/compile/測定・保存結果数値再解析0。224 shots/seed20261109/streamは未採択案。旧SOURCE/結果/STOP・C1凍結を保全し、公開照合後mandatory STOP。


2026-11-07：[O2-SCV224採択済み一campaign](research/o2_scv224_science_20261107/REPORT.md)。保存H4/1 Å/STO-3G/legacy DF12/L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y、N1 actual Q/N2保証state/exact Fraction engine。群・平均一回、train64＋共通fresh評価160＝最大224 physical shots、seed20261109。旧SOURCEを変更せず薄いactivation、人工118件PASS。公開SOURCE照合後一回開始し、完了/中断でmandatory STOP。


2026-11-07：[O2-SCV224完了結果](research/o2_scv224_science_20261107/REPORT.md)。保存H4/1 Å/STO-3G/DF12/L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y。群S={df-fragment-3}、224 bit/価格完備、rank2/range合格。X分散点指標は実質不変、Yは約0.02471%低下。学習込み実現RZ1,079,241（train306,990、評価772,251）で、学習費回収の明確な兆候なし。精度/期待費用保証・正式GO/NO-GOなし。約24.26分、全process正常終了、旧SOURCE/結果/STOPとC1凍結を保全しmandatory STOP。


2026-11-07：[Signed support実H4 metadata-only限定検証](research/scv_signed_support_h4_metadata_science_20261107/REPORT.md)。
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


## O2-SCV 224/448の統合・次方向（2026-11-07、静的整理）

[統合報告](research/o2_scv_two_pilot_synthesis_20261107/REPORT.md)から証拠比較、要因仮説、候補/方向比較、最小検証草案へ辿る。提案は未採択、新科学計算・新control実装なし。


## PR改善の重複排除・C-S接口（2026-11-08）

[報告](research/pr_o1_m1_delta_cs_preparation_20261108/REPORT.md)：O1-M1の重複確認とC-S同位置residualの最小接口。既存F1/F2とmatched H4結果を再利用し、新規人工9件PASS、新科学計算0。M1全般のNO-GOや資源優位の認定ではない。

## C-S V0技術準備（2026-11-08）

[保存slot・coarse/finite/strict政策・人工18件・次回有限草案](research/cs_v0_technical_preparation_20261108/REPORT.md)。実H4の数値取得・合成・価格は未実施。

## C-S V0採択・activation（2026-11-08）

[採択契約・起動/保存/STOP・人工結果](research/cs_v0_activation_20261108/REPORT.md)。実科学未開始。旧技術binding/policy文書は複製せず606c40dを参照する。

## C-S V0 science STOP（2026-11-08）

[一campaignの結果・原本・schema不適合](research/cs_v0_science_20261108/REPORT.md)。Stage B binding TypeError、local spec0/strict0、価格/誤差は未評価。科学SOURCEと旧STOPは無変更、retry0。

[C-S V0 schema修復と同条件再実行](research/cs_v0_repair_20261108/REPORT.md)：旧run01を保全、新人工5件、科学条件は変更なし。

[C-S V0修復後の結果](research/cs_v0_repair_results_20261108/REPORT.md)：同じ条件で2spec/7keysを完了、全cap PASS。局所T削減なし、全体優位性未評価。

2026-11-08 O2：[MultilevelからSampling・Return結合への技術調査](research/o2_a2_j1_mechanism_20261108/O2_NEXT_ALGORITHM_REPORT.md)。A2はアクセス未完成、J1は新しい安価な取得構造未特定。科学結果とは区別する。

2026-11-08：[O2-A2のLCUアクセス・測定/費用会計・人工correctness](research/o2_a2_executable_access_20261108/REPORT.md)。新規9件PASS、実H4/compile/shot0。技術資格と科学的改善を区別。

2026-11-08：[O2全成果統合](research/o2_research_results_integration_20261108/O2_RESEARCH_SUMMARY.md)。初期再配分/G6–G10/Parent/C1からS2・752/SCV・A2 LCUまでの到達層・負結果・今後の探索領域。[証拠索引](research/o2_research_results_integration_20261108/O2_EVIDENCE_INDEX.md)。

2026-11-08：[O2新候補の操作・比較・次技術scope](research/o2_new_algorithm_candidate_exploration_20261108/REPORT.md)。label平均observableの測定と局所差分の別測定を設計。性能/新規性の採択・実装・新科学計算は行わない。

2026-11-08：[O2案Mの最小instrumentと人工correctness](research/o2_m_instrument_technical_validation_20261108/REPORT.md)。12件PASS、強いancilla-free/grouping baselineとの差と実H4 metadata-only次案。新科学計算0。

2026-11-08：[O2案M・n=0 header binderと一回metadata-only科学SOURCE準備](research/o2_m_n0_header_binding_preparation_20261108/REPORT.md)。literal adapter/元SOURCE対応、人工10＋関連3、未採択の有限実行契約。実H4の群適格性/群和は未取得。

2026-11-08：[O2案M・固定n=0 slotのmetadata-only結果](research/o2_m_n0_metadata_result_20261108/REPORT.md)。一回正常完了、NO_NONTRIVIAL_GROUP_IN_SELECTED_SLOT。literal十分条件の局所結果で一般NO-GOではない。

2026-11-08：[O2新機構の候補・技術選別・次の有限技術案](research/o2_new_mechanism_exploration_20261108/REPORT.md)。coherent label消去後の早期終了を1候補に絞り、既知法への優位/新規性は未認定。設計のみ。

2026-11-08：[案Eの実装・人工correctness・強いbaseline比較](research/o2_e_early_abort_technical_validation_20261108/REPORT.md)。人工16件PASS、同じearly LCUとの差0、実科学/実compileなし。

2026-11-08：[O2の非同値性を先に確認した候補選別](research/o2_distinct_mechanism_screening_20261108/REPORT.md)。還元された操作と未解決の情報アクセスを分け、実装へ進む候補は残さなかった。文書設計のみ。

2026-11-08：[O2共通basisの演算子接続・誤差・full費用条件](research/o2_common_basis_static_validation_20261108/REPORT.md)。SOURCE静的確認と紙上比較のみ。旧M singleton結果は維持し、群再生成・test・科学計算を行わない。
