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


[A1 N3の事前判定方針・期待compiled RZ費用保証計画](research/o2_a1_n3_decision_expected_rz_20261106/REPORT.md)（2026-11-06）：正式契約03928ce5と整合、結果未閲覧の設計・人工検査。実現/期待費用、未完了/劣位を区別。実科学実行・条件採択なし。
