# 文書索引

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

## Track C H4（2026-11-06）

[PF native境界・合成/残差・Processing pilot](research/track_c_h4_pilots_20261106/REPORT.md)。条件付きSOURCE mean診断、normalization負担、compiler精度不一致とSTOPを記録。最終総Gや優位性の確立ではない。

## Track C compiler A correctness（2026-11-06）

[実装・人工44ケース・H4限定検証](research/track_c_compiler_a_correctness_20261106/REPORT.md)。LD3 S2/LD12 Y4、h=.02、q1の全joint operatorとX/Yが数値PASS。既にcontrolledな回路に適用し、共通global phaseを許容する。全18roundと価格は未資格、mandatory STOP。

## Track C compiler A 18round（2026-11-06）

[局所区間上界と全18roundの誤差会計](research/track_c_compiler_a_18round_qualification_20261106/REPORT.md)。H4 S2 LD3/Full Y4 LD12、h=.02。全36条件の条件付き数値判定は成立。全RPEの保証はUNQUALIFIED、価格追加なしでSTOP。

## Track C 保証検証A・C-N価格B（2026-11-06）

[総合報告](research/track_c_guarantee_cn_price_20261106/REPORT.md)、[保証](research/track_c_guarantee_cn_price_20261106/A_REPORT.md)、[価格](research/track_c_guarantee_cn_price_20261106/B_REPORT.md)。固定行列モデルでの18round条件、元H表現の未資格、有限cohortの価格差0、長q総G未判定を区別する。

## Track C2-B

[固定Full Y4/LD12/h=.02の18round。共役block合成配分は登録非一様policy比で総T 0.5470%減。強baseline下界には勝てず優位性未資格、STOP。](research/track_c2_b_pilot_20261106/REPORT.md)

## Track C2-B強baseline

[数値上の総T削減0.514072%が残るが、全alpha下界とのgapが残り最適性未確定。登録72配分・保存7603 recordだけで検証しmandatory STOP。](research/track_c2_b_strong_baseline_20261106/REPORT.md)

## 新A reference干渉限定pilot（2026-11-06）

[REPORT](research/pr_reference_interference_pilot_20261106/REPORT.md)：同じDF trajectoryの信号復元を確認。Simon–Love後も短qのCX/depth差が残るがRZ/sizeは増加、全18roundは未確定。mandatory STOP。
