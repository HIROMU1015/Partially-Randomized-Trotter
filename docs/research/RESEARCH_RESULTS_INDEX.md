# 研究結果への統一索引

[Track D B6b＋B/C統合準備](track_d_b6b_bc_integration_20261108/REPORT.md)：Referenceのideal coherenceを任意入力へ導出し、whole-V-ray条件とDF半path/最終真空条件を区別。SOURCE13114b2b、人工/回帰24件と人工P0 16 wrapper postcheck PASS。B4-2保存Nの条件付き共通利用・13-round価格coverage・薄い会計adapterを整理。actual N/長q期待価格は未資格、実H4の新計算/価格再解析/N算定/全RPE数値会計0。公開照合後mandatory STOP。

[Track D C7-3結果](track_d_c7_3_q32_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、独立q32/cut16/8 path。sampling/raw SOURCE35bfd、承認済み高速化compile/post SOURCE94e94。旧raw8再利用・新raw/sampling0、全32 P0/post・pairedχ8 PASS。CX差は全8本で−6960、RZ/size/各depthも有限全8本で減、qubits9→17。一次12量は探索的幅内、別population-line6量幅外。統計/全law/q≥64/N/全RPE未資格、旧STOP/raw非公開を維持、公開照合後mandatory STOP。

[Track D C7-2構造検証](track_d_c7_2_cx_structure_20261107/REPORT.md)／[q32未採択案](track_d_c7_2_cx_structure_20261107/C7_3_EXECUTION_DRAFT.md)：SOURCE34929f8a、54人工P0と位相/observable検査PASS、既存回帰12 PASS。加法性と単純lane depth式の反例、実H4全path bridge未取得。分子再計算/価格再解析0、mandatory STOP。

[過去のブランチURLに対応する固定commit・保存原版の案内](branch_reference_navigation_20261103.md)。

[保留7枝の再現用ファイル・保存別版の追加収録（11月4日）](priority7_collection_20261104.md)。

既存成果のmainへの収録は[収録済み成果と原SOURCEの案内](main_collection_20261103.md)を参照。第1回はTrack B G6の結果・文書・再現用SOURCEである。

2026-11-02 JST。複数ブランチに分かれた既存成果を、総括・結果・SOURCE・raw・検証状態へ辿るための入口である。
外部リンクは固定commitを使用し、下記の閲覧用snapshotと科学計算のSOURCEを区別する。
本索引の追加は、科学的な採否・保証・既存STOP・実行許可を変更しない。

## 最初に読む順序

1. [研究全体の科学的到達点と証拠の地図（11月1日）][GLOBAL]で系列間の関係を確認する。
2. [主張・証拠・未解決課題の対応表][GAPS]で、取得済みの証拠と未解決事項を分ける。
3. 下表から研究別の総括を開き、その総括の条件・SOURCE表を読んでから結果を使う。
4. 11月2日の後続記録は、この索引のH4 P3・研究Cの行を追加で読む。11月1日の総合地図だけを最新の全記録とは扱わない。

旧PR-2／PR-3、P-A〜D、FR、N3・表現探索等の経緯は総合地図とその固定出典から辿る。
基盤資料と過去の検証は[このcheckoutの研究概要・現状](研究概要・現状.md)と、
[固定snapshotの事前検証カタログ実施証拠索引][BASE_CATALOG]から辿る。
これらの旧snapshotの「現在」「次段階」は記載時点の意味で読む。

## 研究系列から探す

| 系列 | 総括・主張と限界の入口 | 後続記録・読み分け |
|---|---|---|
| DF資源評価・高次PF・長時間（Track A） | [H4/H6/H8の証拠統合][A_SUM] | [長時間B2の報告][A_B2]と[B3の報告][A_B3]。geometry・分子・比較集合・標本を各報告で確認する |
| Pauli-JW資源評価 | [研究の総合資料][PAULI_SUM]、[11月1日の論文用証拠統合][PAULI_PAPER] | [主張・根拠・図表の対応表][PAULI_FIGURES]。固定条件付き比較と一般的・保証付き主張を分ける |
| H4 P3・DF/Pauli matched対照 | [Stage 3 run03の結果報告][H4_REPORT] | [Stage 4条件付き会計の契約・設計][H4_NEXT]は結果と別に読む |
| D内部PF・outer分割・B2-D（研究B） | [最終成果統合・一時保留handoff][B_SUM] | [Full DF-Yoshida4 q4の実行報告][B_REPORT]等、総括の各系列に続く。総括に記載された一時保留・再開条件を保つ |
| return集約・有限bit・生成法（Track B） | [成果の利用範囲マップ][T_SUM]、[G6の主張・証拠対応表][T_CLAIMS] | [C1の資源証拠と限界][T_C1]、[M1-Sの統合資料][T_M1]。G6・M1・C1それぞれの条件・保証・STOPを分ける |
| Hamiltonian表現・PR-aware DF（研究C） | [10月31日のEvidence Mapと次の判断][C_SUM] | 11月2日の[PR-only reduced限定実行][C_PREV]と、後続の[小step限定実行][C_RUN]を別記録として読む。後続の完了で旧STOP・資格判定を上書きしない |

「研究B／B2-D」と「Track B」は別の入口である。同じBという文字だけで結果・契約・保証を混ぜない。
本表は元報告の道案内であり、数値の再比較や研究方針の新規採択は行っていない。

## 結果からSOURCE・raw・検証状態へ

下表は代表的な原記録への直接リンクである。複数の実験を持つ系列では、総括のSOURCE・原本対応表から各実験へ進む。

| 系列・代表記録 | 結果・公開raw／パケット | 科学SOURCE・入力identity | 検証状態・適用範囲 |
|---|---|---|---|
| DF長時間 | [B2の公開成果物][A_B2_RAW]、[B3の公開成果物][A_RAW] | [B3 scientific provenance][A_SOURCE]。B2は元報告のSOURCE登録を参照 | [B3工程記録][A_STATUS]と[B3報告の限界][A_B3] |
| Pauli-JW | [論文用証拠統合の出典E1以降][PAULI_PAPER]。例：[H6追加検証の公開成果物][PAULI_RAW] | [H6追加検証のSOURCE登録][PAULI_SOURCE]。H2・H4・event拡張等は証拠統合のSOURCE表で個別に特定 | [主張・根拠・図表対応表][PAULI_FIGURES]。例のH6だけで系列全体を代表させない |
| H4 P3 Stage 3 run03 | [結果報告][H4_REPORT]、[公開成果物][H4_RAW] | [SOURCEと入力の記録][H4_SOURCE] | [verification][H4_STATUS]。Stage 4の[設計][H4_NEXT]へ実行完了を外挿しない |
| B2-D・Full DF q4の例 | [実行報告][B_REPORT]、[原本パケット][B_RAW] | [binding][B_SOURCE]。他系列のSOURCEは[総括の登録表][B_SUM] | [status][B_STATUS]、[STOP][B_STOP]。他系列のraw・STOPも総括に登録されている |
| Track B | [利用範囲マップ][T_SUM]と[C1の原本・資源証拠対応][T_C1]から各パケットへ | [G6の保存SOURCE例][T_G6_SOURCE]。M1のSOURCEは[M1-S統合資料][T_M1]で別に特定 | [G6主張・証拠表][T_CLAIMS]と[C1の限界][T_C1]。共通の科学SOURCE・保証へまとめない |
| 研究C・小step限定実行 | [実行報告][C_RUN]、[run01原本][C_RAW] | [固定科学SOURCE commit][C_SOURCE]、[初期入力identity][C_IDENTITY] | [postflight][C_STATUS]と実行報告の資格・STOP。[旧reduced実行][C_PREV]も保持 |

公開rawのリンクは、そのGit snapshotにあるパケットを指す。非公開原本・別hostのcache・欠落資料の扱いは各総括の制限事項を参照する。
公開ファイルの存在、人工検査PASS、数値correctness、科学的保証、最終的な優位性は別の証拠段階である。

このcheckoutの[VALIDATION_STATUS](../../VALIDATION_STATUS.md)と
[validation manifest](../../artifacts/validation_manifest.json)は、このcheckoutに登録された証拠の状態を確認する入口である。
他ブランチの全結果を包括する単一の最新台帳とは扱わず、上表の各snapshotの原記録も併読する。

## 開発入口を探す場合

| 開発系列 | 既存の入口資料 | 結果との区別 |
|---|---|---|
| P1・DF | [DF/H4 pipelineの実装報告][P1] | plan・mock／人工配線の入口。科学結果の取得や実行許可を意味しない |
| P2・Pauli-JW | [Pauli-JW pipelineの実装報告][P2] | 既存kernelへの開発接続。研究結果は上の総括・原本へ進む |
| P3・B2-D | [B2-D研究entry][P3] | plan／mockと既存科学stageを分ける |
| P4・Track B | [Track B研究entry][P4] | G6・M1-S-gamma・M1-S-pi3を独立に扱うreference／descriptor／mock入口 |

## 固定した閲覧用snapshotと保持先

以下のcommitは文書を探すためのsnapshotであり、各計算の科学SOURCEと同一とは限らない。
保持先名は2026-11-02のリモート照合時点の参照情報で、リンク先はブランチ名に依存しない固定commitである。

| 用途 | 照合時点の保持先 | 固定snapshot |
|---|---|---|
| 全体地図・Pauli・H4 P3 | `codex/h4-p3-poststage3-design-20261102` | [60515a33ce62](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/60515a33ce6231e9ff4ffdb40124ed845373351c) |
| DF長時間B3 | `codex/h46-long-time-b3-20261026` | [ebe9e2aa2c83](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed) |
| B2-D／研究B | `codex/p3-b2d-research-entry-integration-20261028` | [f9adc2ce1c37](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/f9adc2ce1c373c7ba311864c733ac6f40b575fbc) |
| Track B | `codex/p4-track-b-research-entry-integration-20261028` | [33424d205aa9](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/33424d205aa918498e1ef54beb94ab67612bdec8) |
| 研究Cの総括 | `codex/research-c-evidence-synthesis-20261031` | [354a090da388](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/354a090da388c6d9718624bb4525ff3c72a22022) |
| 研究Cの11月2日後続 | `codex/research-c-pr-small-step-execution-20261102` | [b8f1429887af](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b8f1429887af5e1255c0801908a7bc31123d15dd) |
| H4 Stage 4の設計 | `codex/h4-p3-stage4-conditional-accounting-20261102` | [a1d3f32aba64](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/a1d3f32aba6416fd1e200cd66a97d1a825b5dd73) |
| DF開発入口 | `codex/p1-df-pipeline-h4-integration-20261027` | [af54aeb7ebac](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/af54aeb7ebac042a3ca02379d61516ac635feef9) |
| Pauli開発入口 | `codex/p2-pauli-jw-pipeline-integration-20261027` | [04f4cd9cdc8e](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/04f4cd9cdc8e348a9f0a182ead9cd07f1ee69696) |

上表より前のSOURCE・結果commitも、その出典文書に記録されたidentityを使う。
新しい閲覧snapshotのcommitを、過去に実行した科学SOURCEへ付け替えない。

## 旧リモートブランチ名から探す

[保存台帳](branch_archive_20261102/README.md)で、旧branch名から固定SHA・資料path・成果物root・保存tagを確認できる。
過去のbranch名を削除しても、原記録のidentityと履歴をこの台帳から辿る。
tagへの移行は科学的採否・元status・STOP・計算SOURCEを変更しない。

## 更新する範囲

新しい実行報告・総括が公開された場合は、該当する系列の総括、結果・SOURCE・raw・検証記録、保持先snapshotを対応させて更新する。
元報告の条件・未資格事項・STOPと、論文化や次段階の未採択事項は保持する。
数値を引用する場合のmodel・geometry・basis・rank／rank policy・分割・delta窓等は、リンク先の科学条件を確認する。
本索引のリンク確認は、科学性能の再検証や重要科学レビューではない。

[GLOBAL]: pr_repository_scientific_landscape_20261101.md
[GAPS]: pr_repository_claim_evidence_gap_matrix_20261101.md
[A_SUM]: track_a_h468_paper_evidence_integration_20261019.md
[A_B2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[A_B2_RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25
[A_B3]: ../../artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md
[A_RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26
[A_SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/scientific_provenance.json
[A_STATUS]: ../../artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/stages_complete.json
[PAULI_SUM]: pauli_jw_resource_evaluation_summary.md
[PAULI_PAPER]: pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md
[PAULI_FIGURES]: pauli_h2_h4_h6_claim_figure_map_20261101.md
[PAULI_RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27
[PAULI_SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/SOURCE.json
[H4_REPORT]: ../../artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run03/REPORT.md
[H4_RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/60515a33ce6231e9ff4ffdb40124ed845373351c/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run03
[H4_SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/60515a33ce6231e9ff4ffdb40124ed845373351c/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run03/source_and_inputs.json
[H4_STATUS]: ../../artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run03/verification.json
[H4_NEXT]: h4_p3_stage4_conditional_accounting_contract_20261102.md
[B_SUM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9adc2ce1c373c7ba311864c733ac6f40b575fbc/docs/research/research_b_final_outcomes_and_pause_handoff_20261026.md
[B_REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9adc2ce1c373c7ba311864c733ac6f40b575fbc/docs/research/research_b_full_df_yoshida4_q4_execution_20261026.md
[B_RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/f9adc2ce1c373c7ba311864c733ac6f40b575fbc/artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once
[B_SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9adc2ce1c373c7ba311864c733ac6f40b575fbc/artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/binding.json
[B_STATUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9adc2ce1c373c7ba311864c733ac6f40b575fbc/artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/status.json
[B_STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9adc2ce1c373c7ba311864c733ac6f40b575fbc/artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/STOP.json
[T_SUM]: track_b_paper_claim_survey_20261018/results_usability_map.md
[T_CLAIMS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33424d205aa918498e1ef54beb94ab67612bdec8/docs/research/track_b_g6_v1_2_draft_review_integration_20261022/CLAIM_EVIDENCE_MATRIX.md
[T_C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33424d205aa918498e1ef54beb94ab67612bdec8/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md
[T_G6_SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[T_M1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md
[C_SUM]: research_c_evidence_synthesis_and_decision_readiness_20261031.md
[C_RUN]: research_c_pr_small_step_execution_20261102.md
[C_RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b8f1429887af5e1255c0801908a7bc31123d15dd/artifacts/research_c_pr_small_step/2026-11-02/run01
[C_SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a426d62f27182526baa9479cd7cf8a953f92a9dc
[C_IDENTITY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b8f1429887af5e1255c0801908a7bc31123d15dd/artifacts/research_c_pr_small_step/2026-11-02/run01/initial_identity.json
[C_STATUS]: ../../artifacts/research_c_pr_small_step/2026-11-02/execution_postflight.json
[C_PREV]: research_c_pr_reduced_limited_execution_20261102.md
[P1]: ../integration/main_p1_c_price_cohorts_20261103.md#p1
[P2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/04f4cd9cdc8e348a9f0a182ead9cd07f1ee69696/docs/integration/p2_pauli_jw_pipeline_20261027/IMPLEMENTATION_REPORT.md
[P3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9adc2ce1c373c7ba311864c733ac6f40b575fbc/docs/research/b2d_research_entry_20261028.md
[P4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33424d205aa918498e1ef54beb94ab67612bdec8/docs/research/track_b_research_entry_20261028.md
[BASE_CATALOG]: prevalidation_catalog_evidence_map.md

## main内の版差・統合索引

[同名37 pathの整理と88原版](main_collection_versions_20261103.md)で、資料索引の統合とSOURCE配置を確認できる。原データ所在と枝別の収録範囲は[main収録記録](main_collection_20261103.md)を参照する。

## サーバー側に残っていた公開成果

[サーバー1・2の照合から追加した成果とSOURCEの対応](main_server_results_collection_20261103.md)を参照する。元の科学SOURCE・条件・statusは各原記録を使う。

## ローカルに残っていた原稿・受領レビュー・SOURCE

[ローカル資料の収録と原版対応](main_local_results_collection_20261103.md)から、Track A原稿v0.2、Track B・研究C等の受領レビュー、研究ノート、SOURCE入口と保存別版へ進める。未コミット原本の収録であり、元の科学SOURCE・条件・status・STOPを維持する。

## サーバー1の準備済み原本からの追補

[過去の草稿・技術SOURCE・索引の保存原版](server1_onsite_collection_20261103.md)から今回の出典対応へ進める。現在の仕様・科学的採否・実行許可は変更しない。

## サーバー2の追加収録（2026-11-03）

[固定SOURCE・保存成果の追加収録と版対応](server2_main_collection_20261103.md)。保存原版と現行実行pathへの採用を分け、元の科学SOURCE・入力・status・STOPを維持する。

## 追加34枝の公開成果（2026-11-04）

[報告・再現用SOURCE・保存原版と元snapshotの対応](unlisted34_collection_20261104.md)から辿る。元の科学SOURCE・入力・status・STOPを維持し、選択収録と実行採択・枝全体のmergeを区別する。
