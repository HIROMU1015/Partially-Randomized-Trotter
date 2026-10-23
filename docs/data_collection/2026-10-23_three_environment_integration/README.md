# 3環境の論文用データ統合索引（2026-10-23）

既存公開packetを固定commit・path・SHAで参照する差分索引。新しい科学結果・数値表のコピー・論文草稿は作成していない。既存paper manifest、CEM、証拠台帳を中心に再利用した。新成果物は専用branchのローカルcommitまで。公開承認は未確認でpushしない。**mandatory STOP**。

## 執筆開始時の読む順序

1. [採択済みpaper manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f75a7c25d584b69d90de454731a7f343a421e059/artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/paper_manifest.json) を中心索引とする。同commitの `claim_evidence_matrix_v2.csv`（21行）、`evidence_ledger_v4.csv`、`source_catalog.json`、`paper_data_manifest.csv` を再利用する。
2. 同packetの `main_table` が指す固定 `1923fce8f7461513d1bc47b5061e87cd70917e87` のH468主表とFigure2を維持する。主表SHA256は `2d910a62451d6a0b3247751625e85c8aebd181b4dea6b493b94a469d4e49d86c`。旧Figure2の34ファイルと主表を含む既存41参照を `preserved_paper_files.csv`→`references.csv` で追跡できる。`e7a63b2...` の統合主表は別版として比較単位・audit_statusを読む。無断差替えなし。
3. 補足の共通PF588候補は同f75採択packetの `dataset.json.gz`、`candidate_index.csv`、全式15/同一式45の保存比較JSONを参照する。科学結果固定commitは `90f1cd55a0d963953696bd0a27d9d041ff939bb7`。これを新しい独立確認結果や全PR/PF同域最適化と呼ばない。
4. `claim_evidence_overlay.csv` と `scientific_profiles.json` で主張の比較単位・科学条件・限界を確認し、`M1_M3_review_mapping.md` と `gaps_and_requests.json` を読む。レビュー原文のSHA/非公開状態は `review_source_receipt.json`。新規採択・最終freezeは行っていない。

## 入力3環境

|環境|固定commit|入口|
|---|---|---|
|GPUA100-gleap-01|e0c942fbb5dd8741dda69af0933fce88e8eaea91|[README](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0c942fbb5dd8741dda69af0933fce88e8eaea91/docs/research/environment_evidence_gpua100_20261023/README.md)|
|GPUA100-gleap02|b8bd64339169ecdf61b90fb7b575a3937fdc0c8c|[README](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b8bd64339169ecdf61b90fb7b575a3937fdc0c8c/docs/research/environment_evidence/gpua100-gleap02_20261023/README.md)|
|abe-Precision-3260|f284c501bd8332fa5e8fc098b5424e34c666174f|[README](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f284c501bd8332fa5e8fc098b5424e34c666174f/docs/data_collection/2026-10-23_abe-precision-3260/README.md)|

3環境49系列の元manifest行を `research_index.csv` で横断する。元実行hostと所持環境を混同しない。入力44ファイルのうち43はGitHub raw本文のbyte一致、残る大きなGPU01 artifact_indexはcommit treeとlocal blob・宣言SHAの一致で確認した。元catalog3,538参照中3,498件の固定local blobはSHA一致、40件は元object未取得。未取得本文を検証済みとはしない。

## SOURCE・結果・監査・既存機械可読データ

`source_result_audit.csv` は49系列の記録済みSOURCE・入力・結果・監査を参照IDで対応付け、`references.csv` が固定commit/path/SHA256/blob SHA1/byte数/URLと取得検査段階を示す。`series_reference_links.csv` は役割別の補助対応表。全catalogの再複製はせず、空欄の詳細を含め以下の固定元表へ戻る。原表にないSOURCEや入力は推測しない。

- GPU01 `artifact_index.csv` と `recorded_source_bindings.csv`：2,834参照とSOURCE fieldの来歴。
- GPU02 `public_index.csv`、`science_source_index.csv`、`source_result_links.csv`：275参照、SOURCE40行（37公開byte対応、H6run02の3行は元SOURCE未取得）。
- ABE `evidence_index.csv`、`source_result_links.csv`、`required_input_candidates.csv`：429参照、125SOURCE結合、旧27候補の不足入力。

|用途|固定取得先（本文・値・単位は元資料を利用）|
|---|---|
|採択済み共通PF固定域|f75a7c25... 採択packet：dataset.json.gz / candidate_index.csv / comparisons_all_formulas_15.json / comparisons_within_formula_45.json / old_vs_common_15.json / denominator_decomposition_10.json|
|H468独立固定endpoint主結果|1923fce8... 主表main_table.csv。e7a63b2... の track_a_h468_paper_evidence_integration/main_table_proposed.csv は対応監査付き別版|
|高次PR探索会計|8613b3ec... track_a_h246_saved_cost_results/2026-10-21/formula_comparison_30.csv。取得SOURCEとshot契約は別参照|
|元PR同士確認とH6失敗|b8bd643... GPU02 candidates/h246_confirmation_run01/ の6ファイル。元commit0d3578...は今回取得不能、公開転記のSHAは検査済み|
|H6run02単価だけのwitness|同b8bd643... candidates/h6_recovery_run02/axis_costs_64.csv と receipt_projection.json。元SOURCE未取得・end_audit不在・集約PASS未認定|
|別再構成PR同士確認|3eca4614... track_a_h246_reconstructed_confirmation/2026-10-22/fixed_evaluation/{results.json,group_statistics_6.csv,paired_samples_118.csv} と各系end_audit|
|S2元H2回収＋新cohort|a590214d... docs/progress/track_a/figures/pr_pf_s2_recomputed_20261022/figure_data.csv/json、coverage、physical_validation_audit。H6/H8補助FAILを条件付き表示|
|N固定CNG比較|b81d4096... docs/progress/track_a/figures/pr_pf_analytic_vs_circuit_CNG_20261022/figure_data.csv/json。N_circuit等nullを維持|
|H8希少層の終了方針|862d732f... docs/research/h8_rare_rte_closure_policy_20261021.md。条件付き有限標本の既存採択範囲、追加sampling不要|
|H2新共通COMPAREの制約|4008ce45... track_a_h2_accepted_compare/2026-10-23/REPORT.md。診断a6788bf...とは別、COMPARE科学実行なし|

短縮commitは表示用。取得には `references.csv` の40桁commit・repository相対path・SHAを使用する。CSV/JSONを別表へ再転記せず、その固定取得先を使う。gzip本体はUTF8 connector制約でremote download未実施。GitHub treeに結合したlocal blob/manifest SHAは一致し、関連CSV/JSONは別途raw取得した。詳細は `integrity_audit.json`。

## 不足と最新状態

O00002はGPU02の公開owner witnessで期待SHA一致・発見済み。過去GPU01/ABEの未発見記録は当時の範囲の事実として保持する。今回private原本を独立再hashしたわけではなく、絶対locatorは非公開。発見だけで新H2 COMPAREの完全identity成立としない。

旧27候補のpreparation、旧H6/H8新A shot行、H6run02終了監査は、それぞれ元cohortの完全再現・軸別shot分解・元run02集約確認を主張する場合の条件。別SOURCE/cohortの公開限定結果を引用する場合に一括回収を前提としない。H6/H8 FAIL、H6旧親process終了未知、H8公開終了receipt未再検証、未観測rare費用は制約を残す。`gap_source_crosswalk.csv` は3環境の既存不足29記録を、最新状態・主張依存性を記載した `gaps_and_requests.json` へ対応付ける。

`duplicate_version_checks.json` はbyte重複291群、同一path異版3群を元catalog行で示す。byte重複を独立標本増加と扱わず、同じsystemやpathだけでcohort・SOURCEを結合しない。

## 公開・保全・停止

既存3環境commitはGitHubから取得できる。各bundle内の承認待ち/未push記録は過去時点の記録で、後続公開の承認receiptまで今回取得したという意味ではない。新統合成果物への承認は引き継がない。公開候補とリスクは `publication_status.json`。ローカルLevel1原文・private原設定・seed原本・raw/archive・内部絶対path・認証情報は含めない。

検査はmetadata、構造、SHA、重複、公開状態のみ。科学SOURCE、保存数値、grant、ledger、marker、旧STOP、主表、Figure2を変更していない。元worktreeの進行中作業を監視・停止・変更せず、4次PF検証へ介入しない。全private原本のbyte再監査や独立科学レビューを実施したとはしない。

次担当は論文データ整理担当：本索引と既存f75 manifestの限定を付け、採用するclaimの固定資料を参照する。ユーザー：新統合bundleを公開したい場合に限り対象範囲を承認する。承認後のCodexのpush・remote commit/byte照合は別作業であり、現時点では実施しない。追加科学計算・最終freeze・草稿執筆の自動開始なし。**mandatory STOP**。
