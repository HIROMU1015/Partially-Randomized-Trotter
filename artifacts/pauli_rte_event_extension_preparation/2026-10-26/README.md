# H4/H6 純RTE総event数拡張の準備（実行無効）

[準備報告・承認対象](../../../docs/pauli_rte_event_extension_preparation_20261026.md)。基点288d2206。新しい実分子科学call0、保存値再解析/新N_G/再ランキング/実データ図0。

- `execution_plan_DRAFT_DISABLED.json`：初期5・最大12候補、適応条件、最大20 target作用/4704 trajectory/9408 axis compile。execute_enabled=false。
- `candidate_plan.csv`：候補の設計値だけ。新N/Gは未取得/null。`PREPARATION_STOP.json`：科学実行承認待ち。
- `fixed_github_documents.json`：実取得した4固定GitHub資料と指定Git objectの一致。
- `inputs.json`：H4/H6を分けた同一raw model/reference・既存結果の役割/SHA。原本は参照のみ。
- `saved_baseline_quotes.csv`：既存の選択済み確認行20件の転記。新min/G/比/統計は計算していない。
- `inputs/h4_mean_d3.py`：H4固定commit d3fdfadの高速Arb codeの原bytes snapshot。旧moduleは変更しない。
- `prospective_SOURCE_inputs.json`：準備code/input/無効契約の固定。承認後にfresh driver/active batch/SOURCEを固定する。
- `runtime_bindings.json`：compilerとArb数値環境の分離。共有package変更なし。
- `artificial_*tests.txt`、`reused_integer_tests.txt`：人工12件PASS。初回人工import失敗も別logに保全。
- `artificial_mean_inputs.json`：2qubit人工データだけ。H4/H6 raw stateは含めていない。
- `verification.json`：構造・重複・人工検証の結果。技術PASSは科学性能や実行許可を意味しない。

研究仮説への結論は未取得。H2/既存Hybridの新科学実行・旧STOP再開は範囲外。承認原文の保存やgrant等は不要で、ユーザーのチャットでの明示許可を待つ。
