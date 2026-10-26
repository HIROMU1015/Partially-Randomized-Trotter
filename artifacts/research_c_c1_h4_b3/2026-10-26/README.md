# H4 B3：技術準備のみ

[技術報告](../../../docs/research/research_c_c1_h4_b3_preparation_20261026.md)／[未承認65-call契約案](../../../docs/research/research_c_c1_h4_b3_science_contract_draft_20261026.json)。新21＋旧101＝122 unique人工／回帰PASS、SOURCE35／直接保存input19と旧bindingの受動preflight PASS。

- `technical/SOURCE.json`：旧SOURCE32＋新B3 module／CLI／test。
- `technical/validation.json`：122 unique PASSの内訳、fixture修正履歴、byte保全、環境、実H4科学call0。
- `technical/tests_B3_final.txt`：新21 PASS。
- `technical/tests_existing_and_fixture_diagnostic.txt`：旧real-Qiskit74はPASS、新17 PASS／4fixture FAILを含む混在batch。最終fixture修正後は上の新21 PASS。
- `technical/tests_legacy_mock.txt`：隔離processの旧mock27 PASS。
- `technical/byte_preflight.json`：分子配列／QPY decodeなし、scientific_calls0。

B3の費用・correctnessは未測定。`run01`は未作成・未消費、別の明示的科学実行許可までmandatory STOP。
