# Hybrid/RTE比較：実行前の保存証拠棚卸し

[準備報告](../../../docs/pauli_hybrid_rte_preparation_20261025.md)。実分子の新ランキング・比・sampling・compileは0。
`execution_plan_DRAFT_DISABLED.json` は実行無効。添付指示§8の一括実行承認を待つ。

- `candidate_evidence_inventory.csv`：候補/ε/cohort単位1580行。保存値の転記だけ。
- `coverage_by_precision.json`：15条件の群別候補数、保存eligibility、G有無、確認欠測。minやG比較ではない。
- `source_manifest.json`：固定ref/GitHub URL/Git blob/SHA256と既存ファイルまたはsnapshotの場所。
- `archive_members.json` / `evidence/`：旧確認plan/results等のSHA付き回収。旧runを復元・再開しない。
- `prospective_SOURCE_inputs.json`：再利用kernel、準備helper、各分子model/reference、既存runtimeの参照。
- `inventory_verification.json`：転記・構造検証。人工テストは `tests/test_pauli_hybrid_rte_preparation.py`。

CSV空欄はnull。元H4 CSVの文字列`null`と元H2/H6空セルを同じ欠測へ接続した。0/∞を補完しない。
`B_exact` と `B_saved_decimal` は元資料の異なる保存形式であり、空欄を新数値計算で補完していない。
`C_cos_SE` / `C_sin_SE` は価格SE、`G_SE` はGのSE。`price_n` はpaired価格標本数で、量子shotではない。
探索・確認・historical_baselineのcohortを分ける。H4旧baselineは256本、実探索140候補とは別。
SOURCE/result commit、detail_source、price_evidence_locatorで元候補・価格へ辿れる。

棚卸しだけの再生成：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/inventory_pauli_hybrid_rte.py
```

このコマンドはnumerical helperや科学runnerをimportしない。実行承認前に実データへmin/ratio helperを適用しない。
無条件の資源優位・母平均winner・実compiledρ・費用tail保証を新たに認定しない。
