# H8 rare-tail / selection readiness — UNAPPROVED

[統合報告](../../../../docs/research/track_a_h8_rare_tail_selection_readiness_20261018.md)が入口。

- 準備SOURCE：432b76e0b915c6b145a7d275b4cdd42953b762bd。production dispatch未接続。
- 36固定Git file＋pilot closure8/補助1をmetadata検査。実H8 science0。
- 73人工/mock PASS。実H8の確率・費用・SE・順位は新算術せず保存literalを転記。
- `evidence_audit_v1.json` / `fixed_evidence_inventory_v1.json`：Git bytes/SHA、law/seed/pair、旧STOP・grant/ledgerの公開証拠。
- `saved_law_literals_v1.json` / `selection_metadata_v1.json` / `*_saved_literals.csv`：保存law、事前/事後選択、元SE・shot・costの転記。CSVの数値文字列は評価しない。
- `SOURCE_closure_v1.json` / `technical_validation_v1.json` / `synthetic_tests_73.txt`：準備SOURCEと作者の技術検査。native成功・独立レビューの証拠ではない。
- `stage_contracts_UNAPPROVED.json` / `grants_INVALID.json`：4つの任意後続stage案。valid grant/予約/one-shot/sealなし。各stageは別承認、次へ自動移行しない。
- `existing_ledger_binding_v1.json`：読取専用の過去receipt。control376/376、過去STOP・未返却予約は保持。新予算の承認や予約を意味しない。
- `draft_execution_rejection.txt`：実科学入力を開く前のexit2拒否。
- `preservation_v1.json`：独立clone限定変更と旧blob/索引履歴の保全。
- `github_reference_verification_v1.json`：参照commitのexact fetch。新資料の公開成功とは別。

実行確認（人工/metadataだけ）：

```bash
python3 -B scripts/run_h8_rare_tail_selection_readiness.py --synthetic
python3 -B -m unittest discover -s tests -p test_h8_rare_tail_selection_readiness.py -v
python3 -B scripts/audit_h8_rare_tail_selection_evidence.py
```

science flag/config/grantを指定すると、実入力を読む前に拒否する。見かけ上validなgrantでも本SOURCEではdispatchできない。新sampling/compile/statistics/感度/費用再会計は行わずmandatory STOP。
