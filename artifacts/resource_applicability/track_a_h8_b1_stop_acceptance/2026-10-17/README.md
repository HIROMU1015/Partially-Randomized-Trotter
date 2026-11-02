# H8 B1 実行済み受入契約の比較証拠

[資料入口](../../../../docs/research/track_a_h8_b1_executed_contract_comparison_20261017.md)。
正本結果 `83a3c4a6688e66576f85471bf41a45e29cbfb705` は完了済み。元B0 run02はSTOPのまま。
このbundleは両状態と契約差分の照合資料であり、正本変更・新承認手続き・B1再実行を行わない。

- `acceptance_contract_comparison_v1.json`：比較専用。既存認可は再開せず、新規runtime_authorization=false/CPU[]/command=null。
- `evidence_inventory_v1.json`：既公開14資料の固定commit/path/bytes/SHA。原資料は再commitしない。
- `technical_acceptance_check_v1.json`：保存metadataの整合性。新たな数値検証結果ではない。
- `bounded_original_bytes_receipt_v1.json`：B0原本53/B1原本34の限定streaming SHAと既存grant記録。
- `implementation_comparison_v1.json`：実装A/Bの静的差分。統合・変更・新選定なし。
- `source_closure_v1.json`：資料照合用のSOURCE。正本の数値SOURCEとは別。
- `limited_validation_v1.json`：人工metadata21件、数値fit/holdout/compile0。
- `preservation_v1.json`：既存作業場所のHEAD/status保全確認。
- `publication_inventory_v1.json`：この作業の公開対象とbytes/SHA（inventory自身を除く）。

読取照合を再現する場合、上記固定Git objectsを取得したcloneで次を実行できる。科学処理は呼ばない。

```bash
python3 -P -B scripts/check_h8_b1_stop_acceptance.py \
  --repository . \
  --inventory artifacts/resource_applicability/track_a_h8_b1_stop_acceptance/2026-10-17/evidence_inventory_v1.json \
  --comparison artifacts/resource_applicability/track_a_h8_b1_stop_acceptance/2026-10-17/acceptance_contract_comparison_v1.json
```

原NPZ・runtime・checkpoint・ledger・grant raw・認証情報は本bundleへ追加していない。M1-S V0成果物は含まない。
