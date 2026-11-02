# H8 bias証拠監査bundle

[資料入口](../../../../docs/research/track_a_h8_bias_evidence_audit_20261017.md)。
固定184候補のmetadata監査のみ。実数値の非相殺診断結果ではない。

- `evidence_audit_v2.json`：最終metadata監査。172保存成分＋12端点復元候補、PR union6欄の参考確認、来歴限定。
- `evidence_audit_v1.json`：reference/provenance補足前のmetadata監査履歴。上書きしていない。
- `candidate_axis_inventory_v1.csv`：候補/軸別368行。科学bias値は転記しない。
- `producer_lineage_v1.json`：固定producerの数式・source、共用reference/scalarと独立性の限界。
- `diagnostic_plan_UNAPPROVED_v1.json`：実装計画。実H8数値実行・grantなし。
- `unresolved_evidence_v1.json`：旧12復元、成分照合、誤差証拠、prefix19、B2/費用gateの未解決事項。
- `limited_validation_v1.json`：人工30件、実数値再解析0。
- `source_closure_v1.json`：source-tree SHAとfiles。metadata監査器・人工prototypeのSOURCE commitを結合済み。実数値runnerのSOURCEとは区別する。
- `parallel_isolation_v1.json`：B2保全・独立snapshot・既存Git無変更。
- `publication_inventory_v1.json`：公開可能な軽量準備filesのbytes/SHA。private rawは除く。

`references/evidence_inputs_v1.json` に固定commit/path/bytes/SHAの入力索引を保持する。
private old summaryは旧D0の固定input receiptに結合し、Phase A gzipは元JSON bytesのhashへ照合した。
科学array・実runtime/checkpoint/ledger/credentialを新たに格納しない。

資料専用branchへのcommit固定のみ実施し、push・正式共通branchへの統合は未実施。実数値解析の実行承認なし、runtime_authorization=false。mandatory STOP。

SOURCE：`efc7e00f5032d3003347e843b4929eb57b7ed47e`。`commit_preparation_receipt_v1.json` は今回のcommit準備照合記録。`parallel_isolation_v1.json` は元の監査段階の履歴を保全したもの。
