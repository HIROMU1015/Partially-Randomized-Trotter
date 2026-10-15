# Track A V2 H6 実行準備（2026-10-16 JST）

[主要報告](../../../../docs/research/track_a_v2_h6_cost_binding_readiness_20261016.md)。

- `execution_plan_draft_v1.json`：固定scope、未承認CPU/RAM/AS/RSS/wall/storage ceiling、STOP。
- `grant_draft_v1.json`：approved=false、runtime_authorization=false、CPU[]、command=null。新grantではない。
- `input_binding_v1.json`：既存input・SOURCE・private configのbytes/SHA。private path/rawは公開しない。
- `review_provenance_v1.json`：利用者が指定・採択したClaude Level 1資料の原本copyとhash。
- `resource_snapshot_v1.json`：共有hostの読取snapshot。CPU割当・将来のadmissionではない。
- `artificial_tests_v1.txt` / `artificial_result_v1.json`：主担当の人工試験。外部CI・独立科学レビューではない。
- `technical_audit_v1.json` / `file_receipts_v1.json`：限定差分・JSON/link/原本保全の検査と公開payload digest。

`candidate_root_plan_v1.json`は既存104候補の保存root対応。`static_adapter_receipt_v1.json`は既存7root記述の静的検査で、primitive/期待費用0。

旧grant/ledger/実行済marker・科学SOURCE/raw・H5は変更しない。新科学計算・新grant消費・subagents0。公開/remote照合後mandatory STOP。
