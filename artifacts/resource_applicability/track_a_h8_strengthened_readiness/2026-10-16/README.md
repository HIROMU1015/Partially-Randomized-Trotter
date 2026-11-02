# H8決定論的対照の準備・V2設計終了

[H8準備状況・候補・承認境界・資源・STOP](../../../../docs/research/track_a_h8_strengthened_deterministic_readiness_20261016.md)／[V2 NO_GO終了・代替案の技術的可否](../../../../docs/research/track_a_v2_residual_design_closure_20261016.md)。

- [未承認計画と旧32/追加152候補](execution_plan_UNAPPROVED_v1.json)
- [無効なgrant template](grant_template_INVALID_v1.json)：approved=false、未seal、SOURCE/command=null、CPU[]。有効grantではない。
- [既存入力・export・固定Git/sourceのhash receipt](existing_evidence_receipts_v1.json)：配列・Gate復元や費用再解析なし。
- [V2現行fit設計の終了決定](v2_design_closure_v1.json)
- [主担当の限定技術検査](technical_validation_v1.json)／[旧worktree・grant等の保全](preservation_receipt_v1.json)
- [公開ファイルbytes/SHA](publication_files_v1.json)

```bash
python3 -P -B scripts/check_track_a_h8_strengthened_plan.py
python3 -P -B -m unittest discover -s tests -p test_track_a_h8_strengthened_plan.py
```

repo rootで実行する。stdlibと提案metadataだけを使い、科学provider、NPZ、実費用・signal、Qiskit、実workerを起動しない。実分子再現・科学runner完成・科学実行承認を意味しない。正本の既存成果は固定commit参照で再利用し、raw・runtime・private path・credential・有効grant/ledgerを収録しない。公開後mandatory STOP。
