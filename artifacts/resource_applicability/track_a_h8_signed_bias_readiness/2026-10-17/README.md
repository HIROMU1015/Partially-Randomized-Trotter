# H8 signed成分復元・数値整合性検証：実行未承認

[資料入口](../../../../docs/research/track_a_h8_signed_bias_readiness_20261017.md)。

- `execution_plan_UNAPPROVED_v1.json`：SOURCE・全26入力receipt・許容差・対象・禁止scope。
- `grant_UNAPPROVED_v1.json`：approved=false/runtime_authorization=false。入力root/output/fresh gate未指定。
- `source_closure_v1.json`：実装4＋既存入力索引2のbytes/SHA/Git blobとSOURCE。
- `input_binding_v1.json`：旧監査を再利用した来歴。今回の実値照合結果ではない。
- `preparation_environment_v1.json`：既存Pythonを変更せず使用。科学compilerは呼ばない。
- `limited_validation_v1.json`：新38＋既存30、全68人工検査PASS。
- `draft_rejection_v1.json`：固定runnerが実draftを入力読取前に拒否した結果。
- `preservation_receipt_v1.json`：B2 STOPと既存資料の読取保全確認。
- `preparation_summary_v1.json`：未実行scopeと次の担当。
- `publication_inventory_v1.json`：今回変更した公開filesのbytes/SHA（自身を除く）。

実H8の復元・照合結果は0。経験的数値整合性と厳密bias保証は別である。
B2の完了/PASSをこの準備や将来のsigned照合の前提にしない。費用・shot・PR比較・順位付けは対象外。
