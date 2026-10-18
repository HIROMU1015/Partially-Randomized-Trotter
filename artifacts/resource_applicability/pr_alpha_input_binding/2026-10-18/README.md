# PR α入力seal（未承認）

[報告](../../../../docs/research/pr_alpha_input_binding_20261018.md)。最新正本は[run03](run03/)。B-1は14/14、B-2は3,660完全＋60欠測。原本19件、人工62＋CLI8件PASS。実データの数値解析0、有効production grant0、mandatory STOP。

- [B-1 binding](run03/B1_input_binding_UNAPPROVED.json) / [保存h/N一覧](run03/B1_saved_shot_scope.csv)
- [B-2 binding](run03/B2_input_binding_UNAPPROVED.json)
- [コピー検査](run03/copy_verification.json) / [原本receipt](run03/source_input_receipts.json)
- [人工検査](synthetic_tests.json) / [CLI](run03/CLI_checks.json)
- [metadata準備の履歴](metadata_preflight_events.json) / [保全](preservation_audit.json)

root直下の旧bindingはrun01の不完全join、run02は依存限定前の履歴。すべて保全し、実行対象は最新のprivate run03 planだけ。private raw・実行plan・有効grantはGitに含めない。B-1/B-2は別途明示承認を必要とする。
