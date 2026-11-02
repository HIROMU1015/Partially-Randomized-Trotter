# H4保存証拠の原因調査

[専用報告書](../../../docs/non_df_h4_evidence_investigation_20261023.md)を正本とする。対象結果`6892e95ffdfc78e75425e7432e9e4ccdb9958df1`、科学SOURCE`0c329b30a81accb35c212be8defa66fee0dd0f12`。

- `github_retrieval.json`：GitHubのbranch/commitと12固定fileの実取得記録。
- `source_excerpts.json`：科学SOURCEの行番号付き静的抜粋。
- `precision_state_inventory.json`：4方式×5精度・40軸の保存値とstatusの転記。N/Gはnullのまま。
- `monitor_inventory.json`：入力native終了、元progress/finalizerの末尾、caseの終了証拠欠落、recovery観測の転記。監視消失原因UNKNOWN。
- `findings.json`：A/B/C/D/E分類と根拠path。
- `next_execution_contract_DRAFT.json`：未承認・未発行・未実行の次段階案。
- `auxiliary_existing_records.json`：既知の`/tmp`補助記録のcopy。元native証拠の代用ではない。
- `preservation_before.json` / `preservation_after.json`：元SOURCE738 filesと、元結果・claim1147 filesのhash。
- `verification.json`：GitHub/ローカル・保存status・CSV・SHAの構造検査結果。
- `report_checks.json`：報告書・JSON・DRAFT flags・保存結果不変の最終構造検査。

`structure_checks.py`はstdlibの読取検査だけで、科学moduleを実行しない。保存値から新しい比率・N/G・allowance・信号を評価しない。元入力・raw・STOP等を書かない。既存の調査出力も上書きせず、新しい出力ファイルを排他的作成する。調査後の保全確認は次のとおり。

```bash
python3 -B artifacts/non_df_h4_evidence_investigation/2026-10-23/structure_checks.py --after
```

既定の読取先は元の`non-df-h4-execution-foundation-20261023` worktree。別環境では`--source-worktree <対象commitの不変checkout>`を明示する。再検査時も既存の`preservation_after.json`を上書きしない。新しい調査出力領域を使用する。

この検査のPASSは静的調査・構造整合性だけを指す。元caseのnative PASS、厳密certificate、N/G確定、詳細科学レビュー、最終evidence freezeを意味しない。専用local commitのみ、新規pushなし。mandatory STOP。
