# Track A 共通ノート追補：受領・限定検証

基準 `4675730e7e40d25ba3618334a4a0e28b4bd2f145` から資料専用branchを作成。科学source/result branchをmergeしない。

- [共通入口](../../../../docs/research/track_a_shared_research_status.md)
- [確認結果・来歴・H6認可/時間方針](../../../../docs/research/track_a_server_information_update_20261015.md)
- [受領監査と時点snapshot](receipt_audit_v1.json)：指定A–Fと後続H6等11固定fileのGitHub/Git blob/server copy照合。公開情報projectionのみで実grantではない。
- [H6後続完了metadata受領](h6_completion_metadata_receipt_v1.json)：元runtimeの再収録なし。科学的費用解析・新採択0。
- [変更fileのbytes/SHA](changed_files_manifest_v1.json)：manifest自身と検証記録を除く変更fileを記録。自己参照hashを作らない。
- [限定検証](documentation_checks_v1.json)：新リンク、固定証拠、原本exact、変更allowlist、既存worktree保全。独立科学review/CIではない。

原本指示50,268 bytesは全文の機微情報確認後にexact copyを追加。指定local48,223 bytes版は未受領。旧公開47,275 bytes版との内容差だけを機械的diffへ保存する。
H6後続認可・seal・実行snapshotと4h政策差を索引化し、既存契約・runは不変。NPZ/runtime/checkpoint/ledger/venvは収録しない。

post-commit SHA/push結果とremote file receiptは最終チャット及びhome配下private publication receiptへ記録する。失敗時に本branch内の資料を公開済みと扱わない。新科学計算・H8変更・新grant・原稿改訂0。MANDATORY_STOP。
