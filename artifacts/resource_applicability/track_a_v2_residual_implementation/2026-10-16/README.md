# V2保存RZ残差runner：実装・人工検証・未承認契約

[実装報告](../../../../docs/research/track_a_v2_residual_implementation_20261016.md)。公開実装SOURCE：`5703dbda58ebb8b7a9d18dce18ba33ed45f3c11c`。新36＋observer/cleanup7、計43 artificial tests PASS。実データfit・残差・compileは0。

現在の262 scope entriesはすべて独立来歴が不足しており、fit対象0。保存receiptへの結合が必要で、metadata条件付き診断もまだ実行していない。数値実行は未承認、公開後mandatory STOP。

|ファイル|意味|
|---|---|
|[policy.json](policy.json)|解析範囲・閾値・登録順序に関する固定政策と資源上限|
|[grant_UNAPPROVED_DRAFT.json](grant_UNAPPROVED_DRAFT.json)|承認false、SOURCE=null、CPU[]、command=null。有効grantではない|
|[source_input_binding.json](source_input_binding.json)|9 source・5 input・runtime closure・private configのhash。数値計算を行わないseal|
|[provenance_audit.json](provenance_audit.json)|262 scope entriesのmetadata分類。残差や較正結果ではない|
|[artificial_validation.json](artificial_validation.json)|43人工検証の範囲、local checkとしての限界|
|[technical_validation.json](technical_validation.json)|SOURCE/input・拒否gate・既存ファイル/worktree保全と実行0の確認|
|[publication_manifest.json](publication_manifest.json)|公開ファイルのbytes/hash台帳（自己参照を除く）|

既存理論832行と保存接続10,480行は正本commit`7e41670565d9284da9aded94063be591fdcaa3dd`のまま再利用する。private config/registry、分子配列、runtimeファイル一覧は含めない。前段の[計画artifact](../../track_a_v2_residual_readiness/2026-10-16/README.md)は履歴として保持した。
