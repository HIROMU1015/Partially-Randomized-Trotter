# V2保存来歴の回収・照合（数値解析なし）

[報告書](../../../../docs/research/track_a_v2_provenance_reconciliation_20261016.md)。262 entriesのうち25件が条件付きmetadata結合済み、237件が未解決。PR158件のseed来歴を回収したが、fit適格stratumは0件。H8の全PRとH6確認PRは固定holdout側が0件であり、条件を変更していない。

|ファイル|内容|
|---|---|
|[reconciliation_summary.json](reconciliation_summary.json)|回収・未取得の内訳、新private registry SHA、manifest参照、実行0|
|[entry_reconciliation.json](entry_reconciliation.json)|262 entriesごとの測定/未測定、回収できた項目、未解決項目|
|[provenance_audit.json](provenance_audit.json)|固定残差libraryのmetadata-only分類。fit結果ではない|
|[fit_count_precheck.json](fit_count_precheck.json)|固定train/holdout件数。来歴無視の上限診断は実行適格性と分ける|
|[source_semantics_receipts.json](source_semantics_receipts.json)|Git blobで確認したSOURCE/path/bytes/SHA。全binary/basis同値の証明ではない|
|[technical_validation.json](technical_validation.json)|入力・旧SOURCE/保護ファイル/worktree保全、科学実行0|
|[publication_manifest.json](publication_manifest.json)|公開ファイルのbytes/SHA台帳（自己参照を除く）|
|[collect_saved_task_metadata.py](collect_saved_task_metadata.py)|task/worker/manifestの読取専用収集手順。新metadata出力だけを作る|
|[reconcile_saved_metadata.py](reconcile_saved_metadata.py)|CSVと原receipt・seed規則・archive/profile/law記述を照合する手順。科学providerはimportしない|

更新private台帳を旧実行configへ設置していない。旧SOURCE・input・config・grant・ledgerを保全、新config/grant・fit・残差・compileは0。公開後mandatory STOP。
