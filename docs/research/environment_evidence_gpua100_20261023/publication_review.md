# 新規公開候補と承認状態

**公開承認未確認。ローカルcommitで停止し、pushしない。**
根拠は今回の依頼第5節「対象ファイルの公開承認が確認できる場合のみGitHubへpush」「公開承認が確認できない新規ファイル…ローカルcommitでSTOP」。
repositoryのpush権限や、過去の別bundleのpublication_scope_auditを、今回新規作成した索引への承認として代用していない。

## 明示的にstageする候補

|候補|開示内容・残るリスク|
|---|---|
|本ディレクトリのREADME、series_manifest、worktree_catalogue|hostname、branch名、既存研究系列・履歴の存在。環境識別情報として公開確認が必要|
|artifact_index.csv、recorded_source_bindings.csv|既存資料のrepository相対path・固定commit・SHA・サイズ。ローカル限定40参照の存在とfingerprintも含むため、その開示確認が必要。本文は含まない|
|remote_retrieval_checks.json|既存公開12ファイルとSOURCE8 commitの取得・照合metadata。機密本文は含まない|
|gaps_and_requests.json、private_custody_summary.json|既存の不足・FAIL・未解決事項とprivate保全の件数/容量。内部絶対パス・原本別fingerprint・seed値は含まない|
|verification.json、SHA256SUMS、MANDATORY_STOP.json|今回の技術確認・候補byte identity・停止状態。独立科学レビューや公開承認receiptではない|
|PROJECT_MAP.md、docs/README.mdの各追記|新しい所在索引へのリンクのみ|

private設定原本、seed値・seed復元情報、token/認証情報、私有内部絶対パス、非公開grant/receipt、private raw/archiveは候補から除外した。
既存の公開済み結果・入力・図・SOURCEは複製していない。科学値、単位、数値条件、順位は変更していない。
SOURCEのcommit参照は保存manifestのpointerを保持し、今回実際に実行したSOURCEと表現しない。

## 原本の保全と検証限界

原本の詳細path・個別SHA・前後照合結果は、今回の公開候補ディレクトリの外にある本人ローカル非公開監査へ保存した。
その監査はstageしない。公的receiptと原private byteは別証拠であり、公開索引だけで原seed/runtime/rawを独立再構成できるとはしない。
全50,159原ファイルの前後hashと既存worktreeのtracked状態を照合した。
50,158件は一致。今回の編集対象外だった主worktreeのAGENTS.mdに報告規則追加1件を検出し、改変せず保全した。科学原本の変更は検出していない。
結果とguideの前後SHAはverification.jsonに記録する。
静的な索引作成とbyte照合だけで、既存科学STOPを解除する新grantやretryは作成していない。

公開判断が必要なのは上記の**新規候補だけ**。既存公開資料の重複push、元ローカルbranch全体のpush、既存の完成branchへの上書きは行わない。
この文書は承認依頼のための具体的候補表であり、承認済みという意味ではない。
