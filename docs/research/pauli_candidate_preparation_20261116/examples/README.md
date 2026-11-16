# 実入口からの保存metadata応答例

固定技術SOURCE `6e4b6bd6a491815c2a3cb8f809cbfc0240b60fdb`。

- `REQUEST_SAVED_AND_PENDING.json`：LD1〜9のmetadata準備要求。新しいH4辞書は実行しない。
- `RESPONSE_SAVED_AND_PENDING.json.gz`：その実CLI応答。保存LD1/2/3/5と未登録5分割を区別。
- `REQUEST_PUBLISHED_KAPPA.json`：B2の公開LD3 κ×2とその保存packet。
- `RESPONSE_PUBLISHED_KAPPA.json.gz`：公開configurationと一致、13round REUSE。
- `.CONFIGURATION.json`：入口が返すcanonical本体の実ファイル。元の固定locatorも応答に保持。
- `.SAVED_PACKETS.json`：信号入口へ渡す保存packet一覧。

圧縮はこの公開例だけの標準gzip。通常の入口出力はJSONである。展開後の`configuration_file`等の絶対pathは
実行時A worktreeの場所であり、別checkoutではrequestをCLIへ渡し直して現地pathを得る。
[短い受入結果](../ACCEPTANCE.json)には圧縮応答から抽出したID・完成区分・round再利用を記録した。
