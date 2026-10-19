# 最終人工回帰とconsumer修正

最終9 suite: **540 PASS = 既存437 + 新規103、FAIL/ERROR/SKIP各0**。`artificial_test_results.json` にcollected/passed ID、SOURCE SHA、log SHA、science-module import監視を保存。ローカルtest結果でありCI・外部再現ではない。producer481件・旧339/356件を加算していない。

新103件は公開正常12＋障害35の分類とcost境界、F01 flush行非commit、重複FAIL/END通知、unknown/未ACK保守会計、P01–P11 pending、P/PF/S/SF、変造・欠損・schema/SOURCE/runtime/offset/state/receipt/STOP/crash偽成功、normalized gate履歴の不正、CLI/no-overwrite/input read-only、archive traversal/link/duplicate、bounded-memoryを含む。既存437件のfake scope reuse・retry/resume拒否・旧F01/one-shot/codec/observer等も再実行した。旧ledger regressionはv3 private atomic historyの代替にしていない。

開発中にconsumer側で検出して修正した点:

- 旧control validatorの「failure通知は先頭連続」という仮定。v3 log cap下ではsubsequenceとなる。原noticeとの一致を確認して明示normalizeし、履歴欠損はpendingにした。
- 障害reportに成功専用measurement_conditionを要求していた。固定SOURCEのmode定義とsettingsから識別し、その証拠種別を明示した。
- tarfile r|gzの内部decompress bufferが高圧縮率入力で増える。GzipFileのbounded readとstream tarを組み合わせた。2 MiB→32 MiBの人工memberでtracemalloc peakは約4.1 MiBのまま（実測v3_bounded_memory.json）。科学計算のspeedup主張ではない。
- v3 protocolが許すFAILED-before-ENTERを、旧coreのstart下限へ無条件適用しない。observed startを増やさず範囲とpending履歴を保存する。

開発初回99件は未修正consumer/fixtureにFAIL/ERRORがあり、修正後99件PASS、その後境界4件を追加した最終103件PASS。開発ログと最終ログを区別してgzip保存した。最終SOURCEのdigestはsource_inventory.jsonとtest reportに結合される。
