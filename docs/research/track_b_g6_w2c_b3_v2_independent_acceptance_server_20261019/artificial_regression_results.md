# 人工検証の件数

|集合|PASS|FAIL/ERROR/SKIP|扱い|
|---|---:|---|---|
|固定旧runner・依存を揃えた原文実行|622|1/0/0|legacy_regressions_r60ws1q6。元FAIL保持|
|新versioned harnessによる旧10suite|623|0/0/0|co_qualname suffix期待値1箇所だけ明示補正|
|新規独立consumer tests|57|0/0/0|new_tests_delivery.json。51 mutation/negativeを含む|
|上記versioned回帰の重複しないID|680|0/0/0|623＋57。原文実行と二重加算しない|
|公開rawの独立検証|108|正常2／技術失敗40／証拠不完全66|unit test件数へ加えない|
|新規外部writer fault|2|S/P1とも期待した非成功状態|unit test件数へ加えない|
|Producer側既済unit|820（申告）|今回の件数へ不加算|Producer側資料の固定参照|

51 mutationにはSOURCE/runtime/contract/schema、scope/claim、counter/cursor/ticket、global/trial identity、RESERVE/COMPLETE重複、STOP後ENTER、新規予約、chain/canonical/prefix/tail、ACK/refund/byte cap/trace自己課金/diagnostic、summary/receipt/marker順序、CPU/wall、保存失敗の隠蔽等を含む。単に同じProducer関数を再実行して独立と呼んでいない。保存済み親/worker crash、in-flight drain、UNKNOWN、F01は108件の独立core経路でも確認した。

consumer processのProducer/scheduler/Admission/補助checker/science importは0。指定人工回帰・外部writer故障のProducer processでも禁止science moduleを検査し0を確認。raw元入力はread-only、SQLiteはimmutable read-only。mutationは新規copyだけで行った。

初期の依存不足・consumer修正前の診断は別名のログ/JSONへ保全し、最終SOURCEでPASSした件数と混ぜない。新rawの273ファイルは5partへ保存し、復元bytes/SHA256の往復一致を確認済み。
