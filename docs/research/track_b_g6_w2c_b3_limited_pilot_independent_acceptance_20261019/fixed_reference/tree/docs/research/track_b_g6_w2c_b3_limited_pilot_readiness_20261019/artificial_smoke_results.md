# 公開人工rawのsmoke

|対象|件数|今回の結果|
|---|---:|---|
|正常raw|12|VERIFIED_COMPLETE|
|障害raw|35|TECHNICAL_FAILURE|
|費用境界|46|VERIFIED_PARALLEL_COST_BOUNDARIES|
|parent crash費用|1|INSUFFICIENT_EVIDENCE、null維持|

各caseの分類・pending・入力hashはconsumer_smoke_results.json、詳細はconsumer_smoke_reports.tar.gz。全47件がCLI exit0で分類可能だったことは、科学実行の承認や全受入条件PASSを意味しない。failure prefixは正常費用比較へ含めない。未ACK creditは返却しない。

旧完全contract不足30件、P06通知不足11件、private authority/history不足、historical runtime不足を保持。failure時に成功receiptがないことは正常なSTOPの一部である。parent crashで完了費用を推定して埋めない。
