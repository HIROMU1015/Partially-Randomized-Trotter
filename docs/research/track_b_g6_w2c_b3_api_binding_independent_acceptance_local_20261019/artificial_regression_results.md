# 人工回帰結果

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

|検証集合|今回の実行結果|件数の扱い|
|---|---|---|
|既存10 suite|623 PASS、fail/error/skip0|今回ローカル原版を再実行|
|既存独立Storage V2|57 PASS、fail/error/skip0|分離runtime・science guard、今回再実行|
|既存独立Bundle|165 PASS、fail/error/skip0|保存43/35/fresh8等を含む。fixture root修正後|
|今回独立API|130 PASS、fail/error/skip0|Replay19、negative40、Static12、Contracts20、Cases4、SavedNegative35|
|今回独立test計|**975 PASS**|845＋130。再実行attemptを重複加算しない|
|Producer候補test|73 PASS、fail/error/skip0|独立test数へ加算しない。SOURCEは固定のまま|

別のcase-level offline集合：port4、公開port mutations22、固定Bundle43＋mutations35＝104件。test subset等と重なるため975へ加算しない。Direct controlsのgate9、32競合、20 binding negatives、親/spawn guard12、ACK unknownも別の観測として保存。

今回のProducer73 command：疎runtimeで `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover -s tests/tracks/algorithm_codesign -p test_g6_w2c_b3_api_binding_v1.py -v`。real科学moduleはguardでload前拒否する。

既存623は基点内limited-pilot packetの `run_independent_tests.py`、165は基点内Global Bundle packetの同helperで、新復元public/fresh inputsと新出力を指定。57は基点に保存された旧fixed testとV2 input/rawを専用runtimeへ配置し、srcを今回worktreeの独立consumerへ向けて実行。旧SOURCE/fixtureは変更しない。全実行logを `regressions/` に保存。

server歴史報告「新73/22/追加4/既存補正版828」、前回local845は過去の参照として区別。今回845は再確認できたが、server828や追加4を今回そのまま再実行したとは表示しない。co_qualname labelの原版/補正版は上書きせず、今回はPython3.10原版623と旧独立165/57を使った。旧3.12向け補正版のlabel差は旧packetに保全。

[最終SOURCE test実行](regressions/new_independent/summary.json)、[旧V2](regressions/baseline_V2_57.json)、[閉鎖人工controls](regressions/direct_closed_controls.json)。新規science・native処理0。ローカル実行をCI/別物理hostによる再現と主張しない。
