# 回帰集合を分けた最終結果

|集合|PASS|FAIL|ERROR|SKIP|
|---|---:|---:|---:|---:|
|旧原版10suite（ローカルPython3.10）|623|0|0|0|
|既存Storage V2独立|57|0|0|0|
|新bundle独立|165|0|0|0|
|重複しない独立ID|**845**|0|0|0|
|別枠Producer Storage V2 unit|41|0|0|0|
|固定server label補正版：対象77のローカル実行|76|1|0|0|
|portable label補正版：対象77|77|0|0|0|

旧原版と補正版77を二重に足さない。共通9suite546と固定補正版77を合成した集合は622/1、portable版と合成すると623/0。補正版623全体を別に全再実行したという主張ではない。serverの歴史原版622/1・補正版623・828/107/41/57は固定参照のまま。

新165には公開43・公開35 overlay・fresh8・新mutation62と正常control/静的契約/CLI/memory等を含む。新mutation62は58 integrity mismatch/4 incompleteで全拒否。2件のwitness mutationのraw採取再実行、live-runtime/archived-runtime modeの再実行をunique数へ加算しない。Source/ID/SHA、全ログ・失敗traceとliteral-only portable sourceを保存した。

通常consumerと新165経路、旧623/57、別Producer41/fresh probeで禁止科学importは0。新SOURCE通常経路ではProducer/scheduler/Admission/self-checkerも0。新SOURCEを追加したためrepository全test discoverは行わず、承認された人工集合だけを実行した。
