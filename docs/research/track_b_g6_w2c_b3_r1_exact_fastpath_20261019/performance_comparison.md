# ローカル人工性能比較

環境：Intel Core i9-13900、Linux6.8.0 x86_64、Python3.10.12、32 logical CPU。
開始時available RAM約49GiB、disk空き約370GiB、load約1.17を確認。変長exact Fraction/JSONのCPU実装で、GPU/native依存は導入していない。
サーバーEPYC/Python3.12の既存倍率とは別の測定。こちらのCPU倍率を本番/サーバーへ移さない。

| 人工operand幅 | 構成 | CPU中央値s | Wall中央値s | paired CPU倍率 | paired Wall倍率 | CPU倍率範囲 | 最大child HWM MiB | 正規化bytes |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 96 | baseline | 0.9953 | 0.9940 | 1.0000 | 1.0000 | 1.0000–1.0000 | 18.08 | 5458203 |
| 96 | fused | 0.9742 | 0.9733 | 1.0216 | 1.0213 | 0.9857–1.0407 | 17.67 | 5458203 |
| 96 | recurrence | 0.9795 | 0.9782 | 1.0057 | 1.0056 | 0.9922–1.0274 | 17.54 | 5458203 |
| 96 | combined | 0.9654 | 0.9639 | 1.0327 | 1.0327 | 1.0136–1.0379 | 18.03 | 5458203 |
| 256 | baseline | 2.9022 | 2.9012 | 1.0000 | 1.0000 | 1.0000–1.0000 | 20.74 | 10846850 |
| 256 | fused | 2.8774 | 2.8762 | 1.0075 | 1.0076 | 1.0005–1.0206 | 20.87 | 10846850 |
| 256 | recurrence | 2.9037 | 2.9026 | 1.0019 | 1.0020 | 0.9936–1.0061 | 20.41 | 10846850 |
| 256 | combined | 2.8874 | 2.8879 | 1.0069 | 1.0069 | 0.9988–1.0131 | 20.95 | 10846850 |

各幅・各構成5反復。倍率は同repeat baselineとの比の中央値で、CPUはparent＋join済み全owned child。
Wallもdispatch〜全join。1測定8 pair×X/Y×64=1024人工trials、variant順を固定seedで並べ替え。
各pairは新規owned childを逐次実行。pool/reuse/pass同時化/8pair並列schedulerは導入していない。
warm-up各4/axisは除外。readbackと正規化比較、初期importはtimed interval外。
処理件数当たりCPU/wall（µs/trial）、raw output bytes範囲、baseline RSS、HWM、全pair cost/traceは[JSON](performance_results.json)と[gzip全測定](benchmark_results.json.gz)へ保存。

serialization単独microは4000ops×5反復、fused CPU倍率 **1.379×**。
組合せの全体CPU/Wallは96bitで約1.033×、256bitで約1.007×。
冪漸化式のみは約1.006/1.002×で小さく、反復rangeは1を跨ぐ。256bit組合せも1を跨ぎ、安定した全体優位は未確立。
serverのrecurrence値にはfusedが含まれたが、今回は4構成を分離した。相乗効果や倍率の積を仮定しない。
5反復は記述的観測でconfidence intervalではない。

## Exact比較と費用scope

各runのraw file/chain SHAを再読照合。run間では実測generator_exclusive_nsだけを除き、残る全payload/bytes/chain、summary/bit/R-C-Aを一致確認した。
非決定的clockを含むraw files同士が一致したとは言わない。fake clockの別testsでは未正規化の全bytes/hashも一致した。
HWMは各child自身のbaseline/finalから得た最大値で、同時aggregate/固有物理memoryのpeakではない。flushはfsyncではない。
原precisionのroot/Evidence費用を含む人工enclosureだが、実G7 packet/law/source guardの費用・発生率は再現していない。
FunctionObserver/Meterを削減せず、このbenchmarkはbaseline形のworkloadだけ。instrumented本番速度や31,710,848 calls所要時間は外挿しない。
