# 高速化prototypeの実測

全て非採択。既存科学SOURCEは変更していない。下表は8 worker、各3反復の中央値。

| Workload | Variant | Wall s | Total CPU s | Wall speedup | CPU speedup | Worker HWM MiB |
|---|---|---|---|---|---|---|
| mixed | reference | 0.4614 | 2.7800 | 1.000× | 1.000× | 20.9 |
| mixed | recurrence | 0.4459 | 2.6984 | 1.035× | 1.030× | 20.9 |
| mixed | gmp | 0.4600 | 2.7604 | 1.003× | 1.007× | 23.4 |
| mixed | orjson | 0.4574 | 2.7312 | 1.009× | 1.018× | 23.4 |
| mixed | orjson +64KiB | 0.4480 | 2.6741 | 1.030× | 1.040× | 23.4 |
| heavy | reference | 0.6027 | 2.3905 | 1.000× | 1.000× | 20.9 |
| heavy | recurrence | 0.5669 | 2.2543 | 1.063× | 1.060× | 20.9 |
| heavy | gmp | 0.4486 | 1.8305 | 1.344× | 1.306× | 23.4 |
| heavy | orjson | 0.4497 | 1.8359 | 1.340× | 1.302× | 23.4 |
| heavy | orjson +64KiB | 0.4478 | 1.8284 | 1.346× | 1.307× | 23.4 |

recurrenceは級数の冪の漸化式＋fused canonical、gmpはGMP級数＋fused canonical、orjsonはGMP級数＋native JSON。combinedの効果を単独変更の効果と混同しない。light/mixed系では初回importや残るgenerator/evidence処理が支配し、GMPのend-to-end利益は小さい。heavyではGMP候補でwall約1.34×、総CPU約1.30×。native JSON追加のend-to-end利益は小さく、GMPのみとの差を優位と断定しない。追加libraryによりworker HWMは約20.9→23.4 MiB。出力の本質内容と正規化bytes/hashは全variantで一致。

## Hot-pathの分離測定

| Synthetic operation | Variant | CPU µs/op | CPU speedup vs reference |
|---|---|---|---|
| Rational 32 bit/J32 | reference | 564.47 | 1.000× |
| Rational 32 bit/J32 | recurrence | 506.69 | 1.114× |
| Rational 32 bit/J32 | gmp | 423.75 | 1.332× |
| Rational 96 bit/J32 | reference | 1708.36 | 1.000× |
| Rational 96 bit/J32 | recurrence | 1574.07 | 1.085× |
| Rational 96 bit/J32 | gmp | 1450.87 | 1.177× |
| Rational 256 bit/J32 | reference | 7350.63 | 1.000× |
| Rational 256 bit/J32 | recurrence | 6892.65 | 1.066× |
| Rational 256 bit/J32 | gmp | 5130.58 | 1.433× |
| serialization | reference | 63.10 | 1.000× |
| serialization | fused | 46.95 | 1.344× |
| serialization | orjson | 20.76 | 3.039× |
| io | reference | 128.64 | 1.000× |
| io | fused | 106.31 | 1.210× |
| io | orjson | 81.58 | 1.577× |
| io | orjson_buffer | 72.64 | 1.771× |
| observer | none | 1.66 | 1.000× |
| observer | stage | 8.35 | 0.199× |
| observer | function | 44.02 | 0.038× |
| observer | stage_function | 76.17 | 0.022× |
| hash padding 512B | concat | 1.94 | 1.000× |
| hash padding 512B | incremental | 1.69 | 1.150× |
| hash padding 16384B | concat | 12.88 | 1.000× |
| hash padding 16384B | incremental | 12.20 | 1.056× |
| hash padding 262144B | concat | 285.07 | 1.000× |
| hash padding 262144B | incremental | 166.57 | 1.711× |

有理数は最終的なFractionへの変換・約分まで含む。GMPは32/96/256 bitで約1.33/1.18/1.43×。serializationはfusedで約1.34×、nativeで約3.04×、Stream全体では約1.58×。microの最大倍率を全体倍率として使わない。hashのincremental更新は入力UTF-8 bytesを事前生成する条件で、大payloadの一時文字列・再encodeを避ける利益がある。新しいchainは依存するため並列化していない。

stage/function observerは小さなhash kernelのCPUをそれぞれ約5倍/26倍、併用で約46倍にした。これは小kernelに対する比で、実G7全体の倍率ではない。別の人工paired passではinstrumented/baseline task CPU比を記録した。FunctionObserverを除けば細かいfunction費用が失われ、stageだけにしても旧R1の同じinstrumentationではなくなる。軽量化はB候補。

64 tiny tasksのpool再利用はfreshの約0.1086秒→0.0347秒（約3.13×）、CPU約0.647→0.223秒。full-Nでは起動費が償却されるためこの倍率を移さない。reuseはprocess/HWM/setup独立性を変えるB候補。

16 pass-overlap tasksは1/8/16 workerで約1.818/0.293/0.226秒、逐次比約6.20/8.03×。baselineとinstrumentedの資源競合・順序・STOP意味論を変更するためB候補。pair分割は今回は優先しなかった。8-pairのload imbalanceには有望だが、bit offset/state/setup/STOPの追加設計が必要。

64KiB bufferingはheavyでnative unbufferedとの差が約0.4%に留まった。accepted-write prefix保証は人工crashで変わるため、R1採用は推奨しない。正常終了時のbytes一致だけでは不十分。GPUは4台のA100を確認したが、この厳密・可変長処理を移植する有利な候補が得られず、実測していない。CPUより遅いと実証したわけではない。
