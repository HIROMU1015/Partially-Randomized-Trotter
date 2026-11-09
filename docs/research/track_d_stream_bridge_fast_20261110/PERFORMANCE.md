# 同条件の旧新性能比較

[科学条件・SOURCE](REPORT.md)。主比較はrun02_paired。固定8 wrapper×旧新の16 fresh processを同じpoolで起動し、同じvenv、native gzip入力、phase入力、arity依存のfrontier観測器を使った。各processは1 case／1 variantのみを処理する。

測る範囲はnative入力のdecode／NativeInstruction構築／旧新bridge／初期・最終T0計数まで。importと入力hash検査はtimer前、出力hash／MPFR phase検査・結果書込はtimer後。peak RSSはそのprocessのgetrusage high-water markで、import／hash検査も含む。旧新とも同じ条件である。raw QPY読込・lowering・full P0に対するend-to-end speedupではない。

## 主比較

| Wrapper | 旧秒 | 新秒 | 旧/新 | 旧RSS MiB | 新RSS MiB | RSS削減 |
|---|---:|---:|---:|---:|---:|---:|
| q4_reference_X | 12.196 | 6.700 | 1.820× | 155.387 | 100.000 | 35.64% |
| q4_reference_Y | 12.240 | 6.682 | 1.832× | 155.984 | 100.000 | 35.89% |
| q4_symmetric_directional_X | 12.115 | 6.470 | 1.872× | 154.500 | 96.500 | 37.54% |
| q4_symmetric_directional_Y | 12.151 | 6.483 | 1.874× | 152.000 | 98.637 | 35.11% |
| q8_reference_X | 38.972 | 18.337 | 2.125× | 229.383 | 107.375 | 53.19% |
| q8_reference_Y | 38.466 | 18.449 | 2.085× | 228.848 | 107.129 | 53.19% |
| q8_symmetric_directional_X | 39.280 | 18.327 | 2.143× | 230.000 | 105.309 | 54.21% |
| q8_symmetric_directional_Y | 39.208 | 18.184 | 2.156× | 229.000 | 107.195 | 53.19% |

16 process全体wallは40.823秒。各variant一回ずつの記述的測定であり、速度倍率の信頼区間や全環境保証ではない。最初の別wave測定もrun01に保持し、主表には混ぜていない。

## Input decodeとcore処理

| Wrapper | 旧decode秒 | 新decode秒 | 旧total−decode | 新total−decode |
|---|---:|---:|---:|---:|
| q4_reference_X | 1.783 | 1.716 | 10.413 | 4.984 |
| q4_reference_Y | 1.793 | 1.706 | 10.447 | 4.977 |
| q4_symmetric_directional_X | 1.688 | 1.684 | 10.426 | 4.786 |
| q4_symmetric_directional_Y | 1.674 | 1.675 | 10.476 | 4.808 |
| q8_reference_X | 3.644 | 3.609 | 35.327 | 14.728 |
| q8_reference_Y | 3.714 | 3.660 | 34.751 | 14.789 |
| q8_symmetric_directional_X | 3.679 | 3.628 | 35.601 | 14.699 |
| q8_symmetric_directional_Y | 3.693 | 3.586 | 35.515 | 14.598 |

decodeはgzip readline／JSON parse／schema構築の累積wall。generatorがconsumerへ制御を渡す間は含めない。残差には初期／最終Counterと全round、共通観測器を含む。前回のW依存観測器による約34秒と直接倍率を作らない。

## 処理量とcontainer

| Wrapper | Round | 1Q走査命令 | CX走査命令 | 新synthesis call | 新cache hit | 最大cache slot | 旧word長 | 新word長 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q4_reference_X | 10 | 631901 | 474247 | 2775 | 128218 | 2536 | 203178 | 60243 |
| q4_reference_Y | 10 | 631906 | 474247 | 2775 | 128218 | 2536 | 203183 | 60243 |
| q4_symmetric_directional_X | 10 | 632060 | 476963 | 1410 | 129133 | 1322 | 200882 | 60482 |
| q4_symmetric_directional_Y | 10 | 632065 | 476963 | 1410 | 129133 | 1322 | 200887 | 60482 |
| q8_reference_X | 20 | 2164443 | 1825588 | 2953 | 489273 | 2559 | 430996 | 127297 |
| q8_reference_Y | 20 | 2164448 | 1825588 | 2953 | 489273 | 2559 | 431001 | 127297 |
| q8_symmetric_directional_X | 20 | 2189672 | 1853811 | 1483 | 494721 | 1332 | 429400 | 128670 |
| q8_symmetric_directional_Y | 20 | 2189677 | 1853811 | 1483 | 494721 | 1332 | 429405 | 128670 |

旧新の全列走査量とround数は同じ。減らしたのは不変runの再合成、使わない中間Counter、全folded listと元input listの保持であり、worklistで走査そのものを減らした結果ではない。cache slotは前pass＋現passのentry slot和で、同じentryへの参照を含み得る。

| Wrapper（新） | 最大live node | retained slot | node list bytes | 最大frontier参照 | 最大wire stack | frontier list bytes | 最大1Q pending |
|---|---:|---:|---:|---:|---:|---:|---:|
| q4_reference_X | 60243 | 60617 | 499944 | 69794 | 5611 | 598384 | 207 |
| q4_reference_Y | 60243 | 60617 | 499944 | 69794 | 5611 | 598384 | 207 |
| q4_symmetric_directional_X | 60482 | 60848 | 499944 | 70721 | 10651 | 599760 | 131 |
| q4_symmetric_directional_Y | 60482 | 60848 | 499944 | 70721 | 10651 | 599760 | 131 |
| q8_reference_X | 127297 | 128123 | 1140552 | 147250 | 11407 | 1252880 | 207 |
| q8_reference_Y | 127297 | 128123 | 1140552 | 147250 | 11407 | 1252880 | 207 |
| q8_symmetric_directional_X | 128670 | 129488 | 1140552 | 150207 | 22725 | 1283760 | 131 |
| q8_symmetric_directional_Y | 128670 | 129488 | 1140552 | 150207 | 22725 | 1283760 | 131 |

node／frontierは旧CX規則と同じで固定幅へ切っていない。node list bytesはlist allocationのみで、NativeInstruction／cache等のobjectを含まない。per-wire参照和ではCXを二wireで数える。process RSSがそれらを含む総コストの指標となる。

## Loweringの準備と実用限界

保存raw8から1gateごとに既存lowererでnative列を生成した準備は 27.226秒（8 process）。最大gate chunkはStrong44命令／Reference53命令。q4は旧native列とhash一致、q8も旧raw hash・phaseに一致した。準備peak RSSはq4約221〜223 MiB、q8約369〜372 MiBであり、上のnative-only engine peakとは別である。raw QCとgate.definition等のcacheが残り、個々の寄与は分解していない。小さいchunkだけでpipeline全体がO(1)記憶になるとは主張しない。

新stream resource adapterのlate phase接続は人工2件で確認した。このadapterを使ったH4 raw→最終価格の新しいend-to-end計測は行わず、準備とnative-only benchmarkを別に記録した。full-wrapper再構築／P0 DAG／CAR target／state simulationは0だが、保存raw QCと可変長frontierは依然必要。

q4→q8で反復は10→20、new時間は約6.5→18.3秒へ増えた。path／cutも異なり、qだけの因果効果とはしない。旧newともO(RL)時間・O(L)記憶、最悪O(L²)。cacheの異なるrunが増えれば記憶も増え得る。q1024〜4096の速度／RAM／全P0資格は未確認で、外挿や新benchmarkは行っていない。
