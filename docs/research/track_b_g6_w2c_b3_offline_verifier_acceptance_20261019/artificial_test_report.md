# 人工検証結果

実収集**360 PASS＝既存275＋新85、FAIL/ERROR/SKIP各0**。[全ID・SOURCE/log SHA](artificial_test_results.json)。
既存SOURCEは変更せず6 suitesをそのまま再実行し、新suiteを加えた7独立processを並列に起動。親のthread poolは投入管理だけ。

| Suite | PASS |
|---|---:|
| Admission | 69 |
| Readiness | 30 |
| Driver | 36 |
| Cost preparation | 48 |
| F01 | 26 |
| Host integration | 66 |
| 新offline verifier/cost/acceptance | 85 |

新規正常系はactual Stream/run_pair/Meter＋既存mock modules、fake bits/clockを使用。24pair、4方式×A/B×X/Y、correctness16/axis、profile18/axis×二pass、R/C/A、summary/receiptを検査。
正常例は832 toy trials。実科学N・31,710,848 callsを走らせたものではない。Full等の実G7/F3 moduleはimportしていない。
旧testsのsaved budget fixtureもmock構築値であり、実科学保存値の数値再解析を行わない。

異常系はchain/mid-file/file SHA、重複/欠落/index/phase/pass/axis/N、kind/angle、summary/contract/receipt、未知schema/record、非exact・悪いhex/rational/duplicate key、partial末尾。
F01は取得前のstale event、完了誤加算、候補flag順、retry/resume、sample/partial write/flush/file hash失敗を人工注入。
費用はcoarse+fine混入、parent wall混入、prepare重複、pass混合、unknown→0、raw generator費用count矛盾、負memory/不整合residualを拒否。
契約差はREQUIRES_CONTRACT_CHANGE、SOURCE不一致はFAIL、未公開parallelと今回未実行fusedはpending。

## メモリ確認

[bounded_memory.json](bounded_memory.json)：2axis×1000trial、1,261,268bytesでverifierのtracemalloc peak174,963bytes。
2axis×10000trial、12,709,268bytesでpeak98,749bytes。入力約10倍に対して検証時Python割当てpeakは増えず、全trial保持がないことを有限人工例で確認。
これはtracemallocが追跡する検証部分のPython allocationであり、process全RSS・OS cacheの上限証明や64GiB実測ではない。
計算複雑度は最大行＋固定pair metadata。最大行そのもののサイズはRAMに影響する。

環境はLinux6.8/x86_64、Python3.10.12、32logical CPU、RAM62GiB/available約49GiB。詳細は[environment.json](environment.json)。
共有OS/venv/cache/他processを変更せず、owned temp/childだけを使用。共通指示§8.4を適用し人工CPU/RAM/worker/wall capなし。
この局所人工検査をimmutable CIや外部再現と呼ばない。power failure/実storage/実科学入力の耐障害性や科学結果は未検証。
圧縮logsはgzip内にraw stdout/stderrを保全し、artificial_test_results.jsonは非圧縮logのSHAを記録する。
