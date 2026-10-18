# 人工障害検証

[機械可読結果](artificial_test_results.json)と各test logが正本。新67＋旧30/36/48＝**181件PASS**。
新SOURCE/旧guard/旧driver/cost-preparationの4 suiteを独立processで並列に実行した。
32 CPU、available RAM約49GiB、disk空き371GiB、local ext4、load約1.18を事前確認。
SQLite write排他が主たる競合箇所であり、32 contendersは前回の比較fixtureと同数。科学worker数の上限ではない。
GPUを必要としないファイルhash・SQLite・mock検証。共有設定・cache・venv・mount・他jobを変更していない。

| 検査 | 結果 |
|---|---|
| 未採択、自己承認、独立authority欠落、承認/元statement hash不一致 | reserve前に拒否 |
| science/input/driver/guard/adapter SHA、angle、seed、cell/N、call budget、schedule不一致 | 拒否 |
| output/claim/scope identity、F01 pending、release/回帰hash未確定 | 拒否 |
| 別adapterの32 thread・32 owned process、同一adapter32 thread | 各1 success |
| digest+claim変更/同scope、同claim/別scope、STOP後retry/resume | 拒否 |
| reserve前・insert未commitの失敗 | rowなし、receiptなし、同adapter再入拒否 |
| commit直後のack失敗・directory fsync失敗 | RESERVED残存、再取得拒否 |
| claim detail中・output作成前・reserve後のbinding/source変化/衝突 | spent保持 |
| failure/STOP保存の両失敗、STOP短縮write | spent保持、prefix保持、自動保存retryなし |
| owned process異常exit：未commit/commit後/ADMITTED後 | rollback/no permission、後2箇所は再取得拒否 |
| store欠落・不正ID・corrupt DB・再初期化 | 新storeへ自動回復せず拒否 |
| production無承認execute、artificial portの旧Boundary接続 | 科学load前に拒否 |

fixtureは`/tmp/b3-r1-artificial-*`のowned private dir、人工authorizationとmock scopeだけ。
SOURCEコピーはbytes照合対象で、G7/F3/return kernelはimport/実行していない。保存結果は原bytes/hashだけを扱い数値再解析なし。
人工receiptはB3_NEW_ claimではなく`admit_once`も提供しない。実science/実authorization/claimは**0**。

partial claim detailはdurable RESERVED後の論理障害。実SQLite VFSのshort-write/torn-pageや電源断は未実施。
人工DB corruptionの拒否やprocess-exit検査をproduction耐久性の完全証明へ格上げしない。
F01 driver回帰は未統合・今回の67件に含めない。実CPU優位・accuracy・母費用confidenceは未検証。
