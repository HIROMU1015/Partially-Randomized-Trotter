# PRICE_REQUEST_COVERAGE：LD1/LD2全13round・X/Y

2026-11-13 JST。正本は[52軸JSON](PRICE_REQUEST_COVERAGE.json)。全cellはREQUEST_READY・科学実行未選択、mean_C/区間/N/時間/RSSはnull。paired requestをX/Y別応答として保持する。

| m | qPF | LD1 R | LD1 event数 | LD2 R | LD2 event数 | X/Y metadata |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 2 | 2 | 1 | 1 | 両候補・両軸対応 |
| 1 | 2 | 4 | 8 | 2 | 4 | 両候補・両軸対応 |
| 2 | 4 | 8 | 32 | 3 | 12 | 両候補・両軸対応 |
| 3 | 8 | 16 | 128 | 5 | 40 | 両候補・両軸対応 |
| 4 | 16 | 32 | 512 | 10 | 160 | 両候補・両軸対応 |
| 5 | 32 | 64 | 2048 | 20 | 640 | 両候補・両軸対応 |
| 6 | 64 | 128 | 8192 | 40 | 2560 | 両候補・両軸対応 |
| 7 | 128 | 255 | 32640 | 80 | 10240 | 両候補・両軸対応 |
| 8 | 256 | 510 | 130560 | 160 | 40960 | 両候補・両軸対応 |
| 9 | 512 | 1020 | 522240 | 320 | 163840 | 両候補・両軸対応 |
| 10 | 1024 | 2040 | 2088960 | 640 | 655360 | 両候補・両軸対応 |
| 11 | 2048 | 4080 | 8355840 | 1280 | 2621440 | 両候補・両軸対応 |
| 12 | 4096 | 8160 | 33423360 | 2559 | 10481664 | 両候補・両軸対応 |

event数は保存configurationの論理schedule metadataで、compiled RZ数ではない。全round compileの採択表ではない。

LD1 m0/m6：既存P2 pilotは参照のみで新identity bridge/全law資格が未取得。LD1全round：旧E[S]は構造量の参照のみ。LD2全round：旧LD1価格/構造平均を再利用しない。今回受入済み価格は全52軸で0。

全cellの取得方法は、明示承認された有限pathに対する既存adapterのfull-wrapper S/C個別取得。未指定のround/path/精度、実preparation・λ/scalar照合、ideal/sampler law差・rare・CI、長q window tail、巨大wordのwall/RSSが残る。実measurement/fingerprint/cache照合は未実行。

[HANDOFF](HANDOFF.md)／[次の有限案](NEXT_SCIENCE_PLAN.md)。mandatory STOP。
