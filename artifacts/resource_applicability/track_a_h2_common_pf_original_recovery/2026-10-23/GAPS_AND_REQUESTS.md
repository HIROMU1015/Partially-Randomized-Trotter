# 未取得資料と保持者への具体的要求

対象4軸の当該実行plan/input/QPY/raw archive/resultはGPUA100-gleap-01でbyte確認済み。新compileで埋める対象原本欠測は見つかっていない。GPUA100-gleap02への私的転送は未実施であり、以下は保持者へ渡せる要求で、送信済みではない。

| 保持者・担当 | 要求する既存資料 | 用途・受入条件 |
|---|---|---|
| GPUA100-gleap-01の本人所有研究領域管理者 | private locator mapと21-member transfer manifestに記載した原plan、入力5件、対象2 result、worker_exit、対象12 gzip | アクセス制限付きの非公開経路。相対member名とSHA/byteを保持し、GPU02が当該projectionに静的結合する。原planのSHAは6eef028a…、2 resultのSHAは4軸表に記載。原本を上書きしない |
| 同保持者、または元workerのlaunch/runtime証拠保持者 | **実行時に既に保存されたものがある場合だけ**、module origin/path・ロードbinary・shared library・dependency/load manifest、launch/provider receipt | PF SOURCE 1b35f9ac…、原plan SHA、対象PID/日時との関係を付ける。秘密pathを非公開mapへ分離し、公開projectionにはrole/relative locator/SHA/bytes/証拠範囲だけ。今回の現存file再hashや新環境loadで過去実loadを代用しない。不存在なら「当時未保存」と明記 |
| 旧snapshotの原保持者（追加確認が必要になった場合のみ） | 容器SHA62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaaの保存原本または実在locator | 歴史入力の来歴補完。共通PFが実際に使った8bb1aa35…容器は既に確認済み。旧容器を今回生成して「復元済み」としない。この任意補完を対象QPYの欠測とは扱わない |

完全historical実load資料は限定したknown run root/固定索引/元SOURCEの調査では未発見。宣言closureの365 entryの全現存byte一致は確認済みだが、完全依存性の保証へ格上げしない。元科学hostnameもcustodyだけで決めない。

GPU02による現在の共通native定義への全stream静的binding、独立diagonal witnessの不足、runtime/cost-object政策の正式採択はこの回収とは別。科学計算の追加要求は作っていない。保存値の再価格取得、probe、COMPARE、新grant、科学retry/resumeは0。mandatory STOP。
