# KEEP refと研究履歴の削除後保全監査

PUBLIC_REDACTED。environment=LOCAL。確認日時：2026-10-27T15:03:30.522029+09:00。

固定330候補を通常の無条件deleteで処理せず、各固定完全SHAを期待値にした条件付き削除で一つのatomic transactionとして除去した。削除後はremote 481本、固定KEEP472本・B3関連別枠7本・監査後追加2本のすべてを確認した。候補外ref消失・更新は0。今回の操作でlocal refs/worktree/Git objectsを削除していない。

## 保持refからの到達性

LOCAL固定監査`5ee9dcff7976e8c353792b8e79fa750d47c6ad82`の全2,268 commitと、選定330 tipおよび全祖先が、削除後remoteのKEEP集合から到達可能だった。以前のDAG/bitset証拠を再利用し、独立したGit rev-listで候補/全監査tipからKEEPを差し引いた未保持commit数が0であることを確認した。候補同士を相互に唯一の保持先とせず、保持先は候補外472 refに固定した。

既存SOURCE/result/CEM・契約索引に対応する369 commitはKEEPから到達し、保持ローカルobject storeのcat-fileでcommitとして取得できた。固有commitを持つ88 remote branchもすべて保持。旧commitのtree/blob各版は持続的KEEP refの全祖先履歴に残るため、SOURCE/input/result・独立cohort・失敗/negative result・STOPを後続内容で置換しない。SHA文字列をCSVへ残しただけを保全根拠にはしていない。

全IDの固定commit・指定KEEP refと現在tipは[削除結果CSV](BRANCH330_REMOVAL_RESULTS.csv)にある。復旧は必要時の別指示で、KEEP refから正確な旧commitへ辿り、同名refが存在しない条件で元remote refを新規作成する案。既存refを上書きせず、元local branch/markerを要求する旧手順では、その名前と状態を保った上で科学実行許可・one-shot/retry/STOPを別に守る。今回ref復旧・workspace再作成・科学再実行は0。

## ローカル資産と原本

| 保護確認 | 結果 |
| --- | ---: |
| 開始時LOCALの実directory/登録保持 | 220件、60共通Git |
| 既存LOCAL branch ref記録の存続 | 496レコード |
| 既存非公開台帳/manifest/復旧原本等のhash不変 | 778 files |
| 既存公開inventoryのhash不変 | 137 files |
| 139件保全＋大容量追加のarchive metadata不変 | 6412固有版 |
| 先行57件archive metadata不変 | 27793 binding paths |
| 主checkout HEAD/branch/status/main ref | 不変 |
| A/B接続・変更・B3アクセス/操作 | 0 |

受入れ済みarchiveの過去SHA/Git blob検証を再利用し、変更がないmetadataを再照合した。payloadの全件再hash・全archive再生成・private rawの再解析はしていない。既存archive・bundle・復旧索引を削除/移動/上書きせず、新しい実行記録だけを本人専用領域へ保存した。B側3bundleの全remote包含を推測せず、今回の履歴保持根拠は候補外KEEP refである。元の実copyは同じローカルdiskであり、独立障害域backupや科学runtimeの完全復元を保証した意味ではない。

SERVER_A/Bの最新固定監査結果を再利用したが、両hostの全private原本やruntimeをLOCALから新規検証していない。B3はユーザー申告の実行中を前提に放置し、本体・sealed catalogue・稼働資産へアクセスしなかった。科学SOURCE/raw/input/result/STOP/契約の編集や新規科学計算は0。

**削除後監査・結果公開後mandatory STOP。**
