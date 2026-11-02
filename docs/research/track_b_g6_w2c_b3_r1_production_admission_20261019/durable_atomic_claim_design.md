# Durable atomic claim：単一SQLite store

## 採用方式と前提

private owned 0700 directory、0600 `claims.sqlite3`、SQLite3.37.2/Python3.10.12、今回の人工領域はlocal ext4。
`journal_mode=DELETE`、`synchronous=EXTRA`、`BEGIN IMMEDIATE`。ネットワークfilesystemや複数hostを保証対象にしない。
store ID/schemaを初期化時に記録し、admissionは`mode=rw`で既存storeだけを開く。欠落・不正mode/owner・symlink・integrity異常・別store IDは拒否。
初期化は明示的な別管理操作。今回呼んだ初期化は専用/tmpの人工storeだけで、実storeは未作成。

scope PRIMARY KEY、claim_id UNIQUE、nonce UNIQUE。scope keyは契約digestから独立する。
同scopeでdigest/claimを変えても拒否し、同claimを別scopeへ移しても拒否する。
予約transactionはscope/claim/formal digest/全binding（承認hashを含む）/nonce/owner PID/status=RESERVEDを**同じrowで一括commit**する。
二ファイルをtransactionと呼ぶ方式ではない。STOPファイルは補助auditで排他条件に使わない。

SQLiteの[atomic commit](https://www.sqlite.org/atomiccommit.html)と[locking](https://www.sqlite.org/lockingv3.html)に依存する。
[EXTRA同期](https://www.sqlite.org/pragma.html#pragma_synchronous)はrollback-journal削除後のdirectory同期も加える。
FULLとDELETEだけで全電源断のcommit保持が確立したとは扱わずEXTRAを選んだ。加えて予約後にdirectory fsyncを要求する。
保証はOS/filesystemの正しいlocking/fsync、storageのflush遵守、DB/journalを削除・巻き戻し・差替えしないことが前提。
人工テストは電源断・媒体破損・SQLite VFS torn-pageの完全証明ではない。別host/移設/backup復元後の自動resumeは未提供。

## Commit pointと状態

`NEW → RESERVED(spent) → ADMITTED(spent) → STOPPED(spent)`。行削除・解放・lease expiry・再発行APIはない。
最初のEXTRA commitがSQLiteのatomic reserveの境界。directory fsyncと再照合が完了するまでpermission receiptを返さない。
commit前のexception/owned process exitはtransaction rollbackでrowなし・receiptなし・scienceなし。
この場合にも同じadapterはSTOPで、コードは自動再試行しない。新しい試行を承認したとは扱わない。
commit acknowledgementの不確実性やcommit直後異常は、成功と推測して再試行せずSTOP。実際にcommitされたrowはRESERVEDのままでもspent。

予約後、claim detail/ADMITTED更新途中、binding変化、output作成前、保存失敗、process異常終了でもrowを解除しない。
SQLiteの未commit detail更新はrollbackされ、先行commitされたRESERVED rowが残る。partial reservationを未使用とみなさない。
最後のSTOP更新が書けなければRESERVED/ADMITTEDが残り、それでも一意制約で再取得を拒否する。
STOPの補助JSONが短く切れた場合、そのprefixを残して自動retryしない。

再起動後は同じ既存storeのSQLite recoveryを使い、rowのあるscope/claimはstatusに関係なく拒否する。
owner PID消滅も解除理由にしない。store欠落/不明/壊れた場合は新storeを自動作成せずfail closed。
実承認が特定store path/IDをpinするため、空の別storeへ勝手に切替えて同scopeを使うことも拒否する。

## 実測範囲

人工32 thread/32 owned processの同scope競合で各1成功。owned processの`os._exit(88)`をcommit前、commit後、ADMITTED後へ注入した。
commit後の2箇所は新adapterの同scope取得を拒否した。共有OS/serviceの再起動・他job killは行っていない。
partial STOP writeは実際のos.writeを短縮し次writeで失敗させた。claim detailはSQL UPDATE後/COMMIT前の例外・process exitと論理的障害点、DB破損は人工DBのbytes差替えで検査した。
これらをSQLite内部の実torn-page injectionやproductionクラッシュ耐久性の全証明とは表現しない。
