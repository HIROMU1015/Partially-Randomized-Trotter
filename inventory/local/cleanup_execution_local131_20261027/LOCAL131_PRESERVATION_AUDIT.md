# LOCAL131削除後の保全監査

PUBLIC_REDACTED。2026-10-27T10:17:58.021921+09:00。environment=LOCAL。
対象Repository: HIROMU1015/Partially-Randomized-Trotter。

固定基準は`b71c6e9cc074ae51aeeab986600d1b1d6354cbbb`の[最大131件計画](../final_local139_bulk_cleanup_plan_20261027/LOCAL131_BULK_CLEANUP_PLAN.md)にある「正確な131対象ID」列の和集合。非公開ID/branch/完全HEAD/正規path表、139件統合CSV、A139/B_PATH_139を対応付けた。所有者は14研究単位の131件をWORKSPACE_END_APPROVEDと明示し、通常git worktree removeを許可した。旧原本の未回答欄は書き換えず、本実行結果へ新判断を別記した。追加8件へは転用していない。

## 削除前・直前の確認

- 正確な131 ID、branch/完全HEAD、LOCAL正規path・owner・device/inode、common Git/admin/gitfile往復参照、登録・lockを照合。sandboxとnativeのmount表示差は従来の同一inode根拠を再利用し、実行namespaceで覆うmountと子mountを照合した。
- 全131件でtracked clean、untrackedなし。ignoredはディレクトリ名だけでなくleaf一覧を比較し、既知の外向きguide symlink・再生成cacheと既知font-cache schemaだけであることを確認した。未保全の重要資料は未検出。
- 可視本人UIDのprocess cwd/argvとopen FDリンクに対象path一致0。Git共有lock、対象adminのlock/merge/rebase等の具体的実行状態なし。可視scope外の全process不存在を証明したという意味ではない。
- 既存control/editable pointerと直近検証済み差分280レコードを再利用。metadata不変229、以前の削除等で利用不能46を区別し、変更head/controlの75資料を限定読取。新対象path参照0、読取error0。A/Bの全件再調査・接続は行っていない。
- 重要131 commitは持続的なheads/remotes参照で到達可能。保持対象6,841 Git blobの存在をbatch-checkで確認。最新GitHubの対象保持refは131件とも既存HEADを保持し、実行準備時の変化0。
- 既存archive6,412固有版はregular/owner/readable、読み出しとmetadata不変を確認し、以前のSHA-256/Git blob照合を再利用。無条件の全payload再hashは0。各IDでarchiveと元sourceが同一inodeでないことを確認した。

## 実行と削除後確認

全131件を初回preflightの一括集合とし、各ID直前にも再照合して通常removeを1回ずつ実行。同じcommon Gitへ逐次適用したのは共有管理領域の競合を避けるためで、小分け再承認ではない。force/prune/clean/reset/gc/手動再帰削除/unlockは0。

成功131、保留0、失敗0。各IDで物理directoryとworktree登録の両方が消失し、残すbranch/完全HEAD/重要commitとarchiveが保持された。成功判定は各IDの実測によるもので、登録総数の差だけによらない。[全ID結果CSV](LOCAL131_REMOVAL_RESULTS.csv)を参照。

主common Git登録は286→155。開始時対象外の全登録を保持した。並行作業による対象外HEAD前進1件、ref追加1件、既存ref前進1件は別記録へ保存し、巻き戻していない。開始時ref削除0。対象131 branchは全て固定HEADのまま。主checkoutのHEAD/branch/statusとmain refは不変。

保護20件、旧SOURCE未解決11件、別枠追加8件の合計39件はdirectory/登録を保持し、開始時の登録内容も不変。追加8件のGit statusもclean。これらは今回の対象外で、残余SOURCE欠落や研究契約を解消・解除した扱いにしない。

## 保全資産

| 検査 | 結果 |
| --- | ---: |
| 元の非公開台帳/manifest等のhash不変 | 325 files |
| 既存公開inventoryのhash不変 | 62 files |
| 139件保全＋大容量追加のarchive metadata不変 | 6412 固有版 |
| 先行57件のarchive metadata不変 | 27793 binding paths |
| 対象131のbranch/重要commit保持 | 全件 |
| 必要Git blobの存在 | 6,841 |
| A/B接続・変更・B3操作 | 0 |
| 新規科学計算・科学再解析・SOURCE契約変更 | 0 |

SOURCE/input/result/独立cohort/失敗履歴/STOP等は選定範囲の実archiveと固定Git参照で保持する。同一HEADや同一blobを根拠に科学的証拠を同一視していない。コード・sparse未materialize分はGit参照保持であり、実コピー済みとは区別する。完全な科学runtime復元・再実行成功や外部private raw全custodyを今回確認したわけではない。archiveは同じローカルdiskにあり、独立障害域backupではない。

SERVER_A/Bには接続も変更もしていない。既存DF-native B3のSOURCE/provider/input/raw/cache/Git/runtime/grant/ledger/marker/STOPと固定実行契約を維持し、旧Track B/G6 B3と区別した。現在のB3稼働状態をローカルから新規確認したとは表現しない。

非公開には元path/branch/完全HEAD/common/admin・保全manifest・各IDpreflight/result・通常再checkout参考情報を保存した。復旧や新科学実行は今回0。元台帳・archive・manifestは上書きしない。公開へ内部絶対path、host/IP、private配置、認証情報、非公開復旧情報やファイル本体を含めない。

**mandatory STOP。GitHub研究branchの削除・統合へは進まない。**
