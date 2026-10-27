# SERVER_A 24件処理後の保全監査

2026-10-27 JST。物理checkoutの通常remove20件と集合保留4件について、削除後に保全・Git・登録をread-onlyで確認した。scientific runner、compile/sampling、pickle/arrayの数値解析は行っていない。

| 監査対象 | 結果 |
| --- | --- |
| 成功20 IDのdirectoryとGit登録 | 各IDで両方消失を確認 |
| 保留H46 4件 | directory・登録・HEAD/ref保持 |
| 指定KEEP研究15＋他環境待ち1＋管理2 | 全18件のdirectory・登録保持 |
| 対象外の新研究worktree | 1件を保持。削除結果と分離 |
| 24 snapshot HEAD | commit object存在、各HEADへの保持refあり |
| 科学SOURCE | 13固有commit全てobject存在・ref到達 |
| branch/tag | 削除前の研究branch/tagの更新・削除0 |
| 既存237保全file | 削除前後に全SHA-256一致、130,363,450 bytes保持 |
| 保全manifest・RESTORE index | 読取可能・既存対応保持。旧copyと最新Git版を区別 |
| 最新0041 | 5ファイルの実bytesと保持Git blob一致、復旧ref保持 |
| 固定KEEP実行path | 25参照の実体保持（参照重複を含む） |
| 旧proposal0025のpath記録 | 承認された用途終了・通常removeにより対象pathは解除、元proposal/対象版はGit保持 |
| 外部custody | 既存9 rootの保持を確認 |
| 保護control metadata | 134ファイルの存在・size/mtime/inodeが開始時と一致 |
| main・objects・他worktree | 保持、主checkout statusに予期せぬ変化なし |

237copyは元tree外の本人所有領域にあり、元とのhardlinkなし。コピーfile bytesを検証し、rawの科学的再評価はしていない。copy本文・非公開file名・保存先は公開していない。同disk上の保全であり、別障害域backupではない。

0041の古い私有copy5件は更新前の歴史snapshotであり、最新原本へ上書き・復元していない。最新scopeの5件は保持commit/個別Git blobを復旧元とし、実fileとのSHA一致を再確認した。今後のrestoreも勝手に実行せず、必要なら保持refと正確な版を使う。

通常removeは同common Gitに対して順次実施した。force・prune・clean・reset・gc・手動再帰削除・unlock・branch削除・履歴書換えは0。Git管理領域への変更は、承認済みの20 linked worktree登録解除に必要な通常Git動作のみで、他objects/refsを変更する操作はしていない。

H46については新しく残った対象外consumerのdraft PLANが0039を参照しているため、集合全体の非影響gateを閉じられず削除0。現在の科学実行に必須と認定したという意味ではなく、未承認の対象外workspaceの役割を推測で終了させないための保留である。

保全・科学SOURCE・失敗/STOP・grant/ledger/marker・独立cohort・原本custody・科学契約は保持する。未知namespace/別所有者/未記録の将来jobを網羅しているとは主張しないが、今回のpreflightと削除後監査の範囲で対象外の予期せぬ消失・変更は検出しなかった。他processや他hostへの操作はしていない。

公開4資料と詳細のID別結果は同じ24集合に限定する。今回の監査を、残存worktreeや研究branchの削除承認へ転用しない。mandatory STOP。
