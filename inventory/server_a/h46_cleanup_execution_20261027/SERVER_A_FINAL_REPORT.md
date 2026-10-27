# SERVER_A H46残存4 worktreeの削除結果

2026-10-27 JST。固定参照は前回20成功/4保留結果 `5916f4baac80b1180c6f40c57a9dcd4e0aadb694` と準備計画 `6956fd0124c34b7bbf8d6de17cffad3fd2ae86c2`。今回の最新指示に基づき、新しく追加された対象外worktreeとPLANを維持し、旧H46専用workspace4件の利用終了・通常removeだけを実施した。

| 承認集合 | 対象 | 削除成功 | 保留 | 失敗 |
| --- | ---: | ---: | ---: | ---: |
| A-CURRENT-H46-4 | 4 | 4 | 0 | 0 |

0038・0039・0040・0041の正確なID/実path・branch・完全HEAD・登録・owner・index/status・ignored・利用/lockを直前確認し、通常git worktree removeを各1回、安全な順序で実行した。各IDでexit 0、実directory消失と登録消失を確認した。強制回避は行っていない。結果と完全HEADは[4件CSV](SERVER_A_H46_4_REMOVAL_RESULTS.csv)。

## 保留条件を閉じた根拠

新しい保持PLANの6参照は、旧0039の入力5件と結果出力root1件だった。入力は宣言hash・保持Git blob・新保持worktreeに同じbytesが存在し、0039が唯一の供給元ではないと確認した。結果16fileもGitで保持し、新保持worktreeの16結果を実bytes照合した。PLANのstatusと現行の実行設計はdraft/未承認の後続実行という文脈で、旧pathを必要とする現在の可視job/controllerは検出しなかった。

最新指示に従い、履歴/draft PLAN保存と古いpath永久維持を分けた。PLANをそのまま保全し、元worktree削除後の将来利用制約・復旧方法を記録できたため、旧4件の解除gateを閉じた。現在有効な科学実行契約を勝手に解除したのではない。具体的な方法・6参照の根拠は[PLAN保全報告](SERVER_A_PLAN_PRESERVATION_REPORT.md)。

## PLANと将来利用

保持する新worktreeとPLANは、branch・HEAD・本文SHA/Git blob・clean状態が削除前後で不変。PLANの私有copy1件も同一bytesで追加保存した。PLAN/SOURCE/入力設定の書換えは0。

**保存PLANに書かれた旧絶対pathは削除後に存在しないため、現在のまま即時実行可能とは扱わない。** 将来利用する場合、保持ref/Git blobから元pathへworkspaceを再作成するか、別の具体的許可の下でpath設定・実行契約を整理する必要がある。今回restore、PLAN実行、Phase B、新規科学計算は行っていない。

元4件のbranch・重要commit・Git objectsは保持。13固有科学SOURCEとPLAN追加4参照commit、入力5件・結果16件・0041最新5blobも取得可能である。0041の古いcopyは歴史snapshotとして残し、最新状態へ上書きしない。

## 保護対象と残存数

共有研究Git登録は21→17。同じ2研究Git系統では22→18、独立公開cloneまで含めた作業コピーは23→19。残存の内訳は研究17件・管理2件である。新しい対象外worktree1件、固定KEEP_REQUIRED研究15件、他環境確認待ち1件、管理2件が残る。4件以外の削除は0で、対象外の登録内容・main status・PLAN/HEAD・既存branch/tagは保持した。

前回20件成功に今回4件を加え、元の承認24件は累計24件の通常remove成功となった。ただし本タスクで実施した削除は4件だけである。以前の20件を再操作せず、その他のworktreeや研究branchへ削除対象を広げていない。

## 保全監査と公開範囲

既存237copy・manifest・RESTORE情報は保持した。前回の全SHA一致監査を再利用し、今回前後のcopy size/inode/mtime・権限に変化がないことを確認した。以前の全件調査や237copy作成を繰り返していない。同disk保全であり独立障害域backupではない。詳細は[保全監査](SERVER_A_H46_PRESERVATION_AUDIT.md)。

公開は新規directoryのCSV・PLAN報告・監査・本報告4資料だけ。内部実path、PLAN内部の非公開path/hash対応、host/IP、個人名、private原本配置、process識別、credential、非公開復旧先は公開しない。本人所有非公開原本に、6参照の正確な位置・元相対path・保持commit/blob・復旧情報・実行journalを保存した。

正式共通指示2026-10-25版の実ファイルを確認し、§8.1・§8.4・§10を適用した。既存の監査・SHA/CSV/Git/remove helperを限定再利用し、承認証明書・receipt/grantは新設しない。SOURCE/input/result・失敗/STOP・grant/ledger/marker・固定科学契約、共有環境・他job・SERVER_B B3・LOCAL環境に変更/操作は行っていない。

最新remote履歴を保持した独立公開cloneで4資料だけをfast-forward公開し、GitHub内容読み戻しとlocal/remote SHAを確認する。PLAN実行、GitHub研究branch削除/統合、残存worktree整理へ自動進行しない。結果保存・監査・匿名化公開後mandatory STOP。
