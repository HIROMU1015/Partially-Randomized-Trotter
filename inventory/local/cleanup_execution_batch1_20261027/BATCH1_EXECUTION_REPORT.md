# Batch 1：図表・スライド用7 worktreeの削除結果

PUBLIC_REDACTED_EXECUTION_RESULTS。2026-10-27 JST。事後検証：`2026-10-27T02:04:20.745349+09:00`。Repository：HIROMU1015/Partially-Randomized-Trotter。

**対象7件すべてが最終preflightに合格し、通常のgit worktree removeを各1回実行した。7件ともcheckout directoryと対応登録の除去を確認した。最終保留0・失敗0。branch・重要commit・共有Git・保全済み2,052ファイルとmanifestを維持した。Batch 2〜4と旧SOURCE未解決11件は処理していない。**

固定計画commitは`45c78a67992cc2534bd1a9bb85a2fc1985cdf2eb`。[4バッチ実行前計画](../final_three_environment_cleanup_20261027/LOCAL_64_BATCH_EXECUTION_PLAN.md)、[64件固定表](../final_three_environment_cleanup_20261027/LOCAL_64_BATCH_READINESS.csv)、同directoryの75件最終照合表・報告を読み、非公開原本のID/正規path/branch/完全HEAD/Git common/backlink/保全manifestと対応付けた。対象をbranch名や匿名IDから推測していない。今回のユーザー指示によるBatch 1限定実行許可を適用し、所有者の利用終了再質問や承認証明書は要求していない。

正式運用指示`research_project_instructions_unified_2026-10-25.md`を実在ファイルで確認し、§8.1・§8.4・§10を適用した。科学計算・sampling・compile・数値再解析・詳細科学レビュー・旧SOURCE再探索/再構築は0。

## 対象別結果

| ID | 最終preflight | 結果 | 維持した保存ファイル | 操作後確認 |
| --- | --- | --- | ---: | --- |
| LOCAL-WT-0203 | 合格 | REMOVED_VERIFIED | 230 | branch/commit・archive保持 |
| LOCAL-WT-0204 | 合格 | REMOVED_VERIFIED | 509 | branch/commit・archive保持 |
| LOCAL-WT-0205 | 合格 | REMOVED_VERIFIED | 49 | branch/commit・archive保持 |
| LOCAL-WT-0206 | 合格 | REMOVED_VERIFIED | 109 | branch/commit・archive保持 |
| LOCAL-WT-0207 | 合格 | REMOVED_VERIFIED | 133 | branch/commit・archive保持 |
| LOCAL-WT-0208 | 合格 | REMOVED_VERIFIED | 509 | branch/commit・archive保持 |
| LOCAL-WT-0209 | 合格 | REMOVED_VERIFIED | 513 | branch/commit・archive保持 |

各IDの実行有無・正常終了・登録/directory除去・保持確認・最終保留理由は[BATCH1_EXECUTION_RESULTS.csv](BATCH1_EXECUTION_RESULTS.csv)に記録した。0207は別公開保持refから固定HEADへの祖先到達を確認し、0208/0209の固有local/remote branchと完全HEADも保持した。branch削除やHEAD変更はしていない。通常removeの呼出しは計7回・各ID1回で、force/remove再実行/prune/clean/reset/gc/手動の強制再帰削除/lock解除は0。

## 直前確認と差分への対応

各操作の直前に、本人所有の正規path、mainとの分離、linked登録・branch/完全HEAD・common Git・gitfile/admin backlink・lock、symlink/mount実体、tracked/untrackedとignoredの全leaf、可視本人UID processのcwd/引数、既存依存metadataの新差分、local branch/commit objects・必要remote保持ref到達、archive/manifestの可読性と独立性を確認した。7件はtracked clean、untrackedなし、ignoredファイルなし、lockなし、対象pathを使用する可視process一致なしだった。未知の全namespace・未記録jobの不存在を証明したとは扱わない。

読み取りsandboxと削除実行側でcovering mountの表示が異なったため、削除前に切り分けた。両方のnamespaceを区別し、7件すべてで記録済みLOCAL directoryのdevice/inode・正規path・HEAD・Gitfileが同一で、対象や配下への追加mountがないことを確認した。実行側の確認済みmount viewを直前に再照合してから進めた。mount/共有設定の変更は0。

既存206 control metadataと既知editable pointer 12件は前回metadataから不変だった。前回以降にHEADが変化した別研究の最近のcontrol/doc 28件を限定照合し、対象pathの新参照0・読取エラー0。途中でその一部が更新されたため、0203は最初のpreflightで一時保留し、削除は実行しなかった。更新されたmetadataだけを再読し、安定した読取と対象path参照なしを確認した後、改めて直前preflightを通した。正常remove拒否や強制回避ではない。現在値の差分確認を後続IDでも行い、未解決の保留は残っていない。

A_R1/B_PATH_1の3環境結果は固定計画を再利用し、A/Bの全件再調査・SSH・自動転送は行わなかった。具体的な新外部依存・SOURCE/provider/inputの対象path束縛は今回の限定差分で検出していない。未記録alias/mount・別namespace・外部private custody等の既存scopeの限界は維持する。

## 登録数と並行作業

対象共有Gitでの登録数は**343→337**。今回7件を除去した一方、並行作業で対象外の新規登録1件が追加されたため、見かけの差は6件である。全clone/common directoryの再棚卸しを行った総数ではない。事前登録から失われたものは今回の7件だけで、新規追加を削除/巻き戻ししていない。

対象外の既存worktreeで、並行作業によるHEAD前進1件も観測した。branch等の登録fieldは維持され、旧HEADが新HEADの祖先として残ることを確認した。各7回のremoveの直前/直後では、他の既存worktree登録に変更なし。バッチ全体の観測時間内で生じた他作業のHEAD前進を「不変」と表現せず、非公開原本に分けて保存した。対象外のrefs追加/更新も記録し、戻したり科学内容へ介入したりしていない。

残る元75件内の68 checkoutとそのbranch/HEADは維持した。Batch 2のQC18、Batch 3のResearch B34、Batch 4のN1/N3検証済み5、旧SOURCE未解決11を含み、今回の削除へ混入させていない。Research Bの未解決供給環境・その他の研究worktree・主checkoutも除去していない。

## 科学資料・保全先・Git履歴

7件のlocal branchと完全HEAD、必要なGit commit objects・科学SOURCE/result/STOPの公開参照を保持した。主checkoutのHEAD/status・main、共有Git本体/objectsを確認し、今回それらへの履歴変更・branch削除・設定変更をしていない。科学SOURCE/input/result・provenance/STOP/grant/ledger/markerの保持条件を変更していない。

図表・スライド・原稿・生成script/選定入力等の**2,052 files／234,267,578 bytes**は、削除対象外の本人所有・非公開保管領域に維持した。全保存fileがregular/readableで、元checkoutへのsymlink/hardlinkでないことを削除前に確認。過去のSHA-256/Git blob検証を再利用し、全2,052件のsize/mtime/inode等が以前の検証から不変だったため、payloadの再hash/再コピーは0。削除後も全2,052件のmetadata不変と可読性を確認した。既存manifestと非公開原本104ファイルもhash不変だった。

これにより保存済み図表・スライドは元checkoutなしで引き続き参照できる。元checkoutのtracked内容は保持branch/commitから再作成可能な状態を保つ。保存先は同じローカルdisk上で、別障害域の独立backupではない。外部private raw全体や科学runtimeの可搬再実行まで保証した意味ではない。Git管理外の研究データの追加削除・移動、保全files/原本/manifestの上書きは0。

旧SOURCE3版・55宣言の未解決状態は変更しない。サーバーBのDF-native B3は保護対象のまま、job/SOURCE/input/raw/cache/Git/runtime/grant/ledger/marker/STOPへ操作していない。今回Bの現在稼働を独自に再観測したとは主張しない。

## 記録・復旧可能性・公開

本人所有の新規非公開領域へ、元path・branch・完全HEAD・common Git/backlink・保全manifest/保存先・直前preflight・各removeの開始/終了・事後確認・復旧参考手順を保存した。削除成功済みIDは再実行しない状態で記録した。公開失敗時にもremoveを再実行せず、記録を保持して公開だけを保留する方針である。

必要時の通常Git操作による再checkout参考手順を残した。保持branchが記録HEADを保ち、元pathが空き、別worktreeとのbranch使用競合がないことを確認して復旧する案で、今回は復旧操作0。path/branch競合をforceで上書きしない。元のignored cache、案内先や科学runtimeまで完全復元する保証ではない。

今回公開するのは本directoryの結果CSVと本報告の2ファイルのみ。内部絶対path・host/IP・非公開原本/保存先・private raw配置・credential・内部pathの単純hash・保全file本体を載せない。既存の独立公開cloneから最新remoteを確認し、既存inventoryを維持した通常fast-forward更新だけで公開する。commit/remote一致とGitHub読み戻しの結果は非公開公開記録と完了通知に記載する。

**mandatory STOP。Batch 1は7件成功・最終保留0・失敗0。Batch 2〜4の削除と旧SOURCE11件の整理は未承認であり、自動進行しない。次はユーザーが今回の結果を確認し、必要なら次バッチの具体的操作を別途判断する。**
