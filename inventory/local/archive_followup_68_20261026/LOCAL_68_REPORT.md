# ローカル68件：研究ファイル実保全報告

PUBLIC_REDACTED_ARCHIVE_68。2026-10-26 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。入力commit `950a866f4756e29c06a691c6b80e7b0e431777eb`。2026-10-25正式運用指示を実在ファイルから確認し、§8.1・§8.4・§10を適用した。

## 結果

| 群 | worktree | HEAD版の保存パス | 保全scope検証済み・他環境待ち | 旧SOURCE未解決 |
| --- | ---: | ---: | ---: | ---: |
| QC | 18 | 18,216 | 18 | 0 |
| N1/N3 | 16 | 14,391 | 5 | 11 |
| Research B B2D Outer | 34 | 8,094 | 34 | 0 |

対象68件すべてで選定した現存研究資料を保存・検証した。保全scopeを確認できた57件と、旧開発SOURCEの残条件を持つ11件を区別した。所有者の専用workspace利用終了は全68件で確認済み。再承認・利用終了の再質問はしていない。

実コピー**8,284ファイル／511,618,500 bytes（約512 MB）**。保存先内だけのhardlink32,419を加え、版別の参照先は**40,703パス**。68 HEAD版40,701と固定履歴SOURCE2を含む。全実コピーSHA-256・Git blob一致、全版別対応でsource安定hashと保存inode／size／modeを確認した。失敗0、上書き0、原本へのhardlink0。異なるcohort／SOURCE／失敗を同一結果として統合していない。

本人所有・owner-only ACL、空き容量、新規namespaceを確認し、既存の非公開保管領域の新規68件directoryへ保存した。QC／N1_N3／Research_Bごとに元相対階層を残した。各ID／branch／完全SHA／sourceとdestination／hashの完全対応は非公開manifestにある。

## 保全内容と限界

既存75件・234件の索引、SOURCE/input/preservation index、study関連文書、静的local import／明示入力参照を再利用した。研究コード・script・設定・tests、保存済み結果と対象内tracked raw、原稿、固定SOURCE／結果／STOP、独立cohort・失敗／中断記録をコピーした。repository全体を68回複製していない。過去の主checkout全体のhash一覧から無関係なファイル全体を再コピーしなかった。

source/input宣言2,466対応のうち2,411対応のbytesを特定・保全した。うち10対応は旧固定commitからの2ファイル回収である。**残55対応・3旧開発ファイル版は未特定で、11件に未解決条件を残した。** 元のsynthetic記録・ledger等は保全済み。具体的ID／public path／期待hashは[整理計画](LOCAL_68_CLEANUP_PLAN.md)にある。SOURCEの再構築・科学再解析・再実行はしていない。

現在の68 HEADは前回と一致し、tracked clean、untrackedなし、ignored差分なし、index hash不変。現在GitHubの同名tip68件とも一致。別環境の利用やprivate rawの独立backup、固定path契約の解消は確認していない。保存コピーは同じローカルdiskにあり、別障害域のbackupではない。外部private rawの大量転送・共有Git／venv／cacheの移設は0。

新規保存対象のLFS pointer／submodule／symlink通常コピーは0。既存guide symlink68記録は追従コピーせず保持し、環境設定136記録のtracked版を保管した。固定SOURCE・契約・STOP・grant/ledger/markerのbytesや実行条件は変更・消費していない。生成script／scientific runtimeの可搬実行は検証していない。

## 原本保護と並行作業

検証時点 `2026-10-26T12:33:06.528420+00:00`。対象68件のHEAD・branch・status・ignored・indexは開始時と一致。既存research refs・worktree登録・主checkoutも既存entryは不変だった。対象外の並行追加をbranch 1件／worktree登録 1件検出し、追加を巻き戻したり内容へ介入したりしていない。既存研究worktree数を減らしていない。

旧非公開原本・前回資料 25ファイルと既存公開資料21ファイルはhash不変。前回7件の2,052保存ファイルはsize／mtime／inode不変、manifest等のhashも不変。前回保管領域へ書込み0。A/BへのSSH0、B3干渉0、process操作0、科学計算・sampling・compile・数値再解析0。

公開4資料は以下のみ。ファイル本体、非公開絶対path、内部host/IP、private raw位置、非公開ファイル名、credentialは公開しない。manifestに載せた相対pathは、検証した公開commitのtracked pathだけである。

- [LOCAL_68_ARCHIVE_MANIFEST.csv](LOCAL_68_ARCHIVE_MANIFEST.csv)：実コピー単位8,284行。対応ID集合・公開path／commit・hash、保存先は匿名item ID。全40,701版別path対応は非公開原本で解決する。
- [LOCAL_68_PRESERVATION_STATUS.csv](LOCAL_68_PRESERVATION_STATUS.csv)：68件、旧分類・今回保全scope・所有者方針確定・旧SOURCE gap・他環境条件。
- [LOCAL_68_CLEANUP_PLAN.md](LOCAL_68_CLEANUP_PLAN.md)：研究成果／SOURCE／原本の保持、57候補／11保留、他環境照合と後続の具体的操作条件。
- [LOCAL_68_REPORT.md](LOCAL_68_REPORT.md)：本報告。

## 次の担当と公開

**サーバーA担当**の並行参照確認と、**サーバーB担当**のB3非干渉の確認を後日照合する。**該当研究／環境担当**は11件の旧SOURCE gapと外部custodyを確認し、**ローカルCodex（次の指示後）**が保持ref・完全manifest・具体的なcheckout整理条件を突き合わせる。所有者の利用終了を再確認する作業は残さない。**ユーザー**による対象別の削除指示と、該当Codexの直前差分確認が揃うまで全68件を維持する。

公開用の独立cloneでcommit直前の最新remote HEADを取得し、既存履歴・A担当の追加があればそれも保持して、新規directoryだけを通常fast-forward pushする。競合時は最新HEADから新しい独立公開checkoutで同じ新規資料だけを追加し、force／研究branchのrebase・merge／既存資料上書きを行わない。公開SHA・remote一致は最終通知と非公開PUBLICATION_RESULTへ記録する。

**mandatory STOP。実際の削除・登録解除・原本移動・branch削除・main mergeは行わない。**
