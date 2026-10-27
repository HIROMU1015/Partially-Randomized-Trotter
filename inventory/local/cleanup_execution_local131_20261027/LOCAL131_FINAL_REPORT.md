# LOCAL131一括削除・最終報告

PUBLIC_REDACTED。2026-10-27 JST。environment=LOCAL。

**固定計画の131 worktreeは全て直前preflightに合格し、通常git worktree removeにより131件を削除確認した。保留0、失敗0。対象branch・重要commit・保全資料を保持し、保護20件・旧SOURCE11件・別枠8件と全対象外登録を維持した。**

## 入力・所有者判断・対象範囲

RepositoryはHIROMU1015/Partially-Randomized-Trotter。整理branchは`codex/pr-environment-inventory-20261026`。固定commit`b71c6e9cc074ae51aeeab986600d1b1d6354cbbb`の[131計画](../final_local139_bulk_cleanup_plan_20261027/LOCAL131_BULK_CLEANUP_PLAN.md)、139件統合CSV・大容量custody CSV・最終報告を実際に確認した。正式運用指示research_project_instructions_unified_2026-10-25.mdを読み、§8.1/8.4/10を適用した。

14研究単位について所有者の今回の利用終了判断をWORKSPACE_END_APPROVEDとして扱い、同じ指示による131件の通常remove許可を適用した。承認証明書/receipt/新grantは作成していない。科学研究の廃棄、Git branch/履歴の削除、固定契約の変更を認定した意味ではない。過去の資料は変更せず、本実行結果に新判断を残した。

対象は固定表の正確なID集合から抽出し、既存非公開対応表のbranch/HEAD/実pathと照合した。連番やbranch名から追加していない。別枠LOCAL-WT-0002/0095/0096/0097/0099/0100/0101/0103、保護20件、旧SOURCE11件、主checkout、A/B/B3と全対象外worktreeは削除集合に含まない。

## 14研究単位の実績

| 研究単位 | 対象 | 削除確認 | 保留 | 失敗 |
| --- | ---: | ---: | ---: | ---: |
| Track B D/E履歴 | 16 | 16 | 0 | 0 |
| Track B DF correction | 19 | 19 | 0 | 0 |
| Track B M1/M1S | 13 | 13 | 0 | 0 |
| Track B parent allocation | 15 | 15 | 0 | 0 |
| Track B handoff/diagnostics | 9 | 9 | 0 | 0 |
| Track B RA D0 | 7 | 7 | 0 | 0 |
| Track B BF/R1/SP履歴 | 14 | 14 | 0 | 0 |
| Track B G系列 | 16 | 16 | 0 | 0 |
| 旧Track B G6 | 6 | 6 | 0 | 0 |
| Research C C1 | 4 | 4 | 0 | 0 |
| Research C C4 | 5 | 5 | 0 | 0 |
| DF-native Track A | 1 | 1 | 0 | 0 |
| 表現アルゴリズム履歴 | 3 | 3 | 0 | 0 |
| その他の開発・中間検証 | 3 | 3 | 0 | 0 |
| 合計 | 131 | 131 | 0 | 0 |

[ID別結果](LOCAL131_REMOVAL_RESULTS.csv)は131行、[研究単位別結果](LOCAL131_RESEARCH_UNIT_RESULTS.csv)は14行。全IDに通常removeの一回性とdirectory/登録消失を記録した。再実行・force回避は0。

## 保全と非干渉

主共通Gitの登録は286から155。各IDごとに削除を実測し、並行HEAD/ref変化と分離した。対象131 branchと重要commitの保持、Git object可用性、主checkout/対象外登録の保持、保全archive/manifest不変を確認した。[保全監査](LOCAL131_PRESERVATION_AUDIT.md)にscopeと件数を示す。

6,412固有archive版と先行57件の27,793 binding pathはmetadata不変。325既存非公開資料と62公開資料のhash不変。過去SHA/Git blob検証を再利用し、payloadの全件再hashや再コピーは行わなかった。Gitのみのコード/sparse分は実コピー済みと区別した。研究SOURCE・input・raw/result・grant/ledger/marker/STOPの契約は変更していない。

A/B実パス照合は完了済みの固定結果を再利用し、今回A/Bへ接続も変更もしていない。AのLOCAL-WT-0184は歴史参照として保存し、元checkout物理供給契約と混同していない。B3のjobへ介入していない。既知scopeの依存未検出を全環境での依存不存在証明にはしていない。

## ローカルの残存状況

2026-10-27T10:18:06.685237+09:00に、既存棚卸しに記録された60研究関連common directoryだけを読み取り集計し、全60箇所へアクセスできた。**残存登録214、物理directory確認214**。うち主common Gitは155、他の既知common Gitは59。最初の389 IDに対応する残存は194、当初索引以後の新規登録pathは20。先行64件と今回131件の計195 IDが整理済みである。

これは既存研究関連common directoryの範囲の現在値で、新規cloneを全filesystemから再発見した完全棚卸しではない。公開用独立cloneや調査一時領域は含めていない。残存worktreeの新しい削除判定を今回一律には行っていない。

## 残る整理課題と次の担当

- ユーザー：別枠8件について、将来のworkspace利用終了と具体的な削除対象を別途判断する。今回は保全済みでも削除しない。
- 研究担当：旧SOURCE未解決11件の3旧版/55宣言と保護20件の固定SOURCE・契約条件を維持する。研究原本再生成や契約変更は本作業に含めない。
- 次の具体的指示を受けたCodex：GitHub branch整理を検討する場合、保持branch/重要commit、代表成果・独立cohort・失敗/STOP、A/Bの利用・契約、未push/private custodyを対応付け、削除対象を明示する案を作る。checkout削除成功だけでremote branch削除可とはしない。
- ユーザー：remote branch削除・統合は対象確定後に別途許可する。今回自動的には進めない。

archiveは同じローカルdiskで独立backupではなく、未記録alias/mount/namespace/jobや外部raw custody等のscope外残余リスクを完全には排除していない。具体的な保全要求が判明した場合はその対象に絞って扱う。

## 保存・公開

新しい本人所有owner-only領域へ131件の詳細preflight/result、正規path/Git identity、archive対応、復旧参考情報、保全監査を保存した。旧原本を上書きしていない。公開するのは本directoryの匿名化4資料だけ。内部絶対path、ユーザー名/内部host/IP、private raw/原本の具体的位置、秘密情報、非公開復旧情報、保全ファイル本体を追加していない。

既存独立公開cloneを利用し、最新remote履歴を保持するfast-forward可能な通常commit/pushで新規追加する。既存報告書上書き/force push/main mergeは0。commit SHA・remote一致と公開4ファイルの読み戻しは完了時の非公開PUBLICATION_RESULTとチャットで報告する。

**mandatory STOP。削除・保全監査・匿名化結果公開で終了し、GitHub branch削除・統合を開始しない。**
