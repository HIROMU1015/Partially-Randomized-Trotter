# 固定SHAによる条件付き削除の監査

PUBLIC_REDACTED。environment=LOCAL。2026-10-27 JST。

今回のユーザー指示は、固定330件のremote branch削除refspecに限り、期待SHAを明記したforce-with-leaseの使用を明示許可した。以前のdelete_authorized=Falseを改変せず、新しい実行結果にcurrent_remote_delete_permission=Trueを別記した。承認証明書・receipt・新grantは作成していない。

## 実際の条件と実行

- 元集合：`5ee9dcff7976e8c353792b8e79fa750d47c6ad82`のDELETE_CANDIDATES.csvの330行。candidate ID/branch ID/公開alias/完全SHA/指定KEEPを非公開実名対応表で照合。対象の追加・置換0。
- 対象外：KEEP472・未解決7・main・整理branch・新規追加2。削除refspecとの交差0。
- 実行前snapshot：remote811本。全候補tipは固定SHAに一致。不一致0、既に消失0。KEEPと保留7の存在・保持先からの包含を確認した。
- コマンドはshell文字列の展開を使わず、Python subprocessの引数配列で構築した。各refに完全40桁SHAの明示leaseを1つ、同じrefへの空source削除refspecを1つ対応付け、330対が固定集合内であることを検証した。
- 条件付きdry-runが330 refすべてに成功し、atomic対応を確認した。実操作は同じ条件の330 refを一つのatomic transactionで処理し、exit code 0。Git porcelainの削除成功330行とremote再観測の330消失が一致した。
- 実削除transactionは1回、各refの削除対象化は1回。固定SHA不一致の更新、拒否後の新SHA再試行、無条件deleteへの切替、無条件force、期待SHA省略lease、refspecの+、commit push/tip更新との混在は0。

`--force-with-lease=<exact-ref>:<fixed-complete-SHA>`はこの330削除refspecにだけ用いた。Gitの広告ref/明示期待値照合とreceive側のold object検証により、先行確認後の並行tip更新を無条件に消さない形にした。履歴上書きpushを一般に許可・実行した意味ではない。KEEP/監査branchへの更新はこの削除transactionに含めず、結果公開の監査branch前進だけを後の通常fast-forward pushで別に行う。

## preflightで閉じた条件

A固定`90064f1726cc671496e40b1a330da58c30537cac`、B固定`6170f3535f7e71a891dd76af17578f9703d9bbdd`の全330行をcandidate ID/完全SHA/保持先で結合した。両方でNO_CURRENT_NAMED_REF_DEPENDENCY_IN_VERIFIED_SCOPE 330、名前保持必須0、個別保留0。過去manifest/PLAN/復旧名・local branch guardは、現在のremote名の永続要件と区別した既存判断を再利用し、原文や契約を書き換えていない。

LOCALは220登録・候補checkout0。前回確認済みcontrol1515 locatorはmetadata不変、新規/変更headのcontrol31資料を限定確認して候補名の新しいliteral0。GitHub APIはopen PR0、protected branch0、ruleset0、workflowのin_progress/queued/waiting/requested/pending各0。mainのworkflow1資料にも候補名selectorなし。これは当該scope/時点の確認であり、未記録の将来利用不存在証明ではない。

## 結果とSTOP条件

成功330、保留0、失敗/不確定0、固定SHA不一致0、対象外remote削除0。811→481、並行追加/対象外更新0。IDごとの受信結果と消失を判定し、本数差だけで成功と認定していない。

個別のSHA/利用/契約変化・条件拒否があればそのrefだけ保留する設計だった。KEEP欠損、共通object/到達性異常、対象外への影響、B3保護交差、条件付き方式の破綻は全STOP条件とした。今回は発生していない。詳細なコマンド引数、期待SHA、receive応答、前後snapshot・復旧実名対応・一回性journalは新規の本人専用領域へ保存した。認証情報/内部path/非公開名は公開しない。

**今回の限定的削除許可を、local refs/worktree削除・B3/科学操作・別branch整理へ拡張せずmandatory STOP。**

## 公開直前の並行追加

削除直後のremoteは481本だった。その後、公開準備中に削除対象外のbranchが1本追加され、公開直前snapshotは482本となった。実削除330・保留0・失敗0は変わらない。新しい1本を削除集合へ追加せず、削除前から残した追加2本とともに保持した。これは削除transaction中の並行追加0と区別する。最新値は公開後にも再観測して最終通知・PUBLICATION_RESULTへ記録する。
