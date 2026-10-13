# Track A サーバー側情報の棚卸し

2026-10-14 JSTの読取監査。ローカル側未照合の暫定版。[共通入口](track_a_shared_research_status.md)、[同期ルール](track_a_environment_sync_policy.md)。
既存worktreeはGIT_OPTIONAL_LOCKS=0で調査し、index refresh・checkout・reset・merge・削除を行っていない。調査範囲はこのrepositoryの登録worktreeと明示したH4/H6/H8証拠保存先で、home全域の完全探索ではない。

## Repository・worktree・公開状態

Repositoryは `HIROMU1015/Partially-Randomized-Trotter`。GitHubを共通参照先とする。
登録worktree67件（今回の独立棚卸しworktree1件を含む）を確認。読取snapshotで33件のbranch HEADがGitHubと一致、33件は同名remote branchがなく、1件はremote HEADと異なった。
**branchが未公開でもcommitが他の公開branchの祖先として取得可能な場合がある。** 66/67 HEADはfresh remote HEADからの到達性を確認し、1件はローカルGit objectから到達性を確立できなかった。これはGitHub全historyに存在しないことの証明ではない。

|識別対象|状態|意味・保全|
|---|---|---|
|H4 result branch98cdb35c|GitHub HEAD一致|元全240結果を取得可能|
|H4 manuscript branch47dc17a8|GitHub HEAD一致|前回の認証失敗報告後、今回観測では公開済み。v0.4はdraftのまま|
|H6確認321cd8c3、pilot e0e6473d、rare eee1343f|GitHub HEAD一致|固定結果・technical evidenceを取得可能|
|H8 production eaeeefc3 / resume01 96b9560e / resume02 38d386c8|各GitHub HEAD一致|文書は各時点の設計/起動snapshot。最新実行状態は別に読取確認|
|H6 migration worktree76a8ac62|同名remoteはe51f19b6|local HEADはremoteの祖先。現在の作業treeを更新せず両SHAを保持|
|H4 production-run03 a3248055|同名remote branchなし、他fresh remote祖先への到達性未確立|local commitとして保全。旧停止履歴に属し、無断push/mergeしない|
|その他のremote branchなしのHEAD|多くは別公開branchの祖先|branch公開状況とcommit取得可能性を分ける|

全67件のHEAD/remote/dirtyと選択file metadataは[worktree JSON](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/worktree_inventory.json)。一覧tableは[bundle](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/README.md)。file単位の固定証拠19件は[commit/blob/bytes/SHA索引](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/fixed_evidence_index.json)。

## Dirty・untracked・Git管理外

読取時点で既存dirty worktree12件、tracked変更969件、untracked19,981件。件数だけで未公開研究結果と認定しない。

|worktree識別名|確認した状態|扱い|
|---|---|---|
|partially-randomized-trotter（main）|staged deletion958件|原因/意図は未確認。復元・commit・resetせず保全。canonical作業場所に使わない|
|h4-monitor-fix-a-20261007|tracked未commit11件、untracked18件|旧SOURCE checkoutと現file bytesを同一視しない。file単位hashを選択収録、無断統合しない|
|h4-common-panel-h8-preflight-20261013|untracked6件|公開result summary等と別worktree上の集計資料を区別。未commitfileを元固定結果と呼ばない|
|H6 confirmation/exploration/pilot関連9 worktree|多くがlaunch output・checkpoint・cache・lock|公開summaryとprivate runtimeの所在を区別。Git untrackedでも原証拠なので削除しない|

untrackedはcopy/pushしない。大きいruntime一覧はprivate棚卸しとして保持し、公開はfamily件数と必要な少数metadataのみ。
4つのGit管理外GPTレビューMarkdownをworktree親directoryに確認した。いずれも[公開版とbyte-identical](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/external_review_inventory.json)であり、単にGit管理外の置き場所にあることを「サーバーにしかない科学判断」と呼ばない。

## 研究結果と科学レビュー

H4 geometry/B0+I、H6探索/確認/s2 pilot、H8 feasibilityの固定報告とpublic projectionsを確認。数値は[共通入口](track_a_shared_research_status.md)と元証拠を参照。
GPT判断はH4 B0+I採択、H6確認後/後続pilot後の固定レビューに区別する。v0.4主要claimの結果review、H8 core結果reviewは未確認。
独立確認はH6の事前固定8構成×32本である。H4の302 precision投影やB0+Iの決定論compileを独立Monte Carlo確認と呼ばない。

## H8現在状態の読取監査

最新確認branchは `track-a-h8-resource-map-resume02-20261014`、HEAD `38d386c8732295e6fab5d6fcbd47fbe586dc207d`。
実行SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、seed SOURCE `cb8a56c6b2a772827275f9200b02638030f33f72`、H8 linear1Å/STO-3G/DF28/sector4900、native/symmetric-directional、P=0。
既存grantは92 signal→eligible探索592 group→top2 union最大8の独立32確認というcoreのみで、今回の棚卸しによって範囲を変更していない。

読み取りsnapshotでsignal92/92、探索COMPLETE203/592（祖先126＋新77）、新77件のcheckpoint/completeはsource/task/result digestとnative exit0/reap fieldが整合。8 active worker生存、coordinator PID/start identity一致、first STOP理由null。source closure265の現在bytes一致も確認。
独立確認directory・最終結果/complete markerは当snapshotで未確認。現在の探索partialを最終費用mapへ集計しない。
ledgerのknownとreserved、歴史unknown=null、reset/refund=falseをsnapshotに記録。予約をactual実数と呼ばず、H4/H6/H8の異なるcarry系列を足して共通上限を新設しない。
この監査はmetadataの非atomic読取で、全rawのSHA再照合・最終cleanup・continuous監視を代行しない。H8の停止・再起動・grant/checkpoint/ledger/source/worktreeへの書込みは0。

## サーバーに保持する情報・公開対象外

|識別名/種類|確認状況|GitHub共有する部分|本体の扱い|
|---|---|---|---|
|H4凍結NPZ6|存在と各期待bytes一致。既存SHA receiptを参照|H/DF/state/input identity、bytes、記録SHA、source/freezeの来歴|コピー/array読込/新hashなし、private原本保持|
|H4 map・B0+I原本|完了public receiptとprivate保存場所の存在を確認|軽量結果projection・native receipt・file hash|raw/runtime/cache/lockは原本保存先に保持|
|H6探索/確認/pilot runtime|Git untracked/管理外の原本とpublic summaryを確認|固定結果とmetadata来歴・known/reserved会計|trajectory/NPZ/checkpoint/cacheを無条件コピーしない|
|H8入力/state NPZ|snapshot1,299,688 bytes、記録SHA d4c43461…、file presence確認|H/state/input sourceとreceipt SHA|array/load/再hashなし、private原本保持|
|H8実行checkpoint/ledger/grant|read-only scalar snapshot確認|識別名・source/hash・coverage・known/reserved・scope|現在file本体を公開copy/更新しない|
|private venv2系列|H4用とH6/H8 native用の存在確認|既存固定profile、version/RECORD差・未検証事項|venv/activation/local cacheを共有せずinstall/upgrade0|
|ローカル設定/credential|内容を今回収集していない|共有対象なし|token/秘密/endpoint/SSH/個人設定を公開しない|

公開資料では絶対保存先を識別名へ置き換えた。[保存存在・identity](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/private_storage_presence.json)は本体移行の代替ではない。既存public資料にpathがある場合も、新しい共有資料へ転載しない。

## 未確認と同期候補

ローカル側のbranch/commit/dirty/review・研究方針差は未確認。H8最終map/result commitは未取得。main staged deletionの原因、旧dirty11件の完成/放棄の意図も未確認で、現状態を壊さず担当へ返す。
候補は固定public資料の取得、レビュー4件の公開コピー参照、H4/H6/H8の最新結果branchと時点差の統合、必要ならa3248055の履歴が必要か確認すること。大容量原本移行は独立再現の具体的必要性が出たときだけ、bytes/SHA manifestと専用認可を用いて設計する。
新規科学計算、共有venv/Git設定変更、他branch mergeは今回0。
