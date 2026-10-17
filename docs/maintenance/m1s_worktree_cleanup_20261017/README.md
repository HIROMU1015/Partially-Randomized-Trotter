# M1-S worktree条件付き整理：実施結果と保全記録

2026-10-17 JST。**削除0件・C01–C31すべて保留。管理記録の公開後mandatory STOP。**

対象：`HIROMU1015/Partially-Randomized-Trotter`。
監査基点：[893da184e74e682dfb1bc97102c61bbe8ce2895e](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/893da184e74e682dfb1bc97102c61bbe8ce2895e/docs/maintenance/m1s_worktree_audit_20261017)。
今回の条件付き削除承認と利用終了確認は[依頼・利用状況の記録](request_and_utilization_context.md)を参照。
科学研究の結論・契約・STOP状態は変更していない。

## 結果

| 対象・確認 | 結果 |
|---|---|
| C01–C31 | 全31件保留。通常のworktree removeも未試行 |
| 削除済みID・branch・path | なし |
| 残存worktree | 217件。M1-S 42件、対象外175件 |
| 全42 M1-S branch | local／GitHub同名remoteのHEADが固定値と一致。全て保持 |
| 元repositoryの全227 local branch | 前後の名称・HEADが一致 |
| P01–P03、H01–H08 | 登録・HEAD・index・全ローカルファイルmanifest一致。削除・再分類なし |
| 対象外175 worktree | 登録・branch・HEAD等のmetadataが前後一致。個別content監査・変更なし |
| 親worktree | HEAD・index前後一致。既存dirty変更をstage／commitしていない |
| C候補の追跡ファイル | 延べ228,390件をHEADのGit blobと照合、差異0 |
| 管理記録の整合性検査 | 34件PASS・0件FAIL。科学的な数値testではない |
| 新規科学計算・数値再解析・SOURCE変更 | 0件 |
| force／prune／clean／強制削除 | 0件 |

保全検査の状態は `PASS`。これは**削除可能性のPASSではない**。
全31候補は未追跡項目があるため、今回の削除条件に対して不適格である。

## 全31件を保留した理由

旧監査のcandidate CSVではstaged／unstaged／untrackedが全て0件だった。
今回は各候補に次の未追跡symlinkがあり、準備時・最終確認時の両方で確認した。

| 未追跡path | type | link先 |
|---|---|---|
| `.agent/docs` | symlink | `/home/abe/Project/Partially Randomized Trotter/.agent/docs` |
| `CLAUDE.md` | symlink | `/home/abe/Project/Partially Randomized Trotter/CLAUDE.md` |

ユーザーからC01–C31の利用終了は確認済み。しかし、利用終了の確認は未追跡項目の除去承認ではない。
未追跡linkをcacheとみなしたり、削除条件から除外したり、ignore設定を変更したりしていない。
通常の `git worktree remove` の失敗を試す必要もなく、preflightで保留した。
誰が追加したかは断定しない。link先を追跡して内容を開くことも、削除することも行っていない。

C31の既知のignored Python bytecode4件は旧監査と同じpath・SHA256で保持された。
今回の直接の保留理由は他30件と同じ未追跡link2件であり、bytecodeだけを別途削除していない。
将来C31を整理する場合も、通常のremoveで安全に扱えなければ保留する条件は維持される。

各候補のID・branch名・絶対path・固定HEAD・保留理由は[candidate_outcomes.csv](candidate_outcomes.csv)。
残存217件の一覧は[remaining_worktrees.csv](remaining_worktrees.csv)。対象外行はmetadataのみで、個別監査を意味しない。

## 検査方法と証拠

1. 旧監査README／CSV／plan／inventoryのbytesを固定commitと照合した。
   [basis_identity.json](basis_identity.json)にbytes・SHA256を記録した。旧監査は上書きしていない。
2. 元repositoryの全local ref・worktree登録、全42 M1-S remote HEAD、親HEAD/index、
   P/H全11拠点のファイルmanifest・indexを準備時と最終時に記録した。
3. C01–C31のbranch／HEAD／登録、staged／unstaged／untracked／ignored、index flag、lock、
   全追跡fileの実Git blob、特殊file／入れ子repositoryを読み取り専用で調べた。
   `GIT_OPTIONAL_LOCKS=0`、`GIT_NO_LAZY_FETCH=1`を適用し、index refreshを要求していない。
   保全のためのhash照合であり、科学SOURCEのimport・実行・科学値の再評価はない。
4. 読み取り可能な同UIDのprocess cwd・fd・明示path参照は0件だった。
   `sd-pam`／`sshd`の4 processではcwdまたはfdにPermissionErrorがあり、他UIDのprocess内容は調べていない。
   process snapshot単独による全session不使用の証明とは表現しない。利用者確認は別途取得した。
5. 全候補に未追跡項目を認めたため、**削除コマンドを発行する経路へ進まなかった**。
   候補の個別remote再照会直後に削除したという記録もない。全42 remote HEADは前後に一括照合している。
6. 前後検査で、元repositoryのbranchとworktree登録は全件一致した。
   P/H全ファイルmanifestも一致し、対象外研究へmutationを行っていない。

| 記録 | 内容 |
|---|---|
| [preparation_checks.json](preparation_checks.json) | 準備時の31候補検査。利用者回答前なので当該時点の状態はPENDING |
| [final_checks.json](final_checks.json) | 利用終了確認後の最終31候補検査、process観測、remote HEAD、削除0件 |
| [unexpected_untracked_inventory.json](unexpected_untracked_inventory.json) | 31候補×2 linkのtype・target。link先は追跡しない |
| [before_snapshot.json](before_snapshot.json)／[after_snapshot.json](after_snapshot.json) | 全ref／登録の前後記録とP/H各拠点のfile manifest digest |
| [preservation_checks.json](preservation_checks.json) | 全42 branchの固定値、local・remote前後比較、P/H比較、保全結果 |
| [verification.json](verification.json) | 管理記録の整合性・schema・非変更検査 |
| [publication_receipt.json](publication_receipt.json) | payload固定commitのGitHub公開bytes・SHA256照合 |

P/Hの全ローカルfile（tracked・untracked・ignored・symlinkを含む、Git adminはindex/lock別記）は
延べ85,212件の前後manifestを比較した。symlinkはtarget文字列を記録し、先を追跡しない。
24 MB規模の全path manifest2組をそのまま重複公開せず、各拠点の件数とcanonical SHA256を公開した。
完全manifestはこの管理作業の独立した`/tmp/m1s-cleanup-before-full-20261017.json`／
`/tmp/m1s-cleanup-after-full-20261017.json`に保存し、科学証拠の唯一の保存先としては用いない。
canonical JSONの規則はsnapshot内に記録した。対象外研究のcontent全体を検証したという主張はしない。

## 科学成果・provenanceの保全

全42 branchのHEADが不変で、同じHEADがGitHubにも保持されているため、各branchの歴史と固定SOURCE・raw・
契約・authorization・grant・ledger・marker・レビューへの到達関係を維持する。
削除や履歴改変を一切行っておらず、失敗試行・消費済みmarker・未解像結果も変更していない。
HEADの追跡file照合を独立科学再現・数値correctness再検証とは扱わない。

主要anchorのHEADと保全拠点：

- P01 Stage S：`f2bfb8770e6858de56a63e50e68f07ff2d6464b6`。
- P02 統合成果：`5f079dbe67bb35eff35613857b169fdf812af4b8`。
- P03 別履歴の先行研究調査：`4301c0f87a16be45b620304c3fa935d5aa2d721a`。

C23の固有の旧独立correctness準備SOURCEも、当該branch
`track-b-m1s-gate0-v5-independent-correctness-preparation-20261016` の
`d6d35cfb79b97f9cb218c9169b427aacf6a4a334` のまま保全する。
主要anchorだけに集約してbranchを捨てる判断はしていない。

## 公開領域と停止・次の担当

管理記録は既存の独立clone `/tmp/prt-m1s-worktree-audit-20261017`、
管理専用branch `track-b-m1s-worktree-audit-20261017` に新しいdirectoryとして追加した。
このbranchは元repositoryの固定42 branchに含まれない。今回その42 branchも親worktreeも切り替えていない。
元repositoryにworktreeを追加せず、既存の管理branchのみに管理記録をcommit・pushする。

**mandatory STOP。追加削除・branch整理・H01–H08の再分類は行わない。**
次の担当はユーザー。保留のまま維持するか、C01–C31の未追跡symlinkをどう保全・扱うかを判断する。
新たな具体的指示がある場合だけ、Codexがその範囲を再確認し、あらためて削除直前の安全性を検査する。
今回の記録はlink除去や追加削除の承認ではない。新規科学計算や次期研究段階への移行も承認しない。
