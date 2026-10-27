# H46残存4件：削除後保全監査

2026-10-27 JST。科学的な再計算・詳細レビューではなく、直前差分とfile bytes/Git/ref・登録の技術監査。既存の前回24件preflight・remove helper・237copy/13SOURCE監査を再利用し、今回4件へ限定適応した。

| 対象 | 結果 |
| --- | --- |
| 0038・0039・0040・0041の通常remove | 全4件exit 0、物理directoryと登録の消失を個別確認 |
| 主checkout/common root混入 | 0。全4件は承認台帳と一致するlinked worktree |
| 直前tracked/index/untracked | HEAD/branch一致、clean、index差分・未追跡なし |
| ignored非cache・submodule・Git lock | 新規重要資産なし、submoduleなし、対象Git lockなし |
| 新しいPLANと新worktree | bytes・Git blob・branch・HEAD・clean状態が削除前後で不変 |
| 元4HEADのcommit/ref | 削除後も全4件保持 |
| 既存科学SOURCE | 13固有commitを再照合、object/ref保持 |
| PLAN追加参照commit | base/provider/previous execution/original resultsの4参照をrefから保持 |
| PLAN入力5件 | 宣言hash・Git・新保持worktree同一bytesを確認 |
| 旧0039結果 | 16fileの元Git版保持 |
| 新保持worktree結果 | 16fileのGit blobと実bytes一致 |
| 最新0041 | 5fileの最新Git blob保持・復旧可能。古いcopyで置換しない |
| 既存237保全file | 前回の全SHA一致監査を再利用。今回前後のsize/inode/mtime・owner-only権限に変化なし |
| PLAN追加copy | 1件、同一bytesで私有read-only保全、非hardlink |
| 研究branch/tag | 既存ref更新・削除0 |
| 対象外登録 | 全17件が開始時と同じ登録内容で残存（共有Git系統） |
| 主checkout status | 削除前後で不変 |
| 元の非公開監査 | 10既存原本のhash不変 |

以前237copyを最初から作り直したり、必要性なく全payloadを再hashしたりしていない。前回削除後のSHA監査と、今回保存fileのmetadata差分を別の証拠として記録した。PLAN入力/結果と0041最新blobは、今回の具体的削除gateに必要な対象bytesだけ照合した。数値解析・科学module import・runner/compile/samplingは0。

copy・manifest・RESTORE index・private原本は削除tree外に保持。今回PLAN私有copyを1件追加したが、既存237copyは移動/変更しない。同disk上の保全であり、独立障害域backupとは認定しない。

削除はconsumer0041、旧準備0038、旧screening0040、provider0039の順に、同common Gitに対して順次1回ずつ普通のremoveを行った。force・手動再帰削除・prune・clean・reset・gc・unlock・branch削除・merge/rebaseは未実施。新しい対象外worktree/PLAN、固定KEEP群、他環境待ち、管理用2件は削除しない。

科学SOURCE/input/result・失敗/STOP・grant/ledger/marker・独立cohortの原版は保持Git/既存原本に残る。旧checkoutの物理copyだけを解除し、科学契約やPLANの条件を書換えていない。SERVER_B DF-native B3、LOCAL、他process・共有venv/cache/mountへ操作せず、他host接続も0。

PLANは保存されたが旧絶対pathは消失した。将来の実行には復旧workspaceや別の承認済みpath設定が必要で、即時実行可能性を偽って認定していない。復旧索引の具体的な実path/hash対応は非公開原本だけに保存する。今回の結果を他worktree・GitHub branch整理へ拡張せずmandatory STOP。
