# SERVER_B 83件の一括削除準備・171件の追加保全

2026-10-27 JST。公開版はID・件数・非機密保全manifestだけを追加し、詳細path/branch/HEAD/復元bindingは本人専用領域に保持。固定資料は[34f09aa44931の既存報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34f09aa449314b16e9a239026e10779ff1969673/inventory/server_b/worktree_branch_preparation_20261027/SERVER_B_WORKTREE_REPORT.md)。旧資料を上書きしない。

**Task A：固定83件すべてREADY_FOR_BULK_REMOVAL_APPROVAL。Task B：171件すべてで今回確認した管理外資産・履歴の保全条件を解消し、うち86件を新たにTECHNICALLY_READY_PENDING_OWNERとした。残る85件は具体的依存を保持する。実削除・branch削除・merge・科学実行は全て0。最新登録数は379件のまま。**

## Task A

83件の利用終了意向は今回のユーザー指示で確認済み。元分類は[TASK_A_83_READINESS.csv](TASK_A_83_READINESS.csv)のoriginal_classificationへ保持し、今回の判断を別fieldに記録した。準備完了83、保留0。正確なID・研究単位別件数・一括順序と、削除直前preflight/個別保留/共通STOP/削除後監査/復元は[BULK83_EXECUTION_PLAN.md](BULK83_EXECUTION_PLAN.md)と[READY83_IDS.csv](READY83_IDS.csv)。実削除の明示指示と直前preflightが残る。

live GitHub tipとlocal保持refが83 HEADを保持し、tracked clean・untracked0・ignored0を現在の限定確認で照合した。同じGit tree/履歴からtracked SOURCE/code/文書/結果/図表/失敗/STOPを取得できる。既存の[公開worktree表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34f09aa449314b16e9a239026e10779ff1969673/inventory/server_b/worktree_branch_preparation_20261027/SERVER_B_WORKTREES.csv)をIDで辿り、詳細な最新branch/HEAD/実path/復元表は非公開で保持する。外部private raw/checkpoint等はGitHub保持と同一視せず、削除対象外に原位置保持する。未公開commitや固有の管理外資料を持つ別IDを83件へ混入させていない。

## Task B

[TASK_B_171_PRESERVATION.csv](TASK_B_171_PRESERVATION.csv)は元171件をそのまま保持する。保全条件解消171件、追加の技術候補86件、DEPENDENCY_UNRESOLVED84件、B3の既知SOURCE/input bindingを持つKEEP_REQUIRED1件。保全未完了0件。利用終了判断は171件すべて未確定であり、新候補も実削除未承認である。

管理外資料のない48件、初期のbytecode等だけの37件、その他の差分・ignored資料を区別し、全171 checkoutの一律コピーはしていない。3件の専用runtimeについては最初にサイズと参照を確認し、合計約356MiBの小さいpayloadを追加保存した。共有runtimeやリンク先をコピーせず、symlinkはliteralで保持した。再生成可能bytecode/cacheはSOURCEの存在を確認した範囲で除外した。

保全archive94個、ファイル対応延べ198,475件、論理8,290,854,448 bytes、archive 2,331,537,802 bytes。うち前回の47ファイル・4,076,634 bytesは検証して再利用し、新規file-copy対応は延べ198,428件。SOURCE/input/result/raw/checkpoint/STOP/失敗log/旧grant/ledger/markerを科学的に統合せず元worktreeごとに保存した。すべて保存byteのSHAとarchive読戻しを照合。staged/unstaged差分を持つ3件では原indexとbinary patchを保存し、非公開の別indexでstage treeの完全一致とその後のworking差分適用を検証した。原worktreeでadd/stash/reset等は行っていない。

live tipで到達性が閉じていなかった76 HEADには、3個のfull-ancestry Git bundle、計584,875,649 bytesを作成。元object store/alternateを参照しない別bare repositoryへ取り込み、ref一致とconnectivityを確認した。detachedで保持refがなかった3件もこの中に含む。Git履歴保全とprivate原本保全を別に検証した。詳細は[PRESERVATION_MANIFEST.csv](PRESERVATION_MANIFEST.csv)。

**保全ファイルは原本と別inodeだが同じfilesystem上にあり、独立バックアップとは表現しない。** 元のraw・研究結果・SOURCE・index・HEAD・branch・STOPは移動/変更しない。数値の再解析、計算や科学SOURCEの実行もしていない。

## 残る具体的依存

旧4077件の依存監査、B側4474件の既知path索引、editable/runtime/Git alternate、本人UIDの対象path使用を再利用。以前の依存監査外だった55 HEADだけSOURCE/controlの追加差分を確認し、2,364 unique blob・111,681,913 bytesを静的に読んだ。単なる文章引用やworktree観測と固定bindingを区別し、実環境への影響が閉じていないものを[DEPENDENCY_HOLDS.csv](DEPENDENCY_HOLDS.csv)へ残した。これは全環境の依存不存在証明ではない。

| 保留する研究単位 | 件数 |
| --- | ---: |
| BASELINE_OR_OTHER_UNRESOLVED | 2 |
| DF_H2 | 6 |
| DF_H4 | 43 |
| DF_H46_PF_PR | 2 |
| DF_H6 | 14 |
| DF_H8 | 4 |
| DF_NATIVE_TRACK_A | 4 |
| TRACK_B_G6_HISTORY | 10 |

B3保護3 IDを候補へ追加せず、B側の既知binding1 IDもKEEP_REQUIRED。[B3保護報告](B3_PROTECTION_REPORT.md)は今回の操作scopeを記録し、現在の稼働や科学完了を新たに判定していない。旧63件は[影響と引継ぎ](OTHER_ENVIRONMENT_IMPACT.md)のとおり未解消のまま保持し、SERVER_Aへの接続や全件再調査はしていない。

最終確認で対象254 HEAD/indexと列挙済み195,237 source-leaf metadataに変化を検出せず、対象pathに本人UIDの利用を検出しなかった。これは観測時点の確認であり、削除直前のfresh確認を省略する根拠ではない。他chatの無関係なGit進行を凍結していない。

## 次の承認事項

[NEXT_ACTIONS.md](NEXT_ACTIONS.md)。ユーザーが固定83件の実削除をまとめて指示するか判断し、別途86新候補の利用終了を判断する。依存85件は具体的な固定pathの役割が解消されるまで保持する。公開cloneの新directoryだけを通常fast-forward更新し、local/remote SHAと公開byteを照合後mandatory STOPする。
