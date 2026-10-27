# SERVER_A 研究単位別一括削除計画（未実行）

本計画は削除実行の承認ではない。対象はsnapshotのID・branch・完全HEAD・私有対応表で特定し、連番・branch名から物理pathを推測しない。LOCALの承認・再開不要判断をAへ適用しない。

## 正確な候補集合と推奨順序

| 計画集合 | 件数 | 位置付け・owner判断 |
| --- | ---: | --- |
| A-LEGACY-20 | 20 | 旧研究の先行推奨一括集合。Q1のH4旧準備9＋Q2の停止済み証拠11。利用終了未確認 |
| A-CURRENT-H46-4 | 4 | 新しいH46研究の別判断。既存計画のconsumer/provider関係を一体で終了するか、供給側を残す。Q3 |
| A-ADMIN-1 | 1 | 旧棚卸しlinked checkoutのみ。研究集合と区別し、旧orphan refを保持。任意Q4 |

A-LEGACY-20の正確な集合：SERVER_A-WT-0003, SERVER_A-WT-0004, SERVER_A-WT-0005, SERVER_A-WT-0006, SERVER_A-WT-0007, SERVER_A-WT-0008, SERVER_A-WT-0010, SERVER_A-WT-0011, SERVER_A-WT-0014, SERVER_A-WT-0017, SERVER_A-WT-0018, SERVER_A-WT-0019, SERVER_A-WT-0021, SERVER_A-WT-0022, SERVER_A-WT-0023, SERVER_A-WT-0024, SERVER_A-WT-0025, SERVER_A-WT-0027, SERVER_A-WT-0028, SERVER_A-WT-0034。

A-CURRENT-H46-4の正確な集合：SERVER_A-WT-0038, SERVER_A-WT-0039, SERVER_A-WT-0040, SERVER_A-WT-0041。

A-ADMIN-1の正確な集合：SERVER_A-WT-0037。

研究の技術候補は計24件。20件の利用終了が承認され、直前条件も満たすなら20件を1つの論理batchとして扱える。固定の小バッチ上限は置かない。H46の4件はサイズ制限で分割するのではなく、現在の研究単位の利用判断が独立なため別集合とする。全24件の用途終了が明示され各gateが通る場合は、後日の具体的削除指示がその正確なunionを対象にしてよい。

0039のprovider/input pathは0041内の現在のdraft PLANが参照する。0041を継続するのに0039だけを不用意に解除しない。旧proposalの0023→0025も同様に、利用終了のowner判断と実際の固定実行契約を分ける。いずれも現行SOURCE・契約を書換える計画ではない。

保持集合16件：SERVER_A-WT-0001, SERVER_A-WT-0002, SERVER_A-WT-0009, SERVER_A-WT-0012, SERVER_A-WT-0013, SERVER_A-WT-0016, SERVER_A-WT-0020, SERVER_A-WT-0026, SERVER_A-WT-0029, SERVER_A-WT-0030, SERVER_A-WT-0031, SERVER_A-WT-0032, SERVER_A-WT-0033, SERVER_A-WT-0035, SERVER_A-WT-0036, SERVER_A-WT-0042。

他環境確認集合1件：SERVER_A-WT-0015。

これらは現在の一括削除候補へ含めない。KEEPの中にmain/common root、専用interpreter、固定SOURCE/実行path、永続handoff、custody/index供給元を含む。branch/refは候補を含め全件保持する。

## 保持するものと復旧可能性

1. 各IDの現行branchと完全HEADを指すref、detached0034の保持handoff ref、13科学SOURCE commitを保持するrefとobject storeを残す。worktree解除とbranch/ref削除を結び付けない。独立cloneのrootをlinked worktreeとして削除しない。
2. 原科学SOURCE・input/result・失敗/cohort・grant/ledger/marker/STOPはversioned bytesと履歴・bindingを保持する。旧失敗を後続成功で置き換えない。
3. 私有の保全manifestと元相対pathを残した237コピーを保持する。候補に属する入力・ログ・未追跡analysisのsource/copy SHA一致を確認済み。元worktreeへのsymlink/hardlinkを保存先として使っていない。WP11のvenvやmain rawは移動しない。
4. 非公開RESTORE indexには元path、共通Git、ref、HEAD、追加コピーの相対path・SHA・保存先を記録した。必要な復旧は別承認後、保持refからcheckoutを再生成し、元と同じ相対階層へ検証済みcopyを戻す。既存path・SOURCEの上書きを行わない。今回復旧や科学runnerでの再現実行はしていない。
0041の最新5件は保持された新commitから復旧する。古い私有copy5件は履歴snapshotであり、最新checkoutへ上書きしない。元のcopyと最新Git版の関係を非公開restore indexで区別した。

5. 同disk上のコピーとGit保持はworktree-only整理の保全であり、別障害域backupの認定ではない。

## 削除直前preflight案

以下は後日、利用終了判断と具体的削除指示を受けた実行担当が、対象の差分だけ確認する。今回実行済みとは扱わない。

- 対象集合が承認された正確なID集合に含まれ、私有path対応・環境SERVER_A・root ownership・realpath/common directoryが一致する。新規registrationや別pathへ機械的に横展開しない。
- 各HEAD・branch・status/index・untracked/ignored leaf・lock/prunableがsnapshotから変わっていない。通常cacheだけか、保全済み科学資産/未追跡sourceかを区別する。新規fileやbytes変更があればそのIDを保留する。
- 各HEAD/SOURCEへの保持refが存在し、必要objectsを保持する。local-onlyを含むbranchを残す。共有Git・alternate・他worktree登録を破壊しない。
- 全追加コピーの存在・regular file・ownership・source/copy独立性・受入れ済みSHA/sizeを確認する。原本がまだある段階で差分を確認し、コピー先が対象treeの内側でないことを確認する。破損/欠落時に元を削除しない。
- 本人の可視processと既知controller/再開設定・SOURCE/provider/input/result/STOPの新規path束縛を確認する。旧test/文書の記載を実行中依存へ自動昇格しない。既知の新bindingがあれば責任者へ返す。別所有者jobやB3を停止・変更して解除しない。
- ownerの用途終了判断が対象研究単位に適用される。旧proposal/draftを継続するconsumerがある場合、providerを単独解除しない。固定KEEP群のcontractを暗黙解除しない。
- normal git worktree removeで受理される普通のlinked checkoutだけを扱う。main/独立clone common root、lock、dirty、所有者不一致、common異常をforce/prune/clean/reset/gc/手動rmで回避しない。

## 個別保留と全体STOP

特定IDだけの新しいdirty/untracked、保全コピー欠落、利用中process、対象外path、保持ref不一致は当該IDを保留して、原因・根拠を記録する。他の独立IDは同じ承認集合と共通gateを満たす場合だけ進める。

共通Git/ID対応の不整合、コピー先/元の取り違え、固定科学SOURCE/STOPの破損、許可範囲の不一致、共通providerを損なう可能性、複数IDに共通する保全異常、共有環境への影響を検出した場合は全batch STOP。新たな科学計算やbranch書換えで回避しない。

普通のremoveが拒否したIDをforce・prune・clean・reset・gcや手動強制削除で通さない。削除後のrefやdataを片付ける別の整理へ自動進行しない。

## 削除後に行う保全検証案

- 成功IDだけ登録解除されたことと、失敗/保留ID・KEEP群・他環境の登録/branchに予期せぬ変化がないことを比較する。
- common Git・各保持branch/HEAD/SOURCE ref・objectsと、私有237コピー・manifest・restore index・外部custody/原本・STOPが残ることを確認する。
- rawや科学数値を再解析せず、必要なmetadataと保存byte/SHAで保全を確認する。復旧が必要でも勝手に新科学実行やcontract再採択をしない。
- 結果CSVへ成功/個別保留/拒否理由、作業前後登録数、保持ref・保全検証結果を記録し、匿名化公開後mandatory STOPする。

**この計画ではremove等の破壊操作は実行していない。今回の承認範囲は準備と公開まで。**
