# 先行64件：4バッチの実行前整理計画

PUBLIC_REDACTED_FINAL_THREE_ENVIRONMENT_PLAN。2026-10-27 JST。固定LOCAL計画`e5651ab7b7f1682dac51a756e6b277f4a2d9c217`、A_R1 `5eef8de248d454cc2497ef9631bfc171b1a78aae`、B_PATH_1 `33fd56d573c3d6022609b9141daf2429d19c75b4`。

**64件すべてをBATCH_APPROVAL_CANDIDATEとした。既存3環境の確認範囲と今回の限定差分では、64件内の具体的阻害要因0、追加保全必須0、限定追加確認必須0、具体的物理保持義務0。実際の削除許可は0であり、全75件を維持した。** Bの実パス照合未実施という前回gateは解消済み。未確認の副条件は残余リスクと直前preflightへ分ける。一般的なUNKNOWNの不存在証明や同じ全件調査の反復を、次の承認候補化の条件にしない。

## 対象集合と推奨順序

前回の固定[64件CSV](../pre_cleanup_75_20261026/LOCAL_64_CONDITIONAL_CLEANUP_PLAN.csv)の正確なID集合とバッチを再利用した。連番やbranch名から追加していない。各行のbranch・完全HEAD・保持ref・SOURCE/STOP・保全manifestは[今回64件表](LOCAL_64_BATCH_READINESS.csv)にある。実path・common Git・backlink・archive/manifestの実位置・具体的な将来コマンドは非公開原本にのみ保存した。

| 推奨順序・バッチ | 件数 | 正確な対象ID | 現在の主判断 |
| --- | ---: | --- | --- |
| Batch 1：図表・スライド | 7 | LOCAL-WT-0203, LOCAL-WT-0204, LOCAL-WT-0205, LOCAL-WT-0206, LOCAL-WT-0207, LOCAL-WT-0208, LOCAL-WT-0209 | BATCH_APPROVAL_CANDIDATE |
| Batch 2：QC | 18 | LOCAL-WT-0023, LOCAL-WT-0024, LOCAL-WT-0025, LOCAL-WT-0026, LOCAL-WT-0027, LOCAL-WT-0028, LOCAL-WT-0029, LOCAL-WT-0030, LOCAL-WT-0031, LOCAL-WT-0032, LOCAL-WT-0033, LOCAL-WT-0034, LOCAL-WT-0035, LOCAL-WT-0036, LOCAL-WT-0037, LOCAL-WT-0038, LOCAL-WT-0039, LOCAL-WT-0041 | BATCH_APPROVAL_CANDIDATE |
| Batch 3：Research B B2D Outer | 34 | LOCAL-WT-0271, LOCAL-WT-0272, LOCAL-WT-0273, LOCAL-WT-0274, LOCAL-WT-0275, LOCAL-WT-0276, LOCAL-WT-0277, LOCAL-WT-0278, LOCAL-WT-0279, LOCAL-WT-0280, LOCAL-WT-0281, LOCAL-WT-0282, LOCAL-WT-0284, LOCAL-WT-0287, LOCAL-WT-0288, LOCAL-WT-0289, LOCAL-WT-0290, LOCAL-WT-0291, LOCAL-WT-0292, LOCAL-WT-0295, LOCAL-WT-0297, LOCAL-WT-0299, LOCAL-WT-0302, LOCAL-WT-0305, LOCAL-WT-0306, LOCAL-WT-0307, LOCAL-WT-0308, LOCAL-WT-0310, LOCAL-WT-0311, LOCAL-WT-0312, LOCAL-WT-0314, LOCAL-WT-0315, LOCAL-WT-0316, LOCAL-WT-0318 | BATCH_APPROVAL_CANDIDATE |
| Batch 4：N1/N3保全検証済み | 5 | LOCAL-WT-0003, LOCAL-WT-0004, LOCAL-WT-0006, LOCAL-WT-0007, LOCAL-WT-0008 | BATCH_APPROVAL_CANDIDATE |

旧SOURCE未解決11件、Research BのLOCAL-WT-0283/0286、共有Git本体・主checkout・他の研究worktreeはこの64件から除外する。今回64件からの新たな個別除外は0。将来preflightで具体的な問題が出たIDだけを保留し、他の承認済みで条件充足したIDは独立して扱う。

## 各バッチの判断材料

### Batch 1：図表・スライド7件を最初に推奨

版別の図表・スライド・原稿・画像・PDF/SVG・作成script・選定入力・設定・provenance/STOP、関連参照文書を2,052 files／234,267,578 bytes保管済み。元の相対階層と非公開manifestのID/branch/完全SHA/hash対応を残す。元checkoutなしで保存資料を閲覧できるという既存確認を維持し、作成scriptや科学runtimeの可搬実行までは保証しない。現在の保管rootと代表ファイルの可読metadataも確認した。

0208/0209は自身の固有branch/HEADを保持する。0207の同名remote branchはないが、local branchは元HEAD`7c0396be147887d0b33580236b4dce1890e4f230`で保持され、remote `refs/heads/codex/pr-resource-slides-current-20261022`のtip`6c68efbaabd57c3a7f9f3b0fca48ed5c46436bd5`から祖先到達することを再確認した。0207自身のlocal branchも削除しない。

Aは0203〜0206の公開Git等への論理参照4件、残り3件は確認scope内で必須依存未検出。Bは7件とも必須依存未検出。ローカルにも具体的path束縛・未保存資産・lock・可視process一致は検出しなかった。追加の具体的保全阻害要因はない。バッチ承認後の直前差分と残余リスクの扱いを満たせば、通常のcheckout整理へ進む候補になる。

### Batch 2：QC18件

QC strategy handoff `fb635213d99e666852b87212c051ab08a6977c41`を成果の入口とし、18件それぞれのbranch/完全HEAD・固定SOURCE/input・診断・失敗・STOPを保持する。選定HEAD版18,216パスの保全対応を再利用する。代表文書や祖先関係だけで他の科学証拠を代替しない。18件は同名GitHub tipと現在も一致する。

LOCAL/A_R1/B_PATH_1の確認scope内に必須物理依存を検出せず、現在のローカルstatus/ignored metadata・archive可読性・lock・可視processにも新たな具体的問題はない。未記録job/alias/契約/private custodyは範囲外のリスクとして残し、兆候が出た対象だけ確認する。ownerの利用終了再承認やQC全件の再レビューは不要。

### Batch 3：Research B34件

最終成果・pause handoff `e87375fa9a998dce80792be0f15597c747f00d83`を成果の入口とし、34件それぞれの保持ref・独立cohort・SOURCE/input/control・失敗/STOPを保全する。選定HEAD版8,094パスの対応を再利用する。対象外0283/0286のSOURCE供給責任は今回解除せず、そのworktreeを維持する。

34件は同名GitHub tipと一致し、3環境の確認scopeでは必須物理依存未検出。Research Bのpauseは現在のサーバーBのDF-native L_D=0 B3停止を意味しない。B3のjob・SOURCE/provider/input・結果/raw/cache・Git管理領域・runtime・grant/ledger/marker/STOPをこの整理対象へ含めない。B_PATH_1が既知B3 chainを照合済みという条件は解消したが、未知の別名や未記録再開までは認定しない。

### Batch 4：N1/N3保全検証済み5件

対象は上表の0003/0004/0006/0007/0008のみ。N3 closure `ccc575ee339be091fd4031272c38bee13866f097`を入口に、各自身のbranch/HEAD・SOURCE/input・validation/失敗/STOP・保存資料を残す。非祖先の0003はclosureだけでは代替できず、固有refを保持する。5件は同名GitHub tipと一致し、3環境の確認scopeでは必須物理依存未検出。旧SOURCE3版・55宣言未解決の11件は入れない。

## 残条件を3種類へ分ける

| 種類 | 今回64件での扱い | 後続の判断 |
| --- | --- | --- |
| 具体的阻害要因 | 確認済みの固定path束縛・未保存重要資産・稼働job依存・ref欠落・lock等は0件 | 新たに確認されたIDだけ候補から外し、具体的な保全/契約/依存対応へ戻す |
| 操作直前preflight | 全64件で将来実施が必要、今回の確認を完了扱いにしない | 承認範囲と対象を固定した後、下記の差分だけを確認する |
| 範囲外の残余リスク | 未記録mount/export/bind/alias・共有実体、別namespace・他job/再開、外部private custody・未索引契約等 | 不存在を全証明したとは扱わず、具体的な兆候がなければ無期限の再棚卸しへ拡張しない |

固定SOURCE/path契約や共有環境への具体的な危険が判明した場合、一般的な削除承認だけでその保護を解除しない。その対象だけ保持し、必要なら契約当事者の対応と適用される明示指示を得る。所有者の将来workspace利用終了方針はすでに確定している。

## 将来の限定preflight：実操作の直前に行う

1. 非公開対応表で承認されたIDを正規完全pathへ解決する。本人所有・登録済みlinked worktree・branch/完全HEAD・gitfile backlink・common Gitを照合する。対象がarchive、主checkout、共通Git、他worktree、未知のmount/aliasへ変わっていないか確認する。worktree lockがあれば勝手にunlockしない。
2. HEAD/branchとtracked・全untracked差分を確認する。**ignoredはdirectory名の一致だけで済ませず、その配下に新しい重要資産がないかを変更scopeで確認する。通常のworktree removeもignored filesを失わせ得るため、Gitの拒否だけに依存しない。** 既知guideと再生成可能cacheを研究raw/独立結果・STOP・private原本と混同しない。必要な未保存物が出たら、そのIDだけ非破壊保全の具体案へ戻す。
3. 現在利用・可視の関連processのcwd/引数・既存の再開記録・SOURCE/provider/input/path束縛の新差分を確認する。共有環境・他jobを操作しない。未知全namespaceの探索やA/B再SSH・全面調査を既定にしない。具体的な外部変更情報があれば該当環境担当の限定確認を待つ。
4. 各自身のlocal branch/完全HEAD、必要なremote保持refの到達、共通Git objects、元SOURCE/result/STOPを保持する条件を確認する。0207の祖先保持と0208/0209/0003の固有refを落とさない。branch tipが動いた場合、古い対象SHAを黙って新tipへ変更しない。
5. 保全manifest・archiveの可読性と新しい破損/変更兆候を確認する。既存hash/Git blob検証を再利用し、必要なら変化したファイルだけを確認する。全10,336 filesの無条件再コピー/再hashや大規模独立backupを前提にしない。外部private rawへの新しい具体的依存が出た場合だけ対象・容量・保存先を具体化する。
6. 問題がないIDだけ実行候補を確定し、承認された正確なバッチID集合と照合する。今回の主判断はpreflightの合格や実行許可を代行しない。異常IDを外す際もbranch/原本/契約/他worktreeへ変更を波及させない。

## 将来の登録解除・checkout整理方法

今回コマンドは実行しない。非公開資料に、対象ごとの正確な将来コマンド文字列を保存した。承認と直前preflightを満たしたIDへ、通常の`git worktree remove`を1件ずつ適用する計画である。この操作はcheckout directoryと登録の双方を除去する。branchや共通Git objectsの削除はしない。

`--force`、prune、clean、gc、手動の強制再帰削除で拒否を回避しない。拒否・dirty・新しい重要資産・lock・ref変化・未確認の具体的契約や稼働依存が出たIDはSTOPして維持する。残りの明示承認済みで条件充足したIDだけ続行可能とし、無関係なjob/branch/ファイルを変更しない。動作しているB3の停止・再起動・SOURCE変更は一切行わない。

操作後は承認IDだけの登録・directory消失、全保持ref/必要object・共通Git・archive/manifest・他worktree/mainの維持を確認する。想定影響は不要checkoutと登録の減少。disk回収量は未測定で、branch/Git本体やarchiveを減らす計画ではない。

## rollback・再checkout

非公開原本に元の正規path・branch・完全HEAD・common Git・登録backlink・known guide・archive/manifestのbindingと、再checkout参考コマンドを保存した。必要なbranch/objectを保持したまま、pathが空き、branchが別worktreeで使用されておらず、記録HEADに一致することを確認して、通常のworktree再作成を行う案とする。path/branch競合やHEAD変化をforce/resetで上書きしない。

復旧対象と操作を具体的に指示された場合に行い、科学的な再実行は含めない。tracked checkoutはGitから再作成できるが、ignored runtime/cacheやguideの再配置、動的実行環境・全scriptの可搬実行まで完全復元できる保証はない。保存資料そのものは元checkoutなしで読める状態を維持する。原本や保存コピーを移動/上書きする復旧は計画しない。

## ユーザーが後日承認する対象と範囲

最初は**Batch 1の正確な7 ID（上表）について、対応するLOCAL登録済みcheckoutのdirectoryとworktree登録だけを、直前preflightに合格したものに限り通常のGit操作で除去する**範囲を推奨する。Batch 2/3/4もそれぞれ上表の正確なID集合で独立に判断できる。所有者の利用終了を再質問する必要はない。

承認範囲にbranch削除・main merge・科学SOURCE契約変更・SOURCE欠落の受容・原本/保全archiveの移動削除・shared Git/venv/cache変更・他環境やB3の操作を含めない。承認証明書、ID、署名、grant等を新設しない。これは後日の実行範囲案であり、今回の実行許可ではない。

旧SOURCE未解決11件は別判断を維持する。欠落bytesの回収問題、既存証拠の保全、物理checkout保持必要性を切り分け、64件の削除対象へ混入しない。

**mandatory STOP。今回は計画・匿名化公開まで。全75 worktreeを維持し、実際の削除・登録解除・branch整理・原本移動を行わない。**
