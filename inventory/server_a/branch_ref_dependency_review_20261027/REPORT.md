# SERVER_A：GitHub削除候補330 branchの現在の名前依存確認

PUBLIC_REDACTED。2026-10-27 JST。最終scope観測：2026-10-27T14:31:39.686416+09:00。

**固定330候補の全行を確認し、SERVER_Aの確認範囲で現在必要なGitHub branch名・remote ref依存は検出しなかった。NO_CURRENT_NAMED_REF_DEPENDENCY_IN_VERIFIED_SCOPE 330、KEEP_BRANCH_NAME_REQUIRED 0、TARGETED_CONFIRMATION_REQUIRED 0、OUTSIDE_SERVER_A_SCOPE 0。実削除0。** これは330本の削除承認や、SERVER_Bを含む全環境の依存不存在を意味しない。

## 固定入力と結果

LOCAL全面監査の固定commitは `5ee9dcff7976e8c353792b8e79fa750d47c6ad82`。取得した[固定候補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5ee9dcff7976e8c353792b8e79fa750d47c6ad82/inventory/branches/github_branch_cleanup_review_20261027/DELETE_CANDIDATES.csv)、分類・IMPORTANT_HISTORY_RETENTION・計画・REPORTを実際に確認した。候補の追加・置換は0。candidate ID・branch ID・完全tip SHA・元分類・KEEP保持先・元の副条件を残し、SERVER_A判定を別fieldへ追加した。元environmentはoriginal_environment=LOCALとして保存した。

| SERVER_A判定 | 件数 |
| --- | ---: |
| NO_CURRENT_NAMED_REF_DEPENDENCY_IN_VERIFIED_SCOPE | 330 |
| KEEP_BRANCH_NAME_REQUIRED | 0 |
| TARGETED_CONFIRMATION_REQUIRED | 0 |
| OUTSIDE_SERVER_A_SCOPE | 0 |

[330行CSV](SERVER_A_BRANCH_REF_REVIEW.csv)が正確な結果集合。[現在の名前保持必須CSV](SERVER_A_REQUIRED_BRANCH_NAMES.csv)と[個別保留CSV](SERVER_A_TARGETED_HOLDS.csv)は、いずれもheaderのみの0行であり、欠損出力ではない。330候補外のmain・PLAN・現行研究branchなどのKEEPを削除候補へ変更していない。

Git履歴の候補外KEEPからの到達可能性はLOCALの固定監査を再利用した。全809 branchのDAG再取得・全祖先再解析は0。read-onlyのremote heads snapshotは810本で、330候補の完全tipはすべて固定CSVと一致、候補tip差分0だった。remote全数は並行作業で変化するため、この数を将来の実行直前値とは扱わない。

## SERVER_Aの最新作業コピーと並行研究

前回H46削除完了時は研究17・管理2＝19作業コピー。今回開始時は研究19・管理2＝21、確認中に公開用研究checkoutが1件追加され、**最終確認は研究20・管理2＝22作業コピー**となった。共有研究Git登録は20件、別研究Git1件、独立公開clone1件で合計22件である。候補branchの現在checkoutは全22件で0。

前回からの追加はH46境界研究の実行・証拠統合・公開用の3作業コピー。既存の証拠統合workspaceも並行作業によりlocal-full branchへ移った。これらを凍結・巻戻し・checkoutせず、確認済み起点からの変更19 locator（存在する17file）だけ追加照合した。差分の17一致は更新されたversioned manifestの過去branch欄であり、新しい現在必須remote refはなかった。正確な実path・前後HEAD・別名は本人専用記録に保存した。

今回の全Git読取はSERVER_Aの既知3 Git系統だけ。独立公開clone以外のGit refs・index・設定へ書込まず、研究branchの変更・研究worktree操作・原本移動は0。

## 確認方法と範囲

既存SOURCE catalogue・branch/固定path索引・保全/復旧記録・PLAN監査を再利用した。特に準備 `6956fd0124c34b7bbf8d6de17cffad3fd2ae86c2`、前回20件実行 `5916f4baac80b1180c6f40c57a9dcd4e0aadb694`、H46残存4件実行 `22507bd3c039ee6662a4d6be3f757754137496b3` の非公開対応を引き継いだ。

- 既存8,065 metadata locatorを基礎に、現在登録worktreeのSOURCE・PLAN・control・STOP・manifest・handoff・実行script/設定を確認した。22,367現存text fileを読み、同一bytesはまとめ、4,356固有内容を一回ずつ330名の複数patternで照合した。330 branch×全資料の反復全文検索は行っていない。
- 現在のsparse checkoutで未展開の契約・script等1,397 locatorは、既存Git blob 1,224件からread-onlyで取得した。未取得blob0、checkout展開・lazy fetch0。重複を除く追加固有内容1,021件を照合した。
- 並行研究の変更17固有内容を追加し、合計5,394固有内容を照合した。一致は167候補・256固有内容・6,320 locator。163候補は確認資料でbranch literal一致なしだった。一致件数は必要なref依存の件数ではない。
- Gitのcheckout登録、local/upstream設定、同一所有者の可視processの明示branch引数も確認した。候補checkout0、明示候補引数0。credential/environment変数や他hostのprocessを取得していない。

未読範囲も区別した。削除済みworktreeの7,488歴史locatorは現存しないため読了としない。sparse領域のその他1,795 text locatorは今回のSOURCE/契約/script選択外で、large numerical ledgerの15 locatorも本文再走査をしていない。raw・大容量数値payload・全過去履歴の内容監査・科学的再解析は対象外。既存の確認で不足を特定しなかった候補を、一般的な未知だけで一律保留にはしていない。一方、この範囲外に将来/未知の依存がないとは断定しない。

## 名前が現れた資料の判断

1. versioned manifest・STOP・SOURCE/input/resultのbranch欄は、固定commit/hashと並ぶ過去の出所や生成branchの記録だった。原文・科学条件・失敗/STOPを保持したまま、現在のremote ref存続義務とは分けた。
2. private復旧索引の旧branch/HEAD/refは、元のworkspaceを復旧するための対応記録だった。元local branchとorigin tracking refを今回削除・pruneしない。将来古い手順を使う際は、保持commitから元名のbranchを再作成する前提が必要な場合がある。記録された文字列を消したり、現在のまま必ず再開可能と認定したりしていない。
3. 旧H4契約finalizer、旧BF1 read-only recovery、旧T02 diagnosticには同名**ローカルbranch**の検査があった（DELETE-CAND-0170/0258/0322）。現在の候補checkoutや実行引数は0で、これらの歴史的手順にGitHub remote head存在の検査はなかった。ローカルbranchの保持と、候補GitHub refの削除可否は別条件としてCSVへ副記録した。旧固定契約を解除・書換えた判断ではない。
4. 11本の旧branch upstream設定は削除済みworkspaceに対応する受動的な設定だった。現在のcheckoutや候補refを取得する現行scriptは検出しなかった。将来この旧local branchでgit pull等を使う場合、remote refを削除した後なら元名の再作成等が必要になる。設定・local refを今回変更していない。
5. 旧公開用push例、originから読んだ過去のsnapshot、発表資料の固定commit参照は、現在稼働のremote依存としなかった。同じ文字列が保管directoryに含まれるだけの一致も、branch/ref契約と混同していない。

## H46 PLANと復旧情報

保持branchは `codex/pr-boundary-phase-a-20261027`、保持HEADは `a4373f51d7acb5d70fda894b4ad66f8bc781177d`。PLANの状態は `PHASE_A_FIXED_AWAITING_USER_EXECUTION_PERMISSION` のまま。本文SHA・Git blob・branch・HEADは確認前後で不変、科学的条件・path・SOURCE・入力・結果を変更していない。

PLANで候補名h246-reconstructed-confirmation-20261022が現れる4箇所は、seed/exclusion/registryの**保管directory component**で、GitHub branch selectorではない。そのdirectoryや入力の保管を変更せず、同名remote branchの存続とは分けて判断した。新しい実行用PLANについても候補への現在必須named refは検出しなかった。PLAN実行・path修正は0。

旧H46 0038/0039/0040/0041のbranch名は非公開復旧索引に残り、保持local ref/完全commitも現在取得可能だった。4件は今回候補に含まれるが、復旧記録を残すこととGitHub headの永久存続は分けた。旧絶対pathが存在しないことだけを理由にremote削除可能とは判定せず、PLANと固定SOURCEのcommit参照、代替データ供給、復旧名の用途を確認した。

将来旧PLAN/旧復旧手順を使うなら、保持commitからworkspaceと必要な元branch名を再作成するか、別の具体的許可の下でpath/実行設定を整理する必要がある。今回それらを実行・変更せず、旧PLANの即時実行可能性を認定しない。科学SOURCEの各版・独立cohort・入力・結果・失敗/STOPを後続版で置換していない。

## B3・保全・非干渉

SERVER_BのDF-native B3への接続・アクセス・状態調査・再開・停止・設定変更はすべて0。候補外の未解決7 branchは照合対象に追加せず、引き続きKEEP扱いで、そのsealed catalogueやruntimeを読んでいない。330候補中に新しいDF-native B3 bindingを疑う具体的根拠は検出しなかった。過去の別研究で使うB3という表記だけを、SERVER_Bの現在jobと同一視しない。

保護確認では、参照した非公開原本metadata47件のSHAが不変、既存237保全copyのsize/inode/mtimeが前回監査時から不変、保持PLANのbytes/branch/HEADと330候補に対応する研究refsが不変だった。237copyのpayload SHA全件再計算や保全原本作成を繰り返さず、以前の保全監査を再利用した。並行研究による候補外ref変化は別記録にし、停止・巻戻ししていない。

正式共通指示2026-10-25版は実在を確認し、以前の全文読取時と同じSHAであることを確認して§8.1/§8.4/§10を適用した。既存bulk_dependencies/local75のmetadata選択・JSON locator抽出、worktree/ref・hash読取を候補名の単一走査へ適応した。新しい処理は330 join、未展開契約の既存blob読取、並行19 locatorの差分確認、公開4資料作成のみ。承認証明書・receipt・新grantは作らない。

## 公開・次の境界

公開は指定の新規directoryへ本REPORTと3CSVだけ。既存LOCAL/A/B報告は上書きしない。実path・host/IP・ユーザー名・private raw配置・内部process・非公開path対応/復旧先・credentialを公開しない。詳細な照合根拠と復旧name対応はowner-onlyの非公開領域に新規保存し、原本は変更/削除しない。

最新remote履歴を独立公開cloneへ取り込み、4ファイルだけを通常fast-forwardで公開する。並行成果の履歴を上書きせず、force push/main merge/研究branch変更は0。公開後、GitHub内容読み戻し・local/remote HEAD一致・新規4ファイルだけの追加を検証する。

本結果はSERVER_Aの確認時点のnamed-ref依存差分である。後日のremote削除には、別の具体的承認、LOCAL/B側の確認、最新候補tip・KEEP保持先・新しい利用/契約のpreflightが必要。今回branch/worktree削除・解除・prune・clean/reset/gc・原本移動・PLAN実行・新科学計算はすべて0。匿名化公開と結果保存を完了したら**mandatory STOP**。
