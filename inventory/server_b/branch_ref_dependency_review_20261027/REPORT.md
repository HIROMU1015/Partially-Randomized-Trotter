# SERVER_B：固定330候補のbranch名・固定ref依存確認

LOCAL固定版`5ee9dcff7976e8c353792b8e79fa750d47c6ad82`の正確な330候補について、SERVER_BのB3以外の保持59件と限定した静的SOURCE/control・復旧資料を照合した。B3はユーザー申告の稼働中を前提に、本体や非公開稼働資産へアクセスしなかった。GitHub/local branch・worktreeの実削除は0件。

| 項目 | 結果 |
| --- | ---: |
| 共通Gitの登録directory数による保持worktree数 | 64 |
| 現在branch/HEADを照合した非B3保持worktree | 59 |
| 固定候補 | 330 |
| NO_CURRENT_NAMED_REF_DEPENDENCY_IN_VERIFIED_SCOPE | 330 |
| KEEP_BRANCH_NAME_REQUIRED | 0 |
| TARGETED_CONFIRMATION_REQUIRED | 0 |
| 候補330内のB3_SCOPE_DEFERRED | 0 |
| 別枠DEFERRED_UNTIL_B3_REVIEW_AUTHORIZED | 7 |
| B3本体・非公開稼働資産アクセス/操作 | 0 |
| branch/worktree実削除 | 0 |
| 科学計算・数値再解析・SOURCE/契約変更 | 0 |

## 固定集合と確認方法

元のLOCAL分類・完全tip SHA・KEEP保持先を維持し、SERVER_Bの判断を別fieldへ記録した。対象の追加・置換0件。候補名と保持先tipは独立公開cloneから取得したremote metadataと一致し、匿名KEEP1件は固定完全tipと一意のremote refを本人専用対応表で解決した。匿名の実名は新たに公開しない。LOCALが行った全graph保持証拠を再利用し、809本の全履歴解析をやり直していない。

保持数64は既知4 common Gitの登録directory件数から観測した。非B3の59件だけ現在のbranch/HEADを読み、330候補のcheckoutは0件だった。B3の5 checkoutやそのHEAD/status/index/ignoredを照会していない。研究用共有Gitへの書込みは0、remote取得と公開は独立cloneで行った。

既存のworktree/ref/path索引と削除済み315件の復旧対応を再利用した。59件の選択したGit SOURCE/control metadataは共通blobを重複排除して読み、変更/新規controlと既知の外部control locatorだけを補った。全64 checkoutやignored/raw/runtimeを全面走査していない。内容照合は4,010固有内容・241,173,516 bytes、候補literalを含む内容は135件、literalは936件。これはbranch参照の技術確認であり、保存数値の再解析や科学的妥当性の再レビューではない。

common Git/alternateは保持objectの配置、origin fetch refspecは共有取得規則として確認した。現在の固定候補名を要求する共有Git pointerや個別fetch refspecは確認されなかった。runtime関係は選択した非B3の既存宣言/controlを対象とし、B3を含むruntime/venv内の追加走査はしなかった。

## literalの役割と判断

| 静的照合で確認した役割 | 参照数 |
| --- | ---: |
| COMPLETED_ONE_SHOT_CONTRACT_LOCAL_BRANCH_IDENTITY | 2 |
| DATED_RESEARCH_PROVENANCE_OR_OLD_PUBLICATION_INSTRUCTION | 94 |
| HISTORICAL_CHECKOUT_AND_RESTORE_LOCATOR | 540 |
| HISTORICAL_LOCAL_BRANCH_GUARD_NO_REMOTE_NAMED_LOOKUP | 1 |
| PHYSICAL_PATH_LOCATOR_NOT_REMOTE_REF | 26 |
| PHYSICAL_SOURCE_ROOT_LITERAL_NOT_REMOTE_REF | 1 |
| PRESERVED_MARKER_DIRECTORY_LITERAL_NOT_GIT_REF | 1 |
| SAVED_MANIFEST_OR_SOURCE_PROVENANCE_BRANCH_LABEL | 270 |
| SAVED_SOURCE_REMOTE_VERIFICATION_PROVENANCE | 1 |

provenance・過去のremote SHA照合・古いpush指示は、現在もremote branch名を永久保持する義務と同一視しなかった。物理worktree名を含む絶対pathとGit ref名も区別した。SOURCE・PLAN・STOP・入力・復旧索引の本文や条件は変更していない。

特に`track-b-bf1-read-only-recovery`は終了済み一回recoveryのlocal branch identityと共通Git内のmarker directory名、`track-b-ra-d0-t02-exact-certificate-20261008`は終了済みT02のlocal branch identityを持つ。対応runner/契約を静的に読み、remote branch名で取得する条件ではないことを確認した。元local refs・marker・契約・STOPは今回変更していない。将来の別途許可された復旧では必要な旧local branch名・workspace・marker状態を維持/復元する必要があり、KEEP tipだけへそのまま切り替えて旧契約を満たすという判断ではない。

旧H6の`remote_branch`記録は固定SOURCE commitの当時の取得/公開証跡であり、継続するnamed remote predicateではなかった。H4の固定source-root文字列は物理path locatorであり、同名remote branchの存在要件ではなかった。旧Track B/G6に出てくるB3という語を現在のDF-native B3と混同していない。

## 範囲と後続条件

`NO_CURRENT_NAMED_REF_DEPENDENCY_IN_VERIFIED_SCOPE`は上記の非B3確認範囲に限る。B3 sealed catalogueや5保護worktreeの非公開ref・科学資産を確認済みとはしない。別枠7本は[B3_DEFERRED_BRANCHES.md](B3_DEFERRED_BRANCHES.md)の通り全件保持し、今回追加調査を延期した。一般的な未知の可能性を理由に330件全体を保留していない。

後日の削除には、正確なremote ref集合についてユーザーの具体的許可、候補tip・KEEP保持先・各環境の利用/新しい固定ref条件のfresh確認、並行更新を失わない受信側の比較条件が必要である。本監査自体は削除許可ではない。旧名を必要とする復旧では固定commitから正確なref名を復元して旧手順/契約へ戻し、科学実行・one-shot・retry/STOP条件を別途守る。remote branchを消してもlocal refs・保全archive・入力/結果/失敗/STOPを削除してよいことにはならない。

## 公開と終了

[全330判断](SERVER_B_BRANCH_REF_REVIEW.csv)、[名前保持が必要な候補](SERVER_B_REQUIRED_BRANCH_NAMES.csv)、[対象限定の保留](SERVER_B_TARGETED_HOLDS.csv)を出力した。内部path・host/IP・private原本/復旧配置・認証情報を除去し、詳細な参照元/field/role/完全SHA対応は本人専用領域へ保持する。

整理専用branchの最新remote履歴を取り込み、本directoryの5資料だけを通常fast-forwardで公開する。他環境の並行commitを保持し、研究branch変更・force push・main mergeは行わない。公開後のGitHub読み戻しとlocal/remote一致は本人専用結果と最終回答へ記録する。

次の担当：ユーザー/全環境の整理担当が、必要なら今回の限定結果を統合して将来の削除範囲を判断する。B3保留7本は別の明示許可まで調査しない。Codexは公開・照合後mandatory STOPし、branch/worktree削除へ進まない。
