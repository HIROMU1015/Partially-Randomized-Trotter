# SERVER_A 最大24件worktree削除結果

2026-10-27 JST。固定計画commit `6956fd0124c34b7bbf8d6de17cffad3fd2ae86c2` と非公開台帳を使い、ユーザーが明示承認した正確な24 IDについて差分preflightを実施した。A-LEGACY-20とA-CURRENT-H46-4の専用workspace利用終了はWORKSPACE_END_APPROVEDとして扱った。対象の追加・置換、承認証明書やreceipt/grantの新設は行っていない。

## 実行結果

| 承認集合 | 対象 | 通常remove成功 | 保留 | 失敗 |
| --- | ---: | ---: | ---: | ---: |
| A-LEGACY-20 | 20 | 20 | 0 | 0 |
| A-CURRENT-H46-4 | 4 | 0 | 4 | 0 |
| 合計 | 24 | 20 | 4 | 0 |

旧研究20件は同じcommon Gitに対し順次、通常のgit worktree removeだけを実行した。全20件でexit 0、物理checkout消失とGit登録消失をID別に確認した。登録数の単純差だけを成功根拠としていない。コマンド拒否、force、unlock、手動削除、prune/clean/reset/gc、branch削除は0。

削除成功ID：0003・0004・0005・0006・0007・0008・0010・0011・0014・0017・0018・0019・0021・0022・0023・0024・0025・0027・0028・0034（すべてSERVER_A-WT接頭辞）。exact branch・完全HEADと結果は[24件CSV](SERVER_A_24_REMOVAL_RESULTS.csv)へ記録した。

## H46の4件を集合保留した理由

0038・0039・0040・0041のHEAD/status/保持ref・資産保全はpreflightを通った。0041の最新5ファイルも保持Git blobと実bytesの一致を再確認した。以前の古い私有copyを最新状態として上書き・復元していない。

ただし、前回準備の後に対象外の新しい研究worktreeが1件追加されていた。この保持対象のPLANにも、0039の公開結果出力先とH4/H6入力の絶対path参照が6箇所残っていた。参照が過去に複製されたdraft文脈であることは確認できるが、今回の24件利用終了は、その対象外workspaceでの計画利用終了まで含まない。現行/将来の必須実行bindingと断定もしていない。

したがって「0039/0041だけのdraft役割を終了すれば集合外へ影響しない」というgateを閉じられなかった。ユーザー指定どおり4件を部分削除せず、A-CURRENT-H46-4全体を保留した。SOURCEやPLANのpathを書換えて削除可能にする操作は行っていない。workspace終了承認そのものを取り消した扱いでもない。

残条件は、対象外の現在のconsumerがこの旧PLAN/input/result pathを今後使うのか、履歴として保持するだけなのかの具体的確認である。歴史参照だけと確認できれば同じ承認範囲の再開に必要な差分確認が可能だが、今回のmandatory STOP後に自動で追加削除しない。実際の固定実行契約が必要と判明した場合は、その契約・SOURCE・STOPを保持し、別の具体的対応を判断する。

## 差分preflightと保全

ID・実path・所有者・登録・branch・完全HEAD、index/clean状態、untracked、ignoredの科学資産と通常cacheを区別し、重要な新規差分がないことを検証した。全24件が通常のlinked checkoutでmain/common rootではない。submoduleなし、可視本人processでの対象path利用なし、対象のGit lockなしを確認した。削除直前にも各IDのHEAD/index/status/資産SHA/利用/保持refを再確認した。

既存237ファイルをコピーし直さず、受入れ済みcopyのSHA-256を実行前と削除後に照合した。合計130,363,450 bytesで全件一致。対象内のignored科学資産8ファイルも保存bytesと照合し、copyが削除tree外にあり、元へのhardlinkに依存しないことを確認した。通常のPython/test cacheを科学原本と同一視していない。

全24件のHEADとbranch/保持ref、13固有科学SOURCE commitは削除後もexisting objects/refから取得できる。未公開commitを含む研究branch・履歴を書換えていない。原本custody9 root、保護された固定実行pathの25参照、対象外control metadata134件の存在・file metadataを確認し、予期せぬ変化は検出しなかった。元27参照の残り2件は、削除を承認された旧proposal0025のSOURCE root記載であり、固定KEEP対象とは分けた。元proposalと対象版のGit履歴を保持する。詳しくは[保全監査](SERVER_A_PRESERVATION_AUDIT.md)。

同じdiskの別directoryへの保全であり、独立障害域backupが成立したとは報告しない。科学SOURCE/input/result・失敗/cohort・grant/ledger/marker/STOPの内容や固定条件を変更していない。物理checkout内のtrackedコピーを通常removeで解除した後も、原版は保持Git履歴、対象内の非Git科学資産は既存私有copy、外部原本は元custodyに保持される。

## 残存数と対象外保護

前回準備は共有Git側40＋独立旧Track B root1＋現行公開clone1＝42作業コピーだった。実行開始時には対象外の新研究1件が増え、共有Git41＋独立root1＋公開clone1＝43（研究41・管理2）だった。今回20件の成功により、共有Git登録21、初期と同じ2研究Git系統では22、公開clone込みの全作業コピー23（研究21・管理2）になった。新規追加1件と今回の削除20件は別に記録した。

指定KEEP_REQUIRED研究15件、他環境確認待ち0015、管理0037/0042の計18件は、全て実directoryと登録を保持した。保留H46の4件、新しい対象外worktree1件も保持した。対象外の削除・登録解除は0。main・共有objects・独立clone root・共有venv/cache/mount・他job・SERVER_B DF-native B3・LOCAL環境を変更/操作していない。他hostへの接続も0。

## 残る課題と公開

1. H46は対象外consumerのPLAN利用役割・input/result実path必要性を確認する。4件は現在のまま保持し、path変更や新科学実行で迂回しない。
2. 既存KEEP_REQUIRED15件の固定SOURCE/provider/interpreter/custody/common rootの理由は継続。今回承認をこれらの削除へ拡張しない。
3. 0015の既知他環境provider/runtime文脈、旧管理0037の任意整理も今回の削除範囲外として残す。
4. 全研究branchやGitHub branchの削除・統合、science再開・retry、別のworktree整理へ進まない。

公開するのは本directoryの結果CSV・研究単位CSV・監査・本報告の4資料だけ。内部絶対path、host/IP、個人名、private原本配置、credential、process識別子、非公開復旧先・内部pathの単純hashを含めない。詳細preflight、実行journal、保全manifest/restore対応、対象外追加とH46参照の根拠は本人専用非公開領域へ保存した。

独立公開cloneで最新remoteを保持して新規4資料だけをfast-forward公開し、GitHub読み戻し・local/remote SHAを確認する。既存inventoryを上書きせず、main merge・force pushは行わない。正式共通指示2026-10-25版の§8.1・§8.4・§10と既存科学契約・STOPを維持した。既存調査237copyを最初からやり直さず、必要な直前差分と監査だけを実施した。

**削除・保全監査・匿名化公開を終えたらmandatory STOP。**
