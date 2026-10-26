# ローカル75件：サーバーB実パス照合

2026-10-27 JST。PUBLIC_REDACTED。照合snapshot：2026-10-27 01:02 JST。

**受領した非公開対応表の75件すべてでID・branch・完全HEADが固定依頼と一致し、正規pathの照合を完了した。主判断は75件とも `NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE`。今回確認したB側の候補・宣言・既知B3参照先では、対象checkout・共通Git・外向きガイドリンク先へのpath一致を検出しなかった。これは全環境での非依存認定や削除許可ではない。未知のmount/alias、別namespace、未記録job・再開契約、private custodyはUNKNOWNとして残す。**

対象ごとのbranch・完全HEAD・A側結果・B側判断・副条件は [SERVER_B_LOCAL75_ACTUAL_PATH_RESULTS.csv](SERVER_B_LOCAL75_ACTUAL_PATH_RESULTS.csv) に保存した。実パス・host・mount詳細・process/job識別子・private原本配置は公開していない。

## 1. 固定資料と前回成果の再利用

- 固定依頼：commit [`e5651ab7b7f1682dac51a756e6b277f4a2d9c217`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e5651ab7b7f1682dac51a756e6b277f4a2d9c217/inventory/local/pre_cleanup_75_20261026) の75件、readiness表、64件計画。保存bytesを同commitのGit blobに照合した。対象を連番やbranchから追加していない。
- 前回B調査：commit [`34015c1f58ceda0e0409d858d0f606831e178115`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/34015c1f58ceda0e0409d858d0f606831e178115/inventory/server_b/local75_dependency_check_20261027)。開始時のGitHub branch HEADも同commitだった。旧報告とCSVは変更せず、今回分を別directoryへ追加する。
- A側結果：commit [`5eef8de248d454cc2497ef9631bfc171b1a78aae`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/5eef8de248d454cc2497ef9631bfc171b1a78aae/inventory/server_a/local75_actual_path_reconciliation_20261026) の前回検証済み保存結果を再利用した。Aの71件の限定非依存・4件の論理参照はAの判断として転記し、Bの判断へ置き換えていない。

前回の非公開helperのCSV結合・所有者確認・metadata読取り方法と保存索引を再利用し、受領表の検証とpath成分境界付きjoinだけを小さい非公開scriptに追加した。前回の静的38ファイルを読み直したり、静的調査script全体を再実行したりしていない。科学moduleのimport・科学計算・数値結果再解析は0。正式共通運用指示2026-10-25版の§8.1・§8.4・§10を実在ファイルから確認した。

## 2. 受領ファイルの検証

ユーザーがBへ直接配置した本人所有のregular CSVを使用した。読取り可能で、ID重複0、固定75件との差分0、branch不一致0、完全40桁HEAD不一致0。原本を移動せず、本人専用の非公開領域へbyte同一のコピーを保全した。

受領bytesのSHA-256は `6fc2ab36122422a9bbd5e7b786a9ea31d9cd6aa70a3de79443f7fe647e321a05`。転送前の期待SHAは提示されていないため、独立した元digestとの一致は主張しない。これは受領bytesの識別値であり、承認証明ではない。

必要fieldとの対応は次の通り。

| 意味 | 受領field |
| --- | --- |
| worktree ID | `local_worktree_id` |
| branch・完全HEAD | `branch`・`head_commit` |
| 正規絶対path・形式 | `canonical_absolute_path`・`path_flavor` |
| 環境scope | `environment`・`host_scope_id` |
| 観測日時 | `identity_and_listed_links_verified_utc`、補助として`path_alias_observed_utc` |
| 既知checkout別名 | `verified_aliases_json` |
| 外向きリンク | `verified_symlinks_json` |
| 共通Git | `shared_git_common_directory` |

全75件はLOCALのPOSIX絶対pathであり、列挙されたcheckout別名は0件。34件に計68本の外向きガイドリンクがあるが、明示された役割に従いcheckout別名とは扱わない。LOCALでの最終観測以降の状態や、列挙されていない全別名の不存在は未検証である。

各fieldの型・path形式・日時・公開locator形式を確認した。既知credential/private-key形式は検出せず、受領内容はpath・identity・公開locatorのmetadataだった。不要なcredentialやraw数値本文は公開成果物へ転記していない。LOCAL/AへのSSH・SFTP・scpや自動転送は行っていない。

## 3. B側照合範囲と判定方法

`B_PATH_1` は、前回の `B_STATIC_1` が保存した2,023 path occurrence、2,146件の既存SOURCE/runtime宣言索引、既知12参照先、runtime metadata、既存process snapshotを対象とする。参照元・行・文脈・役割を非公開原本に保持した。現在のB3については、既知参照先のmetadata、既知2 service、controllerのcwd/exe/引数、可視mount metadataだけを差分確認した。

正規pathと列挙済みcheckout別名をPOSIX path成分で照合し、単なるprefix部分文字列を一致としない。ancestor一致があっても保持義務とは自動判定しない。共通Gitと外向きガイドリンク先も別枠で比較した。空白を含むpath、隣接名の誤一致、ancestor区別の人工チェックを実施した。

全75件でpath一致0。共通Gitへの一致0、外向きリンク先への一致0。提供されたpath文字列はBの観測namespaceでは75件とも存在を検出せず、同じ文字列にあるB実体をLOCAL実体と誤認する事例もなかった。ただし、別hostのpathをBで無条件realpath化して同一性を認定していない。同じdevice/inode番号のhost間比較も同一実体の証拠にしていない。

前回と現在の可視mount表は一致し、観測したB3 controllerのmount表は調査processのものと一致した。既知B3参照先の正規化先・device/inode、runtimeの既知3 metadataも差分を確認した。これだけで、別名でのexport/bindや他namespaceによるLOCAL実体への接続を全て除外することはできない。未提供の共有実体対応はUNKNOWNである。

**抽出範囲の制約：前回の保存path候補は引用された絶対`/home`系pathが中心である。任意の一時path、相対path、動的構成path、未記録metadata・別namespace・他所有者jobを網羅しない。受領表にある一時directoryのpathも、この保存候補・既知bindingとの照合として評価した。候補非検出を任意のpath参照全体の不存在へ拡張しない。** raw/compiled cache/checkpoint再帰走査、全Git object/履歴探索、科学runtimeの実行検査は行っていない。

## 4. 件数とA側結果との関係

| 項目 | 件数 |
| --- | ---: |
| 受領ID・branch・完全HEAD検証 | 75 / 75 |
| 正規pathの限定照合完了 | 75 / 75 |
| PHYSICAL_DEPENDENCY_CONFIRMED | 0 |
| NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE | 75 |
| LOGICAL_REFERENCE_ONLY（B主判断） | 0 |
| HISTORICAL_REFERENCE_ONLY（B主判断） | 0 |
| PATH_IDENTITY_UNRESOLVED（B主判断） | 0 |
| PRESERVATION_OR_CONTRACT_GATE（B主判断） | 0 |
| VERIFICATION_DEFERRED_FOR_B3_SAFETY（B主判断） | 0 |
| 未提供alias・共有実体・他namespaceの副条件 | 75 |
| private custodyの未確認副条件 | 75 |
| 旧SOURCE欠落の副条件 | 11 |
| 削除・登録解除・branch削除・原本移動 | 0 |

主判断と副条件は別である。実パス未受領という前回の理由は解消したため、全件の主判断を引き続き `PATH_IDENTITY_UNRESOLVED` とする扱いは終了した。一方、cross-hostの全実体同一性は未確定であり、副条件として保持した。

A側の4件の論理参照と、Bでbranch/完全HEAD参照や必須物理依存を検出しなかったことは矛盾しない。Aのscope内判断をBに代入していない。ローカル/Aの記録と矛盾する具体的なB物理依存IDは今回検出しなかった。

## 5. 先行64件と旧SOURCE未解決11件

先行64件の内訳は図表7、QC18、Research B34、N1/N3保全検証済み5。各IDの今回B照合は完了し、元checkoutの必須物理依存は確認範囲内で検出しなかった。ローカル側でBの限定scope結果を64件計画へ統合できる段階である。ただし未確認の具体的なjob・alias・契約・custody条件や整理直前差分を代行認定せず、計画CSV自体も更新していない。削除実行の許可は今回ない。

別枠11件はLOCAL-WT-0005、0009〜0016、0018〜0019。今回のpath主判断も限定非依存であるが、旧SOURCE3版・55宣言の欠落は引き続き未解決。旧SOURCEの広域再探索・回収は実施せず、この照合を回収の代用にしていない。欠落だけを理由に元worktreeの永久保持義務を認定せず、先行64件にも混入させない。

## 6. DF-native B3への影響と非干渉

現在の対象はlinear H4/H6・1Å・STO-3G・tol-only DF10/19、`L_D=0`でone-bodyを保持するDF-native B3。旧Track B/G6/W2CのB3とは区別した。観測時点で取得・終了監視serviceはactive/running、保存statusは `SCIENTIFIC_ACQUISITION_RUNNING`。最終取得完了や科学的性能を示すものではない。

B3が必要とするSOURCE/provider、保存入力、結果領域、runtime/editable供給元、共通Gitの既知bindingを今回対象のLOCAL75と照合した。12参照先は前回と同じB内device/inodeで、観測runtime metadataにも変更がなかった。前回13 scriptの宣言hash一致結果は前回観測の証拠として再利用した。今回SOURCE hashを再取得したり、13本の不変性を新規認定したりしていない。

既存のbase SOURCE `155529e20a8fac39e00786c40b57830b71203c4f`、記録済み復旧SOURCE `ae3d301e2318a3f9f1331c047c68d310771b6849`、provider snapshot SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe` の区別を保持する。checkout HEADが科学SOURCEそのものとはしない。

今回、B3のSOURCE/input/raw/checkpoint/cache/結果/grant/ledger/marker/STOPを変更せず、controller・workerへのsignal、停止・再起動・retryは0。既存のone-shot・1回の復旧・後続失敗はSTOPという条件も変更していない。B3のworktree/branch/共有Git object store/venv/共有設定への書込みは0。本人専用非公開領域と独立公開cloneだけへ保存した。

## 7. 保存・公開・次の担当

非公開原本には受領CSV、field対応・SHA検証、ID別path/alias・B参照元/実体metadata・判断根拠、B3差分観測、未確認条件を保存した。公開は今回directoryのCSVと本報告の2ファイルのみ。旧LOCAL/A/Bの成果物・統合計画は変更しない。

GitHub公開直前に整理branchのHEADを再確認し、他者の更新を保持するfast-forwardで公開する。公開成功とlocal/remote一致は実測後の最終通知・非公開公開記録で示し、この事前文から成功を推定しない。

次の担当：**ローカルCodex**が今回のB限定結果を75件資料・先行64件計画へ統合し、残条件を区別する。**ユーザー**のworkspace利用終了方針は確定済みで再質問不要。実際の整理は、具体的な対象・操作を許可した場合の別作業であり、本調査は開始しない。旧SOURCE11件の判断は別枠を維持する。

**本照合と匿名化公開後はmandatory STOP。新科学計算、worktree削除・登録解除、branch削除、原本移動、prune/gc、main mergeは行わない。B3の既存科学jobは変更せず継続する。**
