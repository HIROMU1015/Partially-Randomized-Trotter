# ローカル139 worktree：サーバーA実パス依存照合

2026-10-27 JST。固定入力は [139件保全状態](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96cf3dae3bd03d6f64f384fefedc6e88e782eef6/inventory/local/next_group1_139_preparation_20261027/NEXT139_PRESERVATION_STATUS.csv)、同commitのarchive manifest・整理計画。元集合は `2caa236fa7332872b5a346f0744318d48d59f6ad` の234件followupでTECHNICALLY_CLEAR_PENDING_OTHER_ENVとなった214 IDから、先行75件の正確なID集合を除いた139 IDである。最新139件CSVとの集合一致・重複0を確認した。先行75件の依存判定・利用終了方針を139件へ転用していない。

正式運用指示 `research_project_instructions_unified_2026-10-25.md` はサーバーAの実在ファイルから全文確認し、§8.1・§8.4・§10を適用した。ユーザーの指示を実行許可として扱い、新しい承認証明書・receipt・grantは作成していない。後続の科学計算や削除を承認済みとは扱わない。

## 受領と完全性

ユーザーが直接配置した非公開CSVを確認した。本人所有regular fileでsymlinkではなく、ユーザー提示の期待SHA-256と受領bytesは完全一致。重複ID0、ID・branch・完全40桁HEADは固定資料と139/139件一致し、正規pathも139件で一意だった。記録pathとcanonical pathは同一。POSIX絶対path、LOCAL環境識別、timezone付き観測時刻を検証した。

元ファイルは内容を変えずowner-onlyへ権限制限し、別のowner-only非公開領域へ同一bytesの保全コピーを作成した。原本移動は行っていない。想定fieldだけの最小表で、private raw本文fieldやcredential等の既知パターンは検出しなかった。公開SOURCE/result/STOP locatorも固定資料と139行すべて一致した。受領ファイル名・hash値・内部絶対path・host識別情報は公開しない。

受領表の検証済みcheckout別名は0件。これは別名不存在の証明ではない。既知guideリンクは外向き案内でcheckout別名ではないというscopeを保持した。LOCALのGit common directoryの具体値は今回の最小表に含まれず、正規pathから推定しなかった。

## 主分類と確認範囲内の物理依存

| 主分類（相互排他的） | 件数 |
| --- | ---: |
| PHYSICAL_DEPENDENCY_CONFIRMED | 0 |
| NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE | 130 |
| LOGICAL_REFERENCE_ONLY | 0 |
| HISTORICAL_REFERENCE_ONLY | 1 |
| PATH_IDENTITY_UNRESOLVED | 0 |
| PRESERVATION_OR_CONTRACT_GATE | 8 |
| VERIFICATION_DEFERRED_FOR_SAFETY | 0 |
| 合計 | 139 |

139/139件で正規実パス照合を完了した。既存A候補・追加の限定メタデータ、確認済みA別名、閲覧可能な本人process、既知Git/alternate/editable pointerの範囲では、**有効な対象path参照一致0、必須物理依存確認0、元LOCAL checkoutを必須とする束縛未検出139件**。この139は主分類130＋歴史参照1＋ローカル保全gate8を合わせた、確認scope内だけの評価である。

主分類のPATH_IDENTITY_UNRESOLVED 0は、未記録mount/alias/別namespaceをすべて確認済みとする意味ではない。全139件で未確認job・再開・共有実体・custody等を副条件に保持した。具体的なA側固定SOURCE/path保持義務は新たに確定していない。8件のgateはローカルの追加保全判断であり、Aが元checkoutを必要としている証拠とは別である。

### 歴史参照1件

LOCAL-WT-0184は、公開 [source manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/09fe527274e8de5c0d808ada5890fa06a59f78ea/artifacts/track_b_rte_reallocation_r1_source/2026-10-06/source_manifest_v1.json) の `Track_A_reference_only/commit` に記録された `4c23453c541700c6a41ba71fc5ec9323b53858d6` と、Aの過去準備資料3件の旧証拠commitが一致した。科学SOURCE供給契約やLOCALの物理directory参照ではない。元checkoutのHEAD `09fe527274e8de5c0d808ada5890fa06a59f78ea` と、この参考commitは別である。

この1件はHISTORICAL_REFERENCE_ONLYを主判定とし、論理的な共有参照であることを別fieldに残す。既存SOURCE/result/STOP・Git保持は継続する。他138件では、今回の具体的SOURCE/control等の照合キーに一致する研究参照を検出しなかった。全139件の完全HEADを保持する取得済みGit参照は確認したが、単なるGit保持参照を科学jobの必須物理依存に数えていない。

## 確認scope A139

前回の処理・CSV join・path正規化・参照元pointer判定・hash照合を再利用し、入力と件数だけを139件へ適応した。旧helperの75件固定条件を139件へ変更し、現行§8.1に反する承認receipt要求を除いた。元helper・原本は変更していない。

既存4,043 path候補のうちJSON pointer由来の誤検出を参照元fieldで除外し、今回helperの有効候補2,657件を照合した。原本の候補一覧自体は保持した。case fold・Unicode変更・他hostのrealpath・環境変数展開・branch名からのpath生成は行わず、path component境界で比較した。prefix文字列一致だけで依存を認定していない。

同じ139件の正規path・branch・完全HEAD・記録済みSOURCE/resultを使い、既存索引のメタデータと最近の既知運用領域だけを一括で静的照合した。2,154テキスト/環境pointer資料を読み、large metadata index2件を除外、読取りエラー0。確認中に稼働観測メタデータ1件の並行更新を検出し、当該1件だけを対象139件のpath・branch・HEAD・SOURCEで再確認したが追加一致はなかった。公開SOURCE/contract関連JSON78件から記録commitを抽出して照合した。取得不能0。SOURCE field・参考commit・結果commit・文書を含むcheckout HEADを区別し、科学moduleやrunnerは実行していない。前回75件のPASS/非検出を139件の証拠として引き継いでいない。

既知A symlink別名4件と選定mount4件をA内で確認した。旧索引から字句展開できたA別名5表現にも対象path一致はなかった。共有Git common directory・既知alternates・venv/editable install pointerをread-onlyで確認し、元LOCAL checkoutの正規pathを必要とするpointerは検出しなかった。別hostで同じ文字列・device/inodeであることだけを同じ実体の根拠にしていない。

本人UIDで閲覧可能なprocessのcwd/引数スナップショットにも対象path一致はなかった。process環境変数は読まず、停止・変更・再配置はしていない。閲覧できないprocess、他所有者、別namespace、将来の再開予定は確認外である。既存のA custody9索引を再利用したが、raw全件探索・payload再hash・大規模archive展開は行っていない。

## ローカル保全131件とgate8件

固定資料の131件はARCHIVED_VERIFIED_CLEANUP_CANDIDATEを維持する。これは選定scopeの保全評価であり、全code・sparse metadata・外部raw・runtimeの独立backupまで確認した意味ではない。Aはローカル保全コピーを再検証していない。

追加保全判断8件はLOCAL-WT-0002、0095、0096、0097、0099、0100、0101、0103。大容量tracked checkpoint/resultの15 binding・9固有版が未コピーというローカル側条件をそのまま保持する。GitHubのcommit保持は実ファイルの追加custody判断とは別で、今回コピーや保全責任の変更は行っていない。この8件もAの依存照合は完了し、A scopeで必須物理依存は未検出だが、削除可能へ格上げしない。

## Track B優先群と返却条件

Track B116件中の選定scope保全済み候補109件に対し、今回A側で追加の必須物理保持義務は検出されなかった。うち先行優先群72件（D/E 16、DF correction 19、M1/M1S 13、parent allocation 15、handoff/diagnostics 9）も同じ限定結果で、Aから新たな一律保留を追加する根拠はない。旧証拠参考commitの一致1件はBF/R1/SP側のLOCAL-WT-0184であり、固定Git/証拠を保持する条件と元checkoutの物理保持を区別する。G10関連7件の追加保全は8件gateの一部として別枠に残す。

旧Track B G6 6件は、現在のサーバーB DF-native L_D=0 B3とは別群。Research C C1/C4の9件は、ユーザー・研究担当が現行作業や他Codex/再開予定を判断する条件を特に維持する。系列名や可視process非一致だけで利用終了とは認定しない。

CSVの残条件コードは各行の正確なID・branch・完全HEADに適用する。

- U1：139件の将来workspace利用終了は未確認。研究単位でユーザー・研究担当が判断する。以前75件の所有者方針は転用しない。
- U2：未記録job・別所有者/namespace・controllerの動的参照・再開/復旧予定は未検証。新たな具体的path束縛が見つかったIDだけ、その根拠を確認する。
- U3：未記録shared mount/export/bind・別名・LOCAL common Git/runtimeの実体関係は未確定。今回最小表にない値は推測せず、必要な場合だけ非公開で照合する。A内のrealpathやdevice番号だけで他hostとの同一性・非同一性を確定しない。
- U4：固定SOURCE/provider/input/result/STOP、private原本/custody、Git参照と実ファイル保全の条件を維持する。今回その既存契約を解除・書換えしていない。
- U5：公開準備中にB側139件報告がcommit `ee30e8b435be6b15bba201de923b61eb02d9c54e` へ追加されたため取得・読了した。[B側報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ee30e8b435be6b15bba201de923b61eb02d9c54e/inventory/server_b/local139_actual_path_reconciliation_20261027/SERVER_B_LOCAL139_PATH_REPORT.md)もB確認範囲では必須物理依存未検出139件としており、ローカル担当がA/Bのscopeと未確認条件を統合する段階へ渡す。A側判定はこのB結果から転用していない。今後具体的なDF-native B3の再開/共有実体条件が見つかった場合はB担当の既存結果と照合する。AからBへのSSH・自動転送・B3干渉は行わない。
- G8：該当8件だけ、未コピー大容量fileの保全scope/custody判断をローカル担当へ残す。

返却判断は「A139 scopeで元LOCAL checkoutの必須依存を検出しなかった」であり、「削除してよい」ではない。131件の優先候補は、上記の所有者判断・A/B結果の統合・契約/custody・直前確認と具体的な削除指示が揃うまで候補状態を維持する。8件は追加保全判断も必要。139件全体の実体・現在/将来jobの不存在を主張せず、未知条件は全行に残した。

## 保護と公開

A自身のworktree・従来12候補（KEEP_REQUIRED 3、UNRESOLVED 9）、固定SOURCE provider、共有Git/venv/cache/mount、未公開commit、input/raw/result/grant/ledger/marker/STOPを変更していない。LOCALの削除済み64件・保護20件・旧SOURCE未解決11件は今回の対象外で、再調査・状態変更をしていない。新規科学計算、sampling、compile、数値再解析は行っていない。

公開は指定された新規ディレクトリのCSV・本報告2ファイルのみ。非公開対応表・内部絶対path・host/IP・mount詳細・個人識別情報・private原本配置・credential・内部pathの単純hash・受領ファイルdigestはPublic GitHubへ追加しない。詳細根拠と実path対応は本人所有owner-onlyの非公開原本に保存する。

独立cloneで公開直前のremote HEADを確認し、B等の並行追加を保持したfast-forward更新のみ行う。既存inventoryを上書きせず、main merge・force push・研究branch変更は行わない。公開後にGitHubから読み戻し、local/remote SHAと公開bytesを確認する。139件の削除・登録解除・branch削除・原本移動は未実施。完了後mandatory STOP。
