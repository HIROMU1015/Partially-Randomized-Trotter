# ローカル75件：実ファイル保全と整理の次段階

PUBLIC_REDACTED_BATCH_75。入力は固定commit `2caa236fa7332872b5a346f0744318d48d59f6ad` の234件CSV。対象はCSVのstudy_family・直前判断の正確な集合から抽出した。図表7、QC18、N1/N3 16、Research B確認済み34の計75。LOCAL-WT-0283/0286は対象外。元分類・前回判断を変更せず、batch_decisionを別に追加した。

| 群 | 件数 | 今回の判断 |
| --- | ---: | --- |
| 図表・スライド | 7 | ARCHIVED_VERIFIED_PENDING_DEPENDENCY |
| QC | 18 | VERSIONED_ONLY_PENDING_OWNER |
| N1/N3 | 16 | VERSIONED_ONLY_PENDING_OWNER |
| Research B B2D Outer | 34 | VERSIONED_ONLY_PENDING_OWNER |

PRESERVATION_REQUIRED／KEEP_REQUIRED／UNRESOLVEDは今回の75件内で各0。ただし削除可能確定は0。全75件のcheckoutは現状維持し、共通Git・研究branch・SOURCE・STOP・外部custodyを保護する。

## 図表・スライド7件の保全

所有者の「専用worktreeを再使用せず、実ファイルを残す」方針をこの7件に適用した。既存の本人所有・owner-only非公開追跡資料領域を再利用し、新規batchサブディレクトリへ各ID／完全commit／元repository相対階層でコピーした。旧資料・旧manifestは変更していない。従来の公開packet名とファイル階層を保持し、別の管理システムは作っていない。

`docs/progress/track_a/` の各版には、構成案・Q&A・参照メモ、PNG/SVG/PDF、保存入力・出典コピー、作成script、requirements、provenance・既存STOPが含まれる。これに参照用のPROJECT_MAP、研究概要、研究目的、VALIDATION_STATUS、validation manifestの各版を加えた。これらは既存bytesの保管であり、現在の科学仕様・成果として再認定していない。

| ID | 保存file instances | bytes | SHA-256・Git blob |
| --- | ---: | ---: | --- |
| LOCAL-WT-0203 | 230 | 32,398,399 | 全件一致 |
| LOCAL-WT-0204 | 509 | 52,835,051 | 全件一致 |
| LOCAL-WT-0205 | 49 | 10,182,944 | 全件一致 |
| LOCAL-WT-0206 | 109 | 15,430,487 | 全件一致 |
| LOCAL-WT-0207 | 133 | 17,526,167 | 全件一致 |
| LOCAL-WT-0208 | 509 | 52,871,318 | 全件一致 |
| LOCAL-WT-0209 | 513 | 53,023,212 | 全件一致 |

合計**2,052ファイル／234,267,578 bytes（約234 MB）**。画像1,357、作成script26を含む。重複する版を含むsource file instancesであり、固有の科学結果2,052件ではない。Git blobの異なりは517。別版を同一科学証拠と認定しない。

全件についてexclusive create、コピー前後のsource SHA-256／destination SHA-256、HEADのGit blob一致を確認した。コピー元とのhardlinkはない。同名上書きなし。非公開manifestに完全なsource/destination、branch、SHA、元相対path、fileごとのhashを記録した。保存先の本人所有・owner-only ACL、十分な空き容量、新規namespaceの非衝突をコピー前に確認した。

図表・原稿と同梱画像はcheckoutに依存せず直接参照できる。旧SOURCE prose内の相対リンク137件は元repository位置を前提とする引用で、元の固定GitHub locatorを使う。一般文脈リンク35件の宛先は各版へ追加コピーした。生成・検証scriptにはGitや旧一時pathを前提とするものがあるため、runtime全体の移設・全scriptの可搬実行を保証していない。描画・科学再解析・既存verifierは実行しなかった。科学raw・独立実行環境のbackupではない。

7件のうち0208/0209は既存統合snapshotの祖先ではないため、自身のbranch／commitと今回の版別コピーを残す。0207は同名の現行GitHub branchではなく、保持branchの祖先として追跡できる。いずれもbranch整理へ拡張しない。

## QC18件・N1/N3 16件

全34件はVERSIONED_ONLY_PENDING_OWNER。cleanで、現在の既知scopeに新しい固有Git外資産は検出していない。各branch／固定HEAD、SOURCE記録、結果・診断・失敗・旧STOPは保持する。既存preservation_index等は保全確認metadataであり、実ファイルの独立archiveがある証明にはしない。

QCの代表入口は四系列strategy handoff `fb635213d99e666852b87212c051ab08a6977c41`。18件のうち17祖先・1同一HEAD。C1準備／診断、C2/C3統合、C4 audit、mixed-access、onebody／PF関連等は各CSV行のHEAD subjectと固定参照で区別する。

N1/N3の代表入口はN3 closure `ccc575ee339be091fd4031272c38bee13866f097`。16件のうち14祖先・1同一HEAD・1非祖先（LOCAL-WT-0003）。N1回復、generator／occupation transfer、旧family診断、saved attribution／width比較の各記録を後続成功で置換しない。0003は自身の保持refを残す。

**所有者**は34件の再使用予定と利用終了・履歴保持役割を研究用途単位で判断する。その後も全SOURCE／cohort／失敗／STOPへの保持refを残す。代表文書だけを残して他branchを消す方針は今回の計画に含めない。

## Research Bの34件

VERSIONED_ONLY_PENDING_OWNER。最終成果／pause handoff `e87375fa9a998dce80792be0f15597c747f00d83` のGit祖先34件。design／adapter／preflight／stage1、static cost／Exact-R／full DF-PF、outer A0/B0、独立D1・recovery、IID／rare、shot／RTE／splitting等の完成・中間記録をCSV行の固定HEADと後続関係へ対応付けた。Git祖先をraw・独立cohort包含の証拠にしない。

現在のローカル既知scopeで追加実コピーを必要とする固有ファイルは未特定。今回この68件の科学rawや契約SOURCEを再archiveしなかった。外部private rawの保管・backup・復元性はUNKNOWNで、不要と認定しない。大規模転送を必要とする新たな具体的対象・容量・保存先も今回のscopeでは特定していない。

**所有者**はpause後の用途・将来再使用・中間checkoutの終了を判断する。0283/0286のSOURCE供給責任は前回UNRESOLVEDを維持して別扱い。現在実行中と申告されたサーバーBのDF-native B3とResearch B／旧Track Bの名称を混同しない。

## 実際の整理前の順序と担当

1. **所有者**：QC/N1/N3の34件とResearch Bの34件について、利用終了・再使用・履歴保持役割を判断する。図表7件のworkspace再使用方針は今回の指示で確認済み。
2. **サーバーA/B担当**：候補IDに対応する実directory、固定SOURCE、生成元、結果／STOP、shared Git／環境への実参照を各自の環境で確認する。ローカルCodexはSSHしない。B3への影響が否定できない対象は保持する。
3. **ローカルCodex（次の指示後）**：非公開ID対応表で対象pathを解決し、保管資料の参照性・外部custody・SOURCE/path責任・branch保持・shared Git保持を対象別に照合する。元worktree、branch、Git本体、保存コピーは別の整理単位として扱う。
4. **ユーザー**：残条件が閉じた対象について、checkout登録解除／directory削除の具体的範囲を別途指示する。今回の保全承認や利用終了回答から削除許可を推定しない。branch削除・データ移動・固定契約変更は別途対象を指定する。
5. **該当環境のCodex（具体的操作の許可後）**：実操作直前にHEAD・status・untracked／ignored・使用状況・SOURCE参照・保持ref・copy hashの差分を確認する。今回のチェックは将来のBEFORE_ACTION_DELTA完了の代替ではない。差分／依存が残ればその対象を維持する。

次段階まで**全75件を現状維持**。既存234件のKEEP_REQUIRED7件、未解決13件、主Git・現行研究環境も対象外として保持する。公開資料には実path・非公開ファイル名・private raw位置・内部host/IP・認証情報を含めず、公開済みtracked file名と匿名archive item IDだけを使用した。ファイル本体の公開は0。

**mandatory STOP。worktree削除／登録解除・原本移動・branch削除・merge・main変更・科学SOURCE/raw/grant/ledger/marker/STOP変更・科学計算・他環境操作は実施しない。**
