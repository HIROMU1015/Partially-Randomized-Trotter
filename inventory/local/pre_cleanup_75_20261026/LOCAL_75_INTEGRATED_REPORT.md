# ローカル75件：保全・環境間依存の統合と条件付き整理計画

PUBLIC_REDACTED_PRE_CLEANUP_75。資料系列日付は2026-10-26。統合入力確定時点（UTC）：`2026-10-26T15:03:07.491683+00:00`。Repository：HIROMU1015/Partially-Randomized-Trotter。

**75件の統合を完了した。64件はPREPARED_PENDING_SERVER_B、11件はHISTORICAL_SOURCE_UNRESOLVED。全75件の既存worktreeを維持し、削除・登録解除・branch整理・原本移動は実施していない。** 所有者の全75件に対する将来workspace利用終了は確定済み。同じ質問や承認証明の作成を要求しない。

## 入力と判定の意味

開始時のGitHub専用branch HEADは、指定の`5eef8de248d454cc2497ef9631bfc171b1a78aae`と一致した。Aの受領済み実パス結果2資料をこのcommitで取得し、既存の独立公開cloneの同一資料と照合した。基準集合は既存75件CSVの正確なID集合であり、branch名や連番から再構成していない。Research BのLOCAL-WT-0283/0286は対象外。最新版`research_project_instructions_unified_2026-10-25.md`をローカル実在ファイルで読み、§8.1・§8.4・§10を適用した。A自身では同正式版が未取得だったという記録を、ローカルの読了で上書きしていない。

- 234件追跡入力：`2caa236fa7332872b5a346f0744318d48d59f6ad`、[群別判断](../followup_group1_234_20261026/LOCAL_234_GROUP_DECISIONS.md)。
- 図表7件・75件基準：`950a866f4756e29c06a691c6b80e7b0e431777eb`、[75件保全報告](../batch_cleanup_75_20261026/BATCH_75_REPORT.md)。
- 68件実保全：`d908f578a41c8ee235ef5bcbb75af6435f473278`、[68件保全報告](../archive_followup_68_20261026/LOCAL_68_REPORT.md)。
- ローカル実パス・旧SOURCE限定探索：`3fc92cad436c400d2f850cdb533d77d18a8ee648`、[前回のscope](../path_and_historical_source_followup_20261026/LOCAL_PATH_RECONCILIATION_REPORT.md)。
- Aの最新R1：`5eef8de248d454cc2497ef9631bfc171b1a78aae`、[A実パス照合報告](../../server_a/local75_actual_path_reconciliation_20261026/SERVER_A_LOCAL75_ACTUAL_PATH_REPORT.md)。
- 初回統合：[統合棚卸し](../../integrated_local_server_a/INTEGRATED_REPORT.md)。

元の保持分類は{"KEEP_EVIDENCE":56,"HOLD_UNKNOWN":19}のまま保持した。234件・batch75・archive68・前回path確認の判断と、今回の新判断をCSVの別fieldへ分けた。過去のPATH_VERIFICATION_NOT_POSSIBLE／PREPARED_MAP_NOT_RECEIVEDは履歴であり、現在のA結果には使わない。古いVERSIONED_ONLY_PENDING_OWNERも履歴で、今回所有者への再質問条件ではない。

ファイル保全、物理checkoutへの依存、Git履歴・科学SOURCEの保持をそれぞれ独立して評価した。PREPARED_PENDING_SERVER_Bは条件付き計画状態であり、READY_TO_DELETEや削除許可ではない。具体的な追加保持義務は既存scope内で未検出だったため、PRESERVATION_OR_CONTRACT_GATE／KEEP_REQUIRED／UNRESOLVED_OTHERは今回75件内では各0。既存契約が解除されたという意味ではなく、Bや差分確認で具体的束縛が見つかったIDだけを該当gateへ移す。

## 保全済みscopeと限界

| 保全群 | 件数 | 過去に実施・検証した内容 | 今回の新判断 |
| --- | ---: | --- | --- |
| 図表・スライド | 7 | 2,052 files／234,267,578 bytes、画像・PDF/SVG・原稿・作成script・選定入力・provenance/STOP | PREPARED_PENDING_SERVER_B |
| QC | 18 | 選定HEAD版18,216パス、コード・tests・設定・研究packet・失敗/STOP | PREPARED_PENDING_SERVER_B |
| Research B B2D Outer | 34 | 選定HEAD版8,094パス、実装・入力/result・cohort・契約記録 | PREPARED_PENDING_SERVER_B |
| N1/N3検証済み | 5 | 既存16件のうちgapのない正確な5 ID、版別現存資料 | PREPARED_PENDING_SERVER_B |
| N1/N3旧SOURCE未解決 | 11 | 現存資料・validation/失敗/STOPを保存、旧3版55宣言は未解決 | HISTORICAL_SOURCE_UNRESOLVED |

QC等68件の実コピー8,284 files／511,618,500 bytes、保存先内hardlink32,419を含む40,703パスを維持する。68 HEAD版40,701パスと、固定commit`d6c831da0b75ab48d1e54471e321ce6694bc0027`から回収済みの旧SOURCE2ファイルである。図表7件と合計すると実コピー10,336 files／745,886,078 bytes、版別参照42,755パス。これらは科学的に独立な結果の件数ではない。SOURCE・cohort・失敗/STOPのbindingを共用bytesだけで統合しない。

既存manifestとSHA-256/Git blob・inode独立性の検証結果を再利用した。前回42,753 HEAD版保存パスのregular file・元checkoutへのsymlink/hardlink 0という確認と、別保存2旧SOURCEの検証により、**選定済み保存ファイルの閲覧に元checkoutを必要としない**条件を引き継ぐ。全ファイルの再コピー・再hash、scientific verifierの実行は0。対象内の新しい未保存Git外科学資産は既存scopeで未特定であり、全filesystemでの不存在を保証しない。

保存先は同じローカルdisk上にあり、別障害域の独立backupではない。外部private raw／原本の保管責任・独立backup・未記録の契約は未確認のまま。具体的な未保存物やpath束縛が出たIDは保全先行gateへ移すが、一般的なUNKNOWNを根拠なく全件の追加大量転送へ拡張しない。独立backupが必要なら容量・保存先・対象を別途具体化し、今回は作成しない。

図表の引用相対リンクは元の固定GitHub参照を併用し、保存階層から閲覧する。Gitや一時pathを前提とする作成script、dynamic import、共有venv/cache、科学runtimeの可搬実行は未保証。SOURCE/input/grant/ledger/marker/STOPのbytes・固定path条件・one-shot/retry/resumeは変更しない。

## Aの最新結果で解消した条件と残る条件

ユーザー経由で受領した最小非公開対応表について、Aは75/75件のID・branch・完全HEAD・正規pathの受領整合性と照合を完了した。主判定はNO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE 71件、LOGICAL_GIT_REFERENCE_ONLY 4件（LOCAL-WT-0203〜0206）、PHYSICAL_DEPENDENCY_CONFIRMED 0。**AのR1範囲で元checkout必須束縛を検出していない75件**という結論へ更新する。以前のpath未受領・照合不能を現在のgateから除いた。

4件は公開Git/SOURCE・図表文書への論理参照として残す。Git保持refや論理SOURCE参照の存在は、元LOCAL directoryを永久保持する根拠ではない。Aは既存候補4,043からJSON pointerだけの1,385候補を区別した2,658候補と既知alias/process snapshotを照合し、有効な元path/配下参照一致0を報告した。この既存結果を再利用し、2,135資料や候補一覧の再検索は行っていない。

全75件にAの副条件ACTIVE_OR_RESTART_DEPENDENCY_UNRESOLVEDとPATH_OR_MOUNT_IDENTITY_UNRESOLVEDを残す。R1の主判定に未解決0とあることを副条件の解消へ読み替えない。未知のmount/export/bind/symlink同一性、未記録job・他Codex・scheduler・再開予定、別namespace、Bの依存は未確認。新たなA全件検索は要求せず、各担当は既存scopeから外れた実依存と直前差分だけを必要に応じ確認する。

ローカルでは旧409件のpath文脈が歴史snapshot／出版・作業場所記録で、実行directory必須束縛を示さなかったという評価を再利用した。前回206変更metadata照合の新一致0も再検索していない。今回75件のHEAD・branch・clean/untracked状態・以前と同じignored directory一覧・共通Git関係を限定的に再確認し、前回から差分0。これは将来の操作直前gateの完了や、現在の全job不使用の証明ではない。

## 先行64件：4バッチ

正確な集合は基準CSVと保全結果のID対応から抽出した。各行の保持対象・実施済み確認・残条件・将来操作・rollback・承認は[64件計画CSV](LOCAL_64_CONDITIONAL_CLEANUP_PLAN.csv)に記載した。実pathと具体的な操作対象表は非公開原本だけに保存する。

| 推奨順序・バッチ | 件数 | 正確なID集合 |
| --- | ---: | --- |
| BATCH_1_FIGURES | 7 | LOCAL-WT-0203, LOCAL-WT-0204, LOCAL-WT-0205, LOCAL-WT-0206, LOCAL-WT-0207, LOCAL-WT-0208, LOCAL-WT-0209 |
| BATCH_2_QC | 18 | LOCAL-WT-0023, LOCAL-WT-0024, LOCAL-WT-0025, LOCAL-WT-0026, LOCAL-WT-0027, LOCAL-WT-0028, LOCAL-WT-0029, LOCAL-WT-0030, LOCAL-WT-0031, LOCAL-WT-0032, LOCAL-WT-0033, LOCAL-WT-0034, LOCAL-WT-0035, LOCAL-WT-0036, LOCAL-WT-0037, LOCAL-WT-0038, LOCAL-WT-0039, LOCAL-WT-0041 |
| BATCH_3_RESEARCH_B | 34 | LOCAL-WT-0271, LOCAL-WT-0272, LOCAL-WT-0273, LOCAL-WT-0274, LOCAL-WT-0275, LOCAL-WT-0276, LOCAL-WT-0277, LOCAL-WT-0278, LOCAL-WT-0279, LOCAL-WT-0280, LOCAL-WT-0281, LOCAL-WT-0282, LOCAL-WT-0284, LOCAL-WT-0287, LOCAL-WT-0288, LOCAL-WT-0289, LOCAL-WT-0290, LOCAL-WT-0291, LOCAL-WT-0292, LOCAL-WT-0295, LOCAL-WT-0297, LOCAL-WT-0299, LOCAL-WT-0302, LOCAL-WT-0305, LOCAL-WT-0306, LOCAL-WT-0307, LOCAL-WT-0308, LOCAL-WT-0310, LOCAL-WT-0311, LOCAL-WT-0312, LOCAL-WT-0314, LOCAL-WT-0315, LOCAL-WT-0316, LOCAL-WT-0318 |
| BATCH_4_N1_N3_VERIFIED | 5 | LOCAL-WT-0003, LOCAL-WT-0004, LOCAL-WT-0006, LOCAL-WT-0007, LOCAL-WT-0008 |

今回の64件内に追加の具体的な物理保持義務を持つIDは検出せず、バッチからの追加除外0。旧SOURCE未解決11件は初めから別枠とした。Bや直前確認で問題が判明したIDだけをバッチから除外し、残りの正確な集合で条件充足を評価する。75件を1件ずつ所有者に再承認させる方式にはしない。

Batch 1は原稿・図表・生成元を版別に残す。0208/0209の自身のbranch/HEADを統合commitだけで代替しない。0207の同名remote branchは存在しないが、local branchは保持され、固定HEAD`7c0396be147887d0b33580236b4dce1890e4f230`はremote保持ref`refs/heads/codex/pr-resource-slides-current-20261022`の祖先として今回再確認した。

Batch 2はQC strategy handoff`fb635213d99e666852b87212c051ab08a6977c41`を入口に、18自身のref・診断・失敗・STOPを残す。Batch 3はResearch B pause handoff`e87375fa9a998dce80792be0f15597c747f00d83`を入口に、34自身のref・独立cohort・SOURCE/controlを残す。対象外0283/0286の供給責任を変更しない。

Batch 4は0003/0004/0006/0007/0008。N3 closure`ccc575ee339be091fd4031272c38bee13866f097`だけでは非祖先の0003を代替できない。各自身のbranch/HEADを残す。後続commitの祖先であることは、rawや独立cohortが科学的に包含されたことを意味しない。

## 旧SOURCE未解決11件を独立して扱う

対象：LOCAL-WT-0005, LOCAL-WT-0009, LOCAL-WT-0010, LOCAL-WT-0011, LOCAL-WT-0012, LOCAL-WT-0013, LOCAL-WT-0014, LOCAL-WT-0015, LOCAL-WT-0016, LOCAL-WT-0018, LOCAL-WT-0019。旧SOURCEの期待bytesは以下3版、未解決宣言55対応のまま。既にローカルとAで限定探索済みで、同じ探索・再構築・再実行は今回0。

| gap | 公開済み相対path | 旧版の期待SOURCE内容SHA-256 |
| --- | --- | --- |
| LOCAL-68-HISTORY-GAP-001 | `src/trottertracks/representation_exploration/n3_occupation_transfer_v1.py` | `6799702fddd0c54784082d0f5648ff1ad6b8b6faa084dee03d41487850db9b28` |
| LOCAL-68-HISTORY-GAP-002 | `scripts/run_n3_occupation_transfer_v1.py` | `d43628daa116c12517e95786820885c1cd144988a10f0bca72e2be848e127795` |
| LOCAL-68-HISTORY-GAP-003 | `tests/test_n3_occupation_transfer_v1.py` | `6533f3fb82e79f8409a87ef2bbf01e9617e310150603890762f536a60da7b2bc` |

旧bytes回収、過去validation/失敗/STOP保全、現在checkoutの保持必要性を分ける。現存ファイルと過去記録は版別archiveとGitに保全済みで、現在の同名ファイル33個の前回限定確認でも旧期待bytesは未発見だった。現checkoutを保持するだけで欠落が解消するとは扱わず、欠落だけで物理worktreeを永久保持必須にもしない。

11件もAのR1内で必須物理依存は未検出、B/job/alias/契約差分は未確認。旧SOURCE回収と独立にcheckoutを整理できる可能性はあるが、旧宣言・失敗・STOP・既存ref・現存archive・欠落状態を将来も保持し、物理checkoutが供給元ではないことを確認したうえで、**11件を別に指定した将来の明示指示**が必要。先行64件の操作候補には含めず、今回は維持する。科学契約の変更やSOURCE欠落の受容を今回代行していない。

## B担当へ依頼する最小確認

[B向け75件CSV](SERVER_B_75_DEPENDENCY_REQUEST.csv)にID・branch・完全HEAD・研究系列・既存の公開SOURCE/result/STOP参照と具体的質問を記録した。SOURCE/provider/input、現在job・他Codex・再開予定、固定path・shared mount/export/bind/symlink、共通Git/alternate・editable runtime、元checkoutの実必要性、現在のDF-native L_D=0 B3への影響を分けて回答してもらう。公開参照が空のIDはSOURCE不存在と推定しない。

旧Track B/G6/W2CのB3と現在のDF-native B3を混同しない。B担当は既存記録で非干渉に確認し、B3 job/SOURCE/input/raw/cache/Git/grant/ledger/marker/STOPへ変更を加えない。ownerの利用終了判断の再承認は不要。実pathは既存非公開原本を再利用した最小表をローカルに準備したが、**自動転送0、Bへの接続0**。受け渡しが必要な場合はユーザーが行い、方法を指定しない。全75件でBの非依存は未確認。

## 将来の操作手順・保持条件・rollback

1. **B担当／各環境担当**：上記の実依存と、全75件の未確認job/再開・未記録alias/共有実体を、必要な既存scopeと差分だけで回答する。具体的なSOURCE/provider/input・固定path・private原本の束縛があれば、そのIDだけ保全/契約gateへ移す。新しい全件探索や科学実行へ拡張しない。
2. **ローカルCodex（後日の指示後）**：B結果と非公開75件表を照合し、64件の候補集合から残gateのあるIDを外す。保持branch/commit/SOURCE参照・manifest・保存資料を読む条件を確認する。一般的なbackup限界を明示し、必要な追加保全があれば別途具体化する。
3. **ユーザー**：閉じたgateと正確なID集合を指定し、バッチ単位で将来のworktree登録解除・directory削除を明示許可する。今回の計画公開許可や利用終了方針は削除許可ではない。branch削除・原本移動・共有Git/環境の整理は含めない。
4. **該当Codex（具体的操作の指示後、直前）**：非公開の正規path・登録backlink・common Git・owner/alias/mountを再照合する。HEAD/branch、tracked/untracked、ignored内の新資産・保存scope、保持ref到達、archive可読性と変化したファイルだけの必要hash、現在job/再開・SOURCE/path依存差分、worktree lock状態を確認する。将来のbranch tipが動いていれば古いSHAを黙って置換しない。
5. **該当Codex（別途許可された将来操作のみ）**：通常の`git worktree remove`を非公開表の正規pathへ1件ずつ適用する案。この操作はcheckout directoryと登録を共に除去する。登録だけ解除して残directoryを手動再帰削除する方式や、force/prune/clean/gcで拒否を回避する方式は計画しない。拒否・dirty・未知資産・lock・依存があればそのIDを維持して報告する。共有Git/objects、branch、main、環境、archiveと旧原本へ触れない。
6. **該当Codex（将来の操作後）**：許可対象だけの登録・directory変化と、branch/共通Git/保全資料/他worktreeの維持を確認する。問題時は保存した正規path・branch・完全HEAD・backlink・guide情報とmanifestからtracked checkoutを再作成する案を使う。復旧作業にもその具体的指示が必要。path/branch競合を強制上書きしない。ignored runtimeや科学実行環境まで完全復元できる保証はない。

将来の想定影響は承認対象checkoutと登録の減少であり、disk回収量は測定していない。必要なデータと履歴はそのまま保持する。原本・archive・common Gitを消して空き容量を増やす計画ではない。74件は現在も同名GitHub tipと一致、0207は上記保持refの祖先で追跡可能。全75 local branch、固定SOURCE refs、科学STOP/one-shot/retry/resume条件を維持する。文書へSHAを書くことだけを永久Git保管の代わりにしない。

## 成果物・非公開原本・検証

- [LOCAL_75_CLEANUP_READINESS.csv](LOCAL_75_CLEANUP_READINESS.csv)：75件、旧分類・保全・現在A_R1・B/契約/private custodyの残条件と新判定。
- [LOCAL_64_CONDITIONAL_CLEANUP_PLAN.csv](LOCAL_64_CONDITIONAL_CLEANUP_PLAN.csv)：4バッチ64件。削除実行リストではない。
- [SERVER_B_75_DEPENDENCY_REQUEST.csv](SERVER_B_75_DEPENDENCY_REQUEST.csv)：B担当の将来照合質問75件。
- 本報告：統合根拠・限界・操作条件・担当。

既存の本人所有・owner-only資料領域の新規統合directoryに、75件の実path/branch/SHAとarchive/manifest/backlink/guide/保持refの対応、64件の将来操作文字列、全75件のrollback参考情報、B向け非公開最小表、入力と検証結果を保存する。コマンド文字列は計画資料であり実行しない。旧非公開原本・manifestを上書きしない。公開へ実path、username、内部host/IP、private raw配置、非公開ファイル名、credential、pathの単純hash、非公開表本体や保全ファイル本体を含めない。

既存行政用CSV/Git/hash/匿名化helperを再利用し、新規処理は75件のID join・独立した状態field・4バッチ抽出と今回指定4資料の生成/検証に限定した。科学moduleのimport・SOURCE解析・科学的採否・計算・sampling・compile・数値再解析0。archive payloadの再hash/再コピー0。A/BへのSSH・自動転送0。

公開は既存の独立cloneのinventory専用履歴で新規directory4ファイルだけを追加する。push直前の現在remoteを再確認し、並行追加があれば最新を保持した独立公開領域から追加する。force push・研究branch変更・main merge・既存inventory編集はしない。commit/remote一致と既存原本保護の最終結果は非公開PUBLICATION_RESULTと完了通知へ記録する。

**全75件を維持。現時点の実削除可能確定0・削除許可0。mandatory STOP。次の担当はB担当の非干渉依存確認であり、結果統合と実際の整理実行はそれぞれ後日の具体的指示後に行う。**
