# ローカル75件：3環境最終照合と64件の整理準備

PUBLIC_REDACTED_FINAL_THREE_ENVIRONMENT_RECONCILIATION。2026-10-27 JST。入力観測：`2026-10-27T01:26:13.336756+09:00`。Repository：HIROMU1015/Partially-Randomized-Trotter。整理専用branch：`codex/pr-environment-inventory-20261026`。

**75件の統合を完了した。先行64件はすべてBATCH_APPROVAL_CANDIDATE、旧SOURCE未解決11件はHISTORICAL_SOURCE_UNRESOLVED。64件内の具体的阻害要因・追加保全必須・限定追加確認必須・具体的物理保持義務は各0。全75件のworktreeを維持し、削除許可・実行は0。**

## 固定入力と再利用

開始時の現在remote HEADは指定の`33fd56d573c3d6022609b9141daf2429d19c75b4`と一致した。既存の独立公開cloneだけに取得し、A/B公開資料・従来計画・保全索引と必要な非公開原本を実際に読んだ。A/Bホストへ接続していない。

| 資料 | 固定commit・参照 |
| --- | --- |
| Bの75件実パス照合 | `33fd56d573c3d6022609b9141daf2429d19c75b4`、[B_PATH_1報告](../../server_b/local75_actual_path_reconciliation_20261027/SERVER_B_LOCAL75_ACTUAL_PATH_REPORT.md)と同directoryの75件CSV |
| LOCALの従来75件・64件計画 | `e5651ab7b7f1682dac51a756e6b277f4a2d9c217`、[従来統合報告](../pre_cleanup_75_20261026/LOCAL_75_INTEGRATED_REPORT.md)と同directoryの4資料 |
| Aの75件実パス照合 | `5eef8de248d454cc2497ef9631bfc171b1a78aae`、[A_R1報告](../../server_a/local75_actual_path_reconciliation_20261026/SERVER_A_LOCAL75_ACTUAL_PATH_REPORT.md)と75件CSV |
| 図表7件・75件基準 | `950a866f4756e29c06a691c6b80e7b0e431777eb`、[batch75](../batch_cleanup_75_20261026/BATCH_75_REPORT.md) |
| QC等68件実保全 | `d908f578a41c8ee235ef5bcbb75af6435f473278`、[archive68](../archive_followup_68_20261026/LOCAL_68_REPORT.md) |
| ローカル実パス・旧SOURCE確認 | `3fc92cad436c400d2f850cdb533d77d18a8ee648`、[path/source報告](../path_and_historical_source_followup_20261026/LOCAL_PATH_RECONCILIATION_REPORT.md) |
| 元の統合棚卸し | [統合資料](../../integrated_local_server_a/INTEGRATED_REPORT.md) |

最新版`research_project_instructions_unified_2026-10-25.md`をローカル実在ファイルで確認し、§8.1・§8.4・§10を適用した。旧SOURCE・科学契約・one-shot/retry/resume・SOURCE/raw/grant/ledger/marker/STOPを変更しない。これは科学的採択や詳細科学レビューではない。

今回の64件は固定LOCAL_64_CONDITIONAL_CLEANUP_PLAN.csvの正確な集合を再利用した。75件は旧75件表・非公開ID逆引き・原本へjoinし、A/BのID・branch・完全HEADと全行一致した。元保持分類と各過去判断は別fieldとして保持した。元の保持分類の書換えや、同一HEADからの同一科学結果認定はしていない。

## 3環境の結果と今回解消した条件

| 環境・scope | 照合完了 | 必須物理依存を検出せず | 論理参照のみ | 必須物理依存確認 |
| --- | ---: | ---: | ---: | ---: |
| LOCALの既存索引・path文脈評価と今回限定差分 | 75 | 75 | 別fieldで保持 | 0 |
| A_R1 | 75 | 71 | 4 | 0 |
| B_PATH_1 | 75 | 75 | 0 | 0 |

Aの論理参照4件はLOCAL-WT-0203〜0206であり、公開Git/SOURCE等への参照として維持する。これを元checkoutの物理保持義務へ格上げしない。Bには4件を代入せず、B自身のscopeでは75件とも限定非依存という結果を保持する。

Bは75 ID・branch・完全HEAD・正規pathの照合を完了し、受領path・共通Git・外向きguideリンク先への一致は各0。今回LOCALでも、B報告にある受領bytesのdigestと、既存ローカル最小表原本のdigestが一致することを確認した。B自身が転送前の期待digestを未提示としていた事実は変更せず、今回のローカル比較として区別する。digest値や非公開ファイル名・実pathは新しい公開資料へ載せない。

前回のPREPARED_PENDING_SERVER_Bは履歴として残し、**B実パス照合未実施・未受領という条件は75件で解消済み**に更新した。A/Bの既知範囲とローカルの証拠が整合したことにより、64件はバッチ承認候補として判断できる。これは全環境で非依存を数学的に証明した状態や、削除承認ではない。

Aは既存候補/既知alias・限定process snapshotを照合したR1。Bは保存path occurrence 2,023、SOURCE/runtime宣言2,146、既知12参照先・runtime metadata・既知B3 chainを対象としたB_PATH_1である。Bの抽出候補は特定形式の引用絶対pathが中心で、任意の相対/動的path・未記録metadata・別namespace・他jobを網羅しない。2,135資料・候補索引・38静的資料・raw/cache/checkpoint等の全面調査を今回繰り返していない。

## 75件の新判断と4バッチ

| 新主判断 | 件数 | 意味 |
| --- | ---: | --- |
| BATCH_APPROVAL_CANDIDATE | 64 | 必要な選定file/Git refの保全と3環境の限定照合が整合し、具体的阻害要因なし。バッチ実行承認と直前preflightが必要 |
| TARGETED_CHECK_REQUIRED | 0 | 対象を限定する具体的な依存/保全疑義は今回未検出 |
| PRESERVATION_ACTION_REQUIRED | 0 | checkoutだけにある新しい重要資産・ref欠落等の具体的追加保全必要例は今回未検出 |
| KEEP_REQUIRED | 0 | 今回75件内で具体的な物理保持義務は未検出。全75件を今は維持するという操作方針とは別 |
| UNRESOLVED | 0 | 上記以外の具体的情報不足で主判断を保留した例はなし |
| HISTORICAL_SOURCE_UNRESOLVED | 11 | 旧3版/55宣言の保全問題を別枠維持。先行64件に入れない |

| バッチ | 件数 | 正確な対象ID | 主判断 |
| --- | ---: | --- | --- |
| Batch 1：図表・スライド | 7 | LOCAL-WT-0203, LOCAL-WT-0204, LOCAL-WT-0205, LOCAL-WT-0206, LOCAL-WT-0207, LOCAL-WT-0208, LOCAL-WT-0209 | BATCH_APPROVAL_CANDIDATE |
| Batch 2：QC | 18 | LOCAL-WT-0023, LOCAL-WT-0024, LOCAL-WT-0025, LOCAL-WT-0026, LOCAL-WT-0027, LOCAL-WT-0028, LOCAL-WT-0029, LOCAL-WT-0030, LOCAL-WT-0031, LOCAL-WT-0032, LOCAL-WT-0033, LOCAL-WT-0034, LOCAL-WT-0035, LOCAL-WT-0036, LOCAL-WT-0037, LOCAL-WT-0038, LOCAL-WT-0039, LOCAL-WT-0041 | BATCH_APPROVAL_CANDIDATE |
| Batch 3：Research B B2D Outer | 34 | LOCAL-WT-0271, LOCAL-WT-0272, LOCAL-WT-0273, LOCAL-WT-0274, LOCAL-WT-0275, LOCAL-WT-0276, LOCAL-WT-0277, LOCAL-WT-0278, LOCAL-WT-0279, LOCAL-WT-0280, LOCAL-WT-0281, LOCAL-WT-0282, LOCAL-WT-0284, LOCAL-WT-0287, LOCAL-WT-0288, LOCAL-WT-0289, LOCAL-WT-0290, LOCAL-WT-0291, LOCAL-WT-0292, LOCAL-WT-0295, LOCAL-WT-0297, LOCAL-WT-0299, LOCAL-WT-0302, LOCAL-WT-0305, LOCAL-WT-0306, LOCAL-WT-0307, LOCAL-WT-0308, LOCAL-WT-0310, LOCAL-WT-0311, LOCAL-WT-0312, LOCAL-WT-0314, LOCAL-WT-0315, LOCAL-WT-0316, LOCAL-WT-0318 | BATCH_APPROVAL_CANDIDATE |
| Batch 4：N1/N3保全検証済み | 5 | LOCAL-WT-0003, LOCAL-WT-0004, LOCAL-WT-0006, LOCAL-WT-0007, LOCAL-WT-0008 | BATCH_APPROVAL_CANDIDATE |

最初の実行候補はBatch 1の図表・スライド7件。その後QC18、Research B34、N1/N3検証済み5を順に推奨する。各バッチの根拠・保持対象・STOP/復旧条件・将来の承認範囲は[実行前計画](LOCAL_64_BATCH_EXECUTION_PLAN.md)に記載した。ユーザーが75件の利用終了を再回答する作業は残さない。

## 限定ローカル差分と残条件の整理

今回75件のHEAD/branch、tracked clean・全untrackedなし、過去と同じignored directory一覧、既知guideリンク、正規path/所有者・common Git・登録backlinkを確認し、差分なし。現在のignored leaf名も既知cache/guideの範囲で、未分類の重要候補0。worktree lock 0、可視の本人UID processのcwd/引数への対象path一致0、既知のcovering mount metadata差分0。process環境変数や全job情報を収集せず、操作・停止もしていない。

75 archive rootと各代表1ファイルのregular/readable・source inode独立性をmetadataで確認した。これは全保存fileの現時点の再hashやruntime試験ではなく、過去の完全性確認を再利用した限定的な可読性差分確認である。今回の確認日時・scopeはCSVへ記録し、将来の直前確認を完了扱いにしていない。

残条件を次に分けた。

1. **具体的阻害要因**：先行64件内では0。SOURCE/path束縛、未保存データ、稼働job、lock、ref消失などが新たに具体化した場合は、そのIDだけ候補から外す。一般的な実行承認で固定契約保護を解除しない。
2. **直前に確認すればよい条件**：HEAD/branch、tracked/untracked/ignored配下の新重要資産、利用状態・lock、保持ref/objects、archive/manifestの可読性・変更、SOURCE/provider/input/path依存の新差分、実path/alias/mount/登録の整合。全64件で将来preflightが未実施であり、必要な差分だけ確認する。
3. **範囲外の残余リスク**：未記録mount/export/bind/alias・共有実体、未知のnamespace・他所有者/他job/再開予定、動的/相対path、未索引契約とprivate custody。これらを主判断から分離し、具体的兆候がある対象だけ追加確認する。同じ全件監査を無期限に繰り返す条件へ変換しない。

## 保全とbackupの区別

| 保持単位 | 確認済み内容 | 限界 |
| --- | --- | --- |
| GitHub上のtracked版 | 74件は同名remote tipに完全HEAD一致。0207は別保持refから祖先到達。各local branchも保持 | 文書にSHAを記すだけでは永続保管にならず、将来も保持ref/objectsを維持する。private rawのbackupではない |
| ローカル非公開実ファイル | 図表2,052＋QC等8,284＝10,336 files、745,886,078 bytes。版別42,755パス、うち68件40,703パスは保存先内hardlinkを含む | 元checkoutから独立して閲覧できるが、同じdisk上で別障害域backupではない。実行環境の完全可搬性は未保証 |
| 外部private raw・未記録原本 | 既存metadataの保管責任をそのまま保持 | 内容・全配置・独立backupは未確認。具体的なcheckout固有依存は新たに未検出 |
| 独立backup | 今回新規作成0 | 一般的UNKNOWNだけで既保全files全件の再コピーを要求しない。具体的必要性が出た対象だけ容量/保存先/理由を別途提案 |

既存42,753 HEAD版manifest bindingと固定回収SOURCE2件の検証を再利用した。HEAD版の各ID/branch/commit対応は今回manifestを読んで一致確認済み。過去SHA-256/Git blob・sourceとarchiveのinode独立性を再利用し、保存payloadの再コピー/再hashは0。過去にsourceへのhardlink 0、保存先内hardlink32,419だったことを維持する。

削除予定checkoutが唯一の重要データ保管先となる具体例は、選定scopeと今回status/ignored差分では未特定。外部raw全体の不存在は主張しない。今回追加backup対象・容量・宛先を具体化すべき新資産は検出していないため、新規大規模backupや原本移動は行わない。

必要なSOURCE/result/STOP公開locatorが指す56 commitは保持common Gitで存在を確認した。回収済み固定SOURCE2ファイルはcommit`d6c831da0b75ab48d1e54471e321ce6694bc0027`への保持remote ref祖先到達を確認した。これは科学SOURCE bytesの再検証や研究契約の解除ではない。

## 旧SOURCE未解決11件

対象：LOCAL-WT-0005, LOCAL-WT-0009, LOCAL-WT-0010, LOCAL-WT-0011, LOCAL-WT-0012, LOCAL-WT-0013, LOCAL-WT-0014, LOCAL-WT-0015, LOCAL-WT-0016, LOCAL-WT-0018, LOCAL-WT-0019。3版・55宣言の未解決状態を維持する。

| gap | 公開済みsource相対path | 旧bytesの期待SHA-256 |
| --- | --- | --- |
| LOCAL-68-HISTORY-GAP-001 | `src/trottertracks/representation_exploration/n3_occupation_transfer_v1.py` | `6799702fddd0c54784082d0f5648ff1ad6b8b6faa084dee03d41487850db9b28` |
| LOCAL-68-HISTORY-GAP-002 | `scripts/run_n3_occupation_transfer_v1.py` | `d43628daa116c12517e95786820885c1cd144988a10f0bca72e2be848e127795` |
| LOCAL-68-HISTORY-GAP-003 | `tests/test_n3_occupation_transfer_v1.py` | `6533f3fb82e79f8409a87ef2bbf01e9617e310150603890762f536a60da7b2bc` |

旧bytesの所在、過去validation/失敗/STOP/ledger保全、現存ファイル保全、物理checkout保持必要性、A/Bの限定照合、将来の別判断を独立して記録した。既存資料・現存bytesはarchiveとGitへ保全済み。現在checkoutに旧期待bytesがないという前回限定確認を再利用し、同じ探索・再構築・科学的補完をしていない。worktreeを保持するだけで回収が進むとは考えず、欠落を受容して科学契約を変更したことにもしない。

11件にもA_R1/B_PATH_1で必須物理依存は検出されなかった。具体的な物理保持義務は既存scopeで未特定だが、旧SOURCE保全問題と物理整理条件を分離した将来の別判断・対象を明示した操作指示が必要。先行64件の実行候補から除外して維持する。

## B3の保護

B報告時点ではDF-native L_D=0 B3がSCIENTIFIC_ACQUISITION_RUNNINGで、既知SOURCE/provider/input・runtime・Git管理chainにLOCAL75の必須束縛は未検出だった。旧Track B/G6/W2CのB3やResearch B pauseとは区別する。base SOURCE`155529e20a8fac39e00786c40b57830b71203c4f`、復旧SOURCE`ae3d301e2318a3f9f1331c047c68d310771b6849`、provider snapshot`c31ef424175c02379eafebe8fc6c7548c1d9aabe`の区別を保持し、LOCAL checkout HEADと科学SOURCEを同一視しない。

今回Bのservice/jobを新しく観測・操作しておらず、B報告以降の現時点状態を独自確認したとは言わない。B3を止める・再開する・SOURCE/input/raw/cache/Git/runtime/grant/ledger/marker/STOPを変更する作業は0。将来もLOCALの承認対象checkoutだけを整理し、Bへ接続・自動転送・環境操作しない。

## 成果物と後日の実行範囲

- [LOCAL_75_FINAL_RECONCILIATION.csv](LOCAL_75_FINAL_RECONCILIATION.csv)：75件、元分類・過去判断・保全・LOCAL/A/B・新判断と独立した残条件。
- [LOCAL_64_BATCH_READINESS.csv](LOCAL_64_BATCH_READINESS.csv)：固定64件、4バッチの保全/保持ref・3環境結果・具体的障害・直前preflight・承認待ち。
- [LOCAL_64_BATCH_EXECUTION_PLAN.md](LOCAL_64_BATCH_EXECUTION_PLAN.md)：正確なバッチ集合・保持対象・順序・STOP・将来の通常Git操作・rollback方針。
- 本報告：統合根拠・限界・旧11件・B3・後日の承認範囲。

非公開原本には新しい専用directoryへ75件の実path・Git管理/backlink・保存先/manifest・保持ref・64件だけの将来操作文字列と復旧情報を保存する。既存原本・manifest・保全filesを上書きしない。公開へ内部絶対path、username、host/IP、private raw配置、非公開ファイル名、秘密情報、内部pathの単純hash、受領表本体、保全file本体を追加しない。

既存CSV/Git/匿名化helperを再利用し、新規処理はB_PATH_1 join、64件判定のA/B/C条件分離と限定差分・今回指定4資料の生成に限定した。旧SOURCE広域探索・再構築、科学module import・計算・sampling・compile・数値再解析・詳細科学レビュー0。既存原本のmetadata hashを保護照合するが、承認証明システムを作るものではない。

**次はユーザーがBatch 1の正確な7 IDについて、LOCAL登録済みcheckoutのdirectoryと登録だけを除去する後日の操作範囲を判断する。** その明示指示後、ローカルCodexが直前preflightを行い、合格した承認IDへ通常のgit worktree removeを個別適用する計画である。拒否や問題が出たIDだけSTOPし、force/prune/clean/gc/手動強制削除を使わない。branch・重要commit・科学証拠・原本/archives・shared Git・他環境/B3を維持する。Batch 2/3/4は上表の正確な集合で独立した承認対象とできる。

今回の新規directory4資料だけを、既存の独立公開cloneから専用branchへ通常fast-forward commit/pushする。開始時とpush直前のremote確認、公開後local/remote SHA一致とGitHub読み戻し、原本・研究ref・登録の保全結果を非公開PUBLICATION_RESULTと完了通知へ記録する。並行更新は保持し、force/main merge/研究branch変更/既存inventory編集をしない。

**mandatory STOP。全75件を維持。今回は3環境統合・実行前計画と匿名化公開までであり、worktree削除・登録解除・branch削除・原本移動は未承認・未実施。**
