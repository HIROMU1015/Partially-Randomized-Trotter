# 固定330 GitHub branch一括削除・最終報告

PUBLIC_REDACTED。environment=LOCAL。2026-10-27 JST。

**固定330本を、完全SHAを期待値にした条件付きatomic transactionで全件削除した。成功330、保留0、失敗0、固定SHA不一致0。remoteは811→481本。KEEP472本・B3保留7本・追加2本、local branch/worktreeと研究履歴・保全原本を保持した。**

## 固定入力・承認範囲

Repository: HIROMU1015/Partially-Randomized-Trotter。整理branch: `codex/pr-environment-inventory-20261026`。

- LOCAL監査：`5ee9dcff7976e8c353792b8e79fa750d47c6ad82`、[固定330候補](../github_branch_cleanup_review_20261027/DELETE_CANDIDATES.csv)と同directoryの分類・全remote・履歴保持・重複/祖先・計画・REPORT。
- A依存確認：`90064f1726cc671496e40b1a330da58c30537cac`、[A報告](../../server_a/branch_ref_dependency_review_20261027/REPORT.md)と全330行CSV。
- B依存確認：`6170f3535f7e71a891dd76af17578f9703d9bbdd`、[B報告](../../server_b/branch_ref_dependency_review_20261027/REPORT.md)と全330行CSV・B3別枠表。

正式共通指示research_project_instructions_unified_2026-10-25.mdの実在と既読版からのSHA不変を確認し、§8.1/8.4/10を適用。ユーザーの今回指示を固定330 remote削除と限定的SHA付きleaseの明示許可として扱った。元CSVのdelete_authorized=Falseは改変せず、新実行記録へ許可状態を別記した。不要な承認証明書・receipt・grantは作っていない。

## 件数

| 項目 | 結果 |
| --- | ---: |
| 実行直前remote | 811 |
| 固定承認対象 | 330 |
| 削除成功 | 330 |
| 保留 | 0 |
| 削除失敗/不確定 | 0 |
| 固定SHA不一致 | 0 |
| 既に消失していて今回成功に算入しなかったもの | 0 |
| 削除後remote | 481 |
| KEEP ref保持 | 472 |
| B3関連別枠7保持 | 7 |
| 監査後追加branch保持 | 2 |
| 対象外remote削除・更新 | 0 |
| 実行中の並行追加・更新 | 0 |

## 研究単位別

| 研究整理単位 | 承認 | 成功 | 保留 | 失敗 |
| --- | ---: | ---: | ---: | ---: |
| ALGORITHM_DEVELOPMENT_HISTORY | 21 | 21 | 0 | 0 |
| BASELINE_OR_OTHER_UNRESOLVED | 1 | 1 | 0 | 0 |
| DF_H2 | 11 | 11 | 0 | 0 |
| DF_H246_HIGHER_PF | 3 | 3 | 0 | 0 |
| DF_H4 | 51 | 51 | 0 | 0 |
| DF_H46_HIGHER_PF_PR | 2 | 2 | 0 | 0 |
| DF_H46_PF_PR | 10 | 10 | 0 | 0 |
| DF_H5_H7_OPEN_SHELL | 4 | 4 | 0 | 0 |
| DF_H6 | 18 | 18 | 0 | 0 |
| DF_H8 | 36 | 36 | 0 | 0 |
| DF_NATIVE_TRACK_A | 62 | 62 | 0 | 0 |
| PAPER_EVIDENCE_PRESENTATION | 8 | 8 | 0 | 0 |
| PAULI_AND_REPRESENTATION | 11 | 11 | 0 | 0 |
| RESEARCH_C_C1_C4 | 6 | 6 | 0 | 0 |
| TRACK_B_ALGORITHM_HISTORY | 72 | 72 | 0 | 0 |
| TRACK_B_OTHER_HISTORY | 14 | 14 | 0 | 0 |
| 合計 | 330 | 330 | 0 | 0 |

正確な全IDの固定tip・実行前tip・受信結果・消失確認・KEEP・理由は[BRANCH330_REMOVAL_RESULTS.csv](BRANCH330_REMOVAL_RESULTS.csv)。単位別集計は[RESEARCH_UNIT_RESULTS.csv](RESEARCH_UNIT_RESULTS.csv)。同一HEAD候補1と祖先候補329を含む固定330集合だけであり、新規追加・置換は0。

## 安全確認と方法

両環境で330全行のNO_CURRENT_NAMED_REF_DEPENDENCY_IN_VERIFIED_SCOPEを取得し、candidate ID・完全SHA・KEEP対応を一致確認した。Aの歴史的PLAN/復旧名やlocal branch guard、Bの非B3限定結果と歴史的local branch identityは、remote名保持義務と区別した原判断を再利用した。B3保留7は330と交差しない。

最新LOCAL220登録に候補checkout0。旧control1515件はmetadata不変、新規/変更31資料も候補名の新規参照なし。open PR・protected branch・ruleset・active/queued/waiting/requested/pending CIは0で、main workflowの候補名selectorも0。KEEP472と保留7の存在・完全履歴包含、元ref/固定SHA/復旧対応を確認した。

無条件のdeleteは使っていない。330対すべてに正確なrefと固定完全SHAの明示leaseを付けた削除refspecだけを、shell展開のない引数配列で構築した。atomic dry-run成功後、実削除を1 transactionで実施し、exit0・porcelain330成功行・remote330消失を突き合わせた。期待SHA省略、無条件force、refspecの+、新しいSHAへの置換/再試行、KEEP/他branchへのcommit push混在は0。詳しくは[条件付き削除監査](CONDITIONAL_DELETE_AUDIT.md)。

## 保全とB3非干渉

全2,268監査commitと候補の全祖先は、削除後も明示KEEP集合から到達可能だった。Git rev-listによる未保持commit数0、固定SOURCE/result/索引369 commitの到達・object取得を確認し、独自commitを持つ88 branchを維持した。科学SOURCE/input/result/失敗・独立cohort・STOPの旧版を後続結果で置換しない。詳しくは[KEEP履歴監査](KEEP_REF_PRESERVATION_AUDIT.md)。

主checkoutのHEAD/branch/status/main refは不変。LOCAL220物理directory/登録、既存local ref記録496レコードを保持。既存非公開資料778 files・公開inventory137 filesのhash不変、archive6,412固有版と先行57件27,793 binding pathのmetadata不変を確認した。過去SHA/Git blob検証を再利用し、全payload再hash/再コピーをしていない。archive/bundle/復旧原本の削除・移動・上書きは0。同じdisk上の保全を独立backupと誤認しない。

B3はユーザー申告の実行中を前提に放置した。5保護worktree本体・sealed catalogue・非公開稼働資産・controller/worker/runtime/service/schedulerへアクセス・操作せず、状態・停止原因・科学的完了・再開条件を新たに判定していない。A/BへSSH・転送・削除指示を出していない。[B3と現行branch保持](B3_AND_ACTIVE_BRANCH_RETENTION.md)を参照。

## 成果物と終了

本directoryへ新規6資料だけを追加する。内部絶対path、内部host/IP、個人情報、非公開branch実名やprivate原本/復旧先、credentialを公開しない。完全な実名対応・期待SHA・復旧情報・コマンド/受信応答・前後snapshot・一回性journalは、新規の本人所有owner-only領域へ保存した。実行後に削除済みrefを再操作していない。

既存独立公開cloneで最新整理branchを取得し、並行履歴を保持する通常fast-forward commit/pushで公開する。削除transactionと公開pushを分離し、main merge・履歴上書きは0。公開commit・local/remote一致・GitHub6資料の完全読戻しは非公開PUBLICATION_RESULTと最終通知へ記録する。既存報告書を上書きしない。

今回の依頼に未処理の削除対象や追加作業はない。残存481本、B3保留7、local refs/worktree、科学研究への別作業は新しい具体的指示があるまで開始しない。

**削除後監査・匿名化公開・GitHub読み戻し完了後mandatory STOP。**

## 公開直前の並行追加

削除直後のremoteは481本だった。その後、公開準備中に削除対象外のbranchが1本追加され、公開直前snapshotは482本となった。実削除330・保留0・失敗0は変わらない。新しい1本を削除集合へ追加せず、削除前から残した追加2本とともに保持した。これは削除transaction中の並行追加0と区別する。最新値は公開後にも再観測して最終通知・PUBLICATION_RESULTへ記録する。
