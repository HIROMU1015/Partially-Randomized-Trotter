# GitHub remote branch全面監査

PUBLIC_REDACTED。2026-10-27 JST。environment=LOCAL。

**開始時808本、全graph取得後・公開準備直前809本。分類はACTIVE_OR_PROTECTED_KEEP 265、IMPORTANT_HISTORY_KEEP 207、REDUNDANT_DELETE_CANDIDATE_PENDING_APPROVAL 330、UNRESOLVED_EVIDENCE 7。削除許可とfresh条件が揃えば最大330本を減らし、479本まで整理できる可能性がある。今回の実削除0。**

## 実際の入力とsnapshot

正式指示research_project_instructions_unified_2026-10-25.mdは前の実行時に全文を読み、今回実ファイルの存在と同じSHAを確認し、§8.1/8.4/10を適用した。PROJECT_MAP.mdのentry pointと既存分類を使用した。承認証明書/receipt/新grant等は作っていない。

開始時remote観測は2026-10-27T13:45:28.759453+09:00。808本を正確に保存した。独立監査Gitへの取得中に1本追加され、809本・808固有tipになった。公開準備直前2026-10-27T13:58:42.970473+09:00のremoteは同じ809本で、取得済み集合からの追加/削除/tip変更は0。開始snapshotと最終snapshotをCSVで区別し、他Codexを凍結していない。監査branch自身は本報告を追加するcommitで前進するため、このCSVの監査branch tipは公開直前snapshot値であって、公開後HEADと同じだとは表示しない。

以前の801分類は全801名を再対応付けた。匿名化されていた2名は完全tipと一意のremote refを根拠に解決し、実名対応を非公開へ残した。現在は新規8名、監査branch tip更新1名を区別する。既存331/465分類は自動的に固定せず、以下へ更新した。

| 旧分類 | 新分類内訳 |
| --- | --- |
| IMPORTANT_HISTORY_KEEP 331 | {"ACTIVE_OR_PROTECTED_KEEP":156,"REDUNDANT_DELETE_CANDIDATE_PENDING_APPROVAL":15,"IMPORTANT_HISTORY_KEEP":159,"UNRESOLVED_EVIDENCE":1} |
| UNRESOLVED 465 | {"REDUNDANT_DELETE_CANDIDATE_PENDING_APPROVAL":314,"IMPORTANT_HISTORY_KEEP":44,"ACTIVE_OR_PROTECTED_KEEP":101,"UNRESOLVED_EVIDENCE":6} |

既存研究SOURCEの全文・raw・数値結果を801 branchごとに再読していない。SOURCE/結果の科学的正しさを新たに承認する監査でもない。

## Git解析と保全の根拠

全headsのcommit/parent graphを独立tree:0 fetchで取得し、2,268 commitを一度のtopological walkで解析。各commitへ到達する先端集合をbitsetで伝播し、同一HEAD・祖先・固有履歴を分離した。全branchペアのmerge-base反復は0。

| 指標 | 結果 |
| --- | ---: |
| 同一HEAD集合 | 1集合・2本 |
| 他tipの祖先である固有tip群 | 720 |
| 独自commitを持つremote branch | 88 |
| 固有tip群だけに属するcommitの変更path metadata確認 | 196 commit |
| 同一HEADの候補 | 1本 |
| 祖先関係の候補 | 329本 |
| 候補外KEEPから到達できない候補commit | 0 |

同一HEADの2名は共通protocol v2の同一完全履歴・既存中間roleが対応し、残存worktree参照0。公開protocol内の両branch名literalは0、公開H46 catalogueのcommon_v2は完全commitで参照していた。今回の計画ではcodex/pr-resource-protocol-v2-adopted-20261019を技術的保持anchorに選び、codex/h2-protocol-v2-update-20261019を候補にした。科学的採択を変更した判断ではない。異なるcohortが同一だと一般化せず、他の祖先群でも科学的同一性は主張しない。

全2,268 commitは明示KEEP集合から到達できる。43行のEVIDENCE_MATRIX.csvと既存synthesis・研究索引・契約catalogueのcommit宣言を再利用し、369の確認済みremote到達commitをKEEPへ対応付けた。公開H46 catalogue27項目は関連公開資料として取得したが、B3の非公開sealed catalogueそのものと同一視しない。40桁宣言にはblob/treeや型未確認値もあり、88値を新しい重要commitと誤認せず別記録へ隔離した。CEMの固定SOURCE/result commitはすべてremote graphで保持される。

名前を残す必要がある現行worktree・main/audit・既存代表role・重要履歴role・正確なSOURCE/result/claim anchorはKEEP。独自commitが他remoteから保持できない88 branchもすべてKEEPに含めた。過去の中間roleで全履歴を別KEEPから取得でき、具体的な独立ref役割が確認されなかった行だけを候補にした。候補は将来の削除承認済みではない。

## 3環境とlocal-only履歴

LOCALは既存60研究Git common directoryの実登録を読み、現在219件を確認した。先の214件から増えた並行追加を保持した。local branchは496 refレコードを読み、8 local-onlyレコード、現在remote graphから到達しないlocal tipは1レコード。後者は同名remoteがなく今回対象外で、非公開対応と重要履歴を保持した。LOCAL/A/Bのみのrefをremote削除候補にしていない。

Aの24件削除結果と新PLAN consumerを取り込み、公開snapshotの残存19件を照合した。Bは83＋86＋146＝315の正確な成功IDを旧379から差し引き、残存64 IDが最新保持報告の一覧と一致した。最新146件の完了を実際に取得し、210残存の古い値を使っていない。A/BをLOCALから現在live照会した値ではない。

全環境の既知残存表に結び付くremote265本はACTIVE_OR_PROTECTED_KEEP。これにはmain/auditとB3の既知public参照を含む。A/Bの未公開・匿名化local ref対応は不明箇所を伏せたまま保持し、科学的な推測で穴を埋めていない。

## 固定branch/ref roleの限定確認と未解決

既存LOCAL control索引のうち、残存checkoutのcontract/catalogue/sealed/recovery locatorと最近の変更locatorを対象にした。3092旧/追加locator中1515読取、268固有内容・68,070,513 bytesを照合した。1577は既に消えた履歴path等で今回読めず、読了や不存在証明には含めていない。新しいraw全探索や各branch SOURCE全文再監査は0。読めたscopeのbranch literalは245のprovenance/履歴参照で、抽出規則に該当する明示的な固定branch/ref義務は新規0。機械的literal判定だけで全契約不存在を証明したとはしない。

研究対応が不足した6名はtipの実変更pathと既存A/研究文脈で限定解消した。B3に関係する7名は[計画の正確な表](BRANCH_BULK_CLEANUP_PLAN.md)へ残した。必要なのはB担当による7 branch名のsealed catalogue/ref/非公開復旧索引との照合だけで、全809本再監査ではない。未取得のB3 catalogue全文を読了として扱わない。

B3保護5 IDはSERVER_B-WT-0003/0123/0124/0128/0130。public main、codex/h46-long-time-pf-pr-20261025、track-a-h46-pr-exploration-execution-20261021を保持し、0123/0128の非公開local-only対応も削除対象外。SOURCE/input/baseline/provider/raw/result/checkpoint/Git/ref/固定契約/STOP/復旧情報は保持する。B3の現在稼働・停止原因・科学完了・再開条件を新たに判定せず、process/runtime/service等の操作0。

## 候補の研究単位と減らせる範囲

| 研究整理単位 | 候補数 |
| --- | ---: |
| ALGORITHM_DEVELOPMENT_HISTORY | 21 |
| BASELINE_OR_OTHER_UNRESOLVED | 1 |
| DF_H2 | 11 |
| DF_H246_HIGHER_PF | 3 |
| DF_H4 | 51 |
| DF_H46_HIGHER_PF_PR | 2 |
| DF_H46_PF_PR | 10 |
| DF_H5_H7_OPEN_SHELL | 4 |
| DF_H6 | 18 |
| DF_H8 | 36 |
| DF_NATIVE_TRACK_A | 62 |
| PAPER_EVIDENCE_PRESENTATION | 8 |
| PAULI_AND_REPRESENTATION | 11 |
| RESEARCH_C_C1_C4 | 6 |
| TRACK_B_ALGORITHM_HISTORY | 72 |
| TRACK_B_OTHER_HISTORY | 14 |

330候補の全履歴が、候補外472 KEEP refの少なくとも一つから到達する。未解決7本も残すため、最大候補を後日実行した場合の残存見込みは479本。これは必要な科学履歴を88先端へ圧縮してよいという意味ではない。named role・現行workspace・固定SOURCEを保つため、graphだけの最小数より多く残している。

## 次の実行条件・復旧・限界

[一括計画](BRANCH_BULK_CLEANUP_PLAN.md)に正確な候補/KEEP集合、固定SHA、保持先、future usage/ref差分、共有STOP、復旧手順を記載した。無条件remote deleteと事前ls-remoteだけでは並行pushとの競合が残るため、未来の実行許可時にreceive側の期待SHA比較条件を確定する。force-with-leaseという名称の手段を現在のforce push禁止の下で勝手に実行していない。

Git object保持、branch名の意味、SOURCE/input/result、独立cohort/失敗/STOP、private原本/復旧、workspace/science再実行可能性は別条件。今回のgraph Gitは全payload backupではなく、Bの3bundleが全remote履歴を保全すると推測していない。復旧は候補外KEEPの持続的refから固定旧commitへ辿って元名を新規作成する案で、同名refを上書きしない。private raw/外部checkpointの全backupや科学的再実行を新規検証していない。

## 成果物と非干渉

- [ALL_REMOTE_BRANCHES.csv](ALL_REMOTE_BRANCHES.csv)：全remoteと初期/最終/旧801の差分。
- [BRANCH_CLASSIFICATION.csv](BRANCH_CLASSIFICATION.csv)：4分類・根拠・研究単位・current worktree/ref/残条件。
- [DUPLICATE_AND_ANCESTRY_GROUPS.csv](DUPLICATE_AND_ANCESTRY_GROUPS.csv)：同一HEADと祖先群・保持先。
- [IMPORTANT_HISTORY_RETENTION.csv](IMPORTANT_HISTORY_RETENTION.csv)：KEEP branchと固定SOURCE/index commitの保持対応。
- [DELETE_CANDIDATES.csv](DELETE_CANDIDATES.csv)：正確な候補・固定tip・候補外KEEP・復旧とfresh条件。
- [BRANCH_BULK_CLEANUP_PLAN.md](BRANCH_BULK_CLEANUP_PLAN.md)：後続承認時の一括計画。
- 本REPORT。

内部branch名/実path・raw/原本配置・host/IP・個人情報・credential・非公開復旧先は公開せず、正確な対応を本人専用新規領域へ保存する。既存原本/manifest/archive/bundleを上書きしない。科学SOURCE/input/result/STOPの変更、branch/worktree削除、merge/rebase/force push/main変更、gc/prune/clean/reset、A/B接続・変更、新科学計算/レビューは0。既存Git/CSV/匿名化/helperと801分類を再利用し、追加処理は全DAG到達性と限定role照合・最新表join・7資料出力に限った。

次の担当：ユーザーが必要な候補集合のremote削除を別途具体的に許可する。A/B担当は候補差分の現在利用・固定ref/private復旧条件と、Bの7名をread-onlyで確認する。その後のCodexが固定SHA比較とKEEP存続を直前検証して実行する。今回はここまで進めず、匿名化公開・GitHub読戻し後**mandatory STOP**。
