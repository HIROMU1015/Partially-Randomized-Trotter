# GitHub branch一括整理の実行前計画

PUBLIC_REDACTED。2026-10-27 JST。environment=LOCAL。Repository: HIROMU1015/Partially-Randomized-Trotter。

**実測809本から、条件付き整理候補330本を選定した。今回の実削除0。後日の具体的削除許可と直前差分確認が揃えば、最大330本を一つの承認集合で扱い、残存479本まで減らせる可能性がある。** 現時点の内訳はACTIVE_OR_PROTECTED_KEEP 265、IMPORTANT_HISTORY_KEEP 207、UNRESOLVED_EVIDENCE 7。未解決7本は削除集合へ含めない。

## 正確な候補・保持集合

正確な候補集合は[DELETE_CANDIDATES.csv](DELETE_CANDIDATES.csv)の全330行。branch IDと完全tip SHAを固定し、匿名aliasがある行は本人専用対応表で実際のrefs/heads名を解決する。連番範囲や類似branch名から増員しない。全remote集合は[ALL_REMOTE_BRANCHES.csv](ALL_REMOTE_BRANCHES.csv)、正確なKEEP集合は[BRANCH_CLASSIFICATION.csv](BRANCH_CLASSIFICATION.csv)のACTIVE_OR_PROTECTED_KEEP/IMPORTANT_HISTORY_KEEPの和集合472行である。UNRESOLVED_EVIDENCE 7行も今回除去しない。

各候補行に指定したKEEP refは候補集合と交差しない。すべての保持先tipは現在GitHubのheadsに存在し、全2,268 commitの親子graphで候補tipを包含する。候補tipだけでなく、その全祖先commitと各版のtree/blobへのGit参照が存続する。候補間の相互依存や、削除予定branchを別候補の唯一の保持先とする循環はない。固定SOURCE/result索引で確認した369 commitもKEEPから到達可能。[保持対応](IMPORTANT_HISTORY_RETENTION.csv)を参照。

これは科学SOURCE・入力・結果・失敗・STOPを後続内容で置き換える処理ではない。必要な旧commitで旧treeを取得する。異なるcohort・独立確認・失敗記録は元commit/相対pathに残り、科学的同一性を認定しない。

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

単位は既存研究索引・B側研究単位・必要なcommit変更path metadataによる運用分類であり、新たな科学的統合ではない。根拠が同じ行はまとめて承認・処理できる。数本ずつの小分け承認は不要。一方、各行の保持先・固定tip・副条件は個別に維持する。

## 削除直前の必須差分確認

1. ユーザーがこの正確な候補集合または部分集合の**GitHub remote branchのみ**の削除を具体的に許可する。今回許可されているのは監査・計画・公開までである。
2. 最新ls-remoteを読み、各候補tipをCSVの完全SHAと比較する。変更・削除済み・branch再作成・新規worktree利用・新しいSOURCE/ref契約があるIDは保留し、今回の旧判断を新tipへ流用しない。
3. 同じsnapshotで全KEEP refの存在・tipと候補の到達可能性を再確認する。KEEPが前進した場合は元保持tipの祖先包含を検証して記録する。KEEP消失/非包含への変更、共通object/保全異常があれば全体STOP。
4. LOCALの現在の全既知研究Git登録とlocal-only/未push refを読み取る。A/B担当には現在の利用・branch名固定・private復旧索引の**候補差分のみ**を確認してもらう。今回の公開残存表はA19/B64のsnapshotであり、未来の全環境不存在を証明しない。LOCALから無断SSHしない。
5. 固定SOURCE・grant/ledger/marker/STOP・科学契約が特定branch名を要求する場合は削除しない。routine worktree復旧用に記録された旧branch名は、保持commitから旧refを再作成して元手順を使えることを今回の対応表で示したが、具体的な永久保持指定やprivate復旧先が新たに判明した行は保留する。既存manifestや契約を書換えて条件を回避しない。
6. B3保護5 IDと既知public refを維持する。未解決7本は今回の削除集合外。B3状態・科学的完了・再開可否をこの監査から判定しない。runtime/controller/service/scheduler/科学SOURCEの操作はしない。

## 並行pushとの競合防止と一括実行案

未来の実行は専用の独立Gitで行い、候補ごとの固定tipを比較条件にしたGit receive側の条件付きref削除を用いる。単に事前ls-remoteと無条件deleteを続けるだけでは、その間の並行pushを失う競合が残る。候補を削除せずに期待SHA比較条件を表現できる標準Gitの `--force-with-lease=<ref>:<expected>` 等は名称にforceを含むため、現在のforce push禁止の下では実行しない。**後続指示で、履歴上書きを一切許さず固定SHA一致時の削除だけに用いる比較条件を明示許可するか、同等のreceive側CASを使える経路を指定する。** この技術条件を削除前に確定し、通常deleteだけで並行更新を安全に保護できるとは主張しない。

大量集合の原子的削除を利用できる場合はGit remoteのatomic対応を先に確認する。いずれの場合も候補以外のref更新、main変更、forceによる履歴上書き、local branch削除、merge/rebase/gc/prune/clean/resetは含めない。keep refsや他Codexの作業を凍結・巻き戻しせず、期待SHA不一致は対象限定で保留する。今回この比較条件付き削除コマンドやdry-runを実行していない。

## 削除後監査と復旧

- 各候補refの消失をID別に確認し、開始時との本数差だけを成功判定にしない。新branch追加・対象外tip前進は別記録にする。
- KEEP/未解決/B3/current refs、重要commit/親子graph、SOURCE/input/result/STOP locator、local refs、private archive/bundle/復旧索引の保持を確認する。研究ファイルの科学実行による検証は不要。
- 誤削除が判明した場合はその時点でSTOPし、保持refから対象の固定完全commitへ到達できること、同名remote refが存在しないことを確認する。別の具体的復旧指示の下で元refs/heads名をそのcommitへ新規作成し、その後に旧workspace復旧手順を適用する。存在する同名refを上書きせず、forceを使わない。
- SHA文字列だけを保存したrefとはしない。復旧元は候補外KEEPの持続的refで、固定旧commitを親graph経由で取得する。今回のtree:0監査Gitはcommit graph用で独立した全payload backupではない。B側3bundleの全remote包含も未検証で、候補保持の根拠には使用しない。

## 個別保留・共通STOP

個別tip更新、current worktree利用、特定ref契約、private復旧binding、比較条件不一致はその行だけ保留。KEEP消失/到達不能、remote認証先相違、共通Git/object破損、保全原本への影響、対象外ref更新、B3等の共通保護異常は全体STOP。retry・履歴上書き・固定SOURCE変更で迂回しない。

## 今回未解決の正確な7本

| branch | 完全tip | 現在のGit保持先 |
| --- | --- | --- |
| pf-common-domain-preparation-20261022 | 3728a618d7f09e67a070632ded1f27dab54b21a4 | pf-common-domain-execution-20261022 |
| track-a-codex2-h46-higher-pf-b-acquire-20261018 | a65e4f6e7bcc83f90a054f6aafc4ee2e0ca5d80a | pf-common-domain-execution-20261022 |
| track-a-codex2-h46-higher-pf-confirmation-readiness-20261018 | 87e6ead0c7b0ab77a7a0fbb420c517acc82ef6e1 | pf-common-domain-execution-20261022 |
| track-a-codex2-h46-higher-pf-integrated-20261018 | 1fc58aa3f5568887e0678a9be939551b0ed6523d | pf-common-domain-execution-20261022 |
| track-a-codex2-h46-higher-pf-readiness-20261018 | 175707427311e1e643e556942c7797cc4dd789d3 | pf-common-domain-execution-20261022 |
| track-a-codex2-h6-higher-pf-confirmation-native-readiness-20261018 | 6e197973d4d4183d4e27b1b8cc95cb02f311bb92 | pf-common-domain-execution-20261022 |
| track-a-h246-higher-pf-confirmation-readiness-20261021 | c7c1d8229508253390212588ae2b25e504eb4915 | pf-common-domain-execution-20261022 |

これらは既存B側研究単位がH46で、B3の公開SOURCE/providerに関係するtipの祖先である。Git履歴自体は保持可能だが、B3の現行sealed catalogueの正確なbranch/ref名指定と、SERVER_B-WT-0123/0128の非公開local-only復旧ref対応はLOCALへ未提供。B担当にこの7名だけをread-onlyで照合してもらう。73 path catalogueの全文を未取得なのに読了と扱わず、B3状態や科学内容を新たにレビューしない。一般UNKNOWNを全branchへ広げない。

**今回remote/local branch削除・worktree削除はすべて0。匿名化公開後mandatory STOP。**
