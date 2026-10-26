# ローカル68件：研究ファイル保全とcheckout整理条件

PUBLIC_REDACTED_ARCHIVE_68。固定入力 `950a866f4756e29c06a691c6b80e7b0e431777eb` の75件CSVからQC18、N1/N3 16、Research B34を正確に抽出した。図表7件、Research B LOCAL-WT-0283/0286は対象外。元保持分類・前回判断・研究契約・旧資料を変更せず、新判断を別fieldへ追加した。

**所有者による68件の将来workspace利用終了は確認済み。再質問は不要。** 現在の他Codex／外部jobの不参照とは区別する。実際のcheckout削除・登録解除、原本移動、branch削除は未承認・未実施である。

| 群 | worktree | HEAD版の保存パス | 保全scope検証済み・他環境待ち | 旧SOURCE未解決 |
| --- | ---: | ---: | ---: | ---: |
| QC | 18 | 18,216 | 18 | 0 |
| N1/N3 | 16 | 14,391 | 5 | 11 |
| Research B B2D Outer | 34 | 8,094 | 34 | 0 |

68件すべてで選定した現存研究ファイルを保存・検証した。57件はARCHIVED_VERIFIED_PENDING_DEPENDENCY。11件はLOCAL_FILES_VERIFIED_HISTORICAL_SOURCE_UNRESOLVEDで、保全の残条件を明示する。これは科学的な成果採否や恒久的な削除禁止判定ではない。全68件を現状維持する。

## 保存単位と残すもの

既存の本人所有・owner-only非公開資料領域に、新規68件用directoryを作成し、QC／N1_N3／Research_B、worktree ID、完全HEAD、元repository相対階層の順に保存した。前回7件の保管領域・manifestへ書き込んでいない。

各群の既存研究subtree、直近変更索引、SOURCE/input/control・preservation index、静的なlocal importと明示入力参照を再利用した。コード・runner・設定・tests、保存済みinput/result/raw、原稿・review/handoff、失敗・中断・独立cohort、元のgrant／ledger／marker／STOPを元bytesで保持した。科学module・runner・verifierは実行せず、数値fieldの科学的解析をしていない。過去の主checkout全体のhash保全一覧は記録として保存し、その全参照先を再帰コピーしなかった。

実コピー**8,284ファイル／511,618,500 bytes**。同一Git blob・modeは保存先内だけのhardlinkで重複を抑えた。追加link32,419、保存先の実パス計40,703。内訳は68 HEAD版40,701パスと、別途回収した旧固定SOURCE2ファイル。元worktreeへのhardlinkは0。同一bytesの物理保存を共用しても、版・SOURCE・cohort・失敗／STOPの対応は各manifest bindingと階層で別々に保持する。

全HEAD版対応でsourceコピー前後SHA-256・Git blob、全保存inodeのサイズ・mode・独立性を検証した。8,284実コピーのhashも再照合した。新規保存ファイルはownerのread-only modeとし、同名上書き0。保存先の所有権・ACL・容量・新規namespaceを事前確認した。これは同じローカルdisk上のworktreeから独立したファイル保全であり、別障害域backupや科学実行環境の移設ではない。

選定対象にLFS pointer／submodule／symlinkを通常ファイルとして複製した例は0。旧索引の案内symlink68記録は共有先の実体へ追従コピーせず、非公開依存資料へ保持した。requirements／pyproject等136環境設定記録は該当するtracked版を保存した。Git本体・venv・不要cache・外部private raw全体はコピーしない。動的import・絶対path・共有runtimeの可搬性は未保証である。

## 旧SOURCEの具体的な残条件

既存source/input記録のfile hash宣言2,466対応のうち2,401は保存済み版と一致、10対応は固定commit `d6c831da0b75ab48d1e54471e321ce6694bc0027` から回収した2ファイルへ一致した。元のNOT_AUTHORIZED記録をそのまま保持し、実行許可や契約を変更していない。

残る55対応は下記3ファイルの旧開発版hash。`development_before_log_guard`／`development_before_unresolved_schema_guard` の既存synthetic_source_validationが参照する版である。現行版のコピーで代用せず、関連する記録・ledger・失敗履歴は保存した。保持Gitの対象path履歴と既存の関連exportを限定照合したが、期待bytesを特定できなかった。現在のlocal disk全体や他hostに不存在とは認定しない。再構築・再実行で埋めない。

| gap ID | 公開済みsource相対path | 旧記録の期待SHA-256 |
| --- | --- | --- |
| LOCAL-68-HISTORY-GAP-001 | `src/trottertracks/representation_exploration/n3_occupation_transfer_v1.py` | `6799702fddd0c54784082d0f5648ff1ad6b8b6faa084dee03d41487850db9b28` |
| LOCAL-68-HISTORY-GAP-002 | `scripts/run_n3_occupation_transfer_v1.py` | `d43628daa116c12517e95786820885c1cd144988a10f0bca72e2be848e127795` |
| LOCAL-68-HISTORY-GAP-003 | `tests/test_n3_occupation_transfer_v1.py` | `6533f3fb82e79f8409a87ef2bbf01e9617e310150603890762f536a60da7b2bc` |

影響11 ID：LOCAL-WT-0005、LOCAL-WT-0009、LOCAL-WT-0010、LOCAL-WT-0011、LOCAL-WT-0012、LOCAL-WT-0013、LOCAL-WT-0014、LOCAL-WT-0015、LOCAL-WT-0016、LOCAL-WT-0018、LOCAL-WT-0019。

**該当研究／環境担当**は、既存の旧SOURCE保管場所・履歴があれば上記公開path／hash／manifestで照合する。見つからない場合、残存履歴での保全scopeの扱いを別に具体化する。これは利用終了方針の再確認ではない。11件の完全保全やcheckout整理可を今回確定しない。

## 研究成果とGit保持

QC18はstrategy handoff `fb635213d99e666852b87212c051ab08a6977c41` に対して17祖先＋1同一。N1/N3 16はclosure `ccc575ee339be091fd4031272c38bee13866f097` に14祖先＋1同一＋1非祖先（LOCAL-WT-0003）。Research B34は最終成果／pause handoff `e87375fa9a998dce80792be0f15597c747f00d83` の祖先。これらは索引入口であり、別SOURCE・raw・独立cohort・失敗の包含を意味しない。

将来checkoutだけを解除する場合も、各自身の保持branch・完全HEAD、固定SOURCE refs、共通Git、今回の版別保存ファイル・完全manifest・原本・STOP／grant消費／retry条件、外部custodyを残す。代表branchだけを残して他branchや科学SOURCEを削除する計画は作成しない。68 HEADは現在GitHubの同名tipと一致した。

## 他環境照合と後続順序

1. **サーバーA担当**は並行確認の結果を対象ID・branch・SHA・固定SOURCE参照へ対応付ける。ローカル保全はその回答を待たず完了した。全68件の他環境実参照は今回未確認のまま記録した。
2. **サーバーB担当**はDF-native B3に干渉しない時点・方法で実参照を確認する。ローカルCodexはA/BへSSHせず、SOURCE・raw・環境・processを操作しない。B3とResearch B／旧Track Bの名称を混同しない。
3. **研究／環境担当**は旧SOURCE gap11件、外部private原本・archive・絶対path・共有実行契約の保管責任を必要な範囲で照合する。非公開実パスが必要な場合、既存の安全な非公開経路の確認事項として残し、Publicへ書かない。
4. **ローカルCodex（次の照合指示後）**は、57件をまず条件付き候補として、A/Bの実参照、private custody、固定path契約、保存資料からの参照性を突き合わせる。11件は旧SOURCE残条件も閉じる。所有者の将来workspace利用終了は全68件で既に確定している。
5. **ユーザー**が具体的ID／pathのcheckout登録解除・directory削除を別途指示した後、**該当Codex**が直前のHEAD・status・untracked／ignored・使用状況・SOURCE参照・保持ref・保全hashの差分を確認する。現在の確認を将来の直前gate完了と扱わない。原本移動・branch削除・科学SOURCE変更へ拡張しない。

**mandatory STOP。全68件の既存checkoutを維持し、実際のworktree削除・登録解除・prune・branch整理・merge・科学実行・原本移動を行わない。**
