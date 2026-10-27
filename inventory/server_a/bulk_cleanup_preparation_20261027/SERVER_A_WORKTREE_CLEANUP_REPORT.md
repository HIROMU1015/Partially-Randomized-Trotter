# SERVER_A worktree整理準備・分類報告

2026-10-27 JST。ユーザー承認範囲は調査・必要な非破壊保全・分類・計画・匿名化公開までであり、削除・登録解除・branch削除・merge・原本移動・科学計算は行っていない。正式共通指示2026-10-25版（§8.1・§8.4・§10）を実在ファイルで確認し、既存helper・索引・copy/hash処理を再利用した。承認証明書・receipt・grantは新設していない。

## 登録数とscope

初期36件と同じ2 Git系統は現在41登録（共有研究Git側40＋独立旧Track B clone側1）で、初期36件の削除は0。追加5件は研究4件と旧棚卸し用linked checkout1件。さらに独立した現行公開clone1件を管理scopeへ含めるため、この報告・CSVは42行、内訳は研究40件・管理2件である。管理用cloneを研究worktreeの増加と混同しない。管理用0042のHEADは調査時点の観測値で、公開作業では既存履歴から今回の公開commitへ進むため、科学対象のHEAD差分とは分離する。初期匿名ID36件を保持し、追加分には実path対応から新IDを割り当てた。IDの番号から物理pathを推測していない。

固定参照は `b71c6e9cc074ae51aeeab986600d1b1d6354cbbb`、A12追跡 `4217e2f43ba363c9687a390657128bd5716d0074`、AによるLOCAL139照合 `8da04de4aae1ccf9c0e0327411ce9081cf78f491`。公開branchはmainと独立したorphan履歴として維持する。LOCALの削除実績・利用終了判断はAへ転用していない。

## 新分類（元分類は別fieldに保持）

| 新状態 | 研究worktree | 管理用 | 合計 |
| --- | ---: | ---: | ---: |
| BULK_APPROVAL_CANDIDATE_PENDING_OWNER | 24 | 1 | 25 |
| PENDING_OTHER_ENVIRONMENT | 1 | 0 | 1 |
| PRESERVATION_ACTION_REQUIRED | 0 | 0 | 0 |
| KEEP_REQUIRED | 15 | 1 | 16 |
| UNRESOLVED | 0 | 0 | 0 |
| 合計 | 40 | 2 | 42 |

技術的な候補は研究24件で、すべてSERVER_Aの利用終了判断待ち。旧研究20件を先行の一括候補とし、新しいH46研究4件は別の研究単位判断へ残す。管理用旧棚卸しcheckout1件は任意の別枠で、通常の研究候補へ混ぜない。削除実行を認定した件数は0。

UNRESOLVED 0は未記録の依存や将来利用が全世界で不存在と証明した意味ではない。確認済みのA技術scopeと具体的な残条件を分け、既知の不足は対象限定PENDING_OTHER_ENVIRONMENT、固定pathとGit/venv/custody rootはKEEP_REQUIREDとした。候補を再び一般的UNKNOWNへ戻さず、利用終了・旧proposal/履歴役割の終了を問う形で保持した。未知namespace/未記録job・他環境の未申告参照はscope限界として全体に残し、新たな具体的根拠が出たIDだけ再評価する。

## 既存12件の更新

従来KEEP_REQUIRED3件（0026・0029・0031）は固定SOURCE/runner/plan/STOPとの関係を引き継ぎ、保持を継続する。従来UNRESOLVED9件は、HEAD不変・tracked clean・重要untracked/新ignoredなし・保持ref・本人process未検出と、既存custody/参照の文脈を確認し、BULK_APPROVAL_CANDIDATE_PENDING_OWNERへ更新した。新たな科学SOURCE実行や契約改訂はしていない。

- 0034：旧一時handoff copy。0030の永続handoff、同一HEAD保持ref、main側bundle・原本保管を残せる。旧copyを履歴として残す役割の終了は所有者判断。単なる同一SHAだけでraw保全を認定していない。
- 0010：外部原本は保持main側custodyにあり、候補の物理削除で原本を移動・削除しない。2既存inventoryの範囲でも候補pathに束縛は検出せず。preparation用途の終了を所有者へ残す。
- 0021/0022：自己test・失敗tracebackの古い絶対path記録。別worktree内のmirrorにも同じbytesの歴史記載があるが、新たな現行科学job依存ではない。旧test用途終了とSOURCE/失敗/STOP/ref保持を条件にする。
- 0023/0024/0025/0027：旧draft/proposal/schemaの将来commandとSOURCE rootを区別した。0023→0025等のproposal関係は、その提案をもう使用しないという所有者判断で一体に終了させる。実行済み固定契約の解除とは区別し、元文書は書換えない。
- 0028：既知incoming実行pathの検出なし。SOURCE checkoutの将来利用終了を所有者へ残す。

対象9 IDは0010・0021・0022・0023・0024・0025・0027・0028・0034。候補に格上げしても実行直前preflightを完了したことにはならず、今回の研究停止・利用終了の承認でもない。

## 全worktreeの保持と保全

WP11（0002）の専用venv interpreterが新しいH46計画で参照されていることを追加確認した。科学ignored資産をコピーしてもvenvや実行契約を移設したことにはならず、0002はKEEP_REQUIRED。main 0001、独立旧Track B clone0036、現行公開clone0042はcommon Gitが内部にあるrootで、通常のlinked worktree削除対象ではない。

0009・0012・0013は既存control/input/resultの物理provider、0016は保全索引の入口、0030は永続handoffとして保持。0020・0026・0029・0031・0032・0033・0035は固定実行path/SOURCE/STOP関係を保持し、ownerの単なる用途終了だけで契約を解除しない。

0015は既知のH2他環境provider/runtime/custody文脈が未解決のためPENDING_OTHER_ENVIRONMENT。旧G6/B3名と現在のSERVER_B DF-native B3を同一視していない。0036は独立repository common rootを守る理由でも保持し、他環境の未確定条件も副fieldに残した。

停止済みのignored科学資産232ファイルと、新しいparameter分析の未追跡5ファイルを、本人所有の非公開領域へ元相対階層を保ってexclusive copyした。合計237ファイル・130,363,450 bytes。元のsize/mtime/inode安定性、sourceコピー前後と保存bytesのSHA-256一致、元へのhardlink0を確認した。コピー先はowner-onlyかつfile read-only。sourceの移動・編集・削除、科学payloadの読み込みによる数値処理はしていない。受入れ時点の未追跡analysis5件を保存した後に、0041の3件のbytes更新とHEAD進行を観測した。新commitの対象subtreeを確認し、現存5件すべてが通常Git blobとして記録され、実ファイルbytesと完全一致し、実在refで保持されることを追加検証した。固有の未追跡資産は残っていないため保全gateを閉じた。古いcopyを最新と呼ばず、0041の最新復旧元は新commitと保持Git blobとする。同じdiskの独立directoryであり、別障害域backupではない。

この保全により候補側の固有ignored入力/ログや未追跡sourceはsnapshotとして保存された。原本へ新しいファイルや変更が追加された場合、将来preflightで差分保留にする。稼働venv・全cacheのコピー、main原本の移設、大規模raw再探索は行わない。追加の大容量コピーを今回の必須gateとして新設しない。

初期SOURCE catalogue22宣言に含まれる13固有commitが、現在も既存Git objectとして存在しrefから保持されることを確認した。全42 snapshot HEADにも保持refがある。0041は並行作業によるHEAD進行を記録して最新の観測値へ更新し、旧研究候補20件のHEAD/statusは不変だった。未公開と記録されたHEADや科学SOURCEも、既存local refsと共有Git objectsを保持する。commit SHAを文書に書いただけの保管ではなく、実在refを残す。branch削除・gc・共通Git削除は計画に含めない。

## 調査scopeと限界

既存の12追跡、custody、SOURCE catalogue、path候補、Git/CSV/hash helperを再利用した。現在のHEAD/status/ignoredと保持ref・本人processを差分確認し、既存metadata代表と新worktreeの関連metadataだけをincoming参照で静的照合した。8,065 metadataを読み、raw本文・全Git履歴を再解析していない。mirror記載件数を独立contract数とせず、test/traceback、旧proposal、provenance locator、実際の固定実行pathを文脈で分けた。

本人processのpath非検出は未来の再開・別Codex利用終了ではない。全環境の動的import・未記録mount/alias・別所有者processを網羅した保証はない。他hostへ接続せず、LOCALで進む整理、SERVER_B DF-native B3、他job・共有環境へ介入していない。

原本と非公開索引は保持し、内部絶対path・host/IP・ユーザー名・private原本の配置・非公開ファイル名・process識別子・秘密情報は公開しない。公開branch名に含まれる内部host部分も匿名化した。詳細な実path・復旧先・元branch/ref・copy SHAは非公開版にある。

## 次の判断と停止

まずQ1/Q2の旧研究20件について、SERVER_Aで専用workspaceの将来利用・再開・配布/旧proposal・履歴保持の役割を終了してよいか、研究単位でユーザーが判断する。新しいH46群4件はQ3として別判断とし、継続するなら保つ。旧棚卸しlinked checkout1件はQ4の任意管理項目。0015はQ5の対象限定確認、固定保持群はQ6〜Q9として後続条件を具体化した。

通常の削除計画とpreflight・異常時STOP・復旧確認は[一括削除計画](SERVER_A_BULK_CLEANUP_PLAN.md)、ID別判断は[CSV](SERVER_A_WORKTREE_DECISIONS.csv)、質問は[一覧](SERVER_A_OPEN_QUESTIONS.csv)、保全とGit保持は[監査](SERVER_A_PRESERVATION_SOURCE_GIT_AUDIT.md)を参照。公開は新規directoryのこの5資料だけ。最新remoteを親に既存成果物を保持し、非forceのfast-forward更新とGitHub読み戻し・local/remote一致を確認する。

削除・登録解除・branch削除・merge・研究データ移動・SOURCE/raw/grant/ledger/marker/STOP変更・科学計算は実施しない。完了後mandatory STOP。
