# 固定SOURCE・公開証拠

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

|役割|固定commit|取得・検証|
|---|---|---|
|直接基点・既存独立consumer|`6bffb5b5819918698b522270ff19b255766d3072`|直接worktree作成。merge/rebaseなし|
|API候補公開|`82896d53fa97235fa848d6c0833eb193af8b99d6`|新packet・契約・SOURCE・rawをGit blobから保存|
|API人工測定|`78c571005fe766c61b32dc867e006531099ff608`|62 source pins全件が測定Git blob・公開HEAD・宣言hashと一致|
|科学参照|`76ca3c4089b201d55da0fede08e1f77733ac38a5`|8 moduleの独立AST、科学6 moduleはimport禁止|
|Global Bundle公開/測定|`d24cccbecdf0928fe977080dd7b28c591df0089e` / `e099dcc461247e81eff981867dc28c3f3b16df5e`|基点内の固定copyとGit/公開bytesを再照合|
|Storage V2公開|`596781daeba253418ba35f4b7964312a54caf233`|基点内の固定copyとGit/公開bytesを再照合|
|共通正式指示|`7eff8a33bef4ee1f29698ab10c6da46988dc8db5`|全文取得、§8.4適用|

[固定Git inventory](fixed_git_inventory.json)278件と[基点内参照](previous_fixed_reference_verification.json)501件を合わせ、[今回の公開bytes照合](fixed_public_byte_verification.json)779件。サーバーの「127件」報告を転記した件数ではない。基点の全history/publication receipt自身が参照する全URLを再取得したとの主張はしない。

`source_inventory.json`の62 SOURCE pins、10 scientific input pinsを照合。数値内容を再解析していない。科学入力・precisionの宣言を保持し、AST inventoryのdigestは公開文書のcanonical JSONに結び付く。独立ASTの追加fieldを含む別JSONは同digestとはしない。

測定SOURCEとpost-measurement helperは[測定照合表](measured_source_verification.json)で区別。`package_evidence.py`、`build_packet.py`、公開/復元管理helper等を測定時runtimeとして数えない。

raw：SHA256 `884a442b394fc16888eb6662c236b0788da2449f19c1a7f6765e2734e5650925`、12 parts、46,334,109 bytes、5,912 files。part/archive/memberを全件照合し、専用ディレクトリへ復元。公開43 Bundle・35差分、port4・22 mutation等を含む。二回目の専用復元と最終helperでの104件offline確認を保存する。

[環境](environment.json)：32 CPU、Python3.10.12、64GiB RAM級。実際の空きCPU/RAM/I/Oを確認し、hash/I/Oは32 threads、独立caseは最大32 processes、testは独立class/suite単位で並列。4portは4件だけなので4process。共有設定・package・他process変更なし。サーバー報告64 CPU/Python3.12.3との差を記録。別物理hostであることは `NOT_ESTABLISHED`。

[Runtime照合](runtime_identity_verification.json)でローカルexecutable/stdlib/extension bytesを保存。サーバーruntimeの宣言hashは保持するが、このhostでその全bytesを取得した証拠はない。`historical_runtime_bytes_proven=false`。保存人工結果を受け入れることとサーバー環境の独立再構築を区別する。

保全補足：親HEAD/indexは不変。親の進捗資料3件（`docs/progress/README.md`、Track A参照メモ、詳細スライド構成案）は開始時hashとの差を検知した。今回の書込対象外であり、内容を戻さず[保全記録](preservation_verification.json)にbefore/observed hashを保存した。原因・編集者は未確定。新worktree以外の親Git statusは開始時と同じ。固定SOURCE/raw/claim/marker/STOPの改変は0。
