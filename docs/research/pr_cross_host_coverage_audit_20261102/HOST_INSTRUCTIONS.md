# Phase 1・環境別の独立実行指示

**Phase 0では以下の3本を実行しない。ユーザーが対象環境へ別途実施指示を出したときだけ使用する。**
各ブロックは一つだけを、その環境のCodexへコピーして渡せる。
`PROTOCOL_COMMIT`は共通仕様・空CSVを先にcommitした実在SHAへ固定済み。科学実行SOURCEとは別である。
本書はPhase 0最終公開commitの版を使用する。仕様固定commitには仕様と空CSVを収録し、
本書を追加した最終公開commitでもその2ファイルのblobが一致することを確認する。
環境ごとの実root・remote・アクセス範囲を確認するまで、絶対pathやhost名を推測しない。

## LOCAL用（Phase 1、未起動）

```text
あなたはLOCALのCodexです。HIROMU1015/Partially-Randomized-Trotterの
アルゴリズム改善研究を除く既存PR検証について、Phase 1のLOCAL棚卸しだけを行ってください。
このブロックの保存・閲覧だけは開始許可ではなく、ユーザーからのLOCAL Phase 1実施指示が前提です。

固定共通仕様：
PROTOCOL_COMMIT=968218c2b0c6280af6c1c02822603e3312aa410c
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/PROTOCOL.md
同commitのINVENTORY_TEMPLATE.csvを使用し、schema_version=PR-COVERAGE-1、environment=LOCALとする。
実際に固定commitを取得して本文・ヘッダを読んでから着手する。branch tipを代わりに使わない。
必須仕様を取得できない、指定版と異なる場合は着手せず、具体的な取得不能を報告してSTOPする。
各worktreeのAGENTS.mdと、親projectsのAGENTS.md、正式共通運用指示
research_project_instructions_unified_2026-10-25.md（特に§8.1/§8.4/§10）も実際に確認する。
規約を未取得のまま読了と称さない。sub-agentを使わずprimary agentだけで作業する。

読取範囲：LOCALでユーザー所有・対象repoとの対応・アクセス許可を確認した研究rootだけ。
SOURCE、result、archive、Git履歴、保全済み証拠をread-onlyで調べる。
SERVER 1/2へSSH・転送・調査を行わない。他ユーザーのファイル/job/processを調べない。
GitHub上の共通資料の取得はよいが、それを他hostの現存確認と呼ばない。
旧SERVER_A/B等とSERVER 1/2の対応は推測せず、UNKNOWNまたはユーザー指定aliasとして記録する。

1. 担当root、観測時点、アクセス制限を出力REPORTへ記す。証拠treeではGIT_OPTIONAL_LOCKS=0、
   暗黙fetchなしのgit show/ls-tree/cat-file等を使う。新しい独立出力branch/worktreeを作り、原本を変更しない。
2. 共通仕様§3の固定研究地図9a0bc7f、Track A索引4591b8d、70カード/SOURCE索引、
   既存環境inventory fa55c162を再利用する。該当版のfull SHAはPROTOCOLに従う。
   LOCALの保全・branch整理記録も既知の所在があれば入口にする。
   server2_repository_inventory_20261102.tsv等はLOCALに保存された所在metadataとしてのみ扱い、
   それでSERVER 2の実験を棚卸ししたとはしない。現存directory一覧を研究全体とみなさない。
3. 既存索引から担当環境の関連証拠へ限定的に辿る。固定Git object・保全archive索引を優先する。
   削除済みbranch/worktreeは既存保全記録・carrier・固定commitで追跡し、cleanupを再実行しない。
   大規模な全域検索、home/projects/tmp横断走査、raw全decode/全hash、archive全展開は行わない。
4. PROTOCOL§4～7の43列を使い、科学条件×run/cohort×保存解析版を記録する。
   A～E（DF、PF/finite RTE、回路/資源、RPE/QPE、Pauli比較）を含め、改善方式自体は除外する。
   改善系列内の既存PR評価・baselineが直接必要ならSUPPORT_ONLYとし、除外理由と関係を残す。
5. REPORT/JSON/CSV/raw/QPY/STOP等を同じevidence_groupへrole付きで接続する。
   原SOURCE、input/state/reference、law/normalization、compiler/測定境界、seed/cohort、
   継続/回復、結果identityを原記録で照合する。コピー/引用/同seed再実行を独立確認と数えない。
   探索と確認、価格標本とsignal、paired axis、複数精度での共有を明示する。
   関係は原記録に基づく候補と根拠だけ。3環境の統合・canonical確定はPhase 2へ残す。
6. 科学判定、技術PASS、実行完了、正式受理、旧結果の失効を別列にする。
   実行前STOPと実行中失敗、未実施と証拠未発見、局所経験値と厳密保証、実価格とproxyを混同しない。
   不明値はUNKNOWN/NOT_RECORDED/NOT_ACCESSIBLE/NOT_FOUND等と限定探索範囲を記録する。
   未発見を未実施と断定せず、原nullを0へ変えない。元の判定と出典を保持し、新しい科学的PASSを作らない。

成果物：新規branch codex/pr-coverage-inventory-local-<作成日と必要な一意suffix> に、
docs/research/pr_cross_host_coverage_audit_20261102/inventory/LOCAL/INVENTORY.csv と REPORT.md。
長い条件・参照だけ必要ならDETAILS.jsonlを使い、PROTOCOLのdetail_ref規則で解決できるようにする。
REPORTにはprotocol固定SHA、読取rootの公開用ID、時点・未アクセス、旧索引の項目ごとの
記録済み/範囲外/未発見/未アクセス/未解決の対応、欠測・alias・関係候補、検査、保全、STOPを記す。
架空の実験行・数値を作らない。条件が特定できない参照はREPORTの未解決一覧へ残す。

保存値のliteral引用と必要最小metadata/hash照合だけを行う。
新科学計算、保存数値の新集計・N/G再会計・fit・再ランキング、sampling/RNG replay、
simulation、Hamiltonian/state/reference生成、回路build/compileを行わない。
coverageの科学的評価、不足A～Eの新分類・優先順位、詳細科学レビュー、主要claim採択を行わない。
既存コードを静的に確認し、適合する読取/転記/参照検査を再利用する。科学runnerをimport/実行しない。
CPU/RAM/workerに恣意的上限を置かず、許可されたmetadata処理を必要に応じ高速化する。
共有設定/OS/package/scheduler/権限、他job/processを変更・停止しない。
原SOURCE/input/raw/result/grant/ledger/marker/STOP、archive、既存研究map・概要・manifestを変更しない。
証拠treeのcheckout/reset/merge/clean/gc/pruneやbranch/worktree/archive削除は禁止する。

公開：成果物の公開コピーだけを独立出力treeで作る。内部host/IP/絶対サーバーパス、資格情報、
private rawは除き不透明IDを使う。必要な非公開の所在対応は環境内で保全してcommitしない。
ヘッダ一致、JSON構文、ID/参照、原引用、欠測、差分範囲と秘密の非混入を検査する。
確認済みの対象repo remoteに通常commit/pushし、remote SHAと固定commitの公開CSV/REPORT/補助本文を照合する。
force pushや既存研究branchの更新は禁止。push失敗はその旨、理由、確認済みcwd/remote/branchを
含むコピー可能な再実行pushコマンドを最終チャットへ示す。未公開を公開済みと報告しない。

終了条件：許可したLOCAL rootと既存索引の参照を状態付きで対応付け、未解決・未アクセスを残し、
形式/参照検査と公開照合を完了する。見えない領域を含めた完全網羅性は主張しない。
アクセス不能項目は記録して他の許可済み読取を継続できる。原本保全違反/秘密公開/他者干渉を避けられない場合、
必須仕様未取得や科学再実行なしに解決できない処理は該当部分をSTOPし、未解決として報告する。
mandatory STOP：公開・照合後はLOCAL Phase 1を終了する。SERVER 1/2、Phase 2/3、科学計算へ自動移行しない。
最終報告はGitHub URL、固定SHAとremote照合、担当範囲/限界、保全、未実施処理、STOP、次の担当を示す。
ユーザーの明示実施指示が許可であり、承認証明・原文保存・grant発行等を新たに要求しない。
```

## SERVER 1用（Phase 1、未起動）

```text
あなたはSERVER 1のCodexです。HIROMU1015/Partially-Randomized-Trotterの
アルゴリズム改善研究を除く既存PR検証について、Phase 1のSERVER 1棚卸しだけを行ってください。
このブロックの保存・閲覧だけは開始許可ではなく、ユーザーからのSERVER 1 Phase 1実施指示が前提です。

固定共通仕様：
PROTOCOL_COMMIT=968218c2b0c6280af6c1c02822603e3312aa410c
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/PROTOCOL.md
同commitのINVENTORY_TEMPLATE.csvを使用し、schema_version=PR-COVERAGE-1、environment=SERVER_1とする。
実際に固定commitを取得して本文・ヘッダを読んでから着手する。branch tipを代わりに使わない。
必須仕様を取得できない、指定版と異なる場合は着手せず、具体的な取得不能を報告してSTOPする。
各worktreeのAGENTS.mdと、適用される親projectsのAGENTS.md、正式共通運用指示
research_project_instructions_unified_2026-10-25.md（特に§8.1/§8.4/§10）も実際に確認する。
規約のパスをLOCALから推定せず、この環境の既知文書またはユーザー提供資料で確認する。
未取得の必須規約を読了と称さない。sub-agentを使わずprimary agentだけで作業する。

読取範囲：SERVER 1でユーザー所有・対象repoとの対応・アクセス許可を確認した研究rootだけ。
SOURCE、result、archive、Git履歴、保全済み証拠をread-onlyで調べる。
LOCALやSERVER 2へSSH・転送・調査を行わない。他ユーザーのファイル/job/processを調べない。
GitHub上の共通資料の取得はよいが、それを他hostの現存確認と呼ばない。
旧SERVER_A/B等とSERVER 1/2の対応は推測せず、UNKNOWNまたはユーザー指定aliasとして記録する。

1. 担当root、観測時点、アクセス制限を出力REPORTへ記す。証拠treeではGIT_OPTIONAL_LOCKS=0、
   暗黙fetchなしのgit show/ls-tree/cat-file等を使う。新しい独立出力branch/worktreeを作り、原本を変更しない。
2. 共通仕様§3の固定研究地図9a0bc7f、Track A索引4591b8d、70カード/SOURCE索引、
   既存環境inventory fa55c162を再利用する。該当版のfull SHAはPROTOCOLに従う。
   旧SERVER_A等のinventoryは所在を探す候補であり、今回のSERVER 1との一致を未確認のまま固定しない。
   この環境で確認できる既存catalog/export/保全・branch整理記録を優先し、現存directoryだけを研究全体とみなさない。
3. 既存索引から担当環境の関連証拠へ限定的に辿る。固定Git object・保全archive索引を優先する。
   削除済みbranch/worktreeは既存保全記録・carrier・固定commitで追跡し、cleanupを再実行しない。
   大規模な全域検索、home/projects/tmp横断走査、raw全decode/全hash、archive全展開は行わない。
4. PROTOCOL§4～7の43列を使い、科学条件×run/cohort×保存解析版を記録する。
   A～E（DF、PF/finite RTE、回路/資源、RPE/QPE、Pauli比較）を含め、改善方式自体は除外する。
   改善系列内の既存PR評価・baselineが直接必要ならSUPPORT_ONLYとし、除外理由と関係を残す。
5. REPORT/JSON/CSV/raw/QPY/STOP等を同じevidence_groupへrole付きで接続する。
   原SOURCE、input/state/reference、law/normalization、compiler/測定境界、seed/cohort、
   継続/回復、結果identityを原記録で照合する。コピー/引用/同seed再実行を独立確認と数えない。
   探索と確認、価格標本とsignal、paired axis、複数精度での共有を明示する。
   関係は原記録に基づく候補と根拠だけ。3環境の統合・canonical確定はPhase 2へ残す。
6. 科学判定、技術PASS、実行完了、正式受理、旧結果の失効を別列にする。
   実行前STOPと実行中失敗、未実施と証拠未発見、局所経験値と厳密保証、実価格とproxyを混同しない。
   不明値はUNKNOWN/NOT_RECORDED/NOT_ACCESSIBLE/NOT_FOUND等と限定探索範囲を記録する。
   未発見を未実施と断定せず、原nullを0へ変えない。元の判定と出典を保持し、新しい科学的PASSを作らない。

成果物：新規branch codex/pr-coverage-inventory-server1-<作成日と必要な一意suffix> に、
docs/research/pr_cross_host_coverage_audit_20261102/inventory/SERVER_1/INVENTORY.csv と REPORT.md。
長い条件・参照だけ必要ならDETAILS.jsonlを使い、PROTOCOLのdetail_ref規則で解決できるようにする。
REPORTにはprotocol固定SHA、読取rootの公開用ID、時点・未アクセス、旧索引の項目ごとの
記録済み/範囲外/未発見/未アクセス/未解決の対応、欠測・alias・関係候補、検査、保全、STOPを記す。
架空の実験行・数値を作らない。条件が特定できない参照はREPORTの未解決一覧へ残す。

保存値のliteral引用と必要最小metadata/hash照合だけを行う。
新科学計算、保存数値の新集計・N/G再会計・fit・再ランキング、sampling/RNG replay、
simulation、Hamiltonian/state/reference生成、回路build/compileを行わない。
coverageの科学的評価、不足A～Eの新分類・優先順位、詳細科学レビュー、主要claim採択を行わない。
既存コードを静的に確認し、適合する読取/転記/参照検査を再利用する。科学runnerをimport/実行しない。
CPU/RAM/workerに恣意的上限を置かず、許可されたmetadata処理を必要に応じ高速化する。
共有設定/OS/package/scheduler/権限、他job/processを変更・停止しない。
原SOURCE/input/raw/result/grant/ledger/marker/STOP、archive、既存研究map・概要・manifestを変更しない。
証拠treeのcheckout/reset/merge/clean/gc/pruneやbranch/worktree/archive削除は禁止する。

公開：成果物の公開コピーだけを独立出力treeで作る。内部host/IP/絶対サーバーパス、資格情報、
private rawは除き不透明IDを使う。必要な非公開の所在対応は環境内で保全してcommitしない。
ヘッダ一致、JSON構文、ID/参照、原引用、欠測、差分範囲と秘密の非混入を検査する。
確認済みの対象repo remoteに通常commit/pushし、remote SHAと固定commitの公開CSV/REPORT/補助本文を照合する。
force pushや既存研究branchの更新は禁止。push失敗はその旨、理由、確認済みcwd/remote/branchを
含むコピー可能な再実行pushコマンドを最終チャットへ示す。未公開を公開済みと報告しない。

終了条件：許可したSERVER 1 rootと既存索引の参照を状態付きで対応付け、未解決・未アクセスを残し、
形式/参照検査と公開照合を完了する。見えない領域を含めた完全網羅性は主張しない。
アクセス不能項目は記録して他の許可済み読取を継続できる。原本保全違反/秘密公開/他者干渉を避けられない場合、
必須仕様未取得や科学再実行なしに解決できない処理は該当部分をSTOPし、未解決として報告する。
mandatory STOP：公開・照合後はSERVER 1 Phase 1を終了する。LOCAL/SERVER 2、Phase 2/3、科学計算へ自動移行しない。
最終報告はGitHub URL、固定SHAとremote照合、担当範囲/限界、保全、未実施処理、STOP、次の担当を示す。
ユーザーの明示実施指示が許可であり、承認証明・原文保存・grant発行等を新たに要求しない。
```

## SERVER 2用（Phase 1、未起動）

```text
あなたはSERVER 2のCodexです。HIROMU1015/Partially-Randomized-Trotterの
アルゴリズム改善研究を除く既存PR検証について、Phase 1のSERVER 2棚卸しだけを行ってください。
このブロックの保存・閲覧だけは開始許可ではなく、ユーザーからのSERVER 2 Phase 1実施指示が前提です。

固定共通仕様：
PROTOCOL_COMMIT=968218c2b0c6280af6c1c02822603e3312aa410c
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/PROTOCOL.md
同commitのINVENTORY_TEMPLATE.csvを使用し、schema_version=PR-COVERAGE-1、environment=SERVER_2とする。
実際に固定commitを取得して本文・ヘッダを読んでから着手する。branch tipを代わりに使わない。
必須仕様を取得できない、指定版と異なる場合は着手せず、具体的な取得不能を報告してSTOPする。
各worktreeのAGENTS.mdと、適用される親projectsのAGENTS.md、正式共通運用指示
research_project_instructions_unified_2026-10-25.md（特に§8.1/§8.4/§10）も実際に確認する。
規約のパスをLOCALから推定せず、この環境の既知文書またはユーザー提供資料で確認する。
未取得の必須規約を読了と称さない。sub-agentを使わずprimary agentだけで作業する。

読取範囲：SERVER 2でユーザー所有・対象repoとの対応・アクセス許可を確認した研究rootだけ。
SOURCE、result、archive、Git履歴、保全済み証拠をread-onlyで調べる。
LOCALやSERVER 1へSSH・転送・調査を行わない。他ユーザーのファイル/job/processを調べない。
GitHub上の共通資料の取得はよいが、それを他hostの現存確認と呼ばない。
旧SERVER_A/B等とSERVER 1/2の対応は推測せず、UNKNOWNまたはユーザー指定aliasとして記録する。

1. 担当root、観測時点、アクセス制限を出力REPORTへ記す。証拠treeではGIT_OPTIONAL_LOCKS=0、
   暗黙fetchなしのgit show/ls-tree/cat-file等を使う。新しい独立出力branch/worktreeを作り、原本を変更しない。
2. 共通仕様§3の固定研究地図9a0bc7f、Track A索引4591b8d、70カード/SOURCE索引、
   既存環境inventory fa55c162を再利用する。該当版のfull SHAはPROTOCOLに従う。
   この環境で利用可能な既存repository inventory、SOURCE索引、保全・branch整理記録があれば優先する。
   server2_repository_inventory_20261102.tsvというLOCAL側記録名から実保存pathを推定しない。
   旧SERVER_B等との対応も確認するまでUNKNOWN。現存directoryだけを研究全体とみなさない。
3. 既存索引から担当環境の関連証拠へ限定的に辿る。固定Git object・保全archive索引を優先する。
   削除済みbranch/worktreeは既存保全記録・carrier・固定commitで追跡し、cleanupを再実行しない。
   大規模な全域検索、home/projects/tmp横断走査、raw全decode/全hash、archive全展開は行わない。
4. PROTOCOL§4～7の43列を使い、科学条件×run/cohort×保存解析版を記録する。
   A～E（DF、PF/finite RTE、回路/資源、RPE/QPE、Pauli比較）を含め、改善方式自体は除外する。
   改善系列内の既存PR評価・baselineが直接必要ならSUPPORT_ONLYとし、除外理由と関係を残す。
5. REPORT/JSON/CSV/raw/QPY/STOP等を同じevidence_groupへrole付きで接続する。
   原SOURCE、input/state/reference、law/normalization、compiler/測定境界、seed/cohort、
   継続/回復、結果identityを原記録で照合する。コピー/引用/同seed再実行を独立確認と数えない。
   探索と確認、価格標本とsignal、paired axis、複数精度での共有を明示する。
   関係は原記録に基づく候補と根拠だけ。3環境の統合・canonical確定はPhase 2へ残す。
6. 科学判定、技術PASS、実行完了、正式受理、旧結果の失効を別列にする。
   実行前STOPと実行中失敗、未実施と証拠未発見、局所経験値と厳密保証、実価格とproxyを混同しない。
   不明値はUNKNOWN/NOT_RECORDED/NOT_ACCESSIBLE/NOT_FOUND等と限定探索範囲を記録する。
   未発見を未実施と断定せず、原nullを0へ変えない。元の判定と出典を保持し、新しい科学的PASSを作らない。

成果物：新規branch codex/pr-coverage-inventory-server2-<作成日と必要な一意suffix> に、
docs/research/pr_cross_host_coverage_audit_20261102/inventory/SERVER_2/INVENTORY.csv と REPORT.md。
長い条件・参照だけ必要ならDETAILS.jsonlを使い、PROTOCOLのdetail_ref規則で解決できるようにする。
REPORTにはprotocol固定SHA、読取rootの公開用ID、時点・未アクセス、旧索引の項目ごとの
記録済み/範囲外/未発見/未アクセス/未解決の対応、欠測・alias・関係候補、検査、保全、STOPを記す。
架空の実験行・数値を作らない。条件が特定できない参照はREPORTの未解決一覧へ残す。

保存値のliteral引用と必要最小metadata/hash照合だけを行う。
新科学計算、保存数値の新集計・N/G再会計・fit・再ランキング、sampling/RNG replay、
simulation、Hamiltonian/state/reference生成、回路build/compileを行わない。
coverageの科学的評価、不足A～Eの新分類・優先順位、詳細科学レビュー、主要claim採択を行わない。
既存コードを静的に確認し、適合する読取/転記/参照検査を再利用する。科学runnerをimport/実行しない。
CPU/RAM/workerに恣意的上限を置かず、許可されたmetadata処理を必要に応じ高速化する。
共有設定/OS/package/scheduler/権限、他job/processを変更・停止しない。
原SOURCE/input/raw/result/grant/ledger/marker/STOP、archive、既存研究map・概要・manifestを変更しない。
証拠treeのcheckout/reset/merge/clean/gc/pruneやbranch/worktree/archive削除は禁止する。

公開：成果物の公開コピーだけを独立出力treeで作る。内部host/IP/絶対サーバーパス、資格情報、
private rawは除き不透明IDを使う。必要な非公開の所在対応は環境内で保全してcommitしない。
ヘッダ一致、JSON構文、ID/参照、原引用、欠測、差分範囲と秘密の非混入を検査する。
確認済みの対象repo remoteに通常commit/pushし、remote SHAと固定commitの公開CSV/REPORT/補助本文を照合する。
force pushや既存研究branchの更新は禁止。push失敗はその旨、理由、確認済みcwd/remote/branchを
含むコピー可能な再実行pushコマンドを最終チャットへ示す。未公開を公開済みと報告しない。

終了条件：許可したSERVER 2 rootと既存索引の参照を状態付きで対応付け、未解決・未アクセスを残し、
形式/参照検査と公開照合を完了する。見えない領域を含めた完全網羅性は主張しない。
アクセス不能項目は記録して他の許可済み読取を継続できる。原本保全違反/秘密公開/他者干渉を避けられない場合、
必須仕様未取得や科学再実行なしに解決できない処理は該当部分をSTOPし、未解決として報告する。
mandatory STOP：公開・照合後はSERVER 2 Phase 1を終了する。LOCAL/SERVER 1、Phase 2/3、科学計算へ自動移行しない。
最終報告はGitHub URL、固定SHAとremote照合、担当範囲/限界、保全、未実施処理、STOP、次の担当を示す。
ユーザーの明示実施指示が許可であり、承認証明・原文保存・grant発行等を新たに要求しない。
```
