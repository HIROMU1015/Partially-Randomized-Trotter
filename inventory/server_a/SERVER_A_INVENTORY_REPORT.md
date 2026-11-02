# サーバーA PR研究棚卸し — 公開用匿名化コピー

environment=SERVER_A。**PUBLIC_REDACTED_COPY。詳細原本は非公開で保全する。整理の実行許可ではない。**

既存4成果物を再利用した公開準備版。元snapshotの36 worktree・分類・CSV行数を維持し、今回追加した整理用worktreeとその後の並行作業は加算していない。ローカル側の公開済みcommit `dc256b1e67433e27192457a2ed3882c61996b769` を基点に、専用branch `codex/pr-environment-inventory-20261026` の `inventory/server_a/` へ追加する。

**今回の利用者の最新指示により、専用branchへのcommit・通常push・local/remote SHA照合後にmandatory STOP。** 既存 `inventory/local/` を変更せず、サーバーBへ接続・操作しない。本体の「調査前後」「STOP」は元棚卸し時点の記録であり、今回の公開作業と区別する。

## 匿名化の範囲

- 絶対パス、内部ホスト名・ユーザー名、private rawの具体的配置、未追跡・ignoredファイル名、process ID、内部環境・mountの詳細を除外または環境内IDへ置換した。逆引き対応は公開しない。
- `SERVER_A-WT-*` はworktree、`SERVER_A-GIT-*` はcommon directory、`SERVER_A-DEP-*` は非公開の保存先・依存先の匿名ID。環境をまたぐ同一性・所在・アクセス権を示すIDではない。匿名化に伴うpath表記の統合は科学証拠の統合を意味しない。
- branch名・commit SHA・study ID・分類・件数・保全上の留保を保持した。内部ホスト名を含むref名だけは非公開表記とし、照合にはSHAを使う。
- SOURCE/結果/STOPへの固定commit付きGitHubリンクは、観測した公開remote refsから到達するcommitにあり、versioned pathの存在を確認できたものだけを残した。未公開commitのSHAと保護理由は保持し、そのファイル名・内容・配置は非公開とした。
- 原本の補助JSON・private対応表・raw・grant本文・外部非公開データの内容・認証情報はこの公開準備版に含めない。公開対象は4成果物だけ。


作成: 2026-10-26T16:05:57.818333+09:00。environment=`SERVER_A`、host=`SERVER_A`、UID_WITHHELD。
読み取り専用調査と、この独立ディレクトリへの報告作成のみ。新規科学計算、数値再解析、compile、sampling、科学レビュー、研究方針変更は実施していない。

## 集計と分類

- worktree総数 **36件**。主cloneとそのlinked worktreeが35件、独立したTrack B cloneが1件。Git common directoryは2系統。
- KEEP_ACTIVE **1件**、KEEP_EVIDENCE **21件**、REMOVE_CANDIDATE_AFTER_APPROVAL **12件**、HOLD_UNKNOWN **2件**。
- 積極的保護対象はKEEP_ACTIVE＋KEEP_EVIDENCEの22件。HOLD_UNKNOWNも保留中は保護するため、整理候補以外の保護・保留は24件。
- KEEP_ACTIVEは端末・言語サーバーがcwdとして使用するmain。今回の観測でPRの科学計算workerは見つからなかった。端末の存在を科学計算の実行とは扱わない。
- local branchはcommon directory単位で39 refs（主repo37＋独立clone2）、同名をまとめると38名。remote-tracking branchは796 refs（主repo776＋独立clone20）。別にsymbolic origin/HEADが2件。
- GitHubの現在のbranchは776名。local/tracking/live全体のbranch名集合は783名。ref種類を併記したSERVER_A_BRANCHES.csvの1613行はbranch実数ではない。
- detached HEADは1件。`SERVER_A-WT-0034`。旧パスalias `SERVER_A-DEP-0301` はmainへのsymlinkで、独立cloneとして重複計上しない。

整理候補はサーバーAで確認した条件付き候補であり、削除可能との最終確定ではない。他環境の利用・未記録依存・再開予定、private rawの保全とユーザー承認を次段階で確認する。branch削除の承認や実行を意味しない。

## 成果物

- [SERVER_A_WORKTREES.csv](SERVER_A_WORKTREES.csv): 36行、元分類と環境内IDを維持。
- [SERVER_A_BRANCHES.csv](SERVER_A_BRANCHES.csv): 1613行、ref種類・SHA・研究区分・保護理由を維持。
- [SERVER_A_STORAGE_DEPENDENCIES.csv](SERVER_A_STORAGE_DEPENDENCIES.csv): 417行、公開参照と匿名化した保存依存・未確認状態を維持。
- [SERVER_A_INVENTORY_REPORT.md](SERVER_A_INVENTORY_REPORT.md): 原本を再利用した匿名化報告。

詳細索引・前後照合・原本hash・匿名IDの逆引きは非公開の作業記録で保持する。このbranchへは収録しない。

## 調査範囲と限界

`PROJECT_MAP.md`、`docs/research/研究概要・現状.md` 全文、validation status/manifestの索引、既存サーバーA証拠カタログを入口にした。2026-10-25版の運用指示は、登録worktreeの既知パス、local全refの当該path履歴、GitHub既定branchおよび統合commitの既知パスで見つからず、**未取得・未読**。取得済み10月23日版を10月25日版と呼ばない。今回のユーザー指示のread-only・原本保護・STOPを適用した。

本人所有の既知研究領域と一時保存領域を限定深度で確認し、対象repositoryと登録worktreeを調べた。別repoは共有依存だけ記録し、詳細な棚卸しを行っていない。未知の保存先まで網羅したとは主張しない。

Git操作は `GIT_OPTIONAL_LOCKS=0`、status、worktree list、rev-parse、for-each-ref、show、ls-tree、cat-file、rev-listなどの読み取りのみ。fetchを使わず、GitHub connectorのGETで776 live headsを取得した。775個の異なるlive tipは両common directoryの既存objectに全て存在し、その到達可能集合とlocal branchを照合した。API404は接続権限で見える当該repoにおける取得不能を意味し、世界中の別cloneやbackupの不存在を意味しない。

raw全ファイル走査、全hash、archive展開・復元、科学SOURCEのimport/実行を行っていない。既存23系列・2834参照・22件SOURCE binding・9 custody rootを再利用した。private custodyの件数・byte数は2026-10-23台帳の歴史値であり、今回はroot存在のみ確認。control metadata134件の限定読取りと、記録済みの2599 locatorの存在確認を行った。locator中の `#sha256=` はpath注記として除外し、誤ったファイル欠落扱いを避けた。

本人processと既存job記録の限定的なmetadataを確認した。process識別子・環境変数・内部scheduler構成は非公開とする。手動の再開予定・別Codexの今後の作業・他hostの利用はUNKNOWN。サーバーBへ接続せず、B3は依頼文の稼働情報を前提に保護した。

## 重要成果とSOURCE・結果の対応

|研究区分|固定結果・入口|サーバーAでの対応と保護|
|---|---|---|
|DF/Pauli統合資料|`6e3436a16b13e539b0faa2a0a3286f6ab3549543`: RESEARCH_SYNTHESIS.md / EVIDENCE_MATRIX.csv|GitHub live branch先端・local Git objectあり。このHEADのlocal checkoutはなし。文書統合を新規科学検証と解釈しない。|
|H4/H6長時間・高次PF/PR|`155529e20a8fac39e00786c40b57830b71203c4f`: REPORT.md / artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/|GitHub live先端。provider SOURCEは`c31ef424175c02379eafebe8fc6c7548c1d9aabe`、baseは`90f1cd55a0d963953696bd0a27d9d041ff939bb7`。private保存先はこのhostに未発見。|
|Pauli-JW総合資料|`b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`: docs/research/pauli_jw_resource_evaluation_summary.md|GitHub live先端。最新原結果はH2 `4aea054b3369709bc0d35ff081071e62b5f77cbe`、event拡張 `10cb3e5cca7834ebf75301bd91bd093ca51d4a6b`。原SOURCEはそれぞれ`bf5f2339f723f4edb7bf91a8ca88f33251bbef57`と`e755ecf581600192fd2745d7066e410cebd933d3`。private rawのサーバーA保全はUNKNOWN。|
|DF-native Track A・共通PF・高次PF/PR|採択`f75a7c25d584b69d90de454731a7f343a421e059`、H6最終`90f1cd55a0d963953696bd0a27d9d041ff939bb7`|SERVER_A-WT-0009/SERVER_A-WT-0011/SERVER_A-WT-0012/SERVER_A-WT-0013を代表証拠として保持。SOURCE catalogue、入力snapshot、raw archive inventory、control/STOPを別に保持。|
|再構成独立confirmation|`3eca46148070ba4d4b0f5815ef45943407cc959e`|SERVER_A-WT-0004のディレクトリ名はh246-confirmation-new-environmentだがbranchはh246-reconstructed-confirmation。名前だけで旧STOP系列と重複判定しない。元入力探索STOPと再構成cohortを区別。|
|既存アルゴリズム・WP11・基盤|main `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`、WP11 `8b574c5431acf8d7fc2059ecc2c3b56ff19c37f0`|SERVER_A-WT-0001/SERVER_A-WT-0002を保護。WP11は公開branch先端より20 commit後方だがHEADは公開祖先。ignored checkpoint等は別保全が必要。|
|失敗・STOP済み中間検証|非DF H2 pilot `bd2f3ce9b2035d9caefac75785e6fe70a3dcbca2`、H4 run02/03/04/05、H6中断・回復|失敗STOP、grant消費、親process終了の未確定履歴を保持。後続成功や公開集計で置換しない。|
|B3に関係し得る資料|独立clone `SERVER_A-WT-0036` とTrack B/G6/W2C/B3等のref|SERVER_A-WT-0036はHOLD_UNKNOWN。branch名にB3がなくてもTrack B関連は保護・他環境確認待ち。|

旧23系列の詳細SOURCE bindingは `PUBLIC_REFERENCE_WITH_HOST_COMPONENT_WITHHELD`、結果と終了監査は同directoryの `series_manifest.json` / `artifact_index.csv` を参照する。各入口のblob/所在確認は PRIVATE_DETAILS_WITHHELD に記録した。

科学的数値・勝敗・係数は今回再評価していない。既存の限定採択、有限標本、dirty local証拠、未検証項目と最終結論は区別し、最終全RPE総cost評価が完了したとは述べない。

## GitHubに未保存の可能性があるcommit・結果

以下4件は現在の全live headsから到達せず、GitHub commit object GETでも404だった。未pushまたはrepo外でしか保全されていない可能性があるため保護する。publishは実行しない。

|local branch|commit|内容と保護場所|
|---|---|---|
|codex/pr-pf-cng-technical-investigation-20261022|`c4cb5f3c9007f84b8d3798be17398ee2ea498e0f`、`f5ae5d6d3fd228f862f8be3eea6c1fb4c0f0ca37`|SERVER_A-WT-0018の技術追補とそのSOURCE。|
|pf-common-formal-adoption-20261022-local|`60b76576263bb0c31b4a3433a306747d62dc5789`|local branchのみ、checkoutなし。公開採択commit f75a7c2と同名subjectでも同一commitとみなさない。|
|pf-common-paper-publication-draft-20261023|`8f0ff62b2bfc6676c31773261b160e3935ffcdae`|SERVER_A-WT-0014の公開配置draft、SOURCE catalogue、manifest、STOP。|

mainにはtracked変更と未追跡6 group entryがある。entry数はdirectory内の全ファイル数ではない。具体的な配置・ファイル名は非公開とし、ignored入力cache、checkpoint・log・worker結果をGitHub集計とは独立の保護対象として保持する。

## raw・STOP・archiveの保全リスク

- `SERVER_A-DEP-0306` は `MANIFEST_ONLY_NOT_TRANSFERRED`、archive_created=false、transfer_executed=false。21 membersの所在一覧を転送完了や独立backupと呼ばない。原本はmain配下のH2実行領域とH246入力/provider側に分散している。
- 長時間研究が指す `SERVER_A-DEP-0291` と記録されたrecovery log / report STOPはサーバーAには存在しない。別hostに同名pathがある可能性を含めUNKNOWN。ローカル棚卸しで原本・途中停止データ・private event/seed・QPY・全trial結果を照合する。
- 旧9 custody rootは今回も存在したが、独立backupの実在・復元性を確認していない。2026-10-23台帳の27089 files / 2804960008 bytesは今回の再計数ではない。
- H4 run01初期化失敗、run02 INPUTS_FROZEN_STOP、run03 MONITOR_STOP_NO_AUTOMATIC_RETRY、run04/05 MONITOR_INTERVAL_FRESHNESS_STOP、H6中断・回復のSTOPを保護する。これらを現在のactive workerと誤認せず、再開の許可とも扱わない。
- handoff bundle、元SOURCE archive、untracked h6 exporter zip、既存GAP recovery/private archiveは保持。bundleの存在だけではprivate rawの全部や他host受領を証明しない。archiveの完全性・復元性はUNKNOWN。
- 既存GAP-001はFOUND_UNVERIFIED、GAP-008はNOT_FOUND_IN_SEARCH_SCOPEという記録を継承する。stale DF screeningやprose-only UWCを科学結果として復活させない。

## 共有領域・他環境・実行環境への依存

- 主Git common directory `SERVER_A-GIT-0001` を35 worktreeが共有する。主ディレクトリの一括削除や共通Git領域の整理は全linked treeを壊し得る。独立cloneのcommonは `SERVER_A-GIT-0002`。このcloneの `objects/info/alternates` はmainの `SERVER_A-DEP-0304` を参照している。common directoryが別でもobject storeは共有依存であり、独立backupではない。B3関連資料を保護するためmainのobject storeも保護する。
- 共有venvを複数研究が利用し、別研究のSOURCE checkoutをeditable参照する依存がある。activation設定と実際のimport元を同一視せず、共有envや外部checkoutを変更しない。具体的配置・内部構成は非公開。
- WP11の `ENV_LOCATION_WITHHELD` は自身のsrcをeditable参照する専用環境。依存版と復元手順を確認するまでworktreeと一緒に捨てない。
- `SERVER_A-DEP-0303` と `SERVER_A-DEP-0025` の観測mountは内部詳細を伏せたfilesystem。ネットワークmountは対象pathのmetadata上では未検出だが、他host共有やbind-exportの不存在までは証明できない。他環境の共有・scheduler・利用状態はUNKNOWN。
- `SERVER_A-DEP-0025` の3 linked treeはmainのcommon Gitへ依存し、独立clone1件とadmin資料もSERVER_A-DEP-0025にある。OSの一時領域運用による消失リスクがある。今回は移動や新backupを行わず、所在と保護要件を記録した。

## worktree別判定

本公開CSVでは絶対パスとprivate配置を匿名化し、HEAD・公開SOURCE/STOP参照・分類を保持する。詳細原本を非公開で保全し、匿名IDや名前だけでworktreeを統合しない。

|ID|worktree|branch|分類|根拠|
|---|---|---|---|---|
|SERVER_A-WT-0001|SERVER_A-WT-0001|main|KEEP_ACTIVE|使用中の端末・言語サーバーあり。AGENTS.md変更、未追跡実行領域・raw・報告・共有Git本体を保持。|
|SERVER_A-WT-0002|SERVER_A-WT-0002|all-r-coherent-opt2-reoptimization|KEEP_EVIDENCE|cleanだが専用venv、ignored ground-state cache、checkpoint・log・worker結果が残存。公開HEADは先行版の祖先でありraw保全とは別。|
|SERVER_A-WT-0003|SERVER_A-WT-0003|codex/h2-common-pf-original-recovery-20261023|KEEP_EVIDENCE|private原本のlocator・保存監査が外部のmain/SERVER_A-DEP-0001に依存。転送manifestは作成済みだが転送未実施。|
|SERVER_A-WT-0004|SERVER_A-WT-0004|h246-reconstructed-confirmation-20261022|KEEP_EVIDENCE|ディレクトリ名に反してbranchは再構成confirmation。独立cohort・元入力探索STOP・private rawの対応を保護。|
|SERVER_A-WT-0005|SERVER_A-WT-0005|codex/non-df-h2-pilot-20261022|KEEP_EVIDENCE|Qiskit失敗・no-retry STOPを含む中間科学証拠。後続成功cohortと置換せず保持。|
|SERVER_A-WT-0006|SERVER_A-WT-0006|codex/non-df-h2-process-recovery-20261023|KEEP_EVIDENCE|独立processの回復結果・grant/ledger/marker/STOPとignored入力cacheを保持。|
|SERVER_A-WT-0007|SERVER_A-WT-0007|codex/non-df-h4-readiness-20261023|KEEP_EVIDENCE|readiness・人工試験STOPとignored入力cacheを保持。分子計算の完了と解釈しない。|
|SERVER_A-WT-0008|SERVER_A-WT-0008|codex/non-df-pauli-fixed-signal-20261022|KEEP_EVIDENCE|固定信号のSOURCE・人工検証STOPとignored入力cacheを保持。後続Pauli研究と同一cohortとみなさない。|
|SERVER_A-WT-0009|SERVER_A-WT-0009|pf-common-domain-execution-20261022|KEEP_EVIDENCE|公開結果に加えてmain外部実行領域のraw・grant・STOPに依存。外部原本の独立backup未確認。|
|SERVER_A-WT-0010|SERVER_A-WT-0010|pf-common-domain-preparation-20261022|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。|
|SERVER_A-WT-0011|SERVER_A-WT-0011|pf-common-formal-adoption-20261022|KEEP_EVIDENCE|採択済み共通PFの代表checkout。公開版とlocal-only採択commitを区別して保持。|
|SERVER_A-WT-0012|SERVER_A-WT-0012|pf-common-h2-execution-20261022|KEEP_EVIDENCE|既存controlからこのcheckout内の入力・保存結果への絶対パス参照が多数残る。private原本の供給元。|
|SERVER_A-WT-0013|SERVER_A-WT-0013|pf-common-h6-metadata-execution-20261022|KEEP_EVIDENCE|H6中断・回復のSOURCE archive、ignored log、外部ledger/STOPを保持。中断前親processの終了履歴には未確定部分あり。|
|SERVER_A-WT-0014|SERVER_A-WT-0014|pf-common-paper-publication-draft-20261023|KEEP_EVIDENCE|HEAD 8f0ff62はGitHub live headから非到達、commit API 404。未公開の配置draft・manifest・STOPを保護。|
|SERVER_A-WT-0015|SERVER_A-WT-0015|codex/pf-order-reversal-h2-common-policy-prep-20261023|HOLD_UNKNOWN|H2共通政策・他環境受領metadataの不足事項が未解決。他環境の原本・再開予定・B3との関係はUNKNOWN。|
|SERVER_A-WT-0016|SERVER_A-WT-0016|REF_NAME_WITH_INTERNAL_HOST_WITHHELD|KEEP_EVIDENCE|既存23系列・2834参照・private保全索引を結ぶ代表worktree。新報告がこの索引を参照するため保持。|
|SERVER_A-WT-0017|SERVER_A-WT-0017|codex/pr-pf-analytic-vs-circuit-cng-20261022|KEEP_EVIDENCE|公開C/N/G保存結果と未公開技術追補は別証拠。独立N未取得などの留保・STOPを保持。|
|SERVER_A-WT-0018|SERVER_A-WT-0018|codex/pr-pf-cng-technical-investigation-20261022|KEEP_EVIDENCE|HEAD c4cb5f3とSOURCE f5ae5d6がGitHub live headから非到達、commit API 404。未公開技術追補・SOURCE・STOPを保護。|
|SERVER_A-WT-0019|SERVER_A-WT-0019|codex/pr-pf-s2-recomputed-theory-compiled-20261022|KEEP_EVIDENCE|ignored PRIVATE_INPUT_CACHE_WITHHELD等の入力・scratchを保持。公開集計だけでは元入力cacheのbackupを立証できない。|
|SERVER_A-WT-0020|SERVER_A-WT-0020|track-a-h4-cross-candidate-run03-20261006|KEEP_EVIDENCE|過去run03のcontrolがこのcheckoutの成果物を絶対パスで参照。停止履歴とSOURCE対応を保持。|
|SERVER_A-WT-0021|SERVER_A-WT-0021|track-a-h4-geometry-contract-20261006|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。|
|SERVER_A-WT-0022|SERVER_A-WT-0022|track-a-h4-geometry-contract-v2-20261006|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。|
|SERVER_A-WT-0023|SERVER_A-WT-0023|track-a-h4-geometry-input-generation-authorization-20261006|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。|
|SERVER_A-WT-0024|SERVER_A-WT-0024|track-a-h4-geometry-input-generation-cpu-launch-proposal-20261006|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。|
|SERVER_A-WT-0025|SERVER_A-WT-0025|track-a-h4-geometry-parallel-source-20261006|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。|
|SERVER_A-WT-0026|SERVER_A-WT-0026|track-a-h4-geometry-resource-observer-fix-20261006|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。|
|SERVER_A-WT-0027|SERVER_A-WT-0027|track-a-h4-geometry-resource-server-prep-20261005|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。|
|SERVER_A-WT-0028|SERVER_A-WT-0028|track-a-h4-geometry-source-20261006|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。|
|SERVER_A-WT-0029|SERVER_A-WT-0029|track-a-h4-input-generation-storage-review-20261006|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。|
|SERVER_A-WT-0030|SERVER_A-WT-0030|track-a-h4-new-server-handoff-20261007|KEEP_EVIDENCE|永続handoffの代表checkoutとして既存handoff記録が指定。bundle・元入力・run05停止記録との依存を保持。|
|SERVER_A-WT-0031|SERVER_A-WT-0031|track-a-h4-signal-compile-plan-review-20261006|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。|
|SERVER_A-WT-0032|SERVER_A-WT-0032|track-a-h4-worker-bootstrap-run02-20261006|KEEP_EVIDENCE|run02で凍結された入力とmandatory STOPを結ぶ科学実行履歴。元入力はmain側に別置き。|
|SERVER_A-WT-0033|SERVER_A-WT-0033|track-a-h4-lazy-identity-run05-20261006|KEEP_EVIDENCE|過去run05のsource checkoutとしてhandoffがSERVER_A-DEP-0025を参照。停止記録・grant消費・入力を保護。|
|SERVER_A-WT-0034|SERVER_A-WT-0034|DETACHED|REMOVE_CANDIDATE_AFTER_APPROVAL|tracked clean、重要untrackedなし、ignoredにもraw/入力/専用venvなし。HEADは現在GitHub公開refから到達。本人processのcwdと今回読んだcontrolの直接SOURCE依存を検出せず。準備文書・保存資料のcheckoutとして条件付き整理候補。他環境の利用・未記録依存・再開予定とprivate原本保全を次段階で確認するまで削除不可。 detached HEADは永続handoff checkoutと同一commit。ただし同一commitだけを削除根拠にせずuntracked/ignored/processも確認した。|
|SERVER_A-WT-0035|SERVER_A-WT-0035|track-a-h4-streaming-monitor-run04-20261006|KEEP_EVIDENCE|過去run04のsource checkoutとしてcontrolがSERVER_A-DEP-0025を参照。monitor STOPと共有予算履歴を保護。|
|SERVER_A-WT-0036|SERVER_A-WT-0036|track-b-g6-v1-2-draft-review-integration-20261022|HOLD_UNKNOWN|独立cloneのTrack B/G6資料。B3に関係する可能性があり今回整理対象外。他ホストの利用・再開予定は未確認。|

## ローカル棚卸しとの照合事項と推奨整理順序

1. まずbranch名ではなくremote repository・common directory・完全HEAD・SOURCE/result commitでローカル側と突き合わせる。最新3成果commitのprivate原本所在、長時間研究の未発見path、H2転送未実施、4件local-only commitを優先する。未公開資料を公開するか、非公開で保全するかは利用者が判断する。
2. B3に関係し得る全ref・成果物・Track B cloneを除外し、他環境担当の利用・共有Git・再開予定を確認する。サーバーBへの接続や変更は今回の後続作業として自動実行しない。
3. 既存custodyとarchive索引を使ってprivate raw、SOURCE/input、grant/ledger/marker/STOPの保全先・復元性を確認する。公開集計だけで削除可としない。必要な保全・移送は別途具体化・承認してから行う。新しい管理制度は作らない。
4. 代表checkoutとしてSERVER_A-WT-0001(main基盤)、SERVER_A-WT-0011(共通PF採択)、SERVER_A-WT-0013(H6来歴)、SERVER_A-WT-0016(環境証拠索引)、SERVER_A-WT-0030(永続handoff)を当面維持する。WP11と固定SOURCE checkoutは専用env・原本依存が解消するまで保持。最新統合/長時間/Pauli branchは成果参照として保持し、新checkoutの作成は今回行わない。
5. 上記照合後、12件の条件付きworktree候補を小さい準備用checkoutから個別に再確認する。tmp detached handoffコピーも永続copyと外部controlを確認するまで削除しない。各候補はbranch削除と切り離して判断する。
6. 最後にlocal branch alias・中間履歴と代表成果branchを分類し、全環境の未push・SOURCE参照を確認した上でGitHub branch整理の具体案を作る。似たbranch名、同一subject、祖先関係を重複の十分条件にしない。

次に利用者が確認すべき事項は、ローカル棚卸し結果、4件local-only commitの保全方針、private rawの独立保全先とH2転送状況、最新研究の実行host・保存先、B3/他hostの参照と再開予定、取得できなかった10月25日版運用指示の所在である。今回の調査・報告完成のための途中承認は求めない。

## 検証と停止

CSVはUTF-8 BOM付き、全行にenvironment=SERVER_Aを含む。参照metadata・分類件数・worktree一意性・CSV行数の検査結果と、調査前後のHEAD/statusおよび限定control metadataのstat比較は非公開の詳細記録へ保存した。この検査は科学的妥当性やraw全体のbyte不変性を保証するものではない。

**mandatory STOP。** 実際の削除・統合・merge/rebase/cherry-pick・branch操作・fetch/prune/gc・ファイル移動・科学SOURCE/raw/grant/ledger/marker/STOP変更・main編集・job操作・サーバーB接続・GitHub公開は行っていない。今後の整理はローカル棚卸しとの統合と利用者の別判断を待つ。

## 今回の公開準備の停止点

元4成果物の行数・分類と原本保全を確認し、公開用コピー4点だけを専用branchへcommit・通常pushする。未公開資料の内容やprivate rawは追加しない。公開完了はpush後にremote SHAとlocal SHAの一致を確認したときだけ報告する。今回も実際の削除・統合・mainへのmerge・研究再開は行わない。

**mandatory STOP。**
