# Track A 正式共通研究ノート：サーバー受領確認

2026-10-14 JST。判定は **SYNC_ACCEPTED_WITH_OPEN_ITEMS**。サーバー・ローカルが同じ固定版の共通研究情報を参照でき、情報同期は成立した。H8最終結果等の取得待ちは後続更新対象として残す。原科学計算の独立再現・母平均保証・新しい科学的採択を意味しない。

## 固定版と受領照合

- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 正式branch：`track-a-cross-environment-canonical-status-20261014`
- 正式commit／取得時remote HEAD：`4675730e7e40d25ba3618334a4a0e28b4bd2f145`（一致、到達性確認）
- サーバー初版：`2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c`
- ローカル照合版：`8e84398995bb162bc74018c1b66fbb14448c7d27`

[正式共通ノート](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/docs/research/track_a_shared_research_status.md)、[同期方針](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/docs/research/track_a_environment_sync_policy.md)、科学review判断・R09原本・環境差分・ローカル参照索引の6資料と依存JSON3件を固定SHAのGitHubから取得し、Git blob／bytes／SHA-256が一致した。
[公開監査receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/artifacts/resource_applicability/track_a_cross_environment_canonical_status/2026-10-14/remote_publication_receipt_v1.json)の対象18件も固定Git blobと一致。receiptが検証した公開commitは`1b9e21e7dfdca6c28c2cc68c5948dfa91ab5e844`で、4675730eはreceipt追記版である。この違いを混同しない。差分・ローカル参照索引は18件の対象外だが、今回GitHubから取得し、保存されたローカル照合版のblobとも一致した。
6資料の内部・固定GitHub fileリンク120箇所を照合した。固定GitHub fileリンクは重複を除いてHTTP HEADも確認。外部論文・APIリンクの網羅調査、既存大量hash監査の再実行は行っていない。
file別結果は[受領監査JSON](../../artifacts/resource_applicability/track_a_cross_environment_server_acceptance/2026-10-14/receipt_audit_v1.json)を参照。

## SOURCE・結果・科学レビューの対応

|研究単位|固定execution SOURCE|結果／記録commit・扱い|
|---|---|---|
|H4 geometry元4方式|`d0726e946f89f8c30a08d383fb8ff48b363629ec`|`7c4483378047b57f514c47dad98905e3f2a9a756`、元集合1812点B2点最小を保全|
|H4 B0+I全面実測|`924b779c2bc6f6e844d6283c355f91c73aed6445`|`98cdb35cd08469714f20d6cc3b982d9c80908f46`、拡張集合397 B0+I／1415 B2を別記|
|H6探索|`78a39051d8b6d41c33d592ecfe172f8824adfaeb`|`df99d6ee28942df8d12aac3c4db7cd202df55d77`、592 group|
|H6独立確認|`87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`|`321cd8c3066194fe98741cbebbcdba1515d1deaa`、固定8構成×fresh32。探索8とpoolしない|
|H6条件付きs2 pilot|`843265b624eac8fbb9acea47ef21cab1207e1880`|`e0e6473d7047158a7f0f788d026a7a89a01898b8`、局所scopeを保全|
|H8 feasibility|`9bfea8bde44cbb2ea9187476ddd496066a8853b3`|`a55da5c2b012912c485184845aa73ca053ac00b9`、core最終mapとは別|
|H8 core resume02|`c31ef424175c02379eafebe8fc6c7548c1d9aabe`|`38d386c8732295e6fab5d6fcbd47fbe586dc207d`は起動資料。最終result commit未取得|

H4 data/seed SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`、H6 science SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12`、H8 seed SOURCE `cb8a56c6b2a772827275f9200b02638030f33f72`はexecutionと区別する。H4 ordinary/structured/legacy DF12とnative H4/H6/H8は別条件の事例であり、同じ統制size比較へ統合しない。

R09原本は[正式公開copy](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/docs/research/track_a_h4_b0i_v04_scientific_review_20261014.md)を受領した。**37,312 bytes、SHA-256 `1c4fb797dbaed0580ab4bf188f8ef4adb4fa042a5ec42731e8a56ff5bd129703`、Git blob `62110a5c92dfcf2c22050de6de08e54ee3f90cd8`**で指定値と一致。
原本0.1–0.3、8.4、10、12–14節と正式共通ノート・review索引・R09 registryの転記を確認した。claim B主／A補助／C別native事例、限定benchmarkとしての採用、追加H4 sampling等を現時点の必須工程にしないこと、新計算認可を含まないことは一致。レビューをやり直していない。
科学reviewの公開版は4675730eで固定参照できる。SOURCE924b779c・result98cdb35c・原稿47dc17a8とreview公開commitは別identity。R09 registry内の自己commit欄nullは、本受領では正式固定commit＋path＋blobで解決している。
旧候補索引に残る「採否待ち」「未採択」は当時の履歴で、正式入口とR09による現在の取得状況と区別する。原本を遡及改変しない。

## H8：最新の確認可能情報

**2026-10-14 10:39:07.889 JST**の非atomic読取snapshot：[保存metadata監査](../../artifacts/resource_applicability/track_a_cross_environment_server_acceptance/2026-10-14/h8_readonly_snapshot_v1.json)。旧203/592 snapshotから推測していない。

- run ID：`H8_CORE_RESOURCE_MAP_20261014_resume02`、SOURCE／seedは上表どおり。
- signal 92/92。探索COMPLETE **556/592**＝祖先受領126＋当run checkpoint/COMPLETE対応430、残36。
- 430件のsource/task/result digest対応、native exit0/reap metadataが一致。全rawの再hash・全科学出力の再監査ではない。
- coordinatorと8 workerのnative start／own UIDが読取時点で一致。observerはactive8、pending28、first STOP理由null、直近journal age約1.11秒。pendingは実行中8件を含まないため残36と整合する。
- 探索summary、独立確認directory／selection／最終result／terminal markerは当snapshotで未観測。独立確認の完了や最終winnerを認定しない。
- fresh remote一覧の既知H8 production／resume01／resume02は従来の固定起動commit。最終core result branch/commitは確認できなかった。GitHub全historyの不存在証明ではない。

known／reserved／unknown=nullとcarryの区別を保持。台帳の返却/reset、進行中H8の停止・再開・条件変更は0。観測後の進捗は本記録の対象外。

## 原本保全・未公開証拠・残る項目

正式統合でサーバー初版の旧2730 pathの削除0、SOURCE/scripts/tests/結果artifact blobの変更0。server inventory、science identity projection、既存原稿・旧STOP/grant・レビューは保存されている。変更は共有入口・policyと索引追補に限られ、主要SOURCE/結果の実質矛盾は見つからなかった。
private原本の責任側はサーバーの利用者管理領域。H4凍結NPZ6／freeze、H4 map・B0+I raw、H6探索/確認/pilotのruntime、H8入力と実行checkpoint/ledger、private venvは本体を移さず、固定公開receipt/summary・identity索引との対応を保持する。今回の存在確認はstatとmetadataに限り、NPZ科学array読込・全raw再hashを行っていない。絶対保存先対応表とnative PID等はprivate監査側に保持する。

|未解決事項|今回の確認・次の担当|
|---|---|
|H8最終探索・独立確認・resultと科学review|H8担当が既存scopeの完了／STOPを監査・固定公開した後、情報同期担当が正式入口へ追補。科学採否はGPTへ|
|H4 run03 `a3248055ec8eda69d43c8911af71c7580f2f95ed`|local commitは存在。同名remoteなし、取得済みremote HEAD祖先で到達性0、GitHub commit API422／固定raw404。公開到達性未確立を保持。履歴公開が必要なら利用者・Git担当の別作業|
|旧export partial450 mismatchのevent/digest|未保存。原因UNDETERMINED、actual unknownを維持。export再実行・event再生成0|
|ローカルonly plan/dirty note、R09補助ZIP/script|hash索引と原review取得状況を保持。取得・採択が必要ならローカル担当へ。本文から補完しない|
|NOT_SAVED／172丸め、希少母平均、旧compiler bit同一性、RQ-P凍結モデル、最終energy費用|既知の科学的保留であり、今回の同期不合格にしない。新検証開始0|

## 終了・公開

正式共通ノート自体の修正は不要。本branchは受領技術記録と軽量監査JSONのみ追加し、既存の実行用worktreeをcheckout/pull/mergeしない。作業前後の既存67 worktreeのHEAD/status保全を監査JSONに記録する。
公開commit／remote照合は最終チャット報告とprivate publication receiptで結合する。HTTPS認証失敗時は未公開と手動push方法を報告し、Git設定・認証情報を変更しない。
新科学計算0、H8実行環境への変更0、既存worktreeへの変更0、新grant0、原稿改訂0。**MANDATORY_STOP、next_stage_authorized=false**。H5/H7・T/精度依存性・追加原稿編集へ自動進行しない。
