# Track A：公開資料とサーバー保存記録の追加照合

2026-10-15 JST。**資料同期は完了、原本版差とH6 wall方針差は未解決として明記する。** 科学レビューの再実施、科学計算、実行準備の追加実装、原稿改訂ではない。

[正式共通入口](track_a_shared_research_status.md)／[科学判断索引](track_a_scientific_review_decisions.md)／[同期ルール](track_a_environment_sync_policy.md)／[受領・限定検証bundle](../../artifacts/resource_applicability/track_a_shared_status_server_update/2026-10-15/README.md)。

## 起点と保全

正式共通branch `track-a-cross-environment-canonical-status-20261014` のremote HEADは基準 `4675730e7e40d25ba3618334a4a0e28b4bd2f145` と一致した。確認したremote branch一覧に別の後続正式共通版は見つからず、この固定commitから資料専用branch `track-a-research-status-update-server-20261015` を作った。

A/C/D/E/FとH6後続実行資料は基準commitの子孫。H8結果B、旧wall方針、旧公開指示は基準と分岐した別履歴である。祖先関係は[受領監査JSON](../../artifacts/resource_applicability/track_a_shared_status_server_update/2026-10-15/receipt_audit_v1.json)に記録した。科学・結果branchをmerge/cherry-pickしていない。

既存rootはmain/HEAD `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`、既存dirty/untracked 958 status entries。index不存在を含む元状態を保持する。H6実行worktreeのHEAD `d1fa0974064ae4feeb1c8168a88b43f89fc58340`、受領worktreeのHEAD `2ef5ecbdbdc222d73d9106618e20dc27b5466ea2` は開始時clean。private監査にbranch/HEAD/status bytes/SHAとindex有無/SHAを保存し、今回の書込先を新worktreeとhomeの専用資料監査directoryに限定した。67/77 worktreeの旧棚卸しや全raw・venv hash検査を再実施していない。

## 指定公開資料の照合

下記6資料を固定GitHub rawから再取得し、指定commitのGit blob・bytes/SHAとサーバー保存copyの一致を確認した。原本を本branchに再収録せず、固定リンクで参照する。file別bytes/SHA/blobは受領監査JSONを正本とする。

|項目|固定資料|今回の追補|
|---|---|---|
|A：正式版の受領|[2ef5ecbdbdc222d73d9106618e20dc27b5466ea2 / server acceptance](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ef5ecbdbdc222d73d9106618e20dc27b5466ea2/docs/research/track_a_cross_environment_server_acceptance_20261014.md)|正式版の受領成立と残る項目を履歴として保持|
|B：H8 core完了|[681ec23c93fdd1090d325908f4b155a79115afdf / resource map result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md)|旧RUNNING snapshot後の探索・独立確認の完了を索引へ反映|
|C：H4/H6 D0|[8dbdb68293c988a999645e21a82b53966f9ef6cd / saved-value reanalysis](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dbdb68293c988a999645e21a82b53966f9ef6cd/docs/research/track_a_d0_saved_value_reanalysis_result_20261014.md)|H4 native fixed16・H6探索92/確認union8の別保存値解析|
|D：H8 D0|[3d910ee1de4597b8895e8506500553dbad5a5085 / H8 saved-value result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_saved_value_result_20261015.md)|Cを再実行せず、H8探索92/確認union6を別SOURCE/認可で解析|
|E：GPT統合review|[3d910ee1de4597b8895e8506500553dbad5a5085 / integrated review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md)|既存R01–R09を保持した追補R10として採否・保留・後続順序を記録|
|F：H6並列準備|[6baf246cb46cf69536db6c16872286e5ed12e85a / parallel readiness](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6baf246cb46cf69536db6c16872286e5ed12e85a/docs/research/track_a_h6_native_b0i_parallel_readiness_20261015.md)|当時の未認可/CPU未定/未sealと、その後の別grant・実行を分離|

H8はsignal92/92、探索592/592group、確認union6×独立32＝192groupで完了している。旧共通初版203/592とAの556/592は途中の非atomic観測であり、Bの完了を否定する情報ではない。完了証拠を得たことは、希少層の母平均保証や一般的PR優位の獲得を意味しない。

Cは旧precision1,088/cost2,659、Dは旧precision944/cost3,576の完全一致後に、ε=.0005を含む5精度・6指標・paired SE・P感度を扱った。C/Dの完了監査metadataも指定blobと照合した。今回解析を実行し直したものではない。C/Dを合算して独立実験数や新しいsignal標本数としない。

対象条件は各原本に従う。H4 nativeはlinear1Å/STO-3G/legacy DF12・固定16panel、H6はlinear1Å/STO-3G/tol-only DF19・prefix5/10/15、H8は同1Å/STO-3G/tol-only DF28・prefix7/14/21。T=.8、q1/2/4/8、δ=.1〜.8、native symmetric-directional、固定compiler、経験的allowance、有限費用標本の比較である。H4 geometry6距離/ordinary系列や、統制したsize-scaling則へpoolしない。

Eの採択は科学的方針であり、Dの実行認可やH6のgrantとは別記録。E原本はH8 D0実行前に書かれたため「H8 .0005未知」と記す。後続Dの結果を原review本文へ遡及挿入せず、取得順序を追補する。R10が採択したH5/H7・T比較・native各系24軸の全面比較を、現H6 6軸grantへ拡張しない。

## 共通作業指示の原本と版差

|版|bytes・SHA-256|取得と公開状況|
|---|---|---|
|利用者が示したlocal版 `AI_AGENT/research_project_instructions_gpt_codex_review_approval_2026-10-12.md`|48,223 / `b00598003bf877116d9ef2e5ffb383b47c86b7ed1d855f80d815205052f894e8`|申告値のみ。今回確認したGit履歴ではこのbytesの実体・固定commit/pathを取得できていない|
|server管理原本|50,268 / `709524f856a3117c85042309c390a3697d9b2c8c8d79af6b6c290e9d20814bcf`|実測一致。[原bytesの公開copy](provenance/research_project_instructions_server_original_20261012.md)を今回追加|
|別の旧公開版|47,275 / `5024ba8df1d2cdc18b76dee495199e814035f41e76abc3d2b50fac5adb00bb7f`|[1901b4b018f05aedf94d0ca5e786f8d91e84e714 / docs/research原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1901b4b018f05aedf94d0ca5e786f8d91e84e714/docs/research/research_project_instructions_gpt_codex_review_approval_2026-10-12.md)を固定GitHub取得で照合。local申告版とは異なる|

server原本は全文を確認し、一般的な研究運用規則で、認証情報・内部SSH/endpoint・private科学データ・個人の内部記録を含まないと判断した。redaction0、改行変換0、原本編集0のcopyである。50,268 bytes版と一致する既公開固定pathは今回の確認範囲では見つからなかったため、このcopyを追加した。既存47,275 bytes版を同一原本と表示しない。

旧公開版とserver原本はUTF-8/BOMなし/LFのみ。旧564行に対しserver601行で、§6に「研究全体の俯瞰と再評価」を37行・2,993 bytes追加した内容差がある。[機械的unified diff](provenance/research_project_instructions_public1901_to_server.diff)を公開する。両方をLF正規化しても一致しないため、この2版の差を改行差だけとは扱わない。

**48,223 bytesのlocal版と50,268 bytesのserver版の直接差分は未確認。** 上記旧公開版との比較をその代用にしない。local原bytes又は固定commit/pathを取得した時点で、改行・形式差と内容差を別々に照合する。今回更新日時で正本を選ばず、両環境の指示版を統一したとも表示しない。

## H6の認可・資源・実行記録

FのSOURCE `7300b39df4c53d854d5bc8934430af19fbdb0085` は準備時の固定版であり、当時は未認可・CPU未固定・未sealだった。その後、利用者の実行指示、修正・再実行指示と別one-shot bindingが保存されている。

|項目|後続の確認可能記録|
|---|---|
|最新公開branch/HEAD|`track-a-h6-pilot-proc-exit-fix-20261015` / `d1fa0974064ae4feeb1c8168a88b43f89fc58340`。remote HEAD一致、固定報告・auth・sealのGitHub bytes取得一致|
|修正後SOURCE|`2e733d8a683d415792ce92a56c7d149b78f42716`。科学provider `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` は別identity|
|scope|`H6_NATIVE_B0I_PILOT6_V1`。linear H6/1Å/STO-3G/保存tol-only DF19/T=.8/S2、p5q1/p10q2/p15q8×cos/sin＝6 compile。matched B0追加・全面map・次stageは未認可|
|承認原文|「本計算をもう一度行って　計算が完了しなかった場合は修正を行って」。保存原文SHA `76651d4b4427915de67de39acd2ca5fff80106862824f32e03d456b0a8ada67c` とgrantのbindingが一致|
|review/binding参照|`88e6c4398c81361416700a3f0f900a1c9df2b88a`。同担当の限定技術確認であり、独立GPT科学reviewではない|
|grant/seal保存commit|`d8cdc1a7141d65658308de0fb66a4ed67233023c`。[authorization_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d8cdc1a7141d65658308de0fb66a4ed67233023c/artifacts/resource_applicability/track_a_h6_proc_exit_fix_v1/2026-10-15/rebinding_v3/authorization_v1.json)／[seal_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d8cdc1a7141d65658308de0fb66a4ed67233023c/artifacts/resource_applicability/track_a_h6_proc_exit_fix_v1/2026-10-15/rebinding_v3/seal_v1.json)|
|sealed契約SHA|`1cf99b426aab08f738e6d71e64e89e001941180f2648e37e08307c54997f9db4`。server private契約実bytesと公開auth/sealのSHAが一致|
|認可状態|approved/runtime_authorization/sealed=true、automatic_retry=false、next_stage_authorized=false。今回は読み取りのみ、新grant0|
|CPU/worker/observer|driver62、workers61/60/59/56/49/47、最大6worker/内部thread1。observerは既存metadata launcher roleで独立観測|
|memory/admission|数値各AS32GiB/RSS9GiB、metadata AS512MiB/RSS256MiB、fresh admission240.5GiB・headroom16GiB、監視5秒上限|
|時間/容量|stage wall14,400秒、数値process合計CPU14,400秒。output512MiB/log32MiB/journal64MiB、output10,000inode、fresh FS2GiB/20,000inode。上限であり完了予測ではない|

[後続の修正・起動報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1fa0974064ae4feeb1c8168a88b43f89fc58340/docs/research/track_a_h6_proc_exit_fix_20261015.md)と公開auth/sealを受領した。元bundle内の過去のpush認証失敗記録はその時点の履歴であり、今回のremote確認を理由に原文を書き換えない。

**2026-10-15 16:52:28 JST頃の非atomic読取snapshot**：run ID `h6-native-b0i-pilot6-20261015-run04-v3`、phase `C_COMPILE_ACCOUNTING`、first STOP null、active workers2、partial cost JSON4/6。terminal/pilot_resultsなし。これは現在永続にRUNNINGという断定でも全6軸完了の認定でもない。4件のpartial内容を新たな費用地図へ解析・採択していない。

**後続追補：17:03 JST以降に完了metadataを受領。** terminalはCOMPLETE_MANDATORY_STOP/first STOP null、3構成×両軸の結果6件、同じ親のdriver wait0、全6worker exit0/reaped、cleanup errors[]。launcher wall約2,262.03秒。terminal内native_exit_code=nullを外部wait receiptの0で書き換えず、正確な終了時刻nullも保持した。[公開した軽量受領projection](../../artifacts/resource_applicability/track_a_shared_status_server_update/2026-10-15/h6_completion_metadata_receipt_v1.json)が原private receipt5件のbytes/SHAと、terminalが列挙するcost6件＋cleanup/resultの8件のbyte照合を結ぶ。新費用解析・科学reviewは実施せず、raw runtimeも公開していない。

旧run01/02/03のSTOP、run04-v1/v2のCPU gate未起動、消費済みgrant/carryは保全。run03のworker API entered6はQiskit実transpile6完了ではなく、PASS payload1/受理pilot結果0、他軸actual未記録という限定も保持する。

## 4時間案と将来wallなし方針

[公開済み将来方針](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/735475b3cb11a8214eefcf14826bacd90ed528fa/artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/future_wall_time_policy_v1.json)は、利用者原文「今後は時間の上限をつけなくてよい」を根拠に、将来の新規・別承認契約のphase/total wall上限をnullとする。既存実行は不変、memory/output/call/監視timeout等を維持する政策である。CPU時間上限とwallは別項目。

H6準備依頼§6にはStage wall4時間が「未承認のSTOP上限」候補として書かれ、Fにも案として残った。その後、一般的な実行指示・修正再実行指示を受けた契約は4hをそのまま固定し、現在のsealed契約にもwall_seconds=14,400が実在する。したがって現状を「4hは未採択の古い案だけ」と報告するのは正しくない。

一方、調べた承認原文・公開auth/seal・準備依頼の範囲では、**既定wallなし方針への例外として4hを明示採択した記録は確認できない**。実行scopeの認可は存在するが、wall政策への明示例外の成立は未確認である。`OPEN_POLICY_CONTRACT_DISCREPANCY`として残し、後続契約を扱う担当に返す。将来wall上限なしが既定、既存sealed契約では4hが実効、という二つの事実を混同しない。

本作業で契約・SOURCE・grant・sealを変更せず、時間上限解除・停止・再起動をしていない。今回の記録は4hへの新たな承認でも、その撤廃指示でもない。

## 未取得・未公開・判断保留

- local指示48,223 bytes版の実体・固定commit/pathは未取得。server exact copyと旧公開版との差分は本資料branchに含めたが、local/server原本の直接差分は未確認。
- H6 6軸pilotの全完了・native wait/cleanup metadataは後続受領済み。最終費用の詳しい公開・科学採否は今回未実施。4h例外の明示根拠は未確認。
- H6/H8各24軸native B0+I、H5/H7、固定δ/T比較は科学採択と実行認可を分離。本同期はそれらの新grant・実行準備・研究設計を開始しない。
- 旧H4 run03 a3248055…の公開到達性、R09補助ZIP/script、旧export event未保存等は旧受領記録の未解決を保持し、今回再調査して解決済みとしない。
- NPZ/raw/runtime/checkpoint/ledger/venv・private絶対path/native identity対応はサーバー管理のまま。公開は軽量な照合projectionと資料だけ。母平均・数値certificate・compiler bit再現性・最終energy費用の保留も不変。

## 限定検証と公開・終了

固定6資料とH6後続公開3file、旧wall方針・旧公開指示の11fileをGitHub bytes取得で照合。C/D/Bの既存完了監査metadataと整合を確認した。科学array読込、全checkpoint/installed file再hash、元解析・compile・人工science testsの再実行は0。
変更対象は明示allowlistでstageし、staged diff・新リンク・JSON/bytes/SHA・原本exact性・旧file非変更を検査する。検証は資料照合であり、独立科学review/CI/原計算の再現ではない。[変更file manifestと検証記録](../../artifacts/resource_applicability/track_a_shared_status_server_update/2026-10-15/README.md)を参照。

公開先はHIROMU1015配下の本repository、non-force pushのみ。認証が使えなければ設定を変えずcommitを保全し、最終報告に実際のworktree/branchを入れたpush commandを示す。push成功時はremote SHAと主要fileを再取得して照合する。push未完了を公開済みと表示しない。

**今回の新科学計算0、H8実行環境変更0、既存認可済みjob操作0、新grant発行0、実行準備の追加実装0、原稿改訂0。** 資料公開又はpush待ち報告でMANDATORY_STOP。追加計算・詳細設計・原稿編集へ進まない。
