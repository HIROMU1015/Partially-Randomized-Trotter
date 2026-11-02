# SERVER 2 Phase 1 — 既存科学証拠棚卸し

観測環境 `SERVER_2`。schema `PR-COVERAGE-1`。最終転記snapshot `2026-11-02T23:12:41+09:00`。repository `HIROMU1015/Partially-Randomized-Trotter`。

**8,761行を登録した限定棚卸しであり、独立実験8,761件の認定ではない。** 価格共有の精度行、同じ確認cohortを使う比較、旧解析と後続解析を含む。全root・全raw・全科学条件の網羅、coverageの充足／不足、科学的採択を主張しない。未展開・未アクセス・未確定は本書に明示する。

[INVENTORY.csv](INVENTORY.csv) / [DETAILS.jsonl](DETAILS.jsonl)。長い`missing_fields`と`evaluation`だけを固定仕様の`detail_ref`で一段外出しした。補助DBや承認証明は作成していない。

## 1. 対象root・観測範囲

SERVER 2という呼称は今回のユーザー指定による。内部host名・IPを照合して旧環境との対応を推測していない。既存TSVの実列を読んでから、各rootの存在、所有者一致、originの対象repository一致を確認した。ファイル名から保存先を推測していない。

|公開root ID|今回の読取|観測HEAD|object store依存|
|---|---|---|---|
|S2-ROOT-01|本人所有・origin対応確認、既存Git objectと明示索引を読取|`0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`|common directoryを保持。独立backupの認定なし|
|S2-ROOT-02|本人所有・origin対応確認、既存Git objectと明示索引を読取|`f9cd14984675fc81505e1c9bc7c1b506cdcb2021`|既存alternatesあり。独立backupとはしない|
|S2-ROOT-03|本人所有・origin対応確認、既存Git objectと明示索引を読取|`fc6bd039dc7171d5674ac04f0b3ad427f1b6495b`|既存alternatesあり。独立backupとはしない|
|S2-ROOT-04|本人所有・origin対応確認、既存Git objectと明示索引を読取|`df7beb8588a3866a7dac2f29add4be7493d092db`|既存alternatesあり。独立backupとはしない|

上記root、登録worktreeの既知文書、既存索引が明記した本人所有の保存先360 locatorの存在metadataだけを対象とした。360件はfile/directory参照の件数であり、rawファイル数ではない。公開IDは`S2-CUSTODY-0001`〜`S2-CUSTODY-0360`、対応表は出力cloneの非公開作業metadataに保持。中身の全走査・再hash・復元試験はしていない。

観測は逐次・非atomic。科学worker、他ユーザー、scheduler、別hostを調査せず、SOURCE/lock/markerを操作していない。出力は独立clone・新branch `codex/pr-coverage-inventory-server2-20261102` に限定した。この出力cloneは既存object storeに依存し、独立保全コピーではない。

## 2. 固定仕様と規約の取得確認

|資料|固定commit|確認したGit blob|
|---|---|---|
|[PROTOCOL.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/PROTOCOL.md)|`968218c2b0c6280af6c1c02822603e3312aa410c`|`3ea836949db06d1f077e8594b187840bf9e23971`|
|[INVENTORY_TEMPLATE.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/INVENTORY_TEMPLATE.csv)|`968218c2b0c6280af6c1c02822603e3312aa410c`|`c62cf3c286a539e41d75b5012f9e3da030a8b986`|
|[HOST_INSTRUCTIONS.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e440e5c06853599b32f678c27c01ab817c1df120/docs/research/pr_cross_host_coverage_audit_20261102/HOST_INSTRUCTIONS.md)|`e440e5c06853599b32f678c27c01ab817c1df120`|`c7e886df38cb38464d52cd39777c0b6ade702268`|

指定commitを実在Git objectから読み、GitHub connectorでも取得してblobを照合した。別branch tipを仕様の代用にしていない。HOST_INSTRUCTIONSのSERVER 2ブロックを適用。PROTOCOL表ではPhase 0担当がLOCALだが、**実際のPhase 0担当はSERVER 2だったという今回のユーザー訂正**をここに記録する。Phase 0を再作成せず、固定PROTOCOLは変更していない。

親projectsのAGENTS.md、対象root/worktreeのAGENTS.md、正式共通運用指示`research_project_instructions_unified_2026-10-25.md`（特に§8.1・§8.4・§10）を取得・確認した。mainと一部の既知親階層にはAGENTS.mdがないことも確認。科学SOURCE/input/provenanceと承認の証明を混同せず、sub-agentは使用していない。PROJECT_MAP・研究概要・目的／方法、status/manifest、個別結果の順に参照した。

## 3. 再利用した既存資料

|PROTOCOL §3の入口|取得版|今回の扱い|
|---|---|---|
|PURPOSE|[9a0bc7f17f47 / 研究目的・研究課題.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究目的・研究課題.md)|固定資料取得。索引・条件・原状態への導線として再利用|
|METHOD|[9a0bc7f17f47 / 研究方法・解析手順.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究方法・解析手順.md)|固定資料取得。索引・条件・原状態への導線として再利用|
|SUMMARY|[9a0bc7f17f47 / 研究概要・現状.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究概要・現状.md)|固定資料取得。索引・条件・原状態への導線として再利用|
|MAP|[9a0bc7f17f47 / PROJECT_MAP.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/PROJECT_MAP.md)|固定資料取得。索引・条件・原状態への導線として再利用|
|STATUS|[9a0bc7f17f47 / VALIDATION_STATUS.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/VALIDATION_STATUS.md)|固定資料取得。索引・条件・原状態への導線として再利用|
|MANIFEST|[9a0bc7f17f47 / validation_manifest.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/validation_manifest.json)|固定資料取得。索引・条件・原状態への導線として再利用|
|LANDSCAPE|[9a0bc7f17f47 / pr_repository_scientific_landscape_20261101.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_scientific_landscape_20261101.md)|固定資料取得。索引・条件・原状態への導線として再利用|
|CLAIMS|[9a0bc7f17f47 / pr_repository_claim_evidence_gap_matrix_20261101.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_claim_evidence_gap_matrix_20261101.md)|固定資料取得。索引・条件・原状態への導線として再利用|
|TRACKA|[4591b8d75a2a / track_a_h4_h8_cross_system_evidence_inventory_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h4_h8_cross_system_evidence_inventory_20261017.md)|19系列の区別を保持。E01〜E32の32資料は原索引Git blobと全て一致|
|CARDS|[a9bdcf1d4406 / STUDY_CARDS.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9bdcf1d44061daceffe896637e2c92fcb3353b4/docs/research/pr_algorithm_improvement_landscape_20261023/STUDY_CARDS.md)|70カードの範囲対応を付録Bに記録。既存の不足優先度は継承しない|
|SOURCEINDEX|[a9bdcf1d4406 / SOURCE_AND_PROVENANCE_INDEX.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9bdcf1d44061daceffe896637e2c92fcb3353b4/docs/research/pr_algorithm_improvement_landscape_20261023/SOURCE_AND_PROVENANCE_INDEX.json)|211参照。対象repo182は記録blob一致、別repo29は確認済みroot範囲外。改善研究を自動算入しない|
|ENVLOCAL|[fa55c162fd8f / LOCAL_INVENTORY_REPORT.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa55c162fd8fe82bd8ccd91f5fb7601d8513791e/inventory/local/LOCAL_INVENTORY_REPORT.md)|過去の環境・保管索引として参照。他hostの現在の実在確認ではない|
|ENVA|[fa55c162fd8f / SERVER_A_INVENTORY_REPORT.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa55c162fd8fe82bd8ccd91f5fb7601d8513791e/inventory/server_a/SERVER_A_INVENTORY_REPORT.md)|過去の環境・保管索引として参照。他hostの現在の実在確認ではない|
|TLOCAL|[9a0bc7f17f47 / track_a_local_inventory.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/track_a_local_inventory.md)|過去の環境・保管索引として参照。他hostの現在の実在確認ではない|
|TSERVER|[9a0bc7f17f47 / track_a_server_inventory.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/track_a_server_inventory.md)|過去の環境・保管索引として参照。他hostの現在の実在確認ではない|

IDEで明示された既存`server2_repository_inventory_20261102.tsv`、`remote_branch_cleanup_handoff_372_20261102.tsv`、`remote_branch_cleanup_result_20261102.tsv`を所在索引として再利用。485 branch/root参照と372 cleanup項目は実験数ではない。過去のcleanup記録はdeleted168／excluded204。今回は整理処理を実行していない。付録Cはその記録と現存Git objectの対応であり、remote branchの再削除指示ではない。

既存`inventory_pauli_hybrid_rte.py`のJSON読取時`parse_int=str, parse_float=str`とCSV DictReader/Writer、`document_pauli_jw_saved_summary.py`のliteral転記、`check_validation_manifest.py`のpath/参照検査方針を静的に確認して再利用した。旧runnerの出力schemaは別目的なので実行・importせず、標準ライブラリの限定metadata転記で新schemaへ対応した。

## 4. 登録した科学条件・run/cohort・保存解析版

CSVの`condition_id`は環境内の仮ID。元candidate/run/analysis/入力identityを別列に保持し、仮IDの一致や異なる番号から科学的同一性・独立性を認定しない。SOURCEが未確認なら`UNKNOWN`のままにし、carrier commitへ置換していない。JSON/CSVの数値文字列を転記し、丸め直し・N/G・比・平均・SE・順位の新計算は行っていない。

|登録系列|観測行数|
|---|---:|
|`LONG_CANDIDATE_ACCOUNTING`|3210|
|`LONG_SAVED_CONFIRMATION_COMPARISONS`|59|
|`B3_CANDIDATE_ACCOUNTING`|960|
|`B3_SAVED_CONFIRMATION_COMPARISONS`|120|
|`H2PAULI_CANDIDATE_ACCOUNTING`|55|
|`H4PAULI_CANDIDATE_ACCOUNTING`|725|
|`EVENT_CANDIDATE_ACCOUNTING`|50|
|`H6HYB_CANDIDATE_ACCOUNTING`|40|
|`H4_GEOMETRY_BIAS_PRICE`|1308|
|`H246_COMMON_PF_STATUS`|588|
|`H468_SAVED_PRIMARY_CONFIRMATION`|6|
|`H4_D1_SAVED_NATIVE_PRICES`|9|
|`H4_D2_SAVED_CONTROL_COMPARISON`|10|
|`H4_SIX_LOCAL_ARB`|6|
|`FINITE_RTE_SIGNAL`|1134|
|`PF_C_SIZE_RESULTS`|4|
|`PF_C_STATE_ACTION_RESULTS`|5|
|`PR2_M1`|210|
|`PR2_M2`|5|
|`RPE4_PHYSICAL_SIGNALS_AND_EXACT_PROBABILITIES`|4|
|`RPE4_EXPLICIT_FRESH_IID_TRAJECTORY_BATCH`|8|
|`M06F_DIRECT_COMPILED`|43|
|`M06F_MODEL_ROUND_ACCOUNTING`|74|
|`H2_FIXED_PILOT_run01`|2|
|`H2_FIXED_PILOT_run02`|2|
|`H4_C_CORRECTED_CANDIDATE_AXES`|40|
|`COMPILED_COST_RTE_BOUNDARY_COST_VALIDATION`|6|
|`COMPILED_COST_RTE_ORDER_STRATIFIED_COST_VALIDATION`|4|
|`COMPILED_COST_RTE_CONNECTED_CLUSTER_COST_VALIDATION`|37|
|`H6_NATIVE_B0I_SAVED_AXES`|18|
|`H6_FIXED_DELTA_SAVED_ACCOUNTING`|15|
|`H8_ORIGINAL_SAVED_PRECISION`|4|

DF legacy rank12とtol-only DF10/19/28、H6 weighted-Hermitian completion、DF fragment prefixとPauli項prefixを統合していない。DF B3は一体項を決定論的に保持し、Pauli純RTEと同一方式にしていない。有限時間C/N/G、短RPE、全roundのモデル費用は別系列。`C_PF_COEFFICIENT`と`C_COMPILED_COST`、trajectory nとshot Nを区別した。

価格取得済みの長時間/B3表はprecisionごとに別行。共有価格cohortは`REUSES`候補を付けた。H4 geometryはbiasと保存平均価格の観測で、反復する302精度投影は今回行へ展開していない。M06-Fはdirect compiled価格とmodel/proxy round会計を分離。局所Arb6件・C候補40軸の正式未受理を保持し、技術PASSやTARGET_METから正式受理へ格上げしていない。

## 5. 保全archive・旧branch/worktreeとの対応

485参照のHEAD objectはそれぞれ指定されたrootで全て読取可能だった。現存worktreeの一覧だけで研究全体を定義していない。SOURCE、原結果、旧STOPは固定Git履歴に保持され、branch名の有無を証拠消失と扱わない。cleanup記録のkeeperは過去の所在情報として保持し、今回keeper選定や祖先関係の再監査は行っていない。

`S2-ARCHIVE-01`は既存索引に明記された旧H6環境exporter zip。本人所有・現存、central directoryの4 memberを確認した。これはcode/exporter保全であり、独立した科学結果や全rawのbackupではない。展開・decode・新hash・復元試験は行っていない。その他のarchiveは元catalogの参照を残し、全archiveの存在・完全性を新認定していない。

選定したH4既存証拠directoryのGit treeでQPY/NPZ1274 locatorのmetadataを確認した。内容のload/decode、回路build/compile、QPY再生成は行っていない。例えば以下は保存programの所在であり、今回の回路検証結果ではない。

|role|固定保存先|照合範囲|
|---|---|---|
|QPY|[artifacts/non_df_h4_a_axes/2026-10-24/run01/trajectories/PF/input.qpy](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_a_axes/2026-10-24/run01/trajectories/PF/input.qpy)|Git tree blob `6a027d6bdd6fdaa3e7fe00c2213316156cce26cc`; 内容未読|
|QPY|[artifacts/non_df_h4_a_axes/2026-10-24/run01/trajectories/PF/output.qpy](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_a_axes/2026-10-24/run01/trajectories/PF/output.qpy)|Git tree blob `1cb58f5a6a082ffe7d85ab3966c0e08319038850`; 内容未読|
|QPY|[artifacts/non_df_h4_a_axes/2026-10-24/run01/trajectories/PR0000/input.qpy](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_a_axes/2026-10-24/run01/trajectories/PR0000/input.qpy)|Git tree blob `cfe0cbac17b57d5adfa6520aa30b1260c1ba9b0c`; 内容未読|
|QPY|[artifacts/non_df_h4_a_axes/2026-10-24/run01/trajectories/PR0000/output.qpy](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_a_axes/2026-10-24/run01/trajectories/PR0000/output.qpy)|Git tree blob `cf0a8571200a6bea0b6a12eea78d4f391d0ef005`; 内容未読|

原本のprivate raw・input NPZ、既存grant/ledger/marker/STOP、archive本体は変更していない。公開Git tableの読取可能性と、private原本の完全性・独立backup・復元性は別の状態である。

## 6. 旧環境alias

`SERVER_A`、`SERVER_B`、10/14のlocal/server、今回の`SERVER_1`/`SERVER_2`の相互対応は`UNKNOWN`。旧索引に同じbranch/OIDがあることは対応候補の根拠に留め、host同一性を確定しない。今回のCSVのenvironmentは観測担当SERVER_2であり、各runの実行hostではない。実行環境が原表に結合されていない行はsampling.execution_environment=UNKNOWN。LOCAL/SERVER 1へのSSH・転送・直接調査は0。

## 7. 重複・独立確認・継続・失効の候補

|関係候補|根拠と限定|
|---|---|
|同candidate価格の精度間REUSES|元表のcandidate ID・cohortと報告上の共有を保持。同じ価格標本の再利用であり精度行数を独立nへ加算しない|
|長時間nomineeの確認|保存confirmation_seed_independenceにseparate_cohort=true、complete_historical_coverage_claimed=false。既知seed照合と全履歴除外を区別。対応はRECORDED_ONLY|
|B3対B2/PF比較|旧baselineをCITES。baseline_uses_independent_confirmationの元値を保持し、比較rowごとにbaseline標本を新規実験として数えない|
|H2 run01→run02|元run01のPauli PF/DF PR成功と、Pauli PR/DF PF失敗を区別。後続2caseは別SOURCEのRECOVERS。旧成功caseの再引用は新runではない|
|H4 C失敗→修正版|旧reference実装欠陥runの失敗/STOPを保持。修正SOURCE e80d7510…の40軸TARGET_METはcandidate証拠で正式未受理。旧nullを0に変更しない|
|H8主表とMechanism alias|固定A6 tableに記録されたcohort/aliasを保持。別文書・別branchへの引用を独立確認として加算しない|
|Arb6件|各summary行を元result_summaryとSTOPへ接続。OLD_QUOTED_NO_REEVALUATIONは追加state-actionではない|

旧DF screeningのinvalidated_pending_regeneration、UWCのprose_only_missing_raw_artifactsは元manifestの状態。未知の条件を埋めた実験行は作らず、付録B/Dの履歴・未解決へ残す。後続の日付だけを理由に旧結果全体を失効させていない。3環境canonicalな重複同定はPhase 2であり今回は行っていない。

## 8. 未展開・未発見・未アクセス・未解決

以下は取得・正規化の状態であり、科学的coverage不足の判定・優先度ではない。

|項目|状態と限定|
|---|---|
|長時間PF/PRの58,160精度行|原表は取得可能。native_cost_status=ACQUIREDの3,210行を登録。他の行のsignal/N等は原表に保持されるが今回個別転記未展開。未実施と判定しない|
|B3の17,160精度行|原表は取得可能。ACQUIRED960行を登録。その他16,200行は固定表に残る。未取得compiled価格と既存signalを区別し、空Gを0にしない|
|H4 geometry精度投影|1,308 candidateのbias/保存平均価格を登録。302点投影、B0I追加版、全precision再会計は原表／索引へ残し、同一価格標本の多数精度を独立確認にしない|
|Track A19系列の旧条件表|E01〜E32の原文書blobは確認。全旧grid・rawの行展開はしていない。個別の対応を付録Aに明記|
|PF係数の個々のdelta・初期cost/RPEの全版|代表サイズsummary・保存holdout・M06-F/短4段を登録。他のmanifest result_setは索引へ対応付けるが全個別条件の転記は未解決。C_useの厳密上界認定はしない|
|H4 Pauli SOURCE.json / event拡張 SOURCE.json|指定result commitの当該候補pathではNOT_FOUND。active_scope等の実在metadataへ切替。SOURCE不存在とは断定しない|
|private原本とarchive全体|360明記locatorの存在・本人所有を確認。全raw identity、全backup、復元性、archive全memberの意味は未照合|
|別repositoryの29 SOURCE索引項目|確認済み対象root外。現環境の別repo所在を探索していない。対象repoのbaselineへ自動算入しない|
|旧SERVER_A/Bと現在のhost|UNKNOWN。別環境の現物確認・環境間同一性判定は未実施|
|SOURCE・run・state/reference・compiler等の欠測|選定表/reportで未記録ならNOT_RECORDED、結合未確定ならUNKNOWN。原表に同じ語があっても意味を推定して埋めない。missing_fieldsは選定参照範囲に限定|

B3のlocal最終`f794a2aa…`と公開carrier`ebe9e2aa…`の差は既存landscape/claim表が記録する論点として保持。今回は公開carrierの表を転記し、private圧縮rawのlocal/remote一致や独立復元を主張していない。旧archive/rawの新decode、RNG再生、新hash全面走査が必要な処理へ進まない。

## 9. 非公開情報

内部host/IP・絶対サーバーパス・資格情報・private rawは公開コピーに含めない。公開Git参照、科学SOURCE、既存input/result identityだけを保持した。絶対locatorを含む原fieldはREDACTED＋不透明IDへ置換し、対応は非公開作業metadataで保持する。実行seedのprivate値を新公開していない。文書に公開済みのseed metadataと、非公開seed一覧は区別した。元TSVやprivate対応表はcommit対象外。

## 10. 技術的整合性確認

主agent自身のmetadata検査で、固定43列・順序、全行environment/schema、必須JSON key、状態enum、ID一意性、10,176件のdetail_ref一段解決、8,761件の固定結果selectorを確認した。5,869 CSV原行・192,930fieldは元文字列と照合。JSON selectorは原文書内で解決し、科学値の再評価はしていない。参照ID/role/locatorの整合と公開情報の混入検査はPASS。

32件のTrack A文書blob、182件の対象repo SOURCE索引blobも既存値と一致。これは存在・metadataの確認で、全科学SOURCEの再監査や独立科学レビューではない。変更対象はINVENTORY.csv、REPORT.md、DETAILS.jsonlの3ファイルだけ。研究概要、PROJECT_MAP、status/manifest、科学コードは変更しない。

通常pushはHTTPS認証情報がなく失敗した（terminal prompts disabled / could not read Username）。remoteに本branchは未発見で、3ファイルは未公開。remote SHA・固定公開本文照合は未実施。接続済みGitHubツールには大きいローカルファイルを直接uploadする手段がないため、REPORTだけの部分公開はしていない。自己参照になるため、このREPORT自身へ最終commit SHAを埋め込まない。公開・照合の完了条件は未達である。

## 11. 原本・STOPの保全

証拠treeではGIT_OPTIONAL_LOCKS=0、GIT_NO_LAZY_FETCH=1でshow/ls-tree/cat-file/rev-parse等の読取だけを行った。checkout/reset/merge/clean/gc/prune、branch/worktree/archiveの削除や再整理は0。出力cloneの作成・新branchでの成果物commit/pushを原本操作と区別する。SOURCE/input/raw/result/grant/ledger/marker/STOP、既存one-shot/retry境界を変更・再開していない。

原runの失敗・STOPと、今回のmetadata検査PASSは別物。科学的PASS、技術的PASS、実行完了、正式受理、失効をCSVの別列に保持した。確認できない状態を推定してPASSにしない。

## 12. Mandatory STOP

**現在は公開待ちSTOP。棚卸し成果物の作成・技術検査・local commitは完了したが、HTTPS認証不足でpushできず、SERVER 2 Phase 1の公開・照合完了は未達である。** 通常pushとremote SHA・公開本文照合が完了した場合もmandatory STOPする。未展開・未解決項目は上記のまま保留する。Phase 2統合、Phase 3 coverage評価、不足A〜Eの分類・優先度、詳細科学レビュー、主要claim採択、新科学計算へ自動移行しない。

次の担当はユーザー：GitHub認証を利用できる環境から本branchを通常pushする。Codex：公開後に同じ許可範囲のremote SHA・3ファイル本文照合を完了してmandatory STOPする。Phase 2/3や新科学計算は開始しない。

## 付録A. Track A既存19系列の対応

|旧study ID|今回の対応|原索引|
|---|---|---|
|H4_GEOMETRY|H4_GEOMETRY_BIAS_PRICEへ登録（精度投影は未展開）|E01;E03|
|H4_GEOMETRY_B0I|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E02;E03|
|H4_NATIVE_ANCHOR|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E04;E05;E24;E03|
|H5_NEUTRAL_PILOT|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E06;E27;E32|
|H6_CONFIRMATION|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E07|
|H6_NATIVE_B0I|H6_NATIVE_B0I_SAVED_AXESへ登録|E08;E07|
|H6_STRENGTHENED136|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E09;E11|
|H6_P7_MEASURED_PAIR|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E10;E09|
|H6_FIXED_DELTA_T|H6_FIXED_DELTA_SAVED_ACCOUNTINGへ登録|E12|
|H6_RARE_SAVED_AUDIT|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E13|
|H6_CONDITIONAL_S2|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E14|
|H7_NEUTRAL_PILOT|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E15|
|H8_ORIGINAL_MAP|H8_ORIGINAL_SAVED_PRECISIONへ登録|E16|
|H8_D0|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E17;E16|
|H8_PHASE_C_SIGNED|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E18;E25;E26|
|H8_NONCANCELLATION_COST|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E19;E25;E26|
|H8_KEY_SIX_AXES|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E20|
|H8_CURRENT_SIGNED|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E21;E22;E23;E20|
|H8_CURRENT_NONCANCELLATION|原資料索引を確認。元grid/条件行の全面展開は未解決；後続の別条件へ同一視しない|E21;E22;E23;E20|

Track A E01〜E32は以下の固定資料へ到達可能で、原catalogのblobと一致した。

|証拠ID|固定参照|
|---|---|
|E01|[docs/research/track_a_h4_geometry_results_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md)|
|E02|[docs/research/track_a_h4_b0i_map_results_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md)|
|E03|[docs/research/track_a_resource_comparison_conditions_20261014_v0_4.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47dc17a83e2fc9216e22b59317969952ef79028a/docs/research/track_a_resource_comparison_conditions_20261014_v0_4.md)|
|E04|[artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/README.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/README.md)|
|E05|[artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/result_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/result_summary.json)|
|E06|[docs/research/track_a_h5_neutral_open_shell_pilot_result_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bde58ba23f1fecdf77ee950795cb80c199c0f86b/docs/research/track_a_h5_neutral_open_shell_pilot_result_20261016.md)|
|E07|[docs/research/track_a_h6_independent_confirmation_result_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/research/track_a_h6_independent_confirmation_result_v1.md)|
|E08|[docs/research/track_a_h6_native_b0i_full_family_results_20261015.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/docs/research/track_a_h6_native_b0i_full_family_results_20261015.md)|
|E09|[docs/research/track_a_v1_reaccount_results_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_reaccount_results_20261016.md)|
|E10|[docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md)|
|E11|[docs/research/track_a_v1_phase_b_run01_results_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_phase_b_run01_results_20261016.md)|
|E12|[docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md)|
|E13|[docs/research/track_a_h6_rare_order_cost_reliability_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md)|
|E14|[docs/research/track_a_h6_conditional_s2_pilot_result_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/docs/research/track_a_h6_conditional_s2_pilot_result_v1.md)|
|E15|[docs/research/track_a_h7_neutral_open_shell_pilot_result_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/162c256ea39b4ab3dadde65a94b3eec73cea4720/docs/research/track_a_h7_neutral_open_shell_pilot_result_20261016.md)|
|E16|[docs/research/track_a_h8_resource_map_result_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md)|
|E17|[docs/research/track_a_h8_d0_saved_value_result_20261015.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_saved_value_result_20261015.md)|
|E18|[docs/research/track_a_h8_phase_c_run01_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086/docs/research/track_a_h8_phase_c_run01_results_20261017.md)|
|E19|[docs/research/track_a_h8_noncancellation_cost_run01_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23d46e489af054735d9d328aad0342ecdbddf305/docs/research/track_a_h8_noncancellation_cost_run01_results_20261017.md)|
|E20|[docs/research/track_a_h8_key_compile_completed_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce6cd8ead5c0311ec72165c04869e7df4050b75b/docs/research/track_a_h8_key_compile_completed_results_20261017.md)|
|E21|[docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md)|
|E22|[artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/comparison_summary_exact.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/comparison_summary_exact.csv)|
|E23|[artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/final_gate.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/final_gate.json)|
|E24|[artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/signal_metadata.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/signal_metadata.json)|
|E25|[docs/research/track_a_h8_b2_completed_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fef2db7a79886ab682b652da7266b463c98c7d3a/docs/research/track_a_h8_b2_completed_results_20261017.md)|
|E26|[docs/research/track_a_h8_phase_b1_run01_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/83a3c4a6688e66576f85471bf41a45e29cbfb705/docs/research/track_a_h8_phase_b1_run01_results_20261017.md)|
|E27|[artifacts/resource_applicability/track_a_h5_neutral_open_shell_pilot_v1/2026-10-16/input_state_signal_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bde58ba23f1fecdf77ee950795cb80c199c0f86b/artifacts/resource_applicability/track_a_h5_neutral_open_shell_pilot_v1/2026-10-16/input_state_signal_v1.json)|
|E28|[artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/summary.json)|
|E29|[docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md)|
|E30|[docs/research/track_a_h6_s2_pilot_followup_scientific_review_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/docs/research/track_a_h6_s2_pilot_followup_scientific_review_20261013.md)|
|E31|[docs/research/track_a_h4_b0i_v04_scientific_review_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/docs/research/track_a_h4_b0i_v04_scientific_review_20261014.md)|
|E32|[docs/research/track_a_h5_neutral_open_shell_readiness_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/803a24ef7a01d8c7394dde88357c74168795bb34/docs/research/track_a_h5_neutral_open_shell_readiness_20261016.md)|

## 付録B. 70研究カードの範囲対応

下表は研究目的に対する対象／除外の索引であり、coverageや研究の優先順位ではない。改善方式自体は除外し、直接関連する既存baselineが今回登録済みかを区別した。

|カード|扱い|今回の参照状態|
|---|---|---|
|REP-01 Pauli groupingとDF回路基盤|baseline/基盤参照候補|改善方式・別repo全体は除外。原カード/SOURCE索引の対応は保留し、自動算入しない|
|REP-02 旧DF rank screening|対象の失効履歴|元manifestの失効状態を保持。条件未確定の新行は作らない|
|REP-03 BLISS型UWC＋grouped PF|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|PF-01 Yoshida・Morales・新4次係数の比較|baseline/基盤参照候補|改善方式・別repo全体は除外。原カード/SOURCE索引の対応は保留し、自動算入しない|
|PF-02 Processed PF／YP8m8|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|PF-03 PF固有位相・D6係数とsurvival代理量|対象の既存検証|関連表をCSVに登録。全旧条件の展開は付録A/Dと§8の限定を保持|
|PF-04 有限時間校正・第一研究S0–S4|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|PF-05 有限時間校正・第二研究Direction C|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|PF-06 FS-C旧入力回収とFS-R1校正follow-up|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|PF-07 BCH高次項・response・direct-scaling|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|CIR-01 WP06-a/b full/support-union基底計画|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|CIR-02 P-A interval-union DP|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|OPT-01 round別r/K・誤差/失敗確率配分|対象の既存検証|関連表をCSVに登録。全旧条件の展開は付録A/Dと§8の限定を保持|
|OPT-02 P-B signal-aware selection|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|OPT-03 P-C geometry tracking／energy差|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|OPT-04 P-D energy/tail-aware高次PF選択|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|OPT-05 R3 split／formulaのmulti-fidelity選択|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|ERR-01 FR 位相・半径分離|baseline/基盤参照候補|改善方式・別repo全体は除外。原カード/SOURCE索引の対応は保留し、自動算入しない|
|PR-01 PR-2 圧縮＋残差random補完|対象の既存検証|関連表をCSVに登録。全旧条件の展開は付録A/Dと§8の限定を保持|
|PR-02 PR-3 tail限定Richardson|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|PR-03 PR-1/4/5/6初期候補群|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|EVAL-01 Track A PM-1/PM-2 precision資源地図|対象の既存検証|関連表をCSVに登録。全旧条件の展開は付録A/Dと§8の限定を保持|
|EVAL-02 Track A H4–H8強対照と希少tail|対象の既存検証|関連表をCSVに登録。全旧条件の展開は付録A/Dと§8の限定を保持|
|EVAL-03 PR高次PF探索と次数逆転の既存証拠|対象の既存検証|関連表をCSVに登録。全旧条件の展開は付録A/Dと§8の限定を保持|
|RB-01 B-F finite-RTE係数設計|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-02 B-M DF multirate／compact BCH|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-03 BS block合成・SP synthesis placement|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-04 R0/R1 finite-mean再配分|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-05 RA-D0 LP・G1–G5固定辞書|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-06 G6 Return Aggregation／General Full|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-07 G7 conditional native economics|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-08 G8 on-demand synthesis／provider budget|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-09 G9 Closed P5 matched-native|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-10 G10 degree-matched native比較|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-11 DF希薄三次補正・generic whole|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-12 Gaussian積生成|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-13 D1–D3 finite-target・境界相殺・保証価格|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-14 Parent allocation／S4 KKT|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-15 M1-S shared ± labels|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-16 G6 finite-bit／B3 live API・古典費用|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-17 C1 G6–DF移植・H4 logical pilot|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|REP-04 N1 representation selector|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|REP-05 N2 signed charge・L1 fit|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|REP-06 N3 occupation／active-energy certificate|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|REP-07 THRIFT／mixed relative evolution|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-01 C1 macrocycle／符号相殺|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-02 C2 branch付きPF外挿|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-03 C3 self-energy／active space|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-04 C4 cost-aware proposal／whole path|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-05 一体再配分・exact-ground PF応答|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-06 H-E1・M-a/M-b PF小改変|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-07 SOS-PF生成・K0/K0.5|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-08 A2 RF0/RF1/EW2有限step相対保証|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-09 DF covariance縮約|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|REP-08 非DF Pauli対DF fixed-signal比較|対象の既存検証|関連表をCSVに登録。全旧条件の展開は付録A/Dと§8の限定を保持|
|EXT-01 P1* bus/CAR固定word|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|EXT-02 J1 joint-loop・echo構成|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|EXT-03 Q1 integrable level insertion|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|EXT-04 集団遷移・pair collision・R1-P|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|EXT-05 G coherent adaptive subspace|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|EXT-06 固定条件付き測定|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|EXT-07 B-01 unknown-gap／energy情報|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|EXT-08 T1 logical accreditation・独立原理P1–P8探索|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|REP-09 core吸収・physical pair表現|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|RB-18 interstep correlated randomization|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-10 H1 先導固有値補正|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-11 H2 mixed correlation／filter|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-12 H3 frame／interaction picture|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-13 H4 copies／phase reference|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|
|QC-14 exact fermion encoding／controlled構成|範囲外：改善方式・独立提案自体|カードへの導線のみ保持。直接必要なbaselineの未結合を不存在とはしない|

## 付録C. 既存branch/root索引とGit objectの対応

下表は485所在参照をそのままroot別に保持する。同じ実験のbranchコピー、文書更新、準備・STOPは別実験へ加算しない。cleanup列は2026-11-02既存記録とのbranch/OID文字列対応だけで、今回のremote実在・削除や新しいkeeper妥当性の認定ではない。

|root|branch|元HEAD・現存object|既存cleanupとの対応|科学条件の扱い|
|---|---|---|---|---|
|S2-ROOT-02|`track-a-codex2-h6-higher-pf-confirmation-native-readiness-20261018`|`6e197973d4d4183d4e27b1b8cc95cb02f311bb92` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-a-h2-higher-pf-outer32-execution-20261021 @ a05f4ed19e87e1eb65f7fc346385b5b4eae69845|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h2-higher-pf-outer32-execution-20261021`|`a05f4ed19e87e1eb65f7fc346385b5b4eae69845` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h2-m5-compiler-phase-guard-20261020`|`5d0be1699dcef8f09f3bdcd8bb00511640571809` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h2-m5-existing-pf-comparison-20261020`|`cbdf57ba0509542193abca5fb7199d53e06b09b5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h2-m5-existing-pf-comparison-fix-20261020`|`ffb62a445c22e01e1ccd832463407e2dec1c2970` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h2-m5-saved-integration-20261020`|`c380397c5a0dd75da24d8a16d5bfcc598218e0fc` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h2-m5-saved-integration-fix-20261020`|`2d415da2b09a73fab5580eded80e822fc44a25e1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h246-higher-pf-confirmation-readiness-20261021`|`c7c1d8229508253390212588ae2b25e504eb4915` / GIT_OBJECT_PRESENT|元OID一致：excluded；keeper codex/h46-long-time-b3-20261026 @ ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h246-higher-pf-confirmation-run01-20261021`|`0d3578a4584a924ed10bdd124db6093e02d90c94` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h246-higher-pf-pilot-evaluation-20261020`|`bc09510b488bcb1e69fb7579b26dcae1459a8810` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-a-h2-higher-pf-outer32-execution-20261021 @ a05f4ed19e87e1eb65f7fc346385b5b4eae69845|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h246-higher-pf-saved-evaluation-20261021`|`d86fa94ae52acdbf4880a74e23689e380f28489e` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h246-higher-pf-saved-evaluation-run01-20261021`|`8613b3ecda2030a677b1690fe7aa3da23dec4873` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-higher-pf-dataset-protocol-integration-20261020`|`489422d38bad4d5892cf98c580d93cc4b602bf0c` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-a-h2-higher-pf-outer32-execution-20261021 @ a05f4ed19e87e1eb65f7fc346385b5b4eae69845|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-higher-pf-pr-design-pilot-readiness-20261020`|`3e7365a606e18efa585252eef23f7b13264b0f38` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-a-h2-higher-pf-outer32-execution-20261021 @ a05f4ed19e87e1eb65f7fc346385b5b4eae69845|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-m5-best-parallel-readiness-20261019`|`f9cd14984675fc81505e1c9bc7c1b506cdcb2021` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-a-h2-higher-pf-outer32-execution-20261021 @ a05f4ed19e87e1eb65f7fc346385b5b4eae69845|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-m5-compile-stop-static-20261019`|`232ec931e2ebad93e85591205b888795223c828c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-m5-cost-sampling-readiness-20261020`|`5764aacb0714c6be42cb6dc7c7d8fdf1c8d98bdb` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-a-h2-higher-pf-outer32-execution-20261021 @ a05f4ed19e87e1eb65f7fc346385b5b4eae69845|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-m5-minimal-comparison-plan-20261019`|`7415e79edf365fcfb60750f1872722e4238a1950` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-m5-protocol-alignment-20261019`|`36e3284f7c0ea864f443e1347a5670df287c8011` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-m5-signal-empty-tail-fix-20261020`|`a960a41da5ebc564b0306e798572257091c8509c` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-a-h2-higher-pf-outer32-execution-20261021 @ a05f4ed19e87e1eb65f7fc346385b5b4eae69845|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-m5-signal-readiness-20261020`|`d09497961a9da5092bec0a075a32cb84295abfc3` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-m5-tol-native-readiness-20261019`|`b38898429af3a2a48d7057d9d828256c057685f8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-pr-design-pilot-execution-20261020`|`2340db01ddcb0493ad7ecbeb1c2884f99691303d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-pr-exploration-execution-20261021`|`605ca07a653241db49348fbd7f4e091f9f18ced7` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper codex/h46-long-time-b3-20261026 @ ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-pr-exploration-readiness-20261020`|`2fcdec15341c0b3371b37ab5d5eb5bec1009de54` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-a-h2-higher-pf-outer32-execution-20261021 @ a05f4ed19e87e1eb65f7fc346385b5b4eae69845|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h46-pr-pilot-storage-optimization-20261020`|`3d0b5477ce04604ce873c079d9c66d02f2782239` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-a-h2-higher-pf-outer32-execution-20261021 @ a05f4ed19e87e1eb65f7fc346385b5b4eae69845|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-higher-pf-confirmation-recovery-run02-20261021`|`8d35e7bfd3e18b76f395e98b5c809e69ec11e4c1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-algebraic-zero-run01-20261019`|`f1093ed1d82e8bd79932b6d072ca2c2146f53db5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-ct01-sine-readiness-20261019`|`81990c6cece6160f3fdea491dd524106b234d114` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-ct01-sine-run01-20261019`|`16e7d1b63092532f2ad0227ca1598285a782fa9f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-full-recovery-20261019`|`282cba7d6047ea6613a9e69c818e467cb962dd65` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-phase-batch-run01-20261019`|`5b90b5e5a77e85b296a95745f17381f5769693a2` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-phase-ledger-run01-20261019`|`3f0ce05668b1f85249b45269ba087b6e2edf7b55` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-phase-ledger-static-20261019`|`fe7f8a38191d4f8026d45e24bc7aca8c5dbb4fc4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-phase-repair-design-20261019`|`39922872592c438d53cd1f3300e52864afad37dd` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-phase-repair-run01-20261019`|`32bd4e571c250c77f1134abd8395aeba218f8703` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-postrepair-phase-static-20261019`|`5050177a414f263e1c391c0980f2d9dee99fa923` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-preserve-oneq-run01-20261019`|`27a5ef64eefebac6edae3c4042f0b6b1a30dbc9b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-saved-phase-plan-20261019`|`a60323947cf39b98618b948093d98568c9a1fa10` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-saved-phase-readiness-20261019`|`3484fc233733e66b4056616d1e04926162a00354` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-m5-saved-phase-run01-20261019`|`71481e485772c0e8d0af657bacffd846bc570483` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-pr-pilot-continuation-20261020`|`99fecd0d1164f555f2f85667bba625ec33c7632e` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-h6-pr-pilot-cutoff-repair-20261020`|`e12fcb8afb4dd3f5bde6672d7b8e1c46823f0aed` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-m5-algebraic-oneq-design-20261019`|`3d11a0122a2a613f3083cb7dd97bf2b383501186` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-m5-local-resynthesis-diagnostic-20261019`|`79ba3fd1d5fa55f7035aee045dfd0bcbc94dbfae` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-pr-resource-protocol-v1-adopted-20261019`|`9036d8a87ad1c38e83b9ff654b412dde7041e9d3` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-pr-resource-protocol-v1-draft-20261019`|`9c54d89f6154525e3c7c814a4da5ecdc37453b8a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-02|`track-a-protocol-compile-tolerance-20261019`|`94d18a979cdc1a390fa7e2eb3ae2f18f700b47d5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-h8-confirmation-indexes-local-20261019`|`6b2261ec3a75e838122be3ea4cd9b6880947802c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-h8-confirmation-parallel-resume-20261018`|`e1fcd9bcbd307e4a45dc1f61edf6e3e611dbb99d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-h8-confirmation-provenance-local-20261019`|`659e555714c8ccf3a6ec7fd1cb0dbf05cfcc342d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-h8-confirmation-result-detail-local-20261019`|`96aa3878a344f479ee064118cb90002a770db627` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-h8-confirmation-result-only-local-20261019`|`00c28938c7e209b5a4e98bf24c1762085a3c21cd` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-h8-confirmation-results-public-20261019`|`e6eea5387a6e84d1e39a9adb3b61975acc74e200` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper codex/h6-pr-confirmation-science-20261019 @ 3cccfcb2051dacf29e91473e2e1152acef1574bd|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-h8-confirmation-table-provenance-local-20261019`|`a0d3503d5409c2f12289d198929b7fd4cc409682` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-h8-new-pr-confirmation-design-20261018`|`434e67a8bb205711d22d5bf1aa5d143fd08b7765` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-h8-new-pr-confirmation-execution-20261018`|`d21647d07bc86343b292b1e1154ba05f887b1593` / GIT_OBJECT_PRESENT|元OID一致：excluded；keeper codex/h6-pr-confirmation-science-20261019 @ 3cccfcb2051dacf29e91473e2e1152acef1574bd|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-h8-pr-candidate-parallel-readiness-20261018`|`f2494ede329e557ecda728100c48431521b3bfe5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-pr-cost-batch-execution-20261018`|`1e3c8450af50f662c6d2f3451d09714cfebcba4a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-pr-cost-batch-results-20261018`|`fc6bd039dc7171d5674ac04f0b3ad427f1b6495b` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-a-protocol-compile-tolerance-20261019 @ 94d18a979cdc1a390fa7e2eb3ae2f18f700b47d5|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-pr-new-a-accounting-local-drafts-20261018`|`47b30e3d7abccbdb38341935c81ecac8b7a56c91` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-pr-new-a-accounting-readiness-20261018`|`37c3357bd156f7b91179e14e257be088de40c413` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-pr-new-a-accounting-review-local-draft-20261018`|`a04ee266afe76bf92308e46ccd4a2a329ce9f0bc` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-pr-new-a-total-cost-result-local-drafts-20261018`|`7f313820a22b7abee356b771e4fa017217ece34b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-03|`track-a-pr-new-a-total-cost-results-20261018`|`64c85f7a833b5950ef2ccae623708b73081edcd6` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-04|`main`|`0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-04|`track-a-final-technical-audit-20261019`|`df7beb8588a3866a7dac2f29add4be7493d092db` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/df-pauli-evidence-paper-plan-20261026`|`6e3436a16b13e539b0faa2a0a3286f6ab3549543` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-accepted-compare-20261023`|`4008ce45d26bb858e092d8654187bfcda2dbe180` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-archived-exploratory-ready-20261023`|`9e5ea6b235d7beecc02651b1393b561948e85789` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-common-native-allowance-20261023`|`b18a9ce5cfa824d68a88f2d8050795692ed32ca4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-common-native-allowance-local-draft-20261023`|`e66af14f0d65c6db0067eb39ee28df3511f59d33` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-common-native-allowance-pre-publication-20261023`|`df689fdce88b133829088eb044a66f07e2dad69a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-common-production-20261023`|`d405b880a679ee4a95d233f5535c4e3bf2db53f4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-common-production-local-docs-20261023`|`db2e210d42e2d0ff9e221304d1587e5aa9e7dd88` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-common-production-local-provenance-20261023`|`510bf0cf99a2dba2bed0625ef8a3c7ad9f458f16` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-common-production-local-source-20261023`|`2011bfec6279922d1337edae80a74b43a6b28445` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-diagnostics-binding-20261023`|`a6788bf710c16e179a644263952c19711c64a2b2` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-outer32-metadata-export-20261023`|`e3531956af6a6d359e2f1fb320e9a5179f41eb98` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-pauli-rte-confirmation-extension-20261026`|`4aea054b3369709bc0d35ff081071e62b5f77cbe` / GIT_OBJECT_PRESENT|元OID一致：excluded；keeper codex/h4-p3-poststage3-design-20261102 @ 60515a33ce6231e9ff4ffdb40124ed845373351c|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-pr-diagonal-reuse-a-20261023`|`43be11d648ca35302bd2c7267e23327fe37e34cb` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-pr-equivalence-contract-20261023`|`65085ff7f818fd92af3424b1980f6bd9dbf1c742` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-pr-identity-recovery-20261023`|`e7a63b2b06a124f9305483aa39bcf1ba8cafdd0d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-pr-original-identity-20261023`|`588d19544a2afdfb55ec4b40475a58114bd47b0b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-pr-resource-preparation-20261019`|`f2d1a86c431f835db700f348ce618d8604e7ee23` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-protocol-v2-update-20261019`|`3826f566b04ad9c2ead9abf1b80c864f7b34b785` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-results-integration-20261019`|`8fbe7d77556c2353be837dbd9ed0fddc02ee81c4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-results-integration-local-20261019`|`25e6f2c91f9e18f08c6c92e4494f3ee0bd1aaf6a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-runtime-cost-reuse-20261023`|`ca29dd9ddc8a09cace84310f0dfdf7124652e07a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h2-saved-cost-static-join-b2-20261023`|`041511aa567c288bcead32d0bf7579fd7f434e44` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h246-order-cost-analysis-ready-20261024`|`d2536c4f959bfb4edd7d2deeceb155585fb53eb0` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h246-prior-art-paper-options-20261024`|`8a2fbe9461d9cc262326fa30f9778b295adeb3f8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h246-reversal-batch-analysis-20261024`|`eebdbfcbd0d5f5608d50bfd91c9cd526c68f0d23` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h246-reversal-evidence-controls-20261024`|`364970df33859e108d90b6ceaacde467db969fb8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-d-preparation-fix-20261019`|`4e8ad1936363266d1b55bb689c36846e6c3dfcdb` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-dp-phase0-20261019`|`504f3b65023e9f3f3497b3881878b90be154d417` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-four-method-exploratory-ready-20261024`|`868ee7ced6818b2531a3aa9c12498f02bce55dc3` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-mc-sample-size-20261019`|`defa5f6dbc72910c0a46ae2fef304a584ae408e0` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-mc-supplement-integration-20261019`|`bee9ae996fffec988d72651e295481b88afd9d73` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p-seal-20261019`|`8a18b277663bf2f8998f79e9b8c0a9e61d2b184c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-finite-bit-law-design-20261102`|`c52bbf2a76708079075f01f57af76676a6d5e7ff` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-finite-bit-law-execution-20261102`|`426ccda5df044d2e94d4a18d98cde75a96ab1e6b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-matched-static-design-20261102`|`eb20e81074aaae3cb365b348f6c3ab9d217d3697` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-poststage3-design-20261102`|`60515a33ce6231e9ff4ffdb40124ed845373351c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-price-confirmation-20261102`|`e44e87a306d05c8549f1a044b7fd056c2e1055a3` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-stage1-execution-20261102`|`5708e37080738c4ebf74590d799a3741246483b2` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-stage2-contract-20261102`|`495ed110ad8379663145791b788a63b64f05eedd` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-stage2-execution-20261102`|`8bcf5851f99f980dc08c5885ace54874b554c815` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-stage3-design-20261102`|`be3de04e6fb501fecc4a07f38ceea7c24ba10472` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-stage3-execution-20261102`|`aab0b483fdd8e57fe17305a11c5cb750a9c80a4d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-stage3-process-recovery-20261102`|`90bc02d458299212d3a1aa7c4bd34a0b98d1e0b5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-stage3-stable-basis-20261102`|`2da49fa4e8578227c130fc6061a0b3061276f893` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-stage4-accounting-design-20261102`|`e4a9b1345a46f1bccccbf451f4e2d26521350fe3` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-p3-stage4-conditional-accounting-20261102`|`a1d3f32aba6416fd1e200cd66a97d1a825b5dd73` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-pr-cost-variability-20261019`|`295b93d50662a0f8af3ec469df43099a3f04af60` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-s-stats-20261019`|`5e18773db3ad18c591b3c0fdc83cc1b5f1a29cfc` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-tol-a-local-history-20261019`|`52bc337d7e2f8b6dde2aeaa788b3d63f8c573858` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-tol-a-science-local-history-20261019`|`1e8bcecdb92d071226e0d35746df9089f5517edc` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-tol-d-local-history-20261019`|`9c6a3a89cc6b7e615469f2ecf06254285eda712a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-tol-only-main-phase0-20261019`|`85cb5b4d1808434165c421115ed09dab3b63dd00` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-tol-p-seal-20261019`|`821cc0f6ced57168293ea166644024867b064ffb` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper codex/h4-pr-cost-variability-20261019 @ 295b93d50662a0f8af3ec469df43099a3f04af60|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-tol-s-stats-preparation-20261019`|`d728d088c010089c6b672eef0521261791b2d209` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h4-trajectory-workers-20261023`|`7752551efe53b7b6dae8917dd7d0b7672192b8c0` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h46-long-time-b3-20261026`|`f794a2aa4c76bc024540c808bf9400d9f523bb76` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h46-long-time-pf-pr-20261025`|`155529e20a8fac39e00786c40b57830b71203c4f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h46-standard-y4-adoption-20261024`|`09e71910ee49e0d9d3a2f2f410249f6623022c0d` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper codex/r2-cs-closure-20261024 @ 270078a87c1febe8944f829270e8c2a00dbed7de|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h46-standard-y4-confirmation-20261024`|`4ce444eb5b7cd1dad7cf61f149aa182949ad60a1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h46-standard-y4-retry-20261024`|`4d117041469fb8eebf2ad55405c0f7d1b70dea7d` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper codex/r2-cs-closure-20261024 @ 270078a87c1febe8944f829270e8c2a00dbed7de|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h468-display-methods-local-backup-20261019`|`917dbd5347da62b5d089a452d0f27b85e2296cca` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h468-paper-evidence-integration-20261019`|`1923fce8f7461513d1bc47b5061e87cd70917e87` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h468-paper-evidence-integration-local-backup-20261019`|`ac42fad5d72869d06c1749830f37bd953284160d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-four-method-exploratory-ready-20261024`|`77ea542152de1c9ff796aefaf9be91c68c7ff1cd` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-hybrid-fairness-extension-execution-20261027`|`00958ae1945c780718b2deef07d5dab1096a5cd8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-hybrid-fairness-extension-readiness-20261027`|`bcc2eb6b201e1434bf47560775f5afb3e14d3ca8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-hybrid-search-fairness-audit-20261027`|`f7757c4320ba34dd36a64de4250c29a69bf38390` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-native-bridge-scalar-phase-audit-20261102`|`0e1c7aae386986f6c1bb9483548671655447f516` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-noncancellation-evidence-investigation-20261020`|`6cfc4579223b0ab4e54adbdacbc3b16a88ecf382` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-noncancellation-stage-a-20261020`|`c53dd16ae16413182c9d3080f82f80237231a248` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-noncancellation-stage-b-20261020`|`c585cef8990ef28f4d43c13dfaf2a0d423fda5a7` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-noncancellation-stage-b-rerun-20261020`|`3983fe4f1ede003953735e757d2c098c12b4644f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-pauli-p1-bias-execution-20261101`|`bfb4ca75437a9b3605a8f57b5f82100615dd034f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-pauli-p2-cost-execution-20261101`|`c9073274daee69c9b69d8db3dc332bf6132c2259` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-pauli-price-tail-audit-20261102`|`0d5558344ed7b94b264f79cbbb9c7deaa6bf0071` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-pr-confirmation-adoption-20261019`|`877e7c2834609c75f585c0cf7561ae7dca5db8d1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-pr-confirmation-final-seal-20261019`|`f79366f68caf8585cc36d2f2b0fefcb1a4bd0d1f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-pr-confirmation-phase0-20261019`|`6d290fc8fb656f4db148e2c9f91b757d80ed83c2` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-pr-confirmation-s-stats-20261019`|`d229ab9f4aa95c3f006e8067d58d448afd992205` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-pr-confirmation-science-20261019`|`3cccfcb2051dacf29e91473e2e1152acef1574bd` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-pure-rte-hybrid-evidence-audit-20261102`|`b3bab573cf17b9dd8ddc1dca5f1a6a6c935ef6d1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-rte-fairness-20261025`|`288d2206d689a9c24516844cba2100c2d9845a32` / GIT_OBJECT_PRESENT|元OID一致：excluded；keeper codex/h4-p3-poststage3-design-20261102 @ 60515a33ce6231e9ff4ffdb40124ed845373351c|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-selected-implementation-execution-20261102`|`71432728ee0140fb3c1f8bf30ecf51b2faba86ec` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h6-whole-law-transfer-feasibility-20261102`|`0e1c7aae386986f6c1bb9483548671655447f516` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-evidence-recovery-20261020`|`9f33ad0b11b4febe48b802000c38c5a0e3fd267c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-evidence-recovery-local-review-20261020`|`253356942845dddf05c2bb6f4582b026e0bce942` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-evidence-recovery-public-build-20261020`|`6a7886cd44bacb2a65300a99d34fa50c2c7e674a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-gpt-share-packet-20261020`|`363199c83b4cebde3d144f173e0e0aae67f7b2df` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-gpt-share-public-20261020`|`a6246f03c6297438ab624fc62e5061a58e53145f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-gpt-share-public-build-20261020`|`fad8c5c473c36bfa91774ba99e834fb8673baee6` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-original-artifacts-final-search-20261020`|`515dd604bfacbda507f3a7beb183fc8acd55bd19` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-original-artifacts-supplement-public-20261020`|`5f267e96ea313f373eb78a75040a43fac3a2ee1b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-rare-rte-c1-pilot-contract-20261021`|`77bf6472f901feab8cdc4871369bc8ddfc04be7b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-rare-rte-c1-pilot-repair-20261021`|`f377f8869c697c4846e3614b0ae7bc9d3ad5cb3c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-rare-rte-c1-pilot-run-20261021`|`d31ac1080e139eaad717301ffaa8d80160bbd378` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-rare-rte-closure-policy-20261021`|`862d732fba0a5f353c522a9219dec78ae2f21a4b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-rare-rte-saved-integration-20261021`|`145b438c05b1888ab479081ef72462b93429f89a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-rare-rte-stage-c0-20261021`|`136bd2730e49938bae697ed32674dfcc53e63f07` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/h8-rare-rte-stage-c05-20261021`|`cb9c994be608233fccc49ff2e44007bbb32e2fb4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-a-axes-execution-20261024`|`564cd71936fa27225624b00a7d5d1f12dc4cc6dc` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-a-cohort-cos-execution-20261024`|`7c5e25dce757b7daa165095839ca879ed0bb0c10` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-a-gate-order2-execution-20261024`|`5eaf32d2f01e95f514634bbbdefd43e971da5d16` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-c-evidence-run01-20261023`|`49db6f628e5ae51b5f5afa014e29d771e6d0d1eb` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-c-production-20261023`|`c4497d8b49a3d796513df439dd025c72fb964a21` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-c-reference-evidence-20261023`|`86b8b002704aced8ca239414080f53793de81658` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-c-reference-fix-20261023`|`450b0a787f376a7a72b5d944685d102aac911ae4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-arb-pf-cosine-20261025`|`78d7f24c80b500fc63c908efc768e683406571b3` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-arb-pf-pr0-sine-20261025`|`21fa1d08cf53422bd6e44e832ce8cc453f6d3ebf` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-arb-pr0-cosine-20261025`|`e90d6cfed1721b59c33c22bb146aebcc1fea3973` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-arb-pr0-preparation-20261025`|`f817c492c527d4e436b81abda9e9341defa1c49d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-arb-pr1-cosine-20261025`|`5c549779536bc272eb500d126a94845ae2bb1d6b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-arb-pr1-sine-20261025`|`06ca7c838808cbde9e1c1b5030c46f260d929c32` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-arb-preparation-20261024`|`5c69e1339955c77273b356df30e94ec3c57458b1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-conditional-cng-20261024`|`388b1e525d3cfbeba8f231acd42c813f67a7f531` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-r0-empirical-20261024`|`f01c4018201de9335f20a8901ec5f40f9cf9e89b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-r1-order2-20261024`|`c7585f2d6e4a0fa3f0cc873fb6e468be1d42bce9` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-r1-sine-20261024`|`36e993e698953f24415f4e90c4cac199a693ee16` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-r2-design-20261025`|`b297571d6949bb8b39152f08589c3a3b9c829ba6` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-central-rho-preparation-20261024`|`2734b97c2dd5f017e93cd81619ff52a6a0c6f16b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-certificate-gate-audit-20261023`|`0369fa2a05aac7820e1c0f7aa7e55f9d048a0efa` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-certificate-gate-foundation-20261023`|`36d0c12de9cec2a4fbb3abc0d8ab59c7f890cb2a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-cl-transfer-20261023`|`934f3d09db03ab9efbdad7da32eaaf432b6eb1ac` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-comparison-local-snapshot-20261025`|`c37ab940b858f12f46b2d5b203c61aaca1a7ac2a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-comparison-synthesis-20261025`|`03efd057a7e77c40ae7e90b9c77885d6606e793b` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper codex/non-df-h4-pauli-pr-search-execution-20261025 @ d3fdfad880902b7ff35b710d22e1c9759f7087c2|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-conditional-cng-plan-20261024`|`c8e7ae0f120edd5d4f93e9dad33c4bccefa2ac03` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-conditional-cng-results-20261024`|`5600f0e4b28c140a6a50b8aba47425860972a932` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-controlled-comparison-redesign-20261024`|`966c2cb323e3fe34032dc3be9f6b962936415e77` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-cost-reversal-structure-20261024`|`6cc5bded8e0ea2a4f0f1ad7e990f909eb1878d13` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-d1-contract-preparation-20261024`|`3282f74cd4d219141e4358c215a225dc2e3ec9c3` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-d1-execution-20261024`|`a5f77bbb99e23f15a2a465e0ab485777e7a6f40d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-d1-reproducibility-design-20261024`|`f173b76c7aba56012cbf1d51cb8f27dd062b3bb2` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-d1-source-execution-20261024`|`4e172d69271e8956e095b8e284766de0a9f7e563` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-d2-execution-20261024`|`b9183b42bf667df2853a05ba8033c92eb1c1e056` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-d2-preparation-20261024`|`57b65ce2d2c707409b2c06367d4bb1107b8ab66b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-data-catalog-20261025`|`7a83dc186bddcc35e528e38032814423ef1cba88` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-evidence-investigation-20261023`|`ebe20403bcb5c4439162614f64d991a769d7ed68` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-execution-foundation-20261023`|`6892e95ffdfc78e75425e7432e9e4ccdb9958df1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-kl-production-20261023`|`80147890fa91ad8270ce7643ab671727b3db5de4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-local-diagnostic-20261023`|`de94417e7e1a0b984b93b81aba673b5ef94fdf98` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-paper-plan-20261025`|`9fd3084b52c7f6c9cc72b5ee04d1c13eb820f196` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-pauli-b0-comparison-20261025`|`fbfbfd21a252a4a32f52e3b40e50acd91b3c626a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-pauli-b0-execution-20261025`|`6ba7d7e2ee0ebc4ff246596e5ecae358c74b6278` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-pauli-b0-preparation-20261025`|`2511bf5bcaccb664b34315a883629753c76d8fa1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-pauli-pr-search-execution-20261025`|`d3fdfad880902b7ff35b710d22e1c9759f7087c2` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-pauli-pr-search-preparation-20261025`|`351dd5fa15c65feb3c14738a933145fc62d2be2c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-pr-mean-cost-design-20261025`|`a5d96ae54d8cc3bc891d684570ad7e82160bab6c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-pr-mean-cost-scalar-execution-20261025`|`07d4f6ff7f1df085f292d36c36e460354df8a740` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-pr-mean-cost-scalar-preparation-20261025`|`fa3337b57e7d1e6ba5296317770a4f49a0cbaaf1` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper codex/non-df-h4-pr-mean-cost-scalar-execution-20261025 @ 07d4f6ff7f1df085f292d36c36e460354df8a740|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-pr-mean-cost-stage1-20261025`|`c5f17b117b9d34c27b4e1d680c02a444fff4e084` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-precision-bc-20261023`|`b0f94185f93cd5614e8f183613a2a9313a5b9d4f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-proof-binding-20261023`|`669a6ee576a032e8bf54d7ff47d570cc70d7eca5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-r2c-compiler-phase-proof-20261025`|`c45247deb9992d08594ad74b1114e6fac505292f` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper codex/non-df-h4-pauli-pr-search-execution-20261025 @ d3fdfad880902b7ff35b710d22e1c9759f7087c2|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-tail-rho-audit-delivery-20261024`|`1caaba2516dd3a30475e2acbd69672810a9fc05a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/non-df-h4-tail-rho-investigation-20261024`|`5600f0e4b28c140a6a50b8aba47425860972a932` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pauli-h2-h4-h6-integration-20261025`|`e2b9215fd7432e7406ef8c7b8b5f55fba563a696` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pauli-h2-h4-h6-paper-evidence-synthesis-20261101`|`4591b8d75a2a6a55c74a51225f0c61648023c8ee` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pauli-h2-h6-native-source-20261025`|`a8c7c7dca5a03aae1b008a09048d89c216bbf89c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pauli-h2-h6-resource-comparison-20261025`|`8fd1ceb283309a7e515ff4a533a253260bdd0f78` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pauli-hybrid-rte-execution-20261025`|`90c1c60820d52ca7ddcd4fb5315e9d21bb4a86c3` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pauli-hybrid-rte-preparation-20261025`|`25d248a1f2ed88de62b0e621c2d1cea8532804c4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pauli-jw-research-summary-20261026`|`b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pauli-pure-rte-cost-mechanism-inventory-20261101`|`5432857b94c3714ea9f4d6031e493e8dfc308a91` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pauli-pure-rte-mechanism-readiness-20261101`|`6f0759df7c1df7fd6d835922b0246753a8a98d6c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pauli-rte-event-extension-execution-20261026`|`10cb3e5cca7834ebf75301bd91bd093ca51d4a6b` / GIT_OBJECT_PRESENT|元OID一致：excluded；keeper codex/pauli-jw-research-summary-20261026 @ f0a035961583d976e8967511a78f9da66758f0bb|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pauli-rte-event-extension-preparation-20261026`|`c6d42d925b258939768a8c6a34ee3f48db27b964` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-direct-compile-contract-split-20261020`|`d510d7beda94d293432328a17fcc60d4286dc637` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-direct-compile-execution-20261020`|`a80f89b435340898f8cc46940e9080c9bcd3b403` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-direct-compile-expansion-plan-20261019`|`db06128a4b302faea0bc42d45399508b241a276d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-direct-compile-retry106-20261020`|`2619688b26c1146ae8c32d31d272282bfd1de407` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-direct-compile-retry216-20261020`|`636208ba6d74af13e7c405970c896b621022af65` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-direct-compile-technical-preparation-20261020`|`ed3aa50976121bb470b64ebb0e38afeec63ed797` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-order-reversal-evidence-audit-20261023`|`366b2fd24f7e86aa82b5b518b6e3774520f65596` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-q-model-comparison-preparation-20261020`|`b777ac7ea16e0a2feff6f8b0864ed766d779770f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-q-model-evidence-integration-20261020`|`80b7a35c4dbf683f952293eeb55944bc10b9efc8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-q-model-quantitative-comparison-20261020`|`7430a386bcd4dc545107e413cd022ab7cb261968` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-selection-comparison-preparation-20261020`|`f63065fb636504b78d295f351ffc339370b9a532` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pf-selection-cost-comparison-20261020`|`85433c48fd4c37d4cb4594391da422558ac3ce7f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pr-evidence-gpua100-gleap02-20261023`|`b8bd64339169ecdf61b90fb7b575a3937fdc0c8c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pr-rare-rte-existing-evidence-20261021`|`42ed0876c0505502428629c1fb4863233247ebaf` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pr-rare-rte-stage-b-20261021`|`3d5ab5bb1cf0d78a3157ad7f742cfa5c7b58efbb` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pr-repository-scientific-landscape-20261101`|`9a0bc7f17f47823e46c3945734127e9f0d09154c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pr-resource-protocol-v2-adopted-20261019`|`1f001cfc4e8e0531279a26613be29a52c7c0640c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pr-resource-protocol-v2-adopted-local-backup-747bfd6d`|`747bfd6df291f646f21ff501f50e51adc055232c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pr-resource-protocol-v2-draft-20261019`|`c0808b254aa02a94c7482f5c6e353b54532a54f8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/pr-resource-protocol-v2-draft-local-backup-ba4362f1`|`ba4362f1877b1fa53ddc319bdcc1fba4b47084b3` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/r2-config-selection-c0-20261024`|`8cb41c40aec9c31c69a5a50e8e5fe93aa6d2eb88` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/r2-cs-closure-20261024`|`270078a87c1febe8944f829270e8c2a00dbed7de` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/r2-one-sided-c06-20261024`|`dee25ba13ee8ebcb23c7d294c659ea678d818277` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/r2-pr-structure-c05-20261024`|`25cbd26d27057bd3d5af9c18d1c4b7be13d9cedf` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`codex/track-a-data-systematization-20261024`|`731b3d6f3572c6aa7950a579b68251fec9a5218b` / GIT_OBJECT_PRESENT|元OID一致：excluded；keeper codex/pr-boundary-evidence-integration-20261027 @ fbf701356cbc4da6ca8e3b7ef53533e1763243f9|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`main`|`0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`pr-beta-contract-audit-20261018`|`3fd8fca966e6b5a6a972aa71de68b31a0129f5ab` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`pr-beta-contract-followup-b03-b08-20261018`|`a8ae65b1edbbb10b20b927304762bca50b14d59b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`research/gap001-gap008-local-search-20261023`|`8dfdfa6460f9e7d25a9595132a672e8de2a5dc7e` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`research/pr-prior-art-novelty-20261021`|`501b4232b3e9ef9778e917f0a86f1d6c1d05f189` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-alpha-contract-audit-20261018`|`e62ddb59173d100c113be7a33581515d78f152ee` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-alpha-followup-preparation-20261018`|`343e0c591d297f99174459d750206b01bf7b9412` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-clean-figure-notes-ja-20261020`|`55b719c8e0881bda848a8ed721b0cd452267b801` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-cross-environment-server-acceptance-20261014`|`2ef5ecbdbdc222d73d9106618e20dc27b5466ea2` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-cross-environment-sync-server-20261014`|`2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-d0-saved-value-preflight-20261014`|`b941565835c539fc414d3a75dcb8ca439561a4d5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-d0-saved-value-reanalysis-result-20261014`|`8dbdb68293c988a999645e21a82b53966f9ef6cd` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-final-pre-review-implementation-20261015`|`2ec66aae5e17c9329fcea8ad962be27130980d31` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-approved-relaunch-20261009`|`ab86c31155c600f24552766ccd9d1505c5fb1de4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-authorized-launch-20261009`|`0bfeb3e6da83ad4bb2af8d6061eb944fb8e35218` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-b0i-compile-preparation-20261013`|`34ad305d55a797b57d00aebed99fc03a49a3ed2f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-b0i-manuscript-v04-20261014`|`47dc17a83e2fc9216e22b59317969952ef79028a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-b0i-map-parallel-run01-20261014`|`4bc56ea6eecf9e65eeaee784e062435c41e7b107` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-b0i-map-parallel-run02-20261014`|`98cdb35cd08469714f20d6cc3b982d9c80908f46` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-b0i-map-preparation-20261014`|`eff4a73f5153674d6fbc5e60629d34d51302fa84` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-b0i-pilot6-20261014`|`c377b074ab6cc442d6eb73ca4417bc56fa663d5c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-cleanup-esrch-fix-20261009`|`6f8e95ad0f69c9d8d49a61aab20bb4669b826ac5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-common-panel-h8-preflight-20261013`|`217c971dd537b1cc7fbc13742bc0f6c263ff7f45` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-encoding-speedup-20261010`|`5706ab29e0ede62a079e471d5ebd1ae8e9765328` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-entrypoint-cache-fix-20261009`|`461ddb2c2d613c063b4358ada8ad3de02b798141` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-gaussian-structure-proposal-20261010`|`56c07727743b4216f70d470793cda51e4357ca79` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-geometry-analysis-20261013`|`c81f7326dde3f40686b3caefe05e2f544d8d1826` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-geometry-results-20261013`|`7c4483378047b57f514c47dad98905e3f2a9a756` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-geometry-technical-integration-20261013`|`f6b0cb5e2ddabd2d7919d0822c58324c7338bac0` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-host-pressure-grace-fix-20261010`|`fa66bfa01ecb3724a3b25dde3c2ebf1c784c2ac7` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-host-pressure-policy-fix-20261010`|`08beb875ef233a1a9250c2cfd47faebc5b9f600a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-inverse-instruction-speedup-20261011`|`28866c82ed5bc6f73de3eeb8063785502de878ba` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-library-cache-fix-20261009`|`f25284607fd5d52da3b7fbddebee3005a22f57de` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-lightweight-speedup-20261009`|`5bb241125717d06e0dabd244247f2a0dc9124228` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-native-proof-seal-20261009`|`5de745199748a6523ba84b07a0ea61b2b831815b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-new-server-monitor-fix-a-20261007`|`b2a5ad89e8b39d72716f7ddb17d263bd0cdedb45` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-new-server-monitor-fix-a-20261007-02`|`3e0494e1e8a0c5ea72ef0bdc1cfc9b08a542edcc` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-new-server-monitor-preparation-20261007`|`814212cb1c25f27e3300294e72fef452d953fbc9` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-per-attempt-budget-20261009`|`91112b6cac2f8a26880d91ba5088bd53270af29c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-prelaunch-preparation-20261007`|`798270161d85af6361a4b08cdee9e4c879e6ffd7` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-production-resume01-20261011`|`d0b4e1734b063c9fef353fdb3a7b63b26c961e72` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-production-resume02-20261011`|`81c78b366d2cf2c699bb31977f72ae9deab4f1d4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-production-run03-20261009`|`a3248055ec8eda69d43c8911af71c7580f2f95ed` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-production-run04-20261009`|`cde398580dda6b8d34f999c06a05288128637095` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-production-run05-20261009`|`34c3fd46cd5dede93473f14187b3e92471f315ea` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-production-run06-20261010`|`00cc443636bef01583a8e61370bae0e390657f62` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-production-run07-20261010`|`84cea869cfb03b3312b52e23946256bd7ab62ba1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-production-run08-20261010`|`8ab82ef6f600a3497fa170149012d97904162c20` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-production-run09-20261010`|`82d1f43e97ee924f737a7b43a50b82a1d076297e` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-production-run10-20261010`|`95c2269263c92688936343b0423e9f1dfa22281c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-receipt-final-review-20261008`|`992c09d60c802525e00cf642eb5e1724b1a8c3e8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-receipt-reseal-20261009`|`c81144bb6678b603c9f320f2600b7d0a6dba06f7` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-resume-preparation-20261011`|`99249f5c032a33c98dc8181fcfc50b16d8217fce` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-resume02-preparation-20261011`|`37e8895b9e2f600987a94c6f835ca3912be808b1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-run07-stop-cause-audit-20261010`|`b1c2e746eee9a50e642eff59ff2c5178c75a4809` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-worker-error-fix-20261009`|`00c6ad5fa7c7affc6c6d702af54ac1de2b275d3d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h4-worker-memory32-fix-20261010`|`4fa8f9c48a14fd585b32e08984125abd02572054` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h46-s4-count-mismatch-investigation-20261018`|`b7385211e789ce9e9ab9515f3d5779fbd117bfb6` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h46-s4-counter-native-compatibility-20261018`|`6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h46-s4-primitive-site-validation-20261018`|`cb501906269ef31fe8733a3391db946b34467163` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h46-theory-compiled-paper-evidence-20261018`|`427480e4f1a61385cbffa724930a2221ed9e5e1d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h46-theory-compiled-total-plots-20261019`|`f15752625613956d087bab221e48f1ce7dd59fef` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h5-neutral-open-shell-pilot-run01-20261016`|`f906a61415d40ec7f362d00aa76e26420e25e7dc` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h5-neutral-open-shell-readiness-20261016`|`803a24ef7a01d8c7394dde88357c74168795bb34` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h5-open-shell-pilot-readiness-20261016`|`043f58ac9336d662a3181caff508526568f1e57a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h5-pilot-counter-fix-20261016`|`d86a4673d6ada701797d3e035f745d0d4faf5150` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h5-pilot-launch-record-20261016`|`bde58ba23f1fecdf77ee950795cb80c199c0f86b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-b0i-blueprint-fix-20261015`|`1f04e41e59566ecc9d7a88eaa34827fde43022d5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-conditional-s2-pilot-preparation-20261013`|`e0e6473d7047158a7f0f788d026a7a89a01898b8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-confirmation-preparation-20261012`|`3b6d733a107273a027c910ee1a61638b5a3c241e` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-exit-observation-fix-20261012`|`ce95a13902ed948885deb0870c99f2e8cbbb2610` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-family-fingerprint-parallel-fix-20261015`|`74ad43e516b18007644d384e3d5abe7dc7ed529d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-family-optimized-run04-20261015`|`e8e182aa5ff89691414b21d6ad30cc8a52a7d221` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-family-startup-cache-fix-20261015`|`f4432009d9a5ca145963563f7fcebba1da3db788` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-fixed-delta-t-dependence-readiness-20261015`|`043f58ac9336d662a3181caff508526568f1e57a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-four-worker-preparation-20261012`|`8eaea37d6ab2d27e633184ca3eba5db1ae259938` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-hash-reuse-eight-worker-20261012`|`321cd8c3066194fe98741cbebbcdba1515d1deaa` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-manuscript-h8-plan-20261013`|`653d64bedba7516d4c17f2c7457cefbc80f877a2` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-native-b0i-full-family-launch-20261015`|`fae54bd492b06d545c8cb1eb2428b8b4ffafb9dc` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-native-b0i-full-family-readiness-20261015`|`ac26bf554427dcc2257d4429236b1e61403edb57` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-native-b0i-full-family-results-20261015`|`34efc914b2f485ec694c7d6016025d5c76829be6` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-native-b0i-parallel-readiness-20261015`|`6baf246cb46cf69536db6c16872286e5ed12e85a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-native-b0i-pilot-readiness-20261015`|`ec6bad241d0ab26dd5628c33d65c94b1536f7e9f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-native-b0i-pilot6-launch-20261015`|`1453ceebe887fbc3040b19e00059e7a4d6766015` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-native-b0i-pilot6-results-20261015`|`ae3b992f22aead88515ef594cde7e5f867aefa87` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-native-b0i-pilot6-run02-20261015`|`c18f01d41d308bb61ec28e41b5578d4eebe20c56` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-native-dependency-fix-20261012`|`169d83c91b05e1efe21301c261579347d81af956` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-native-reproduction-fix-20261012`|`1293109e2c278d6c36c6337f0a9d885ea004b2e9` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-parallel-preparation-20261012`|`6f7bdbc54122455541405158d9cfa22e798356a6` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-pilot-external-monitor-fix-20261015`|`232d60c4240fc51138e4e51d59f6a3e790b96390` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-pilot-proc-exit-fix-20261015`|`d1fa0974064ae4feeb1c8168a88b43f89fc58340` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-rare-order-cost-analysis-20261013`|`eee1343f6e99aa7661892ddfbe1067e974e2b74a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-roundoff-execution-20261012`|`19da6a6de2a33ce27f467e7ca7223ddbe022b766` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-roundoff-resume01-20261012`|`df99d6ee28942df8d12aac3c4db7cd202df55d77` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-server-migration-preparation-20261011`|`76a8ac62bd0820554fa1e4a935815cb2e070695f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h6-server-migration-preparation-20261011-local-history`|`ad02f95c142c3af3a3dc9647e12566ff410e1b99` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h7-exit-monitor-resume01-20261016`|`d9fd36fc630a67fab220752154bfaa9c4b8d60c6` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h7-neutral-open-shell-pilot-results-20261016`|`162c256ea39b4ab3dadde65a94b3eec73cea4720` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h7-neutral-open-shell-pilot-run01-20261016`|`f93ad35b05a75c5e17e2c53b0f3de2af4fc443f2` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h7-neutral-open-shell-readiness-20261016`|`17df7b74896ad59503ca5979d8eb977a3fb22ad3` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b0-anchor-cpu-gate-fix-20261016`|`2b495b4c82742120b2caebe22666a89f92f7c099` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b0-anchor-implementation-20261016`|`847266323ea3d7bd29ac42322b7819b0b2a6c43f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b0-anchor-readiness-20261016`|`93bf0bf3420683d9d20755cea856d3814b951233` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b0-anchor-run01-stop-20261016`|`a6f76beccd009649a8a0ff932f36a8cc00dc83e9` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b0-anchor-run02-stop-20261016`|`af10b55420d31e227905f7762971e4b39c6c6a05` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b0-saved-anchor-audit-20261017`|`b19c337f0b16513d68b59eb3366dc6eff175c313` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b1-stop-acceptance-reconciliation-20261017`|`f0038f7e34331bf88482d3cf6258562b4a9c08de` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b2-concurrent-preservation-recovery-20261017`|`e3ecc03a4d02503ccdf20afd9d750cb49d8ca6ff` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b2-cpu-handshake-recovery-20261017`|`2e604e4a3a04eee5090e03d9bb83818d7afd3c87` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b2-cpu-handshake-running-20261017`|`7fc1d828da8177edf5645f02fb470383b246ff79` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b2-recovery-run01-stop-20261017`|`534ef062363d06672d04947e00d295454e761e67` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b2-sine-psi-observation-20261017`|`f58da9590597f42ff7260c22d771734096a706bb` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b2-sine-psi-running-20261017`|`fef2db7a79886ab682b652da7266b463c98c7d3a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b2-sine-recovery-running-20261017`|`09dd7178d273a504b25274d509f29da0aa02d040` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-b2-sine-spool-recovery-20261017`|`aaf3d909f20f20354016966358ce754dbf5b7aff` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-bias-evidence-audit-20261017`|`b75039c1d5ead475d8c0ff8b62dba04a9978a6fa` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-d0-saved-value-one-shot-20261015`|`3d910ee1de4597b8895e8506500553dbad5a5085` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-duplicate-work-optimization-20261013`|`f8c50394392589434cfb6cffd8350dc2fd564a2a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-fast-retry21-20261013`|`a55da5c2b012912c485184845aa73ca053ac00b9` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-native-b0i-full-family-results-20261016`|`5f79257f432bc7e0317f122090f62fec379e2a1b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-native-b0i-full-family-run01-20261015`|`90cc9505064513bb0b37bd872dccc15fb42c6ee8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-native-b0i-readiness-20261015`|`b654e90e25a59011b1763dbf8ab29f7ae93f6e56` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-noncancellation-readiness-20261017`|`d567e6e1f3b1713b455a80da9b5c5f9e7848f2de` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-noncancellation-readiness-20261017-publish`|`a967538e37fe1b017260fd728830b2572a833833` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-phase-b1-implementation-20261017`|`21e3d13469b508545b28415b13eaf25c60a9d38f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-phase-b1-readiness-20261016`|`b09f408873cad94a371261386f7f3bf9cc0baa91` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-phase-b1-run01-20261017`|`83a3c4a6688e66576f85471bf41a45e29cbfb705` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-phase-b2-readiness-20261017`|`3a450af0fb431269ac94152807992883c1490b26` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-phase-b2-recovery-parallel-readiness-20261017`|`bf56461fa9e0780d435bad5d30c2cf2103549041` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-phase-b2-run01-results-20261017`|`d5f290248147d0127d73afd35a53e2b10ff835df` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-remaining-fast-handoff-20261013`|`adbd67661ed3717f57d904fb479c1253e2a25e8f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-resource-map-production-20261013`|`eaeeefc38be454375d141fabfb8ae229498175b9` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-resource-map-resume01-20261013`|`96b9560e6bc02a71a5917db5d0cf911a69aec3de` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-resource-map-resume02-20261014`|`681ec23c93fdd1090d325908f4b155a79115afdf` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-saved-df-state-preflight-20261013`|`0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-signal-compile-preflight-20261013`|`22197f9901c37d98889fb0df3fbfac081de7680b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-signed-bias-readiness-20261017`|`f9bf3da5f6cffca95f269dd65bdeb806d875234d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-signed-bias-result-20261017`|`8e54303bfe46bbebbf4d37148cbe26ef4013ae38` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-h8-strengthened-deterministic-readiness-20261016`|`01309059636f2432e29b663bcd2eb0e400f5ae0a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-independent-technical-audit-20261015`|`40a4f55464dba80ab2192d777f7d69d2f551676f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-mechanism-a82-factor-results-20261019`|`1ff6ac0b6ffed6231dac5f655275b620598d0201` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-mechanism-a82-factors-readiness-20261019`|`bfa47ba92925c16bcb99001fb28974ddaef57cda` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-mechanism-a82-results-20261019`|`182d7044fb3e61d68d8c7a5dd86d84c0e961ed6c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-mechanism-explanation-fix-readiness-20261019`|`b174dd7144b3fef08a205c3486520895ed976ef8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-mechanism-explanation-results-20261019`|`0edd62403c35c5ee18b2a2a152158faca5c2ebd8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-mechanism-figure2-a82-readiness-20261019`|`eb33de67475e7d33eed56801b74a2a277147e4be` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-mechanism-paper-integration-20261019`|`bd41b3760ff89e428380f55585d78094a9330a56` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-mechanism-results-20261019`|`2bf5d83293b3a47f672805fae956ff2e50df2878` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-mechanism-runner-readiness-20261019`|`85841ae9a94cf5099751d7cd56112ded578fd1d3` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-native-df-h6-h8-metadata-binding-20261016`|`46c06e5653152eb4ce786d69d927ecbf12f9a2de` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-native-df-h6-h8-metadata-completion-20261016`|`b0c7a2855482175e438ffebe6c821ce36b0b2860` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-native-df-theory-evaluation-20261016`|`7e41670565d9284da9aded94063be591fdcaa3dd` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-paper-preparation-20261019`|`0a8ac1d2d03c5a9aef2acab0295226f8c88b38ce` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-pr-pf-absolute-cost-v2-20261019`|`c5ad79457aea87cc36b38022a2821ebb42eb276d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-pr-theory-compiled-h46-completed-20261018`|`7e454b91185ed7b0f92a3790f800ba5d65d5f156` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-pr-theory-compiled-h46-local-stage3-source-20261018`|`34b2da1a6bfc001314e0e31458d803caca80adf2` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-pr-theory-compiled-h46-readiness-20261018`|`f27ae0a5119b4083909720c34558145527d21123` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-pre-review-technical-closure-20261015`|`560ba600dce8737f588e030fbd3f1f6c5559cbbc` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-presentation-figure-updates-20261020`|`0566c3bb6c9c1a2d634519db6aabc5d2ef412798` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-q-model-figure-extension-20261020`|`40e7e135536ac9abbffa04c7c2621dbc44c33bb2` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-research-status-update-server-20261015`|`6b78e5f39917d92d6c60694d3e0b0d06394d7eb5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-resource-evaluation-expansion-design-20261014`|`2ccc79502f043b4885e2337ffa98ff45f1875ae4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-v1-h6-deterministic-readiness-20261016`|`25bbdd33fe758a0e730138cb91ddac8c41b0a298` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-v1-h6-p7q1-compile-readiness-20261016`|`63e5773b6b408506f4c39783b80f3025f14481b9` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-v2-h6-cost-binding-readiness-20261016`|`dd695e6584f15b88341d2446ed010c7bdfef1666` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-a-v2-residual-analysis-readiness-20261016`|`68a010b35d35a1500dcc901c480fcce433fac4a8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-c1-0-technical-preflight-20261020`|`a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-c1-1-d0-partition-hash-20261021`|`8188c8a50f79ea9d17fddb987c651650b58aa22c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-c1-1-input-basis-exact-model-20261020`|`4cfe1c257709f0ac09351eadce3952b81403cf93` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-c1-1-r1-new-partition-exact-model-20261021`|`2937e6b655ba3cd77f982d6901d766355a8b92c1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-c1-2-c1-4-integrated-adapter-validation-20261021`|`e72867b87e12d93d4fc1361f45f105febc6a94c6` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-c1-3-r1-recovery-c1-4-20261021`|`49dae5247ff0ad2c4efb9056e095b70913cc9ddb` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-c1-3-r2-reference-recovery-c1-4-20261021`|`068f0ae8771fccf28c529bda12a84abe8b7e7648` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-c1-4-5-c1-5-technical-preflight-20261021`|`af75af03c0b334c29b5a4311bd2b4578a999460c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-c1-5a-h4-limited-resource-pilot-20261021`|`b022f2ea421649015a4c21ea3c904fe619d0c198` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-c1-5a-p5-worker-recovery-20261021`|`bb9059bacb697e1105e63c1b3826b0c34886ad15` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-c1-freeze-and-manuscript-synthesis-20261021`|`185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-coefficient-type-canonical-audit-20261020`|`f823e909cb7049a612f1bf2c247f3bbe15fc184d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-closed-port-bundle-integration-20261019`|`62967269e611b61e73b70abe20e998d9c7eaaf50` / GIT_OBJECT_PRESENT|元OID一致：excluded；keeper codex/p4-track-b-research-entry-integration-20261028 @ 33424d205aa918498e1ef54beb94ab67612bdec8|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-closed-port-bundle-integration-20261019-local-content-record`|`becf1933abaff59c83fbddabe794a27c33628737` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-closed-port-bundle-integration-20261019-local-final-record`|`317866e36b9a22db20a92a4f38ceb7a7c0d12424` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-closed-port-bundle-integration-20261019-local-source-record`|`41bb71e494f0c5fd0cbdd1fa06cc88112bfd958c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-correctness-512-authorized-run-20261020`|`22ef92cef31f98e3e31f27a209405f81f5c4a0e4` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper codex/p4-track-b-research-entry-integration-20261028 @ 33424d205aa918498e1ef54beb94ab67612bdec8|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-correctness-512-authorized-run-20261020-SOURCE`|`f8aad18d385c9dd787c5be8c794cf3c339335919` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-correctness-512-authorized-run-20261020-local-evidence`|`814bae76f4d2385a7fa530d577cd883138422d5e` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-final-live-connection-simple-run-control-20261020`|`bf86fddc0abf493d4e9e4cc40498fde72c23c194` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-final-live-connection-simple-run-control-20261020-SOURCE`|`a1c4f41b6c29baf691006a8ef39e87b8ab9aa8d8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-final-live-connection-simple-run-control-20261020-filesystem-local-evidence`|`4988bc6a52fe3fe8cdd5f4a482835fea903dbcc1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-final-live-connection-simple-run-control-20261020-local-evidence`|`0a73c3b39e81095329d532354a3eed1d03e5235f` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-limited-pilot-16896-authorized-20261020`|`95b169216ed93e6cad71e65cf529db7c6b7cf2f4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-limited-pilot-readiness-20261019`|`76ca3c4089b201d55da0fede08e1f77733ac38a5` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019 @ 82896d53fa97235fa848d6c0833eb193af8b99d6|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-live-api-correctness-preflight-closed-20261020`|`6c6d3376e7ff206219f5e58355f1e1080fafba04` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-live-api-correctness-preflight-closed-20261020-SOURCE`|`4b4496a86f4ab9e0e1266e7323283aff9fe24f26` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-live-api-correctness-preflight-closed-20261020-local-evidence`|`4a3eaca91040a6c25875707f275aa33f84711fd5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-live-correctness-release-preparation-20261020`|`841ac8920e41e93e2e79574f6ecdc6213a570f88` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-live-correctness-release-preparation-20261020-SOURCE`|`2405f5e2e5fb0dc5add39b102e3ccae3d56df0a4` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-live-correctness-release-preparation-20261020-local-evidence`|`a9e7784682f193730d768faa695610c28182c006` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-parallel-fused-r1-technical-integration-20261019`|`7ae4bd0b951dae05380538cd5c22a3ead5328ab2` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019 @ 82896d53fa97235fa848d6c0833eb193af8b99d6|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-parallel-overhead-optimization-20261019`|`1cd2a7196bc48a4aa1a904d7bd351e3d8fbf45ff` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019 @ 82896d53fa97235fa848d6c0833eb193af8b99d6|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-performance-scaling-pilot-20261019`|`75ba5b944c56669a2136393924a8bb8d22e40fd1` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-pilot-results-technical-organization-20261020`|`78cb65be373b0808641b5a5e44d9b69913bf84e9` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper codex/p4-track-b-research-entry-integration-20261028 @ 33424d205aa918498e1ef54beb94ab67612bdec8|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-post512-metadata-pilot-readiness-20261020`|`f6088e2cc453816b30e985e52b10696830844fa9` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper codex/p4-track-b-research-entry-integration-20261028 @ 33424d205aa918498e1ef54beb94ab67612bdec8|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-post512-metadata-pilot-readiness-20261020-SOURCE`|`f0fe153a22c5df3a1c22b04d0e0599eef196b640` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-r1-parallel-contract-pilot-20261019`|`7b6f526b5563ee1a6401baec131e5eb290f00fb8` / GIT_OBJECT_PRESENT|元OID一致：deleted；keeper track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019 @ 82896d53fa97235fa848d6c0833eb193af8b99d6|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-adapter-independent-server-20261020`|`5c1e13fa74b3c5e4f3166babed785274844655d0` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-adapter-independent-server-20261020-r2`|`0db83adb370d66a2f7f5254c8037dadda94ecdb8` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-adapter-independent-server-20261020-r2-SOURCE`|`74c68e148edc47144268187cad4bd03ce796480c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-adapter-independent-server-20261020-r2-local-evidence`|`c24937540681aae093a98d253067e34db24a5d34` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-api-adapter-contract-closed-v1-20261020`|`80b7bf61fe0a735d4031bb719c4b3e788611c3ba` / GIT_OBJECT_PRESENT|元OID一致：excluded；keeper codex/p4-track-b-research-entry-integration-20261028 @ 33424d205aa918498e1ef54beb94ab67612bdec8|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-api-adapter-contract-closed-v1-20261020-local-evidence`|`e56a2de61eafd12dd50114280e65a7d7da84fde5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-api-adapter-contract-closed-v1-20261020-measured-source`|`b7ad344401b10395c49e4cfa806909282859bab5` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019`|`82896d53fa97235fa848d6c0833eb193af8b99d6` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019-local-content-record`|`ec6a6c1c0dac5cd680879bf13608ff9c4a107f7b` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019-local-final-record`|`39fd7b3afe78258e3fb15aac2cf3c54ef2234513` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019-local-source-record`|`e1e1f437460ba89b6637d32f50148bfed0084b8a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019-local-source2-record`|`7fed70c417f4941c9c51dfcad720a49017251d35` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-trace-reservation-accounting-repair-20261019`|`596781daeba253418ba35f4b7964312a54caf233` / GIT_OBJECT_PRESENT|元OID一致：excluded；keeper track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019 @ 82896d53fa97235fa848d6c0833eb193af8b99d6|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-v2-independent-acceptance-server-20261019`|`77126767c00585d0967558c0756907df2d69d1ff` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-v2-independent-acceptance-server-20261019-local-record`|`7327fe47adde5731ae46e3e8f4dc06326f41b025` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-v2-production-bundle-bridge-candidate-20261019`|`d24cccbecdf0928fe977080dd7b28c591df0089e` / GIT_OBJECT_PRESENT|元OID一致：excluded；keeper track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019 @ 82896d53fa97235fa848d6c0833eb193af8b99d6|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-v2-production-bundle-bridge-candidate-20261019-local-content-record`|`72a21a5b4bbc22d3e9440bab53ed4d6a6f97147a` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-v2-production-bundle-bridge-candidate-20261019-local-final-record`|`b3af84d198dc13f73a216441637b9a0862285a7d` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-g6-w2c-b3-v2-production-bundle-bridge-candidate-20261019-local-source-record`|`017cfdb81794ba8067630b09d21b6e2315b1735c` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-partial-sum-query-audit-20261020`|`df90ed8c8e847f8ea1861d9761b9515e7a895916` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-partial-sum-query-audit-local-20261020`|`48f004727f494098138f16f6bf5f687972a8188e` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-ra-application-feasibility-20261020`|`291e21218c1d1fde062c001a1586ba7c1bb9c988` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|
|S2-ROOT-01|`track-b-return-aggregation-method-note-v1-1-20261020`|`243eea8e840f3c472753f7935e510391626fd325` / GIT_OBJECT_PRESENT|該当記録なし|所在索引。個別SOURCE/条件の対応は登録表またはUNRESOLVED|

## 付録D. 固定結果表と未展開範囲

|系列|元表の行数|登録行|選択／未展開|固定表|
|---|---:|---:|---|---|
|LONG_CANDIDATE_ACCOUNTING|58160|3210|Rows with native_cost_status=ACQUIRED; other rows indexed in REPORT|[candidate_precision_accounting.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/candidate_precision_accounting.csv)|
|LONG_SAVED_CONFIRMATION_COMPARISONS|59|59|All rows|[independent_comparisons.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/independent_comparisons.csv)|
|B3_CANDIDATE_ACCOUNTING|17160|960|Rows with native_cost_status=ACQUIRED; other rows indexed in REPORT|[candidate_precision_accounting.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/candidate_precision_accounting.csv)|
|B3_SAVED_CONFIRMATION_COMPARISONS|120|120|All rows|[independent_comparisons.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/independent_comparisons.csv)|
|H2PAULI_CANDIDATE_ACCOUNTING|55|55|All rows|[candidate_precision_table.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/candidate_precision_table.csv)|
|H4PAULI_CANDIDATE_ACCOUNTING|725|725|All rows|[candidate_precision_table.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/artifacts/non_df_h4_pauli_pr_search/2026-10-25/candidate_precision_table.csv)|
|EVENT_CANDIDATE_ACCOUNTING|50|50|All rows|[candidate_precision_table.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/candidate_precision_table.csv)|
|H6HYB_CANDIDATE_ACCOUNTING|40|40|All rows|[CANDIDATE_RESULTS.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/CANDIDATE_RESULTS.csv)|
|H4_GEOMETRY_BIAS_PRICE|1308|1308|All candidate bias and saved mean-price observations; precision projection fields omitted and indexed.|[saved_candidates_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/saved_candidates_v1.json)|
|H246_COMMON_PF_STATUS|588|588|All rows|[candidate_status_588.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/candidate_status_588.csv)|
|H468_SAVED_PRIMARY_CONFIRMATION|6|6|All saved main-table rows; H8 aliases are reused cohorts.|[main_table_proposed.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.json)|
|H4_D1_SAVED_NATIVE_PRICES|9|9|All rows|[metrics_table.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_d1_source_execution/2026-10-24/metrics_table.csv)|
|H4_D2_SAVED_CONTROL_COMPARISON|10|10|All rows|[comparison_table.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_d2_execution/2026-10-24/comparison_table.csv)|
|H4_SIX_LOCAL_ARB|6|6|All rows|[six_local_results.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/six_local_results.csv)|
|FINITE_RTE_SIGNAL|96|96|All stored q/r/K x state observations; numeric values are literal strings.|[h4_sto3g_d100_rank12_ld0_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld0_v1.json)|
|FINITE_RTE_SIGNAL|96|96|All stored q/r/K x state observations; numeric values are literal strings.|[h4_sto3g_d100_rank12_ld1_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld1_v1.json)|
|FINITE_RTE_SIGNAL|150|150|All stored q/r/K x state observations; numeric values are literal strings.|[h4_sto3g_d100_rank12_ld3_dt0p025_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_dt0p025_v1.json)|
|FINITE_RTE_SIGNAL|150|150|All stored q/r/K x state observations; numeric values are literal strings.|[h4_sto3g_d100_rank12_ld3_dt0p05_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_dt0p05_v1.json)|
|FINITE_RTE_SIGNAL|150|150|All stored q/r/K x state observations; numeric values are literal strings.|[h4_sto3g_d100_rank12_ld3_dt0p1_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_dt0p1_v1.json)|
|FINITE_RTE_SIGNAL|150|150|All stored q/r/K x state observations; numeric values are literal strings.|[h4_sto3g_d100_rank12_ld3_dt0p2_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_dt0p2_v1.json)|
|FINITE_RTE_SIGNAL|150|150|All stored q/r/K x state observations; numeric values are literal strings.|[h4_sto3g_d100_rank12_ld3_dt0p4_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_dt0p4_v1.json)|
|FINITE_RTE_SIGNAL|96|96|All stored q/r/K x state observations; numeric values are literal strings.|[h4_sto3g_d100_rank12_ld3_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_v1.json)|
|FINITE_RTE_SIGNAL|96|96|All stored q/r/K x state observations; numeric values are literal strings.|[h4_sto3g_d100_rank12_ld6_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld6_v1.json)|
|PF_C_SIZE_AND_STATE_ACTION|9|9|All size-summary / state-action-summary rows; per-delta values are indexed separately.|[h2_h6_paper_d6_c_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/pf_c_system_size_validation/h2_h6_paper_d6_c_v1.json)|
|PR2_M1|210|210|All candidate summary rows, excluding raw trajectory arrays.|[pr2_matched_accuracy_m1_b1_compile_map_result_v2.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json)|
|PR2_M2|5|5|All candidate summary rows, excluding raw trajectory arrays.|[pr2_matched_accuracy_m2_transfer_result_v2.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json)|
|H6_NATIVE_B0I_SAVED_AXES|18|18|All rows|[axis_compiled_metrics_v1.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/artifacts/resource_applicability/track_a_h6_native_b0i_full_family_result_v1/2026-10-15/axis_compiled_metrics_v1.csv)|
|H6_FIXED_DELTA_SAVED_ACCOUNTING|15|15|All rows|[primary_RZ_P0_v1.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/primary_RZ_P0_v1.csv)|
|H8_ORIGINAL_SAVED_PRECISION|4|4|All rows|[precision_summary.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/precision_summary.csv)|

以下は元manifestのresult_set全84項目の対応。準備・source_present_no_current_ciは科学実行済みの意味ではない。条件を特定できないもの、今回行展開していないものは実験行を捏造せず索引状態で残す。

|元result_set ID|保存された元状態|今回の状態|
|---|---|---|
|legacy_high_order_pf|historical_evidence|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|df_partial_randomized_screening|invalidated_pending_regeneration|失効履歴。原条件/入力未結合、再生成・有効化なし|
|uwc_grouped|prose_only_missing_raw_artifacts|旧prose-only/raw欠落履歴。科学値を補作しない|
|tests|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|medium_q_rpe_hadamard_benchmark|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|large_q_rpe_hadamard_affine_proxy|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rpe_hadamard_proxy_resource_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rpe_allocation_sensitivity_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rpe_four_round_accounting_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rpe_four_round_phase_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rpe_target_round_horizon_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rpe_delta_round_schedule_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rpe_delta_compiled_cost_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_prevalidation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_ablation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_pf_sensitivity|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_gate_s1|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_structure_pilot|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_sequence_policy|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_full_scope|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_full_scope_extension|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_decision_cost|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_late_round_proxy|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_compiler_transfer|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_uncertainty_break_even|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_wp11_synthesis|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_full_opt2|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_signal_weight_pilot|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|research_direction_geometry_energy_difference_pilot|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|finite_rte_signal_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rpe_round_cost_connection_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rpe_hadamard_failure_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|random_circuit_cost_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rte_boundary_cost_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rte_boundary_pair_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|hierarchical_compiled_cost_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rte_order_stratified_cost_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rte_connected_cluster_cost_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|rte_system_size_cost_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pf_delta_surrogate_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pf_c_system_size_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|research_direction_joint_synthesis_pilot|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|research_direction_theme_selection|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|research_direction_joint_synthesis_blind_validation|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|research_direction_joint_synthesis_formalization|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|research_direction_joint_synthesis_mechanism_validation|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|research_direction_geometry_tracking_breakdown|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|research_direction_energy_tail_pareto|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|research_direction_pd_realization|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|research_direction_pd_fair_comparison|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|research_direction_pd_s1_posthoc_reanalysis|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|finite_rte_phase_amplitude_fr1|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|finite_rte_phase_amplitude_fr_revision_r0|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|finite_rte_phase_amplitude_fr_revision_r1_plans|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|finite_rte_phase_amplitude_fr_revision_fr1a|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|finite_rte_phase_amplitude_fr_revision_fr1b|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|finite_rte_phase_amplitude_fr_completion_contract|source_present_no_current_ci|改善方式自体は範囲外；関連baselineの索引だけ保持|
|pr2_pr3_minimal_pilot|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_s0_s1_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_new_series_v1_v3|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_v4_s2_development|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_resource_contract|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m1_implementation_contract|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m1_preexecution_amendment_v2|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m1_execution_authorization|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m1_a|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m1_b1_contract|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m1_b1_execution_source|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m1_b1_result_validation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m2_transfer_contract|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m2_transfer_contract_revision_v2|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m2_execution_source|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m2_execution_authorization|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_matched_accuracy_m2_transfer_execution|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_track_a_pm0_evidence_attribution|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_track_a_pm1_nearby_discard_preparation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_track_a_pm1_execution_authorization_draft|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_track_a_pm1_authorization_finalization|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_track_a_pm1_discard_execution|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_track_a_pm2_precision_resource_preparation|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_track_a_pm2_precision_analysis_source|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr2_track_a_pm2_precision_resource_analysis|source_present_no_current_ci|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr_pf_s2_recomputed_20261022|LOCAL_COST_PAIRS_WITH_FAILED_AUXILIARY_DIAGNOSTICS_MANDATORY_STOP|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|
|pr_pf_analytic_vs_circuit_CNG_20261022|SAVED_CNG_ACCOUNTING_COMPLETE_N_INDEPENDENT_MISSING_CONDITIONAL_PHYSICS|索引確認。関連する登録系列は上表；全条件/全解析版の展開は未解決|

SOURCE索引の対象repo182項目は記録blob一致、別repo29はroot範囲外。これは科学実験182件の登録ではない。詳細は固定SOURCE_AND_PROVENANCE_INDEXと元カードから追跡する。
