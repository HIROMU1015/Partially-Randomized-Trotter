# PR研究3環境Coverage監査 Phase 2 — 既存実験証拠の横断カタログ

担当はLOCAL Codex、2026-11-03 JST。ユーザーの明示指示に基づき、固定Phase 1成果物のmetadata統合・技術検査・公開を行った。**本成果物は科学的coverageの充足判定、不足A〜E分類、主要claimの採択、独立科学レビューではない。** 科学値のN/G・平均・SE・比・fit・順位を新計算していない。

## 固定仕様と適用規約

[固定PROTOCOL](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/PROTOCOL.md) と同commitの [43列テンプレート](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/INVENTORY_TEMPLATE.csv)、[固定HOST_INSTRUCTIONS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e440e5c06853599b32f678c27c01ab817c1df120/docs/research/pr_cross_host_coverage_audit_20261102/HOST_INSTRUCTIONS.md)を確認した。§4〜7の条件/run/cohort/保存解析版、§9のPhase 2境界、§10の保全を維持する。共通入力schemaは `PR-COVERAGE-1`。固定PROTOCOLにはPhase 0担当LOCALという記述が残るが、実担当はユーザー訂正どおり **SERVER 2**。固定文書は変更していない。

適用される主worktreeのAGENTS.md、親ディレクトリの規約候補、正式共通運用指示 `research_project_instructions_unified_2026-10-25.md`（§8.1/§8.4/§10）を確認した。PROJECT_MAPを入口とし、前段で読了した研究概要と現行の固定索引の時点差を保持する。primary agentだけで実施し、sub-agentは使っていない。IDEのactive file/open tabsは対象・実行許可に使用していない。

## 入力の取得・公開状態・完全性

| 観測環境 | 固定commit | 入力 | 観測行数 | 確認 |
|---|---|---|---:|---|
| LOCAL | `17aacf67ee854657652ade8035c5fdfad1eb077a` | INVENTORY.csv / REPORT.md | 9,665 | GitHub固定directoryの全file/blob/sizeとLOCAL Git object集合が一致 |
| SERVER_1 | `2de84ed7af00e73339886429aabab0bf8180e9f0` | INVENTORY.csv + part002〜part030 / REPORT.md | 100,878 | GitHub固定directoryの全file/blob/sizeとLOCAL Git object集合が一致 |
| SERVER_2 | `8834de0cf938b13490c08e05006e3c566fed7c7a` | INVENTORY.csv / DETAILS.jsonl / REPORT.md | 8,761 | GitHub固定directoryの全file/blob/sizeとLOCAL Git object集合が一致 |

全36ファイルの固定Git objectを読み、GitHub connectorで3環境のREPORT本文と固定directory listingを取得した。全大容量CSVをAPIレスポンスへ一括保持せず、既存LOCAL Git objectからstreaming読取した。SERVER 1の30分割は全て入力に含め、各ヘッダ・record ID連続性・行数・bytes・REPORT保存SHA-256を検査した。複数行CSV fieldはcsv parserで処理し、行末や改行文字の単純分割では復元していない。

SERVER 1をヘッダ1回・同じRFC4180/CRLF表現でstreaming連結したhashは `6e5d2fb858aa9edc8806cd2134097ab148e8edd950ef2c99eefe374a6a1a0912` で、固定REPORTの分割前hashと一致した。1.55GBの復元コピーは書き出していない。SERVER 2のDETAILS.jsonlは **10,176参照**を一段で解決し、record_id/columnとの一致、重複・未解決・未使用entryのないことを確認した。

SERVER 1の固定REPORT末尾の「未公開」は公開前の履歴記載として保全した。SERVER 2のREPORTにも同種の履歴記載がある。今回、指定固定commitの全入力ファイルがGitHubで取得可能で、返されたblob/sizeがLOCAL固定treeと一致することを確認した。過去の記述を修正せず、現在の取得状態と区別する。これは他hostのfilesystem/raw現存確認ではない。

[INPUT_FILES.csv](INPUT_FILES.csv) に固定commit、ファイル、blob、bytes、行範囲・hash・公開照合状態を記録した。固定REPORTにないhashを元REPORT記載済みとはしていない。入力36ファイルの内容・固定仕様・旧STOPは変更していない。

## カタログの単位・確定範囲

| metadata項目 | 件数 | 意味 |
|---|---:|---|
| 入力観測行 | 119,304 | 3環境の元行を全て保持 |
| カタログgroup | 103,141 | 保存結果同一性groupと暫定単独観測の和 |
| 同一保存結果を確認したgroup | 12,632 | 複数観測が同じ保存結果blob/path/検証済みselectorを指す |
| 同一保存結果の追加観測リンク | 16,163 | COPY_OF。独立実験として加算しない |
| 暫定単独観測group | 90,509 | 他観測とのidentity未確定。独立実験認定ではない |
| 公開元結果selectorの解決 | 119,244 | JSON pointer/CSV data rowの実在を原表で確認 |
| 不透明なLOCAL私有観測 | 60 | 元V4 metadata参照を保持。private rawを公開せず再取得しない |

**103,141は独立実験数ではない。** 新しい独立cohort総数の認定は行っておらず、その総数はUNKNOWN。確定しているのは同じ保存結果を観測したコピーの対応であり、未確定のrun/cohort、入力/state/reference、law、compiler境界、正式受理・有効性・raw custodyは別に残る。全groupに少なくとも一つの未解決照合事項があることと、保存結果コピーを確認できたことは両立する。

CAN- は複数観測の `VERIFIED_SAVED_RESULT_IDENTITY`、TMP- は `PROVISIONAL_SINGLE_OBSERVATION` を示す。研究上の世界共通experiment IDを創作したものではない。元experiment_id/condition_id/run_id/analysis_id/source_commitはRECORD_LINKSにそのまま残す。コピー、未知、負結果、旧版を消していない。

## 同一性の照合規則とSOURCEの役割

同一保存結果の結合は、同じfull result Git blob、元repository path、同じ正規化selector、原表でのselector解決、記録された物理・方法条件の両立を必要とする。元の科学SOURCEが異なる場合は、明示されたcomponent bindingで同じ実験の別componentと対応付けられる場合に限り結合する。run IDの既知矛盾は結合しない。branch名、REPORT名、host名、carrier commit、同じ数値、同system/T/精度だけでは結合しない。異なるpathの同一blob、異なる精度・round・state selectorも自動結合しない。

CSV_DATA_ROW=N / CSV:data_row=N / CSV data row N は各原REPORTが定義するヘッダ除外の1-based data rowとして正規化した。JSONの空pointerはRFC6901のdocument rootであり、欠測の空欄とは区別する。元selector表記は固定Phase 1へ戻れる。数値文字列の書式差はmetadata等値比較の補助に限り扱い、原科学値は書き換えない。

同じ結果を指す4,389組で、Phase 1のSOURCE欄がproviderとnative/confirmationを別々に記していた。[SOURCE_COMPONENTS.csv](SOURCE_COMPONENTS.csv) の13 bindingに、保存scientific_provenanceのroleと実在するfull commitを対応付けた。B2ではprovider c31ef424…とnative 27ebabdd…、B3ではprovider c31ef424…、native 638e0a989…、confirmation ae3d301e…を区別する。元SOURCE欄は置換しない。SOURCE役割照合は、全コードの科学的再監査やdirty SOURCEのclean認定ではない。

303の固定結果参照に対し263の異なるGit blobを確認し、公開元表の119,244 selectorを解決した。[REFERENCE_INDEX.csv](REFERENCE_INDEX.csv) は公開locator、S1-T/S1-META/S1-GROUPのREPORT内解決先とmetadata確認を記録する。1,095のlocator/member確認を行い、調べた公開Git objectの未発見は0。private locator・別hostの実体・全raw完全性まで確認した意味ではない。

## 関係の型・独立性・更新

| 型 | metadata関係行数 | 確定範囲 |
|---|---:|---|
| COPY_OF | 16,163 | 同じ保存結果への重複観測をVERIFIEDで接続 |
| REUSES | 11,270 | 元記録の再利用と、同じ候補・cohort・条件に対する精度間の価格共有。RECORDED_ONLY |
| CITES | 9,824 | 元表/文書への引用。独立標本ではない |
| INDEPENDENT_CONFIRMATION_OF | 99 | 元記録99件をRECORDED_ONLYで保持。99独立cohortという意味ではない |
| CONDITION_VARIANT_OF | 124 | 元記録の条件変更候補を保持 |
| RECOVERS | 3 | 旧STOPを保持した部分回復。既存trajectory再利用を独立nへ加算しない |
| CONTINUES | 1 | B3の保存SOURCE履歴と継続条件。元attempt IDは未解決 |
| RERUN_OF | 1 | B3の記録済みevent-only同一seed再構成。新規独立trajectoryとしない |
| SUPERSEDES | 2 | H8の選択六軸の単価置換と会計版だけに限定 |
| INVALIDATES | 1 | 固定manifestが明示した旧DF reduced screening失効だけに限定 |

RELATIONSHIPSのsource/targetはcanonical ID、元環境:record ID、固定文書referenceまたは欠測token。文書レベルの履歴は `DOCUMENTED_HISTORY_METADATA` とし、存在しない実験行・event IDを作らない。元targetを特定できない場合はNOT_RECORDED/UNKNOWNのまま残す。全関係を科学的同一性・独立性の確認済みへ昇格していない。

探索と確認、同一seed再実行、条件変更、回復を分離した。別seed番号だけで独立性を保証しない。長時間B2の保存seed metadataはseparate_cohort=true・known_seed_collisions=0だがcomplete_historical_coverage_claimed=falseであり、既知namespaceの非衝突と全過去seed除外を区別する。Cosine/sine paired、複数精度の共通価格、同じsignalを使う別価格取得を分け、標本のpoolingや新しいn/N/Gを行わない。

H8主表とMechanismのalias候補は既存REPORTのまま残す。B3の106既存PASS再利用、後続の部分回復・event-only再構成は保存provenanceへの関係で、元STOPの解除や新しい独立実験数ではない。H8会計更新を旧raw全体の失効とせず、旧DF失効を後続PF係数・Track Aへ波及させない。

## 状態・研究系列・保全の読み分け

run/scientific/technical/acceptance/validity/preservation/access/independenceの8状態は、全119,304元recordについて個別に一致を確認してRECORD_LINKSへ保持した。canonicalに一つのPASS/採択を与えていない。元inventoryのvalidityは全行UNKNOWNであり、そのまま残す。旧DF失効等の明示文書関係は別に加え、欠けた科学条件の新規experiment行を捏造していない。

[SERIES_INDEX.csv](SERIES_INDEX.csv) から、元環境・元research_series・scope・元表へ戻れる。系列名は観測時のまま保持し、語が似ることを理由に科学的同一性へ正規化していない。主な入口は次のとおり。

| 分野 | 保存証拠への導線 | 保持した区別 |
|---|---|---|
| DF / PR-2 / Track A | PR2旧S0・rebaseline/V4・M1/M2、H4 geometry/B0I、H5/H7 pilot、H6/H8強対照、DF-longtime-B2/B3 | legacy DF12/tol-only DF10/19/28、fragment順、split、探索/確認、旧STOP |
| PF / finite RTE / normalization | pf_delta・PF_SIZE・finite_rte、元state/round/係数selector | C_PF_COEFFICIENTとC_COMPILED_COST、Bとbias、経験包絡と保証 |
| Native/compiled・資源 | 元compiled/correctness・局所Arb・cost-provider・再会計 | 技術PASS/科学判定、実測/model/proxy、有限平均/母平均、ρ仮定/実残差 |
| RPE/QPE | 短round、保存schedule、M06-Fの全roundモデル | 単一Tとround/energy、モデル集計と最終実装実証。最終費用は新採択しない |
| Pauli-JW / DF比較 | H2/H4/H6のcandidate×precision×cohort、event/Hybrid拡張、元D1/D2・実装境界 | DF B3の一体保持とPauli L_D0純RTEを別方式として保持 |

アルゴリズム改善方式のcoverageは引き続き範囲外。Phase 1にあるSUPPORT_ONLY baselineは元scopeのまま追跡し、改善方式全体の支持・主要claimへ転用しない。実測価格を量子実機shotと呼ばず、固定Tの条件付きGをfull energy/RPE評価へ結合しない。33/40や14/15等の既存科学判定を再計算・再採択していない。

## 未解決とPhase 3前の技術的照合事項

[UNRESOLVED.csv](UNRESOLVED.csv) とpart002は科学的coverage不足の一覧ではない。各groupの元record集合へ戻り、以下のmetadata状態を分ける。

- 元データ未展開：Phase 1に未登録の大grid・旧PM全解析版等は、今回の登録行へ追加・推定していない。
- SOURCE/input/state未結合：同一保存結果を確定できても、全scientific identityの再認定はしていない。component roleで説明できない相違は未解決に残す。
- run/cohort独立性：既存99関係は保持するが、完全seed・stream・価格/signal共有・回復trajectoryの閉じた対応が必要な箇所を未知として残す。
- 旧host alias：S1 REPORTは旧SERVER_Aの一部root/UID/HEAD/custodyの一致を記すが、host全体のSERVER_A/B→SERVER_1/2対応へ拡張しない。
- raw/archive保全：各Phase 1の保存・アクセス欄を保持。全raw/QPYの復元・hash・decode・独立backup認定はしていない。
- 失効/更新・正式受理：元行の未知を推測補完せず、明示文書の更新範囲と行単位の結合を後続照合事項にする。
- 未発見：UWC原JSON等のGAP記録は限定探索の履歴。科学実験未実施の証明ではない。
- 権限/公開制約：LOCALの不透明V4 60参照、他hostのprivate実体は未取得。公開入力だけで解けないものを他host調査で埋めない。

Phase 3へ渡す前の技術的事項は、必要なclaimに対応する未展開元表・source/state/law/測定境界・独立cohort・受理/失効・raw custodyの個別joinである。全未知を追加計算の義務や不足A〜Eへ変換せず、判断が必要ならユーザーへ戻す。

## 技術検査・再利用・資源・原本保全

14件の人工identity/CSVテストがPASS。等しい保存blob/path/selector、違う数値ではなく同じ結果、異なるrow/round/path/rank、SOURCE役割、別attempt、同一seed再実行、別seedだけの非保証、明示disjoint確認、複数行CSV、null/欠測token、JSON pointer、匿名化を検査した。全元IDからgroupへの一対一の対応、group membership、関係参照、分割ヘッダ/全行数、原状態・原ID逐語保持、旧結果の非削除、公開差分を確認した。

既存Phase 1のschema/参照検査、check_validation_manifest.py、inventory_pauli_hybrid_rte.py、document_pauli_jw_saved_summary.py等のliteral転記・固定Git参照の方式を再利用した。既存runnerは系列固有出力/科学処理へ接続するためimport・実行せず、stdlib-onlyの小さなadapterへ必要機能を移した。科学runner、NumPy/Qiskit/SCF/DF/state生成、simulation、sampling、compileは0件。

使用可能CPU affinityは32、起動時available RAMは約52GiB、専用一時領域の空きは約231GiBだった。32の独立CSV part処理を32 workerでstreaming解析し、元結果metadata照合も独立file単位で並列化した。人工CPU/RAM capは設けず、不要な1.55GB復元コピー・2GB API一括取得・科学raw再hashを避けた。metadata hashは公開入力/統合成果物の整合確認に限定。共有設定・他ユーザー・他jobに変更を行わない。

証拠treeはGIT_OPTIONAL_LOCKS=0 / GIT_NO_LAZY_FETCH=1の読取だけ。主証拠HEAD・既存tracked statusと固定入力commit/blobは保持されている。並行する別作業のref/未追跡追加は非atomic観測として扱い、凍結・巻戻しをしていない。独立出力Git/indexを用い、元Phase 1・科学SOURCE/raw/input/result/grant/ledger/marker/STOP、研究概要・map・manifestを変更しない。

公開差分には匿名化した統合CSV・本REPORT・必要なmetadata用コード/人工テスト/検査要約だけを含める。内部host/IP、絶対保存パス、資格情報、private rawや逆引きを含めない。公開用の不透明IDを実保存root/別hostの実体確認と呼ばない。

## ファイル・schema・復元順

入力43列をそのまま再複製せず、以下の軽量表と固定commit/record IDで追跡する。Phase 2の出力schemaはmetadataカタログ用であり、共通PR-COVERAGE-1の原定義は変更しない。配列/objectセルはJSON。数値0・科学的null・UNKNOWN等は元入力で保持し、欠測を0へ変換しない。

- CANONICAL_CATALOG：group、anchor、元series/条件、run/SOURCE、原表blob/selector、状態参照、確定範囲。
- RECORD_LINKS：全元record→group、元固定file/data row、元ID・8状態・source値。
- RELATIONSHIPS：type、source/target、metric scope、basis、VERIFIED/RECORDED_ONLY/UNKNOWN、独立性の意味。
- UNRESOLVED：group/global参照、metadata分類、限定理由、必要な照合。科学充足判定NO。
- SOURCE_COMPONENTS / REFERENCE_INDEX / INPUT_FILES / SERIES_INDEX：役割、固定所在、分割・取得、系列別の入口。

分割表は同名の無suffixファイルが第1、part002→part003…が後続。ヘッダを1回だけにしてCSV parser/writerで連結する。下表が全分割の順序と行数・bytes・SHA-256を定める。単一表だけを系列全体として扱わない。

| ファイル（表内の順） | data行数 | bytes | SHA-256 |
|---|---:|---:|---|
| [SOURCE_COMPONENTS.csv](SOURCE_COMPONENTS.csv) | 13 | 6838 | `1ebce130b3607f9440bf2e29a4fccfa04bb2e6cf3c1eae23ec642076e2f5556a` |
| [CANONICAL_CATALOG.csv](CANONICAL_CATALOG.csv) | 37806 | 50331864 | `668c68bf24625dc2a72666241846450db9060d9298a082a2e550f94681f435e7` |
| [CANONICAL_CATALOG.part002.csv](CANONICAL_CATALOG.part002.csv) | 36881 | 50332451 | `2d93f06d95edca0adba940a8ce365022a63c84c79e74ea6e95cba10d313637cf` |
| [CANONICAL_CATALOG.part003.csv](CANONICAL_CATALOG.part003.csv) | 28454 | 36864615 | `d0ec485ca61a97828783c31dc640800d3b70275bf023b6307e20fdabbb76f573` |
| [RECORD_LINKS.csv](RECORD_LINKS.csv) | 61663 | 50332135 | `447c9927f5083358675929a459aaa7a532498b533b24e46fb5cbf9264ffcef92` |
| [RECORD_LINKS.part002.csv](RECORD_LINKS.part002.csv) | 57641 | 46272794 | `2b86958e86307e2b7e5651897358dc414acb963d5b0fdb6cc24fff5fe0b36127` |
| [RELATIONSHIPS.csv](RELATIONSHIPS.csv) | 37488 | 24540066 | `e359618ecaab8875bc5c70174546f5e34116badec12770743393ab641b5150e7` |
| [UNRESOLVED.csv](UNRESOLVED.csv) | 82934 | 50331859 | `431e5736b740b7183be0b69276061eb31640730c3511a05eef16ccf3e3d2230b` |
| [UNRESOLVED.part002.csv](UNRESOLVED.part002.csv) | 20211 | 11992607 | `5997c65ac8a3ffb37c2fa67746fcbe10a15ba9db389ec4636bae88f3fcf46b65` |
| [REFERENCE_INDEX.csv](REFERENCE_INDEX.csv) | 1095 | 698371 | `33120ab4ac8843e9264a0f7fb14d85a3fb6a07d45a12074204ee3a698841eecb` |
| [INPUT_FILES.csv](INPUT_FILES.csv) | 36 | 17230 | `f04410c86ed935a421fd6679d9198e513262684f648cd649b25029061926b525` |
| [SERIES_INDEX.csv](SERIES_INDEX.csv) | 155 | 89339 | `c0c3b9b08cdb53a2e5a11f501174852584df2c86f360b6e51e5a0d2a9f99bdba` |

実装はmetadata_integration.py（入力streaming/schema/detail参照）、integrate_catalog.py（identity/関係）、verify_outputs.py（参照・原状態・membership検査）、test_metadata_integration.py（人工検査）。SCIENCE関数はimportしない。入力は固定Git objectから読み、--repo/--cache/--outは作業領域へ明示する。既存科学treeへの出力先変更やrunner実行の許可を与えるコードではない。

## 公開照合・mandatory STOP

新branch `codex/pr-cross-host-coverage-phase2-20261103` に通常commit/pushする。既存branch更新、force push、main merge、cleanupはしない。公開後にremote SHA、全公開fileのblob/sizeと本文bytesを照合し、その最終SHA・照合結果を最終チャットへ報告する。自己参照のため本REPORTへ自分の最終commit SHAを埋め込まない。

**公開・照合後にPhase 2 mandatory STOP。Phase 3、科学的coverage matrix、不足A〜E分類、追加計算、詳細科学レビューへ自動移行しない。** 次の担当はユーザーであり、必要なPhase 3または限定的な技術照合を別指示で指定する。
