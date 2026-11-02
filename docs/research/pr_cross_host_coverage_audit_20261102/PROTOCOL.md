# PR研究・3環境横断coverage監査 共通仕様 v1

2026-11-02 JST。Repository: `HIROMU1015/Partially-Randomized-Trotter`。
**本成果物はPhase 0の準備仕様。実験カタログ、coverage評価、科学的採択ではない。**
作業基点は `9a0bc7f17f47823e46c3945734127e9f0d09154c`。
[空CSV](INVENTORY_TEMPLATE.csv)と、Phase 0最終公開commitに同梱する
`HOST_INSTRUCTIONS.md`の環境別指示を併用する。
仕様と空CSVを先にcommitしてから、その実在SHAを環境別指示へ固定する。
最終公開commitと仕様固定commitが異なる場合も、仕様・CSVのblob一致を確認する。

## 1. 目的、担当、実行境界

研究目的・主張に対して、検証済み条件、条件の集中、比較の公平性、独立確認、
重複引用、保存証拠の漏れ、失効・更新、未実施と未整理の違いを後続で調査する。
今回はその方法だけを定める。全条件の直積網羅やPRの勝利を完成条件にしない。

| Phase | 担当・作業 | 今回の状態・終了境界 |
|---|---|---|
| 0 | LOCAL Codex：資料確認、範囲・schema・照合規則・環境別指示の作成 | 今回のみ許可。文書検査、commit/push、remote SHA・公開本文照合後mandatory STOP |
| 1 | LOCAL／SERVER 1／SERVER 2の各Codex：担当環境の既存証拠をread-only棚卸し | 未起動。ユーザーの別指示後、環境ごとに公開・照合してSTOP |
| 2 | LOCAL Codex：3環境成果物の同定、重複・継続・失効関係の統合 | 未実施。別指示が必要。カタログ公開後STOP |
| 3 | 別指示で指定する監査担当：主張別coverage・不足候補A～Eを評価 | 未実施。科学的レビューを要する範囲は別の開始許可。新計算は含めない |

Phase 1の指示が許可するのは保存文書・既存値の引用とmetadata照合である。
保存値からの新しいN/G・比・平均・SE・fit・ranking・費用再会計は数値再解析であり、
別の明示的実行許可が必要。simulation、sampling、RNG replay、Hamiltonian/state/reference生成、
回路build/compileも禁止する。Phase 2/3の開始許可はこれらの実行許可を兼ねない。
「次は何をするか」という質問や計画採用だけでは次Phaseを開始しない。

## 2. 対象と除外

| 対象 | 含める科学的検証と境界 |
|---|---|
| A：DF部分ランダム化PF | PR-2、Track A、H4/H6/H8等、geometry移送、DF rank・fragment順・分割L_D、B0/B0+I/B2/B3/Full PF、強い決定論対照、固定候補探索・独立確認。H2、open-shell pilot等も目的が該当すれば含める |
| B：PF・finite RTE | PF次数・係数・bias、K、q/r/T/刻み、law、normalization・attenuation、coherent cosine/sine、数値精度・誤差評価。既知PFの評価をアルゴリズム新規設計と区別 |
| C：回路・資源 | Native correctness、compiler/backend/control条件、実compiled RZ/CX/depth、model/proxy照合、希少event・tail、sampling・選抜、十分shot設計、条件付きG |
| D：RPE/QPE接続 | round別信号・位相・失敗確率、誤差予算、round価格、full interrogationのモデル総費用、energy精度への接続。固定TのGとの境界を保持 |
| E：Pauli-JW比較 | 純RTE/Hybrid、PF/B0/PR、Pauli–DF、P1/P2/P3等の既存原因検証、実装残差・価格tail。DF主研究と目的・条件を分離 |

アルゴリズム改善・新規提案自体のcoverageは除外する。Track B共設計、Return Aggregation、
新event生成・sampling設計、理論的新規性、独立した改善案が該当する。
branch／directory名だけでは決めず、研究目的・主張・評価対象で判断する。
改善系列内の既存PR baseline・誤差評価・回路検証がA～Eの主張へ直接必要なら
`SUPPORT_ONLY`として該当証拠だけを関連付け、改善方式のcoverageには加算しない。
判断不能は`SCOPE_UNRESOLVED`として理由を残し、Phase 3へ渡す。

DFのB3は二体DF prefixが空でも一体項を保持し得る。Pauli純RTEと同じ方法に正規化しない。
DF fragmentのL_DとPauli項prefix、legacy DF12とtol-only DF10、異なるlaw・stateを結合しない。
既存資料にあるP-A等のアルゴリズム設計結果も、関連する比較・検証部分だけを扱う。
repository整理、branch削除、worktree管理は科学的検証ではなく、所在追跡の補助情報である。

## 3. 再利用する既存資料と時点

以下はPhase 0で確認した資料の参照版である。研究結果の再検証や最新性の認定ではない。
相対pathは特記しない限りrepository内。固定commitは文書／保存版のcommitであり、
そのまま科学実行SOURCEとはしない。

| 資料・固定版 | 再利用する内容 | Phase 1へ残す確認 |
|---|---|---|
| [研究目的・研究課題][PURPOSE]、[研究方法・解析手順][METHOD]、[研究概要・現状][SUMMARY]、[PROJECT_MAP][MAP]（基点9a0bc7f） | RQ1～3、有限signalと全round目標の区別、入力・誤差・測定境界、文書の優先順位 | 日付付き追記を当時のsnapshotとして保持し、後続の正式なscope・結果・採択に更新があるか |
| [VALIDATION_STATUS][STATUS]、[validation_manifest][MANIFEST]（基点9a0bc7f） | 失効／raw欠落／再現性／人工検査等の既存区分と証拠への入口 | proseとmanifestの時点差、旧失効と後続回収、schema差を証拠付きで記録。原本の修正はしない |
| [全研究landscape][LANDSCAPE]、[主張・証拠対応表][CLAIMS]（指定9a0bc7f） | 系列、固定SOURCE・result・文書の区別、A01～A13/R01～R02/F01/P01～P05、引用alias・公開差・後続証拠への導線 | 11/01時点からの更新、B3公開版とlocal最終版差、H8独立主表のalias等。旧U01～U10は既存論点であり今回の不足判定ではない |
| [Track A横断inventory][TRACKA]（指定4591b8d） | 19系列・E01～E32等の固定証拠索引、実測/model/finite meanの区別、探索・確認、通常/非相殺、STOP/回復関係 | 10/17対象の旧snapshot。後続H8 rare、強化対照、独立確認、長時間B2/B3等による更新を照合 |
| [70研究カード][CARDS]、[SOURCE_AND_PROVENANCE_INDEX][SOURCEINDEX]（a9bdcf1） | 全研究への探索入口と固定SOURCE/result索引。全カードを対象研究へ自動算入しない | A～Eの評価・baselineに関係するカードだけ抽出。別PF repositoryへの参照は関連する証拠だけ対象 |
| [LOCAL既存環境inventory][ENVLOCAL]、[旧SERVER_A環境inventory][ENVA]（fa55c162、branch `codex/pr-environment-inventory-20261026` の読取版） | 匿名ID、Git履歴・archive・保管依存、削除後も証拠を追跡する方法 | 10/26～27の保全・整理資料であり実験カタログではない。旧SERVER_A/Bと今回SERVER 1/2の対応を推測しない |
| [Track A local inventory][TLOCAL]、[server inventory][TSERVER]（基点9a0bc7f、10/14記録） | 既存export、SOURCE索引、共通Gitと独立clone、公開commitとbranchの違い | 旧host名を現在の担当へ自動対応させない。記載済み所在を各環境の許可範囲で確認 |

LOCALの既知整理資料 `server2_repository_inventory_20261102.tsv`、
`remote_branch_cleanup_handoff_372_20261102.tsv`、`remote_branch_cleanup_result_20261102.tsv`
は、列定義を確認した所在追跡の候補である。内容の実験棚卸し・行の転記はしていない。
これらの絶対path・非公開保存先を公開しない。削除済みbranch/worktreeは既存archive、
公開carrier、固定Git objectの索引で追う。branch不在はcommit／証拠不在の証明ではない。

文書の優先順位は、現行の規範・synthesis → status/manifest → 個別検証資料・原結果 →
日付付き判断履歴 → 旧protocol・presentation類とする。異なる版で食い違う場合は
両版・対象条件・時点を残す。新しい日付だけを理由に旧結果を失効させない。

既存の`check_validation_manifest.py`、`inventory_pauli_hybrid_rte.py`、
`document_pauli_jw_saved_summary.py`を静的に確認した。metadata・literal引用・参照検査の
考え方を再利用する。既存schemaや特定科学系列の出力先へ結合されたrunnerは今回実行しない。
新schema用の管理システムや科学runnerは作成しない。

## 4. 記録単位と識別

ファイル数ではなく**科学的に区別できる条件**を基本にする。
CSVの1行は「条件 × 実行/cohort × 既存解析版」の観測であり、REPORT一個の行ではない。
同じ条件・run・解析に複数metric／axisがあれば同じ行の`evaluation`へまとめてよい。
precision、round、method、cohort、受理状態が変わる場合は分離する。
大きいgridを一行のrangeで済ませず、既存の条件別機械可読表を参照し、
行キーと固定列定義で全条件が一意に復元できる場合だけ補助JSONLへ委ねる。
将来の未実施条件の直積を生成しない。

| キー | 意味・対応 |
|---|---|
| `record_id` | 観測環境内の一意ID。例の形式は`LOCAL-REC-<連番>`。環境間で同じIDに上書きしない |
| `experiment_id` | 原資料の研究実験ID。無ければ環境内の仮IDを付け、元IDが無いことを注記 |
| `condition_id` | 物理入力・方法・比較・精度・state/reference・law・測定境界等の科学条件の識別子。未知なら同一と決めない |
| `run_id` | 実際の一回の実行／attempt。copyや文書引用で新runを作らない。継続・回復は親関係を持つ別attemptまたは明記された同run |
| `analysis_id` | 保存済み解析／費用会計版。新しい解析は行わない。同じrawからの再会計は新標本ではない |
| `evidence_group_id` | 当該条件・run・解析を支える証拠群。REPORT/JSON/CSV/raw/QPY/STOP等をrole付きで束ねる |

各環境の仮IDはPhase 2で元IDを保ったままcanonicalな対応群へ結び付ける。
Phase 1は候補関係と根拠を記録し、他環境との同一性を確定しない。
原資料に記載のない実験・値は作らない。未実施の行は計画等にその条件と未実施の明示がある場合に限る。
証拠未発見の索引項目は、条件が特定できればUNKNOWN付き行、特定不能なら報告の未解決一覧へ置く。

## 5. 共通CSV schema

`INVENTORY_TEMPLATE.csv`はヘッダだけのUTF-8/RFC 4180 CSV（schema `PR-COVERAGE-1`）。
ヘッダ順序は全環境で同じにする。JSONセルはCSV writerで引用し、JSON値を二重に手書きescapeしない。
下表の43列を使用する。`string`は原表記または定義された欠測token、`refs`は§5.2の参照配列。
省略可能な情報も空欄にせず、適切なtokenを用いる。

### 5.1 列名・型・意味

| 列名 | 型 | 定義 |
|---|---|---|
| `schema_version` | string | `PR-COVERAGE-1` |
| `record_id` | string | §4の環境内観測ID |
| `environment` | enum | 観測担当`LOCAL` / `SERVER_1` / `SERVER_2`。実行hostはsampling内で別記 |
| `observed_at` | string | ISO 8601・timezone付き観測時刻。原run日時とは別 |
| `research_series` | string | 元系列名・study ID。PR-2/Track A/finite RTE/Pauli等の区別を保持 |
| `scope_class` | enum | `INCLUDED_VALIDATION` / `SUPPORT_ONLY` / `SCOPE_UNRESOLVED`。除外系列の全行作成は不要、報告に除外理由を記す |
| `claim_refs` | refs | 目的・RQ・原claimへの参照。既存A01等のIDは文書commitと組み合わせる |
| `experiment_id` | string | §4の論理実験ID |
| `condition_id` | string | §4の科学条件ID |
| `run_id` | string | 原実行ID、または欠測token。未実施は`NOT_APPLICABLE` |
| `analysis_id` | string | 保存解析版・posthocかどうかを識別できるID |
| `evidence_group_id` | string | 同じ条件を支える複数ファイルの証拠群ID |
| `source_commit` | string | 科学SOURCEの確認済みfull SHA。dirty等はcode_refsに差分metadataを残し、carrier SHAを代用しない |
| `code_refs` | refs | 実行コード・version・SOURCE binding。別componentのSOURCEもrole別で保持 |
| `input_refs` | refs | 分子/Hamiltonian/state/reference等の保存snapshotの所在・既存identity |
| `contract_refs` | refs | 科学契約・範囲・one-shot/STOP等の原資料。承認証明の保存を求める列ではない |
| `molecule` | string | 分子名・サイズ。人工系は元fixture名と`SYNTHETIC`を明記 |
| `geometry` | object | 配置、単位、geometry ID・snapshot等。§5.3 |
| `basis` | object | 基底関数、活性空間、写像等。§5.3 |
| `electronic_state` | object | 電子数、charge/spin/sector、入力状態とreference。§5.3 |
| `hamiltonian` | object | 表現・生成/切断条件・rank・fragment順・identity等。§5.3 |
| `method` | object | PF/RTE方式と次数・L_D/q/r/K/T/刻み・law等。§5.3 |
| `comparison` | object | 対照・候補集合・選抜規則・公平性に必要な対応。§5.3 |
| `precision` | object | 要求精度・誤差・α・演算精度・保証。§5.3 |
| `implementation` | object | compiler/backend/control/測定境界・価格単位。§5.3 |
| `sampling` | object | seed/cohort・探索/確認・n・IID/依存・共有標本。§5.3 |
| `evaluation` | array | metric定義・原結果参照・根拠種別。新算術は行わない。§5.3 |
| `result_refs` | refs | 保存結果・table/row/field。REPORTだけの場合もその制約を記す |
| `run_status` | enum | §6の実行状態。exit0と科学的成功を分離 |
| `scientific_status` | enum | §6の保存された科学判定。今回新判定しない |
| `technical_status` | enum | §6の実装／metadata検査。科学PASSと分離 |
| `acceptance_status` | enum | §6の正式受理状態。原判定の出典が必要 |
| `validity_status` | enum | §6の旧結果失効／更新状態と出典 |
| `stop_refs` | refs | STOP/失敗/marker等の既存記録。内容を書き換えない |
| `relationships` | array | §7のtype・target ID/ref・scope・basis_refs・照合状態 |
| `identity_basis` | object | 入力・SOURCE・契約・candidate・state/ref・law・compiler・seed・結果identityの照合根拠、不一致・未知。§7 |
| `independence_status` | enum | `NOT_ASSESSED` / `REUSED` / `OVERLAPPING` / `DOCUMENTED_DISJOINT_COHORT` / `DETERMINISTIC_REPEAT` / `UNKNOWN`。独立性の保証ではない |
| `evidence_refs` | refs | 全証拠群のrole付き所在、現存/履歴/archive、アクセスとidentityの確認状態 |
| `preservation_status` | enum | `PRESENT` / `ARCHIVED` / `GIT_HISTORY` / `NOT_FOUND` / `MIXED` / `UNKNOWN`。各参照にもstatusを付ける |
| `access_status` | enum | `ACCESSIBLE` / `PARTIAL` / `NOT_ACCESSIBLE` / `REDACTED` / `UNKNOWN` |
| `reliability` | array | 保存された根拠種別と適用metric/条件・refs。§6。強弱の一値へ潰さない |
| `missing_fields` | array | 欠測fieldのdotted path、token、探索した範囲・根拠ref、後続の照合事項 |
| `limitations` | string | 適用域、未保証・相違、既存値の共有、時点差、除外・未解決理由 |

### 5.2 共通の参照・欠測・補助ファイル

参照はJSON配列とし、各要素に`ref_id, role, locator, version, selector, preservation, access, identity`を持つ。
`locator`は固定commit付きGitHub URL、repo相対path＋full commit、または公開用の不透明な保存ID。
`version`は観測した文書commit／archive ID／snapshot。`selector`は見出し、CSV row ID、JSON pointer等。
`identity`は既存SHA/blob/結果IDと、その確認が`RECORDED`か`VERIFIED`かを記す。未照合は明示する。
必要な少量metadata/hash照合だけを許可し、raw全件hash・全依存hashを要求しない。
REPORT/JSON/CSV/raw/QPY/STOPはroleで区別する。QPYの生成／再compile／実行は行わない。

| token | 用途 |
|---|---|
| `UNKNOWN` | 調査範囲では値・意味を確定できない |
| `NOT_RECORDED` | 確認した原資料に該当情報の記載がない。全保存先に無いという意味ではない |
| `NOT_APPLICABLE` | 当該検証には適用されない。理由をmissing_fields等へ |
| `NOT_ACCESSIBLE` | 存在候補・参照先はあるが権限／接続／許可範囲で読めない |
| `NOT_FOUND` | 明記した限定探索範囲で証拠を発見できなかった。未実施と同義ではない |
| `REDACTED` | 非公開情報を公開コピーから除いた。環境内の対応は保全する |

数値0、空文字、JSON `null`、空配列を欠測tokenの代用にしない。既存の科学的な`null`は
値の文字列`"null"`と元field・理由を記録し、0へ変換しない。集合として要素がないことを
確認した参照・関係配列は`[]`、不明な集合は`"UNKNOWN"`等を用いる。
JSON objectの必須keyにもtokenを入れる。原数値を引用する場合は丸め直さず文字列として転記する。

条件や参照が長い場合だけ、同じIDを持つ`DETAILS.jsonl`（1 object/line）へ当該セルの値を外出しできる。
セルは`{"detail_ref":"DETAILS.jsonl#<record_id>/<column>"}`、補助側は
`{"record_id":...,"column":...,"value":...}`とする。展開後の型・意味は本仕様に一致し、
外出しは一段だけ。公開コピー内で参照が解決し、条件・cohortを一意に特定できること。
外出し先が非公開ならtokenと限定理由を残し、完全な照合可能性を主張しない。
補助DB、grant発行、承認証明、専用管理基盤は作らない。

### 5.3 条件objectのkey辞書

下表のkeyはそれぞれ記録する（該当なし・欠測はtoken）。追加の原fieldは説明付きで保持できる。
`refs`を伴う値は§5.2へ接続し、同じ語が旧契約で別定義なら原定義を優先して記す。

| 列 | 必須keyと意味 |
|---|---|
| `geometry` | `id`：元geometry ID、`configuration`：linear等の配置、`coordinates_ref`：座標snapshot、`distance`：距離の原文字列、`unit`：Å/bohr等 |
| `basis` | `name`：基底関数、`active_space`：活性空間、`orbital_basis`：AO/MO等、`mapping`：JW等、`qubits`：system/ancilla数と順序 |
| `electronic_state` | `electrons,charge,spin,sector`：電子状態、`input_state_ref`：保存状態・ray/正規化、`reference_ref`：exact/FCI等の基準・生成法と版 |
| `hamiltonian` | `representation`：DF/Pauli等、`snapshot_ref`：入力identity、`rank_policy`：tol/cap/legacy等、`rank`：返却rank、`fragment_order_ref`：生成順/norm順と列、`thresholds`：cutoff、`identity_policy`：scalar/tail identity、`one_body_policy`：一体保持等、`decomposition_ref`：分割単位・原定義 |
| `method` | `family`：FullPF/B0/B0I/B2/B3/純RTE等の原定義、`pf_order`：S2/Y4/m5等、`L_D`：分割位置、`q,r,K,T,time_step`：原条件と単位/定義、`law_ref`：signed/paired等のlaw、`normalization_ref`：B/ν/log_B/補正・attenuationの定義と保存値参照、`round_id`：RPE round等 |
| `comparison` | `baseline_refs`：FullPF/B0/B0I/高次PF等の対照、`candidate_id`：元候補、`candidate_set_ref`：探索domain・範囲・適格条件、`selection_rule_ref`：nominee/停止/事前固定・posthoc、`matched_fields`：揃えた条件、`differing_fields`：相違、`comparison_scope`：matched tuple/family-best/transfer等 |
| `precision` | `target`：signal/phase/energyと要求精度・単位、`error_budget_ref`：PF/RTE/統計/数値/実装残差の配分、`failure_probability_ref`：axis/round/total α、`arithmetic`：binary64/Arb bits等、`tolerance_ref`：許容差、`guarantee_scope`：経験値/包含/理論等と仮定 |
| `implementation` | `compiler`：名前/version/opt/seed/basis-gates、`backend`：target/backend/routing、`control`：generic/central/native等、`measurement_boundary`：body/full Hadamard/no-prep/full RPE等、`cost_units`：RZ/CX/depth/T等、`state_preparation_policy`：P=0等、`runtime_ref`：必要な保存環境・compiler条件 |
| `sampling` | `execution_environment`：元run hostの既知alias（観測環境と別）、`role`：EXPLORATION/CONFIRMATION/HOLDOUT/DIAGNOSTIC/DETERMINISTIC等、`seed_spec,rng,cohort_id,sample_count,sample_ids_ref`：seed/stream/index/nの原記録、`price_cohort_id,signal_cohort_id`：共有・別標本、`draw_law_ref`：IID/条件付き層・weight、`dependence_ref`：paired axis/共通乱数/重複、`selection_data_refs`：選抜に使ったデータ、`confirmation_plan_ref`：固定時点・nominee・逐次停止 |
| `evaluation`の各要素 | `metric`：signal/複素bias/係数/B/N/C/G/RPE等、`definition_ref`：定義・単位・符号/分母、`axis_or_round`：cos/sin等、`result_ref`：保存row/field、`basis`：§6根拠種別、`status_ref`：当該metricの保存判定。`C_PF_COEFFICIENT`と`C_COMPILED_COST`、trajectory nとshot Nを明記して区別 |

## 6. 判定状態・証拠種別を分離する

以下は原判定の意味を損なわず正規化する列であり、Phase 1で新しい科学的PASSを発行しない。
各状態の原tokenと出典はresult_refs／status_ref／limitationsで保持する。
軸・metricごとに異なる場合は`MIXED`とし、evaluationまたは別行に内訳を持たせる。

| 列 | 値と意味 |
|---|---|
| `run_status` | `COMPLETED`、`PARTIAL`、`FAILED_DURING_EXECUTION`、`STOPPED_BEFORE_EXECUTION`、`NOT_EXECUTED_DOCUMENTED`、`IN_PROGRESS_REPORTED`、`UNKNOWN`。計画書やSTOPだけで実行を推定しない |
| `scientific_status` | `PASS_REPORTED`、`FAIL_REPORTED`、`UNRESOLVED_REPORTED`、`NOT_EVALUATED`、`MIXED`、`UNKNOWN`。負結果・PR非優位も科学結果として保持 |
| `technical_status` | `PASS_REPORTED`、`FAIL_REPORTED`、`NOT_CHECKED`、`MIXED`、`UNKNOWN`。人工検査、native probe、metadata検査のscopeを区別 |
| `acceptance_status` | `ACCEPTED_DOCUMENTED`、`NOT_ACCEPTED_DOCUMENTED`、`PENDING_DOCUMENTED`、`NOT_APPLICABLE`、`UNKNOWN`、`MIXED`。TARGET_METやexit0から正式受理へ格上げしない |
| `validity_status` | `CURRENT_AS_DOCUMENTED`、`HISTORICAL`、`SUPERSEDED_DOCUMENTED`、`INVALIDATED_DOCUMENTED`、`CONFLICTING`、`UNKNOWN`、`MIXED`。失効は明示的な訂正等の出典が必要。単なる新条件は旧結果失効ではない |

根拠種別（reliabilityおよびmetricのbasis）は
`THEORETICAL_GUARANTEE`、`VALIDATED_NUMERICAL_ENCLOSURE`、`EMPIRICAL_NUMERICAL_CHECK`、
`MEASURED_COMPILED`、`FINITE_SAMPLE_MEAN`、`MODEL_ESTIMATE`、`PROXY`、`SYNTHETIC_CHECK`、
`CONDITIONAL_EVALUATION`、`PROSE_ONLY`、`UNKNOWN`を使用し、適用条件・出典を付ける。
複数種別は併記する。実compiled単価でも、有限平均・ρ=0・経験的allowance等の条件が残り得る。
実回路をcompileした価格と量子実機shotsも区別する。
局所6対象の包含、未使用列長holdout、経験的C_useを全law／全gridの厳密保証へ拡張しない。

## 7. 重複・独立確認・継続・失効の照合

`identity_basis`には次のkeyを持たせ、`value/ref, match_state, basis_refs`を記録する：
`physical_input, source_contract, candidate_condition, state_reference, law_normalization,
compiler_boundary, seed_cohort, continuation, result_identity`。
`match_state`は`VERIFIED` / `RECORDED_ONLY` / `DIFFERENT` / `UNKNOWN`。
必要な入力snapshot・既存hash・保存run ID・result identityを組み合わせる。
文書commit、branch、host、同じ数値だけの一致では同一性・独立性を確定しない。
全依存のbitwise一致は要求せず、科学条件へ影響する差を残す。固定契約の条件は保全する。

関係要素のkeyは`type,target,scope,basis_refs,match_state`。
targetは既存record/run/evidence IDまたは固定参照。不明は不明のまま候補関係として記す。
以下の関係はmetric単位で併存できる（signal再利用＋価格だけ別標本確認など）。

| type | 識別根拠・Phase 2での扱い |
|---|---|
| `COPY_OF` | 同じ保存run/result identityの複写・archive・host間コピー。所在は複数でも同一結果として数える |
| `RERUN_OF` | 同科学条件の新実行。別attemptを保持。同seed/同標本再生は独立標本とはしない。科学に影響するSOURCE差は併記 |
| `INDEPENDENT_CONFIRMATION_OF` | 固定nominee・確認条件・cohortの非重複を原記録で追跡できる別seed/stream標本。seed番号だけでは認定しない。独立の対象metric・探索との関係を限定 |
| `CONDITION_VARIANT_OF` | geometry/rank/state/law/精度/PF次数/測定境界等が変わる新条件。移送や一般性の検証と同条件再現を区別 |
| `CONTINUES` / `RECOVERS` | 親run・carry/index・完了部分の再利用・未取得部分を記録。復旧済み部分を独立runとして加算しない。旧STOPは解除しない |
| `SUPERSEDES` / `INVALIDATES` | 明示的な後続訂正・受理・再会計の出典と対象metricだけを結ぶ。旧raw・値・STOPを残し、有効域と時点を分ける |
| `CITES` / `REUSES` | 複数資料から同じ値を引用、または同じsignal/price/cohortを再使用。独立検証数は増えない |

曖昧なら統合せず`UNKNOWN`と照合事項を残す。異なるbranchにあってもコピー、同じSOURCEでも
条件変更、別seedでも探索への事後利用・標本重複があり得る。
決定論の再実行は再現確認であり、Monte Carloの独立n増加ではない。
探索と確認を混ぜず、確認値の再選抜や逐次n延長は原手順と停止規則を記録する。
同価格標本を共有する5精度・多数precision点は独立試行数へ加算しない。
共通baseline、cos/sin paired、signal共有も依存として残す。
Phase 2で表示する件数は条件数、run/attempt数、独立cohort数、証拠ファイル数、引用数を別にする。

## 8. Coverage matrixの仕様（Phase 3用、今回は空の設計のみ）

一行は「既存claim × 明示された適用域 × 必要な証拠接続」。全軸の直積は作らない。
claimの出典・正式採択／草案／仮説の別を最初に固定する。
条件の集中を見る集計は重複・共有cohortを除いた単位と分母を示し、
分子数・行数が少ないことだけで科学的不足とはしない。

| 軸群 | 記録・照合する条件 |
|---|---|
| 物理 | 分子サイズ、geometry、basis/active space、spin/sector、DF rank/表現・生成条件、入力state/reference |
| 方法 | 決定論PF次数、fragment順とL_D、q/r/K、T/刻み、RTE law/normalization/attenuation、誤差予算・要求精度 |
| 比較 | Full PF、B0/B0+I、B2/Hybrid、B3、Pauli純RTE、強い決定論候補、同精度/α、候補集合・選抜、公平なmatched/family比較、独立確認 |
| 実装・資源 | native correctness、compiler/control/測定境界、実価格とmodel/proxyの適用域、希少event/tail・標本、十分N・条件付きG、RPE全round/energyへの接続 |

matrixの必須列は次とする（後続でMarkdown/CSVの最小表として作成し、今回データは入れない）：
`claim_id, claim_ref, claim_status, claimed_domain, required_evidence_link, applicable_axes,
condition_refs, canonical_evidence_groups, independent_cohorts, comparison_alignment,
coverage_state, assurance_scope, preservation_access, historical_conflicts, unresolved_lookup,
candidate_gap_id, assessment_basis_refs`。
`required_evidence_link`はbias→headroom→N、law→価格→G、round→energy等の必要な接続を記す。
`applicable_axes`は必要な軸だけと理由。条件付き固定T主張へfull energy検証を一律要求しない。

coverage_stateは`SUPPORTED_WITHIN_RECORDED_SCOPE`、`PARTIAL_EVIDENCE`、
`NOT_EXECUTED_DOCUMENTED`、`EVIDENCE_UNLOCATED`、`NOT_ACCESSIBLE`、
`INVALIDATED_ONLY`、`CONFLICTING_EVIDENCE`、`NOT_APPLICABLE`、`NOT_ASSESSED`。
保証種別・前提は別列とし、文書やstatusのPASSでセルを自動充足させない。
異なる科学条件は別セル、同じ保存標本の再引用は同じevidence groupとする。
不適格／負結果はcoverageの証拠であり、未実施へ落とさない。

## 9. 不足候補の分類とPhase 2/3の手順

| 分類 | Phase 3での意味 |
|---|---|
| A | 中心的な研究主張に直接影響する検証不足 |
| B | 一般性・再現性の補強に有用 |
| C | 現在の条件付き主張には不要だが、より強い主張には必要 |
| D | 実施済みの可能性があり、先に証拠照合が必要 |
| E | 今回の研究範囲外 |

不足候補の必須欄：`gap_id, related_claim_ref, present_evidence_refs, missing_link,
effect_on_conclusion, resolvable_with_existing_evidence, requires_new_science,
discriminating_question, category, rationale, priority_and_reason, decision_status`。
既存証拠の閲覧・照合で解決できる場合と、保存値の新しい解析を必要とする場合は分ける。
後者も科学実行許可が必要であり、「既存データだけで可能」は実行許可を意味しない。
分類・優先順位をPhase 0/1で記入しない。既存U01等の優先度も今回の判定として継承しない。

後続の作業順序：

1. Phase 2：各環境成果物の固定SHA、schema版、対象・未アクセス・時点を確認する。
2. 原ID・原状態を保持し、§7のidentityを照合。コピー・引用・再使用を対応付け、疑義を残す。
3. 継続／回復、失効／更新と有効metricを結ぶ。別cohortと共有signal/価格を明示した共通カタログを公開してSTOP。
4. 別指示後Phase 3：支持したいclaimと適用域を確認し、必要な軸・比較・誤差保証を選ぶ。
5. 該当条件・対照・独立cohort・証拠接続をmatrixへ対応付け、未発見／未実施／失効／負結果を区別する。
6. 既存データと資料の対応で解決する論点を先に整理し、残る候補をA～Eと根拠・判別したい問いに分類する。
7. 正式判断が必要な範囲と追加計算案をユーザーへ戻し、実行・主要claim採択・詳細レビューへ自動移行せずSTOP。

## 10. read-only調査・保全・公開

各Codexは自分の担当環境だけを調査する。LOCALからSERVER 1/2へのSSH・転送・一括調査はしない。
GitHubの公開済み共通資料を読むことは他hostへの調査ではないが、その取得を他hostの現存確認と呼ばない。
他ユーザーのファイル/job/processを対象にせず、共有設定・OS・package・scheduler・権限を変更しない。
科学workerにsignalを送らず、他worktreeの凍結も要求しない。
primary agentのみで作業し、sub-agentは使用しない。

最初に、その環境で実際にアクセス可能なユーザー所有root・repositoryとの対応・許可範囲を確認する。
存在未確認の絶対pathやhost名は固定しない。旧SERVER_A/Bと今回の対応も自動推定しない。
既存カタログ・保全記録・固定Git object・archive索引を優先し、記載された関連先へ限定的に辿る。
原本全域やhome/projects/tmpの横断再帰走査、raw全decode、archive全展開は行わない。
pathが必要でも公開版では不透明IDへ置換し、非公開の対応は環境内で保全する。
秘密・資格情報・private raw・内部host/IP・非公開サーバーパスをcommitしない。

証拠treeでは`GIT_OPTIONAL_LOCKS=0`と暗黙fetchなしの読取を用いる。
checkout/reset/merge/clean/gc/prune、branch/worktree/archive削除、封印SOURCEやSTOP等の更新は禁止。
共通資料の取得や成果物のcommit/pushは新しい独立した出力作業領域だけで行う。
archiveは既存索引と必要な少量の文書metadataを優先し、原archiveの書換え・復元配置の上書きをしない。
実行中の可変ファイルはsnapshot時刻・非atomic性を記録し、lock・markerに触れない。

通常の権限で実施し、CPU/RAM/worker数の恣意的上限を設けない。許可されたmetadata処理は
既存実装を確認・適合すれば再利用し、必要に応じ並列化・キャッシュ等で効率化する。
管理者quotaと共有環境の非干渉を守る。この方針は科学実行許可を与えない。
正式共通運用指示2026-10-25の§8.1/§8.4/§10と各worktreeのAGENTS.mdを実際に読む。
取得できない必須規約を読了と称さず、取得不能を報告する。

Phase 1成果物は各環境の独立branchに置く。
`inventory/<ENV>/INVENTORY.csv`と`inventory/<ENV>/REPORT.md`を必須とし、必要な場合だけ`DETAILS.jsonl`を追加。
ここで`<ENV>`は`LOCAL` / `SERVER_1` / `SERVER_2`。
REPORTにはprotocol固定SHA、観測範囲・時点・アクセス制限、既存索引の確認済み／除外／未解決対応、
欠測・未アクセス・旧alias、技術検査、保全、公開範囲、終了状態を記す。
局所成果物の文書上の件数はよいが、科学値の新集計や不足優先度は作らない。
CSVヘッダ・JSON構文・ID/参照・欠測・公開情報・差分範囲を確認し、対象成果物だけをstageする。
通常push後にremote SHAと固定commitの公開本文を照合する。force push/既存branch更新は行わない。
push失敗は失敗理由と確認済みcwd・remote・branchを含む再実行コマンドをチャットへ提示する。
公開失敗・権限不足・一部未照合を隠して完全完了としない。

Phase 1の終了条件は、実際に合意した担当rootと既存索引の参照項目を
「記録済み／範囲外／未発見／未アクセス／未解決」に対応付け、形式・参照検査と公開照合を終えること。
見えない環境や未知の保存先を含めた網羅性は主張しない。
権限外や科学再実行が必要な項目は未解決として残し、他の読取作業は続けられる。
原本破損・干渉・秘密公開のおそれ、必須仕様の未取得・版不一致で意味が変わる場合は該当処理をSTOP。
回避できない場合はその環境作業をSTOPし、具体的な不足・判断事項を報告する。

## 11. 今回の検査・次の境界・mandatory STOP

Phase 0で許可する検査は、CSVがヘッダのみで列辞書に一致すること、文書内参照、
固定Git資料のpath/版、3本の指示の整合、変更範囲、公開情報とremoteの本文照合である。
それらは主agent自身の文書・metadata検査であり、独立科学レビューではない。
既存科学SOURCE/input/raw/result/grant/ledger/marker/STOP、研究概要・map・manifestは変更しない。

作成時の技術検査では、ヘッダ1行・43列と辞書順序の一致、列名の一意性、
3本の指示の環境/schema/出力/STOP整合、相対参照と文書構造を確認した。
§3の固定GitHub参照15件はGitHub connectorで取得可能であり、返されたGit blob SHAは
ローカルの各固定commit/pathと一致した。これは参照の可用性・同一性だけの確認で、
証拠群や科学SOURCEを再監査した結果ではない。公開SHA・最終本文の照合結果は最終チャットで報告する。

Phase 1前にユーザーが決める事項は、公開された本仕様の固定commit、実施する環境と担当、
各環境で許可する読取rootと非公開情報の扱い、旧環境aliasの対応（不明のまま記録でも可）。
本仕様の採用だけでPhase 1を起動せず、対象環境へのread-only実施指示を待つ。
Phase 1の科学callは0であり、既存の科学契約・one-shot/STOPを再開する権限を持たない。
承認はユーザーの明示的な許可そのもの。承認ID・原文保存・証明・grant発行等を新規要件にしない。

**Phase 0成果物をcommit/pushし、remote SHA・公開内容を照合したらmandatory STOP。
LOCAL／SERVER 1／SERVER 2のPhase 1、Phase 2/3、詳細科学レビューへ自動移行しない。**

[PURPOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究目的・研究課題.md
[METHOD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究方法・解析手順.md
[SUMMARY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究概要・現状.md
[MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/PROJECT_MAP.md
[STATUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/VALIDATION_STATUS.md
[MANIFEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/validation_manifest.json
[LANDSCAPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_scientific_landscape_20261101.md
[CLAIMS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_claim_evidence_gap_matrix_20261101.md
[TRACKA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h4_h8_cross_system_evidence_inventory_20261017.md
[CARDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9bdcf1d44061daceffe896637e2c92fcb3353b4/docs/research/pr_algorithm_improvement_landscape_20261023/STUDY_CARDS.md
[SOURCEINDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9bdcf1d44061daceffe896637e2c92fcb3353b4/docs/research/pr_algorithm_improvement_landscape_20261023/SOURCE_AND_PROVENANCE_INDEX.json
[ENVLOCAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa55c162fd8fe82bd8ccd91f5fb7601d8513791e/inventory/local/LOCAL_INVENTORY_REPORT.md
[ENVA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa55c162fd8fe82bd8ccd91f5fb7601d8513791e/inventory/server_a/SERVER_A_INVENTORY_REPORT.md
[TLOCAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/track_a_local_inventory.md
[TSERVER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/track_a_server_inventory.md
