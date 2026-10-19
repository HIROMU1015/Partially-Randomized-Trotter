# Codex指示：Track B W2c-B3 実G7/F3接続SOURCE・初回Correctness検証の実行前準備

## 1. 目的・担当・今回の権限

Repository：

`HIROMU1015/Partially-Randomized-Trotter`

担当：サーバーCodex

これまでに以下が完了している。

- Global Bundle × API PortのEnd-to-End人工統合。
- 実G7/F3接続準備用Adapter・Exact Codec・Bits・Evidence変換の一括実装。
- サーバー内の別Codexセッションによる独立人工技術受入。
- 独立224 checks PASS。
- Producer 132 tests PASS。
- 既存107 regression PASS。
- Semantic mutation 20/20拒否。
- 人工16,896 trialのGlobal Bundle保存証拠照合。
- 実科学処理0件、production gate閉鎖、mandatory STOP。

今回の目的は、**実科学APIを使用した初回correctness検証を将来実施できるよう、必要なSOURCE・接続境界・実行契約・preflight・authority要求・証拠保存設計を整えること**である。

### 今回の作業

1. 固定科学SOURCEと、独立受入済みAdapterの接続仕様を確定する。
2. 実G7/F3を将来呼び出すための新versioned wrapper/factoryのSOURCEを準備する。
3. Canonical 3方式とFull Returnの実API経路を区別して接続設計を完成させる。
4. 実科学Result、Evidence、bit consumption、angle/requery、失敗境界を既存Adapterへ渡す技術契約を作る。
5. 初回の小規模correctness検証について、科学条件・call予算・観測項目・判定基準・STOP・保存契約の候補をまとめる。
6. 実行前preflight、trusted authority、runtime、worker、monitoringの候補SOURCEを整備する。
7. 科学入口を閉鎖したまま、人工backendで必要な接続・拒否・保存テストを行う。
8. 必要なSOURCE・契約・Evidence Packetを固定してGitHubへ公開する。
9. mandatory STOPする。

### 運用方針

今回は関連する技術準備を一括して実施する。

技術的な小修正、静的解析、人工テスト、原因調査はCodex内で完結させてよい。

工程を細分化して、そのたびにGPTへレビューを要求しないこと。

ただし、科学的意味論の変更、正式科学契約の採択、新規科学計算、実authorizationの発行が必要になった場合には、作業範囲を拡大せず、その判断だけをユーザー・GPTへ戻すこと。

**今回の指示は、実科学処理を開始する許可ではない。**

---

## 2. 最優先の禁止事項

今回、以下を実施してはならない。

- 実G7/F3 core科学moduleのimport・load・call。
- 実科学namespaceのbitstream生成。
- 実G7/F3によるsampling。
- 実科学Result・Evidenceの新規生成。
- 新しい科学計算・数値再解析・シミュレーション。
- Native量子回路のbuild/compile/synthesis。
- 正式な科学SOURCE・入力・精度・実験契約の採択。
- 実authorization/grant/claimの発行・消費。
- 実trusted storeの既存claimへのアクセス。
- 既存one-shot grantの再利用。
- 既存STOPの解除。
- Production gate解除。
- 実correctness計算。
- 実科学16,896 callsの限定pilot。
- Full-N 31,710,848 calls。

今回のwrapper/factory SOURCEには将来の実科学接続の仕様を実装してよいが、実科学moduleを呼び出す実行経路は閉鎖したままにすること。

新しいSOURCEも、単にbooleanやfake approvalを変更するだけでは実科学処理に移行できないようにする。

将来の実科学実行には、別versioned SOURCE、正式に採択された契約、必要な実authorization、およびユーザーの明示的な実行承認を必要とする。

---

## 3. 固定SOURCE・必須資料

### A. 最新のサーバー内独立技術受入

Branch：

`track-b-g6-w2c-b3-real-adapter-independent-server-20261020-r2`

最終HEAD：

`0db83adb370d66a2f7f5254c8037dadda94ecdb8`

資料：

`docs/research/track_b_g6_w2c_b3_real_adapter_independent_server_20261020/`

必ず確認する資料：

- `README.md`
- `handoff.md`
- `verification.json`
- `pending_and_required_decisions.md`
- `producer_consumer_discrepancies.md`
- `import_scope_finding.md`
- `source_inventory.json`
- `publication_receipt.json`

固定独立受入結果：

- 224 independent PASS
- Producer 102 unit + 30 integration PASS
- 最終SOURCEに対する旧107 regression PASS
- Semantic mutation 20/20拒否
- 人工22ケースの分類一致
- 人工16,896 trialのdispatch・Storage・receipt結合一致
- 実科学処理0

この独立受入を、今回の技術的な出発点として使用する。

既存の人工受入を最初から再実装・再設計しないこと。

### B. 接続準備済みAdapter原本

Branch：

`track-b-g6-w2c-b3-real-api-adapter-contract-closed-v1-20261020`

最終公開HEAD：

`80b7bf61fe0a735d4031bb719c4b3e788611c3ba`

測定SOURCE：

`b7ad344401b10395c49e4cfa806909282859bab5`

Artifact identity：

`024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb`

資料：

`docs/research/track_b_g6_w2c_b3_real_api_adapter_contract_closed_v1_20261020/`

主要SOURCE：

`src/trottertracks/algorithm_codesign/g6_w2c_b3_real_adapter_v1/`

固定内容：

- Static API binding
- Exact Codec
- F3/driver Evidence sidecar
- Result/Failure adapter
- Bitstream/Seed interface
- Canonical Audit/Angle
- Full F3 Angle/Requery
- Closed artificial backend
- Global Bundle bridge
- Offline verification

### C. Global Bundle人工統合原本

公開HEAD：

`62967269e611b61e73b70abe20e998d9c7eaaf50`

測定SOURCE：

`deade78a19f5b23859d16816d3a7bfb37765dd80`

### D. API Binding原本

公開HEAD：

`82896d53fa97235fa848d6c0833eb193af8b99d6`

測定SOURCE：

`78c571005fe766c61b32dc867e006531099ff608`

今回の最新独立受入で訂正された正しい測定SHAを使用すること。

### E. 科学SOURCE

固定参照commit：

`76ca3c4089b201d55da0fede08e1f77733ac38a5`

主要SOURCE：

- `g7_generator.py`
- `g7_provider.py`
- `return_aggregation.py`
- `g6_f3_contract.py`
- `g6_f3_precision.py`
- `g6_f3_generator.py`
- `g6_w2c_b3_driver.py`
- `g6_w2c_cost_preparation.py`

ただし、import制限は以下のように区別する。

**Core科学6 module：**

- `g7_generator.py`
- `g7_provider.py`
- `return_aggregation.py`
- `g6_f3_contract.py`
- `g6_f3_precision.py`
- `g6_f3_generator.py`

これらは静的読取り・AST・SOURCE hash照合のみとする。

**Driver/helper 2 module：**

- `g6_w2c_b3_driver.py`
- `g6_w2c_cost_preparation.py`

これらについては、前回の利用者による明示的範囲修正に従い、人工runtime・旧回帰・人工negative/mutation testに必要なimportと関数実行を許可する。

ただし、推移的依存によりcore科学6 moduleがloadされてはならない。

親process・spawn workerの双方でload前guardを維持する。

### F. 既存Pilot候補

固定参照：

`docs/research/track_b_g6_w2c_b3_limited_pilot_readiness_20261019/`

特に、

- `pilot_scientific_scope_candidate.md`
- `pilot_measurement_design.md`
- `pilot_contract_candidate.json`
- `pilot_execution_readiness.md`
- `pilot_vs_full_N_decision_matrix.md`

を確認する。

さらに、

`docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/fixed_scientific_reference.json`

の科学入力・precision・acceptance・call budgetを確認する。

既存Full-N契約と限定pilot候補を混同しないこと。

### G. 正式共通運用指示

`research_project_instructions_unified_2026-10-17.md`

固定commit：

`7eff8a33bef4ee1f29698ab10c6da46988dc8db5`

固定path：

`docs/research/qc_relative_form_finite_step_audit_20261017/inputs/research_project_instructions_unified_2026-10-17.md`

実際に取得・確認し、特に§8.4を適用する。

---

## 4. 新しいBranch・SOURCE管理

新しい独立branch/worktreeを作成する。

推奨branch：

`track-b-g6-w2c-b3-live-api-correctness-preflight-closed-20261020`

直接基点の第一候補：

`80b7bf61fe0a735d4031bb719c4b3e788611c3ba`

最新独立受入：

`0db83adb370d66a2f7f5254c8037dadda94ecdb8`

は固定参照として扱う。

無条件のmerge/rebaseは行わない。

新しい技術SOURCEはversioned namespaceへ分離する。

推奨：

`src/trottertracks/algorithm_codesign/g6_w2c_b3_live_preflight_v1/`

固定科学SOURCE、既存Adapter、Global Bundle、Storage V2、独立checkerを直接変更しないこと。

旧SOURCE・raw・grant・ledger・marker・STOPも変更しない。

今後の実科学用SOURCEを設計する際にも、既存の使用済みauthorizationやgrantを継承したことにしてはならない。

---

## 5. Phase A：実G7/F3接続の仕様確定

今回の中心となる作業である。

固定科学SOURCEを静的に読み、将来の実API wrapperに必要な具体的な接続仕様を確定する。

対象方式：

1. Ordinary
2. Partial Return + ordinary tail
3. Closed P3 + ordinary tail
4. General Full Return

Layer A/B・Axis X/Yを保持する。

### A1. Canonical 3方式

以下の既存APIとの接続を確認する。

Layer A：

- `make_generator`
- `CanonicalGenerator.sample`

Layer B：

- `LegacyEndpoint`
- `canonical_snapshot`
- `AuditParity`
- `event_witness`
- `event(group_index, raw_word, child)`
- `enclose_atan`

科学SOURCEに実在するsignatureとcall siteを基準にする。

CanonicalをFull F3 facadeへ無理に統一しないこと。

固定正入力のCanonical F3 preflightで、

`CANONICAL_PREFLIGHT_NOT_IMPLEMENTED`

が存在する境界を勝手に解除しない。

### A2. Full Return

Layer A：

- `make_generator`
- `FullReturnGenerator.sample`

Layer B：

- F3 prepare
- GeneratorSession
- Sample
- Angle/requery

実際のSOURCE上の名前・引数・返却構造を確認する。

Full Return特有の、

- Order draw
- Raw labels
- Reduced/nonreduced
- Packet
- Acceptance
- Child
- Event→packet requery

の依存を明示する。

### A3. Source binding

次のbindingを準備する。

- Scientific SOURCE hash
- Input hash
- Precision/policy
- Arm
- Layer
- Axis
- Trial
- Seed domain
- Generator/session
- Result/evidence
- Event fingerprint
- F01
- Runtime identity

### A4. Wrapper SOURCE

将来の実科学APIに対応するwrapper/factoryのinterfaceを実装する。

必要なら、実科学moduleを将来別releaseから解決するための明示的なinterfaceを定義してよい。

ただし、今回のSOURCEから実moduleがimport・呼出しされないようにすること。

未承認のreal factoryを人工factoryの単なる設定変更で起動できる構造にしてはいけない。

実科学loaderの実行が必要になる段階は、今回の完了範囲外として残す。

---

## 6. Phase B：実Result・Evidence・Codec接続準備

既存の`g6_w2c_b3_real_adapter_v1`を可能な限り再利用する。

### B1. Exact Codec

保持するもの：

- F3 original document
- Original digest
- Driver projection
- Driver document/digest
- Lossless sidecar
- Exact Fraction
- Tuple/list/map type
- Source identity
- Event fingerprint

Driver投影だけでは情報損失がある。

原文・sidecarが存在する受理subsetでのみ、復元可能性を主張する。

### B2. ZERO／EVENT／FAILURE

区別する。

- EVENT / ACCEPTED
- ZERO / RAW_NONREDUCED
- ZERO / COIN_REJECT
- IDENTITY
- Deterministic EVENT_QUERY
- Typed failure

`None`だけからZERO理由を判断してはならない。

分類に必要なtraceやprovenanceが不足する場合は、推測せずtyped error/未確立として扱う。

### B3. Angle/requery

保持するもの：

- Original event
- Event fingerprint
- Parent/child
- Exact ratio
- Source/input/policy
- Precision
- Angle interval
- Representative/radius
- Original Evidence
- Query result
- Typed failure

CanonicalとFullのcontext差を明確にする。

人工intervalの整合検証を、数学的なatan enclosureの保証にしない。

### B4. 完了条件

実科学Resultを渡す場合の入出力契約をSOURCEとして定義する。

人工documentで変換・拒否・failure propagationを検証する。

**実科学live Evidenceとの実互換性はNOT_ESTABLISHEDのまま保持する。**

---

## 7. Phase C：科学Bitstream・Seed・消費契約の準備

科学入力候補：

`P5_general_order`

- p=(1/5,3/10,1/2)
- x=5/7
- m=5
- sigma=+1

精度候補：

- root_bits=256
- probability_bits=160
- eta=1e-12
- rho=1e-12
- epsilon_axis=1/200
- primitive_error=1e-6
- alpha_axis=1/320
- alpha_familywise=1/20

A6の既存math-only条件：

- tau=1e-6
- K_phi=32
- J=32

を固定参照とする。

これらは正式採択済みの新契約ではない。

### C1. Seed identity

次を区別する。

- Science release
- Input
- Scope/claim
- Run
- Mode S/P1
- Arm
- Layer
- Axis
- Phase
- Pass
- Local/global trial
- Seed domain

Modeはglobal trial IDに含めるが、科学seed domainへ含めない。

対応するS/P1、A/B、phase/passのprefix契約を保持する。

Arm・X/Yは分離する。

### C2. Bitstream契約

既存のSHA256 counter方式を静的SOURCEで確認する。

- UTF-8 seed
- uint64 big-endian counter
- Requested width
- High bits
- Discard
- Consumption counter
- Failure position

を記録できるinterfaceを準備する。

人工namespaceだけで検証する。

実科学namespaceは文字列・未採択manifestとしてのみ保持する。

### C3. 実消費順序

実G7/F3では、degreeやreject経路によってbit consumptionが変動し得る。

人工backendの固定draw数を実科学の消費規則として採択しない。

実科学でしか確認できない、

- Variable draws
- Conditional child law
- Rejection path
- Actual bit consumption
- IID条件

はNOT_ESTABLISHEDとして残す。

---

## 8. Phase D：初回Correctness検証の候補契約

**今回は検証計画と実行契約候補を作る。実行はしない。**

### D1. 目的

初回correctness検証では、科学アルゴリズムの優位性やCPU性能を判断する前に、以下を確認する計画とする。

- 実G7/F3が固定入力で呼び出せるか。
- 実Result/Evidenceが新Adapterへ正しく渡るか。
- Exact event invariantsが成立するか。
- A/Bの同一trialで期待するprefix・event identityが整合するか。
- CanonicalとFullの監査経路が正常に機能するか。
- B側angle/requeryが成功または正しいtyped STOPになるか。
- Global storage/receipt/callerまで実科学Evidenceが結合されるか。
- 実際のbit consumptionとSOURCE/runtime identityを保存できるか。

これらは技術・科学correctnessの検証項目候補であり、今回成立したと判断してはならない。

### D2. 規模候補

最初から16,896 science callsを実行しない。

第一候補として、以下の小規模correctness-only実験を契約案にまとめる。

- 4方式
- Layer A/B
- Axis X/Y
- 16 cell
- 16 trials/cell
- S/P1二条件

この形状なら、

16 cell × 16 trials × 2条件 = **最大512 scheduled sampling calls**

となる。

ただし、これは提案であり、ユーザーによる正式な規模採択ではない。

既存Full-N契約のcorrectness256 callsや、限定pilot候補の16,896 callsと区別すること。

また、512 scheduled sampling callsは、constructor、audit、angle、requery等を含めた全科学関数呼出し数ではない。

それらのcall budgetは別に列挙する。

### D3. 独立した実行契約

Correctness-only検証には、限定pilot本体と区別された、

- 新SOURCE
- 新input/runtime binding
- 新scope
- 新grant
- 新seed namespace
- 新call budget
- 新result/log budget
- 新STOP
- 新receipt

を候補として設計する。

旧grantや旧spent scopeは絶対に再利用しない。

初回correctnessを実行した後、そのgrantで限定pilot本体を自動開始する設計にしない。

### D4. 合否条件の候補

少なくとも次を個別に定義する。

- SOURCE/input/runtime一致。
- 許可されたtrialのみ実行。
- 予定したtrialの完了状態。
- Result/Evidenceの完全性。
- Exact rational・event invariant。
- A/B対応prefix。
- ZERO・reject・acceptedの分類。
- B preflightとangle/requery。
- F01・typed failure。
- STOP・UNKNOWN。
- Trace・receipt・caller成功。
- 保存容量・call予算の一致。

`at least one accepted per cell`等、旧Full-Nに存在するcoverage条件についても確認すること。

ただし、少数sampleでacceptedが得られない場合に、恣意的な追加samplingやretryを実行してはいけない。

Coverage不足、技術失敗、科学的不一致を区別する候補判定を提示する。

Coverage不足に対して、

- INCONCLUSIVE
- 固定予算内での判定
- 別途承認を要する追加検証

の扱いを提案し、勝手に採択しないこと。

### D5. STOP

実科学実行時の候補STOP条件を明示する。

- SOURCE mismatch
- Input/precision mismatch
- Authority不一致
- Seed/consumption mismatch
- Invalid event
- Codec/Evidence mismatch
- Angle/requery failure
- F01 failure
- Storage/trace/receipt failure
- Budget exceeded
- Worker failure
- UNKNOWN
- STOP後ENTER禁止

既存のS→P1・one-shot・retry0/resume禁止を維持する候補とする。

---

## 9. Phase E：Authority・Trusted Storeの実行前条件

今回は本番authorityを発行しない。

将来の初回correctnessに必要なauthority仕様を整理する。

### E1. 必須binding

- Science SOURCE release
- Scientific input
- Precision
- Runtime
- Seed
- Global scope
- Global claim
- Sampling budget
- Constructor/angle/requery budget
- Result/log budget
- Worker lineage
- Storage destination
- Monitoring
- STOP
- Retry/resume

### E2. 未決の外部条件

以下について、現在の実装で確認済みのものと、ユーザー・管理主体の判断が必要なものを区別する。

- Trusted storeの所有主体
- Credential/鍵管理
- 失効
- Atomic claim
- Commit ACK unknown
- Power-loss durability
- Worker lineage
- Runtime containment
- Receipt/caller durability

Fake SQLite authorityの技術検証を、実trusted authorityの保証に拡張しない。

### E3. 今回許可する実装

- Authority interface候補
- Preflight候補
- Fake-only test store
- Static binding検証
- 不正grant拒否
- STOP/one-shot negative tests

に限定する。

実credentialや実grantは作成しない。

---

## 10. Phase F：保存・Monitoring・費用契約

初回correctness検証では、実科学のEvidenceと失敗原因を十分に残せる保存契約が必要である。

### F1. 保存候補

検討するもの：

- Trial identity
- Original event/Evidence
- Exact rational
- Original F3/driver document
- Codec sidecar
- Bit consumption trace
- Angle/requery
- F01
- Worker/source/runtime identity
- Ordered global trace
- Global state/counter
- Condition/global receipt
- STOP/UNKNOWN
- Marker
- Delivery/caller observation

小規模correctness-onlyでは、診断に必要なfull Evidenceを保持する案を優先検討する。

ただし、具体的な保存量・予算を固定する前に、既存人工データと静的な構造から必要量を検討する。

人工raw容量をそのまま実科学容量として断定しない。

### F2. 予算

既存の限定pilot候補：

- Result 64 GiB
- Log 256 MiB

は、S/P1共通のglobal budgetである。

ただし、初回correctness-only用の予算として自動採択しない。

初回correctnessには別の明示的なcall/byte budget候補を設計する。

実測されていない実科学データサイズは未確認として扱う。

### F3. CPU/wall・費用

将来の実科学実行時に、少なくとも次を観測可能にする。

- Parent CPU
- 全worker CPU
- Global outer wall
- Generator setup
- Online sampling
- Bit consumption
- Coefficient/child-law処理
- Audit
- Angle/requery
- Codec
- IPC
- Observer
- Storage
- RSS/HWM
- I/O
- STOP/drain/close

Parent＋全workerを二重計上せずに集計する。

Coarse/fine費用はそれぞれouter totalと整合させる。

実G7/F3費用、accepted/reject分布、有理数bit演算費用は今回測定しない。

---

## 11. Phase G：実行前Preflight SOURCE

新しいversioned preflightを準備する。

### G1. SOURCE検査

- Git commit
- Source SHA256
- Scientific API signature
- Input pins
- Precision
- Contract digest
- Artifact identity
- Runtime executable/library hash
- Worker runtime
- Storage destination
- Seed namespace
- Authority binding

を検査する。

### G2. Closed preflight

今回実行するpreflightは、科学moduleのimportを必要としないものに限定する。

実行可能な人工scopeで、

- 正常binding
- Wrong SOURCE
- Wrong input
- Wrong runtime
- Wrong precision
- Wrong seed namespace
- Wrong call budget
- Wrong result/log budget
- Fake grant
- Spent claim
- Retry/resume
- Closed gate

を検査する。

### G3. Production gate

今回のSOURCEは次の状態を維持する。

- `content_adopted=false`
- `run_ready=false`
- `contract_grants_execution=false`
- `science_release=null`
- `scientific_adoption=null`
- `artificial_only=true`
- `mandatory_STOP=true`
- `retry=0`
- `resume=false`

これらを単純に変更するだけでは実科学dispatchに進めないこと。

実科学処理を可能にする別versioned release・authorityが必要な構造を保つ。

---

## 12. Phase H：人工技術検証

新しいSOURCEについて、変更された部分を重点的に人工検証する。

### H1. Adapter

- Canonical/Full routing
- Exact Codec
- Result/Evidence
- ZERO/reject/failure
- Angle/requery
- Bit consumption
- Source/input/precision binding

### H2. Worker・Global Bundle

- Factory binding
- Spawn worker
- Global identity
- S/P1
- A/B
- F01
- Storage V2
- STOP
- Receipt/caller

### H3. Negative/mutation

少なくとも、

- Wrong science SOURCE
- Wrong API signature
- Wrong event type
- Wrong original digest
- Wrong sidecar
- Wrong seed domain
- Wrong counter/width
- Missing angle Evidence
- Requery mismatch
- Wrong trial ID
- Fake approval
- Wrong contract
- Closed gate bypass
- STOP後ENTER
- Spent claim reuse

を検査する。

実科学moduleを実行せず、人工fixture・静的SOURCE・閉鎖interfaceによって検証すること。

### H4. 回帰方針

既存の独立224、Producer132、旧107、人工16,896等を毎回一から再構築しないこと。

今回の変更影響範囲に合わせた差分回帰を優先する。

Global Bundle、STOP、Storage等の重要な意味論を変更した場合には、その影響範囲に応じて広めの回帰を実施する。

最終SOURCE固定後、必要な人工統合回帰をまとめて実施する。

人工技術検証の成功は実科学互換性の証明ではない。

---

## 13. 科学的に未確立の事項

今回のSOURCE準備後も、以下は未確認として保持する。

- 実G7/F3のlive import・call互換性。
- 実Result/Evidenceのruntime型。
- 実law・係数・child law。
- 実variable bit consumption。
- 実ZERO/reject/accepted分布。
- 実angle enclosureの数学的妥当性。
- 実F3 requery。
- 実CPU/wall・保存量。
- IID/独立性に関する条件。
- Full Returnの実性能。
- Native回路への保証。
- Real trusted authorityの耐久性。
- Full-Nの実行可能性。

これらの確認が必要だからといって、今回のCodexが実科学moduleをimport・samplingしてはならない。

未確認事項は`NOT_ESTABLISHED`として保存する。

---

## 14. 資源・共有サーバー保護

正式共通指示§8.4を適用する。

CPU、RAM、worker数、thread数、wall time等に恣意的な低い上限を設けない。

重い技術検証やSHA256照合は、実際の空き資源を活用して適切に並列化・高速化する。

不要な重複計算は削減する。

ただし、他ユーザーまたは共有環境全体に影響する変更・操作は絶対禁止する。

- 共有OS/kernel/sysctl変更
- 共有scheduler/cgroup/quota変更
- 共有venv/package/cache変更
- 他ユーザーprocess/job操作
- 他ユーザーのファイル変更
- 共有サービス再起動
- グローバル設定変更
- 保護対象worktreeの変更

自分が所有する分離worktree、process、venv、cache、temp、outputのみを使用する。

実際の空きCPU/RAM/I/O・pressureを確認し、他ユーザーに干渉しないよう実行負荷を調整する。

科学call数・one-shot・STOP・保存上限等の固定研究契約を、資源最適化の名目で変更しない。

---

## 15. 成果物

推奨branch：

`track-b-g6-w2c-b3-live-api-correctness-preflight-closed-20261020`

推奨ディレクトリ：

`docs/research/track_b_g6_w2c_b3_live_api_correctness_preflight_closed_20261020/`

### SOURCE

- Real API binding specification
- Closed live wrapper interface
- Canonical/Full dispatch contract
- Result/Evidence connection interface
- Bitstream/seed contract
- Correctness-only preflight
- Authority candidate interface
- Worker/runtime binding
- Monitoring/storage contract
- Artificial closed test backend
- Negative/mutation tests

### Markdown

- `README.md`
- `handoff.md`
- `fixed_source_and_input.md`
- `live_api_wrapper_design.md`
- `canonical_full_connection_contract.md`
- `real_result_evidence_interface.md`
- `scientific_bits_and_seed_contract.md`
- `correctness_only_scope_candidate.md`
- `correctness_acceptance_and_coverage.md`
- `call_budget_and_one_shot.md`
- `authority_and_trusted_store_requirements.md`
- `preflight_and_closed_gate.md`
- `monitoring_and_storage.md`
- `artificial_technical_acceptance.md`
- `remaining_scientific_validation.md`
- `source_and_contract_delta.md`
- `pending_and_required_decisions.md`

### Machine-readable

- `live_api_binding_candidate.json`
- `correctness_scope_candidate.json`
- `correctness_contract_candidate.json`
- `scientific_source_input_pins.json`
- `seed_namespace_candidate.json`
- `authority_requirements_candidate.json`
- `preflight_manifest_candidate.json`
- `artificial_acceptance_matrix.json`
- `source_inventory.json`
- `verification.json`
- `publication_receipt.json`

必要なSOURCE・人工raw・trace・runtime identity・tests・SHA256を保存する。

ファイル名や構成は合理的に変更してよい。

### 未決事項

未決事項ごとに、以下を明示する。

- 技術的に準備済みか。
- 人工検証済みか。
- 実科学実行が必要か。
- 正式契約の採択が必要か。
- 実authority等の外部条件が必要か。
- 次の担当は誰か。

---

## 16. 最終報告 Q1〜Q10

**Q1.** 最新独立受入、閉鎖Adapter、固定科学SOURCE、input/precision、Global BundleのSOURCE/hashを確認できたか。

**Q2.** 実G7/F3接続用の新versioned wrapper/factory interfaceとして何を実装したか。実科学moduleをloadしない閉鎖構造を維持できたか。

**Q3.** Canonical 3方式とFull Returnの異なるAPI、監査、angle/requery経路は正しく区別されたか。

**Q4.** 実Result/Evidence、Codec、ZERO/reject/failure、bit consumptionを将来接続するための契約はどこまで完成したか。

**Q5.** 初回correctness-only検証の候補条件、call budget、coverage、失敗分類、STOP、保存範囲は何か。旧Full-N・限定pilot契約とどう区別したか。

**Q6.** SOURCE/input/runtime/precision/seedを結合したpreflightはどこまで実装・人工検証できたか。実科学moduleをimportせず検証できたか。

**Q7.** Real authority/trusted storeの要件と、まだ外部判断が必要な事項は何か。Fake authorityとの境界は閉鎖されているか。

**Q8.** 新規unit/negative/integration、mutation、既存差分回帰はそれぞれ何件PASSしたか。未解決の技術的不整合はあるか。

**Q9.** 正式科学SOURCE・入力・精度・correctness契約を採択するため、ユーザー・GPTが次に判断すべき事項は何か。初回科学実行に必要な追加承認・証拠・外部条件は何か。

**Q10.** 実G7/F3 core import/sampling、科学bitstream生成、新科学計算・再解析、実authorization/grant/claim発行・消費、native処理、旧claim再利用、STOP解除、実correctness、limited pilot、Full-Nがすべて0件であるか。

---

## 17. 完了条件・GitHub公開・Mandatory STOP

次を完了すること。

1. 固定SOURCE・資料の取得と照合。
2. 新versioned SOURCE・worktree作成。
3. 実G7/F3接続wrapper interface準備。
4. Canonical/Full API契約の固定。
5. Result/Evidence・bitstream接続仕様の固定。
6. Correctness-only候補契約作成。
7. Call/byte budget・coverage・STOP候補作成。
8. Authority・trusted store要件整理。
9. Static preflight・closed gate実装。
10. 人工unit/negative/integration/mutation検証。
11. 必要な差分回帰。
12. 未解決事項・必要承認の明示。
13. Evidence Packet作成。
14. Commit/push。
15. Remote HEAD・公開bytes/SHA256照合。
16. 既存SOURCE、raw、grant、ledger、marker、STOP、保護対象worktreeの保全確認。

その後、**mandatory STOP**する。

技術的な不具合は今回の範囲で修正してよいが、科学SOURCE・入力・精度・実行契約の正式採択が必要な変更については、勝手に判断しないこと。

実科学moduleのimport・samplingが必要になった場合も、今回の権限内で実行せず、その必要性と要求する最小実行範囲を記録してSTOPすること。

### 次工程への移行禁止

今回の技術準備が完了しても、

- 実科学APIの動作確認
- 実correctness計算
- Science release
- Real authorityの発行
- Limited pilot
- Full-N
- Production gate解除

へ自動進行しない。

**次の担当はユーザー・GPT。**

今回の最終目標は、実G7/F3の初回correctness検証について、実行SOURCE・科学条件・authority・保存・STOPの候補を具体化し、ユーザーが明示的に採択・実行承認を判断できる状態まで技術準備を進めることである。