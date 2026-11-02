# Codex指示：Track B W2c-B3 実G7/F3接続Release・512件Correctness契約・Real Authority実行前準備

## 1. 目的・今回の位置付け

Repository：

`HIROMU1015/Partially-Randomized-Trotter`

担当：サーバーCodex

本作業では、実G7/F3に対する初回correctness-only検証を実施するための、**新versioned接続SOURCE、科学実行契約、authority境界、preflight、監視・保存、独立検証可能な証拠**をまとめて準備する。

ユーザーは、次の方針による準備作業を了承している。

- 固定科学入力と既存precisionを維持する。
- 4方式×Layer A/B×Axis X/Yを対象とする。
- S/P1の二条件を同一Global Bundleで扱う。
- 初回はcorrectness-only、各cell 16 trials、最大512 scheduled sampling callsとする。
- Coverage不足はINCONCLUSIVEとし、追加sampling・retryを行わない。
- Result 1 GiB／Log 64 MiBを実行契約の上限候補とする。
- S→P1、単一global one-shot、retry0、resume禁止、mandatory STOPを維持する。
- 実科学実行の正式採択と実行承認は、別の手続きとする。

**この指示は、上記方針に基づく技術準備の許可であって、科学実行、実grant発行、production releaseの有効化の許可ではない。**

今回の工程では、実行に必要な技術SOURCEと契約候補を可能な限り完成させる。ユーザーや管理主体による判断が必要な箇所だけを残してmandatory STOPすること。

技術作業を細分化して頻繁にGPTへ戻さず、通常の実装・デバッグ・人工テストはサーバーCodex内で完結させる。

---

## 2. 固定SOURCE・必須参照

### A. 最新のCorrectness実行前準備

Branch：

`track-b-g6-w2c-b3-live-api-correctness-preflight-closed-20261020`

公開HEAD：

`6c6d3376e7ff206219f5e58355f1e1080fafba04`

測定SOURCE：

`4b4496a86f4ab9e0e1266e7323283aff9fe24f26`

資料：

`docs/research/track_b_g6_w2c_b3_live_api_correctness_preflight_closed_20261020/`

必須資料：

- `README.md`
- `handoff.md`
- `verification.json`
- `correctness_scope_candidate.json`
- `correctness_contract_candidate.json`
- `correctness_only_scope_candidate.md`
- `correctness_acceptance_and_coverage.md`
- `call_budget_and_one_shot.md`
- `authority_and_trusted_store_requirements.md`
- `preflight_and_closed_gate.md`
- `fixed_source_and_input.md`
- `scientific_bits_and_seed_contract.md`
- `monitoring_and_storage.md`
- `remaining_scientific_validation.md`
- `pending_and_required_decisions.md`
- `scientific_source_input_pins.json`
- `source_inventory.json`
- `publication_receipt.json`
- `publication_transport.json`

公開GitHub側のcommitとローカル測定commitは区別し、SOURCE tree・bundle・SHA256・publication mappingを照合すること。

### B. 実G7/F3接続準備Adapter

公開HEAD：

`80b7bf61fe0a735d4031bb719c4b3e788611c3ba`

測定SOURCE：

`b7ad344401b10395c49e4cfa806909282859bab5`

Artifact identity：

`024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb`

SOURCE：

`src/trottertracks/algorithm_codesign/g6_w2c_b3_real_adapter_v1/`

### C. 最新のAdapter独立技術受入

公開HEAD：

`0db83adb370d66a2f7f5254c8037dadda94ecdb8`

資料：

`docs/research/track_b_g6_w2c_b3_real_adapter_independent_server_20261020/`

既存成果：

- 独立224 checks PASS
- Producer132 tests PASS
- 旧107 regression PASS
- 20 semantic mutations拒否
- 人工16,896 trialを全件照合
- 実科学処理0

### D. Global Bundle・API Port

Global Bundle公開HEAD：

`62967269e611b61e73b70abe20e998d9c7eaaf50`

API Binding公開HEAD：

`82896d53fa97235fa848d6c0833eb193af8b99d6`

API Bindingの正しい測定SOURCE：

`78c571005fe766c61b32dc867e006531099ff608`

旧SOURCE、raw、grant、ledger、marker、STOPは固定保全する。

### E. 科学SOURCE

固定科学参照：

`76ca3c4089b201d55da0fede08e1f77733ac38a5`

Core科学6 module：

- `g7_generator.py`
- `g7_provider.py`
- `return_aggregation.py`
- `g6_f3_contract.py`
- `g6_f3_precision.py`
- `g6_f3_generator.py`

これらは**静的読取り・AST・SHA256照合のみ**を許可する。

人工runtimeに必要な次の2 moduleについては、以前の明示的な範囲修正に従う。

- `g6_w2c_b3_driver.py`
- `g6_w2c_cost_preparation.py`

この2 moduleの人工用途でのimport・関数実行は許可する。ただし、Core科学6 moduleへの推移的importを実行してはならない。

### F. 正式共通運用指示

`research_project_instructions_unified_2026-10-17.md`

固定commit：

`7eff8a33bef4ee1f29698ab10c6da46988dc8db5`

参照path：

`docs/research/qc_relative_form_finite_step_audit_20261017/inputs/research_project_instructions_unified_2026-10-17.md`

実際に取得・確認すること。特に§8.4を適用する。

---

## 3. 今回の許可範囲・実行禁止

### 実施してよい作業

- 固定SOURCE・仕様・既存証拠の取得。
- 静的科学API解析。
- 新versioned real接続wrapper/factoryの実装。
- 将来のloader・dispatcherに必要なSOURCE構造の実装。
- Contract・manifest・preflightの実装。
- Authority interface・trusted store候補の技術実装。
- 自分が所有する隔離されたfake/test storeでのatomic・STOP・故障検証。
- 人工inputによる統合・mutation・negative tests。
- 保存・監視・CPU accountingの技術実装。
- 実装に必要な技術的原因調査・修正。
- Evidence PacketのGitHub公開。

### 禁止事項

- 実G7/F3 Core科学moduleのimport・load・call。
- 実科学namespaceのbitstream生成。
- 実科学sampling・correctness計算。
- 新科学計算、保存科学値の数値再解析。
- Native回路build/compile/synthesis。
- 実authorization/grant/claimの発行・消費。
- 本番trusted storeへのアクセス・変更。
- 既存grantやspent claimの再利用。
- 既存STOP解除。
- 科学計算が可能なproduction gateの有効化。
- Limited pilot16,896件の実科学実行。
- Full-N31,710,848件の実行。

**Real接続に必要なSOURCEを記述することと、そのSOURCEを実行することは別である。**

今回のrelease candidateは、新しい科学実行用SOURCEを含んでよいが、現段階で実科学へ到達できる有効な実行releaseとしてはならない。

単なるboolean変更、fake approval、fake grant、人工namespaceの置換だけでは科学実行できないよう、実行入口を閉鎖しておくこと。

---

## 4. 新branch・SOURCE管理

推奨branch：

`track-b-g6-w2c-b3-live-correctness-release-preparation-20261020`

直接基点の第一候補：

`6c6d3376e7ff206219f5e58355f1e1080fafba04`

新しい独立worktreeを作成すること。

既存の科学SOURCE、Adapter、Global Bundle、Storage V2、独立checkerを直接変更しない。

新しいversioned namespaceを追加する。

推奨：

`src/trottertracks/algorithm_codesign/g6_w2c_b3_live_release_v1/`

既存SOURCEと新SOURCEの差分を明示し、正式な科学releaseの採択とは区別する。

既存SOURCEへの変更がどうしても必要な場合は、原本を書き換えず新versionでの変更案・互換性・影響範囲を保存すること。

---

## 5. Phase A：実G7/F3接続SOURCEの実装

これまでの工程では、実科学loaderが未実装だった。

今回の主な技術的進展は、**将来の承認済み実科学releaseで必要となる接続処理を、実装レベルで具体化すること**である。

ただし、実moduleを今回import・実行してはならない。

### A1. Canonical 3方式

対象：

- Ordinary
- Partial Return + ordinary tail
- Closed P3 + ordinary tail

Layer A：

`make_generator → CanonicalGenerator.sample`

Layer B：

`LegacyEndpoint → canonical_snapshot / AuditParity / event_witness / event(group_index, raw_word, child) / enclose_atan`

固定SOURCEからAPI signature・call site・返却形式・失敗境界を確認する。

正入力CanonicalをFull ReturnのF3 facadeへ統一してはならない。

既存の`CANONICAL_PREFLIGHT_NOT_IMPLEMENTED`境界を無断解除しない。

### A2. Full Return

Layer A：

`FullReturnGenerator.sample`

Layer B：

`F3.prepare → GeneratorSession.sample / angle_for_event`

実SOURCEで確認できる必要なpacket、acceptance、child、requery、fingerprint、typed failureの意味を保持する。

### A3. Real wrapper/factory

実接続に必要な新しいversioned SOURCEを作る。

- Science SOURCE resolver仕様
- API symbol/signature binding
- Input/precision binding
- Generator/session construction
- Bitstream interface
- Sample result conversion
- Evidence conversion
- Angle/requery
- Typed failure/STOP
- Worker lifecycle
- F01・Storageへの引渡し

実科学moduleが実行時に提供するオブジェクトと、既存Codec/Adapterが受け入れるexact documentの間に明示的な契約境界を置くこと。

Real科学moduleの解決・import・実呼出しを伴う関数については、今回の実行可能入口から分離し、未認可では実行できない状態を維持する。

Real moduleを実際にimportして確認する必要が生じた場合は、その部分を未検証として残す。

### A4. 完了の定義

今回は、

- SOURCE上のAPI接続構造
- 固定入出力契約
- 静的signature binding
- 責務の分離
- 実行前認可境界
- 失敗伝播
- 人工backendによる動作検証

までを完了対象とする。

**実live APIとの互換性が成立したと主張してはならない。**

---

## 6. Phase B：512件Correctness契約の固定候補

ユーザーが了承した方針を、明確なversioned契約候補へまとめる。

### B1. 科学入力

Input：

`P5_general_order`

- p=(1/5,3/10,1/2)
- x=5/7
- m=5
- sigma=+1

Precision：

- root_bits=256
- probability_bits=160
- eta=1e-12
- rho=1e-12
- epsilon_axis=1/200
- primitive_error=1e-6
- alpha_axis=1/320
- familywise alpha=1/20

A6：

- tau=1e-6
- K_phi=32
- J=32
- math-only

既存固定資料の完全一致を確認し、無断変更しない。

A6 math-onlyをnative保証へ変更しない。

### B2. Sampling計画

対象：

- 4方式
- Layer A/B
- Axis X/Y
- 16 cell
- 16 trials/cell
- S/P1二条件

合計：

**512 scheduled sampling calls**

S条件：256 calls。

P1条件：256 calls。

Correctness-only。

Profile・instrumented pass・性能測定用の追加samplingは行わない。

S→P1の順序を維持する。

### B3. Separate call budget

Sampling以外の操作について、既存資料の候補値を固定参照する。

- Primary construction：16
- G7 make_generator：14
- F3.prepare：2
- AuditParity construction：8
- Canonical snapshot：6
- B sampling audit：最大256
- B angle：最大256
- B event requery：最大256
- Canonical angle：最大192
- Full angle：最大64
- Event witness：最大512
- Full sample packet：最大128
- Full angle packet：最大64
- Additional sampling：0

F3.prepare内部constructorなど、既に別項目の内数に入っている操作を二重計上しない。

現在の候補にあるbit draw requests上限案3,200についても、固定SOURCEから根拠と成立範囲を確認すること。

**境界関数のcall予算と、内部科学関数の総call数は異なる。**

Root、law、predicate、packet内部操作などの総実呼出し数が未確立であれば、確立したことにしない。

将来の実科学実行で、何を実際にcount・limit・observeするかを明示する。

予算上界が静的SOURCEだけで保証できない場合は、その項目を`NOT_ESTABLISHED`として残す。

### B4. Coverage

16 trials/cellを固定する。

すべての予定trialを正常完了し、exact event invariant、A/B prefix、Evidence、B監査・angle/requery等に不一致がないことを確認する判定設計とする。

各cellでaccepted eventが少なくとも1件必要という既存のcoverage条件を維持する候補とする。

Acceptedが得られなかったcellがあれば、

`INCONCLUSIVE_COVERAGE`

として扱う。

追加sampling・adaptive延長・再試行は禁止。

Coverage不足と、技術的失敗・科学的な不一致は明確に区別する。

既存Full-Nのcorrectness256、限定pilot16,896と今回512を混同しないこと。

### B5. Contract status

今回の合意は設計方針として記録する。

契約候補は固定可能な形で作成するが、実科学release・実行grant・科学計算の実行承認は未完了とする。

`USER_APPROVED_PREPARATION_SCOPE`と`SCIENCE_RUN_AUTHORIZED`等、異なる承認状態を混同しない機械可読記録を整えること。

---

## 7. Phase C：Real Authority・Trusted Store準備

現在の課題はfake authorityしか成立していないことである。

今回、必要な技術実装と外部判断を可能な限り切り分ける。

### C1. Required authority binding

最低限次を結合する。

- Science SOURCE release
- Input/precision
- API/contract digest
- Runtime executable/library
- Worker runtime・lineage
- Seed namespace
- Global scope/claim
- Sampling/constructor/angle/requery budgets
- Global result/log budgets
- Storage destination
- Monitoring
- Retry/resume
- STOP
- Receipt/caller

### C2. Trusted storeの技術設計

次を具体化する。

- Atomic claim取得
- One winner
- Claim spent
- Double claim拒否
- ACK unknown時の保守的処理
- STOP後の新規実行禁止
- GrantとSOURCEのbinding
- Storage/receiptとの証拠連鎖
- Power-loss時の必要条件
- Worker authorityの最小化
- Credential失効・運用境界

既存repositoryに利用可能な認可・保存契約がある場合は、そのSOURCEと証拠を静的に確認して再利用可能性を判断する。

### C3. 今回の実装範囲

隔離された所有ディレクトリ・人工scopeで、永続化・atomic・ACK unknown・STOPの技術prototypeを作成・検証してよい。

必要なら自分の試験用processを異常終了させる人工故障試験を行ってよい。

ただし、

- 実trusted storeへのアクセス
- 実credentialの発行
- 本番鍵の配置
- 実grant・実claimの発行/消費
- 旧claimの再利用

は禁止する。

Fake環境での成功は、実trusted storeの耐久性・認証強度・管理運用の保証ではない。

### C4. 外部判断事項

少なくとも次を決定待ちとして明示する。

1. Trusted storeの所有主体。
2. Grant発行の権限者。
3. Credential・鍵の管理方法。
4. 失効・ローテーション。
5. Atomicityとcommit ACK unknownの契約。
6. Power-loss durability。
7. Worker lineage/containment。
8. Receipt/callerの耐久性。
9. 実行環境の責任範囲。

所有主体・権限・鍵の具体的な情報が未提示の場合、それを推測して本番実装済みにしない。

**Real authorityの外部条件が成立していない場合も、ほかの技術準備は完了させ、最後に必要な判断事項をまとめてSTOPする。**

---

## 8. Phase D：Preflight・Release candidate

既存のclosed preflightを再利用し、実科学接続に向けた新しいrelease候補を準備する。

### D1. Preflight対象

- SOURCE commit/hash
- Input pins
- Precision
- API signatures
- Adapter artifact
- Runtime executable/library
- Worker runtime
- Seed namespace
- Trial plan
- Call budgets
- Result/log budget
- Storage destination
- Authority binding
- Monitoring
- STOP
- Retry/resume
- Release identity

### D2. New release candidate

正式な実行releaseと未採択candidateを区別する。

今回作るものは、

`LIVE_RELEASE_CANDIDATE_UNAUTHORIZED`

に相当する状態とする。

実行に必要なSOURCEを準備・固定してよいが、science releaseを正式に採択したことにはしない。

### D3. Closed gate

現時点では、

- `run_ready=false`
- `contract_grants_execution=false`
- `science_release=null`
- `scientific_adoption=null`
- `retry=0`
- `resume=false`
- `mandatory_STOP=true`

を維持する。

既存の閉鎖候補の`content_adopted`等も、正式な採択手続きが完了していなければfalseとする。

Boolean変更のみで実科学moduleが動く構造を作ってはならない。

将来の実科学実行には別の正式release・authority・ユーザー実行承認が必要である。

---

## 9. Phase E：Result/Log・Monitoring・STOP

### E1. 保存予算

今回のcorrectness候補では、

- Global result：1 GiB
- Global log：64 MiB

を第一候補とする。

S/P1共通の単一global capである。

これは実科学Evidence容量の保証ではない。

### E2. 保存する証拠

少なくとも次を対象とする。

- SOURCE/input/runtime identity
- Trial identity
- Generator/result
- Original scientific document
- Exact Codec sidecar
- Event/ZERO/reject/accepted
- Fraction・weight・child
- Bits consumption
- Angle/requery
- Audit evidence
- F01・typed failure
- Worker/source identity
- Global trace/state/counter
- Condition/global receipt
- STOP/UNKNOWN
- Storage close
- Delivery/caller observation

Correctness512件では、原則すべてのtrialのfull Evidenceを保持する候補とする。

実データが予算を超える場合、保存粒度を無断で下げて処理を続行しない。

Budget exceededとしてSTOPする。

### E3. STOP

以下を区別して保持する。

- SOURCE/input mismatch
- Authority mismatch
- Seed/consumption mismatch
- Invalid event
- Codec/Evidence mismatch
- Angle/requery failure
- Scientific invariant mismatch
- Worker failure
- Storage failure
- UNKNOWN
- Byte/call budget exceeded
- STOP後ENTER禁止
- In-flight drain

S失敗時にはP1へ進まない。

P1失敗時にはSの保存証拠を保持する。

Unknown creditは返却しない。

### E4. Monitoring

実科学実行時に次を計測できる構造を準備する。

- Parent CPU
- 全worker CPU
- Global outer wall
- Constructor/setup
- Online sampling
- Bits
- Coefficient/child law
- Audit
- Angle/requery
- Codec
- IPC
- Observer
- Storage
- RSS/HWM
- I/O
- STOP/drain/close

ParentとworkerのCPUを二重計上しない。

Coarse/fineのexclusive partitionをそれぞれouter totalへ照合し、残差を明示する。

実科学CPU、実Evidence容量、実path率は今回未測定として維持する。

---

## 10. Phase F：人工統合検証・差分回帰

今回の新SOURCEについて、技術的に変更した範囲を中心に検証する。

### F1. 人工正常

新しい実接続wrapper仕様を人工backendへ結合し、次の経路を検証する。

Artificial Result
→ Versioned live wrapper
→ Exact Adapter/Codec
→ Global worker
→ Pair/F01
→ Storage V2
→ Receipt/caller

正常・失敗の原証拠が実際にこの経路を通ったことを確認する。

### F2. Negative/mutation

少なくとも以下を検証する。

- Wrong scientific SOURCE
- Wrong API signature
- Wrong input/precision
- Wrong runtime
- Wrong seed namespace
- Wrong global trial identity
- Invalid Result/Evidence
- Wrong Codec/digest
- Missing angle/requery
- Wrong consumption
- Unauthorized grant
- Fake-to-real authority confusion
- Double claim
- STOP後ENTER
- Storage/receipt failure
- Budget exceeded

### F3. Coverageと契約

人工caseでは、

- 正常完了
- Coverage不足によるINCONCLUSIVE
- Technical STOP
- Scientific mismatch相当の人工STOP
- Evidence incomplete

を区別する。

Artificial fixtureによる正常分類を科学的correctness成功と表示しない。

### F4. 重複検証の抑制

過去の独立224件、Producer132件、旧107件、16,896人工trialの検証を無条件に再実施しない。

新しいSOURCE変更に直接関係するテストと、STOP・Storage・authority等の重要な依存範囲を優先する。

最終SOURCE固定後、必要な人工統合検証をまとめて実施する。

小さな修正ごとに別セッションへ独立レビューを依頼しない。

ただし、科学的意味論、seed、STOP、one-shot、保存契約を実質的に変更する不具合が見つかった場合は、固定原本を変更せず証拠を保存して判断を戻す。

---

## 11. 計算資源・共有環境

正式共通指示§8.4を適用する。

CPU、RAM、worker数、thread数、wall time等に恣意的な低い制限を設けない。

重い人工検証・SHA256照合・mutation等は、実際の空き資源を利用して十分に並列化・高速化する。

ただし、固定契約に定められたS→P1、one-shot、worker構造、call予算、STOPは無断変更しない。

共有サーバー全体や他ユーザーに影響する操作・変更は禁止する。

- OS/kernel/sysctl変更
- 共有scheduler/cgroup/quota変更
- 共有venv/package/cache変更
- 他ユーザーprocess/job操作
- 他ユーザーのファイル変更
- 共有サービス再起動
- グローバル設定変更
- 保護対象worktree変更

自分が所有する分離されたworktree、process、venv、cache、temp、outputのみ使用する。

実行前・実行中に空きCPU/RAM/I/O・pressureを確認し、他ユーザーへの非干渉を維持する。

共有環境の利用条件・管理者quotaも遵守する。

---

## 12. 成果物

推奨branch：

`track-b-g6-w2c-b3-live-correctness-release-preparation-20261020`

推奨資料ディレクトリ：

`docs/research/track_b_g6_w2c_b3_live_correctness_release_preparation_20261020/`

### SOURCE

- New versioned scientific API wrapper
- Canonical/Full factory binding
- Result/Evidence bridge
- Exact Codec integration
- Bits/Seed binding
- Global Bundle integration
- Correctness-only contract
- Authority/store interface candidate
- Preflight/release candidate
- Monitoring/Storage/STOP
- Artificial tests
- Offline evidence checker

### Markdown

- `README.md`
- `handoff.md`
- `fixed_source_and_reference.md`
- `live_api_connection_implementation.md`
- `canonical_full_runtime_binding.md`
- `correctness_scope_and_adoption.md`
- `correctness_call_budget.md`
- `correctness_acceptance_and_coverage.md`
- `real_authority_requirements.md`
- `trusted_store_implementation_status.md`
- `release_and_preflight.md`
- `monitoring_storage_and_STOP.md`
- `artificial_technical_acceptance.md`
- `source_and_contract_delta.md`
- `remaining_execution_blockers.md`
- `pending_and_required_decisions.md`

### JSON

- `live_release_candidate.json`
- `correctness_contract_candidate.json`
- `scientific_source_input_pins.json`
- `call_budget_manifest.json`
- `seed_namespace_candidate.json`
- `authority_binding_candidate.json`
- `trusted_store_requirements.json`
- `preflight_manifest.json`
- `artificial_acceptance_matrix.json`
- `source_inventory.json`
- `verification.json`
- `publication_receipt.json`

必要な人工raw、logs、trace、hash、runtime identityを保存する。

ファイル分割は合理的に変更してよい。

旧証拠の不必要な重複複製は避け、固定GitHub参照を適切に使用する。

### 実行可能性の分類

完了時には、以下を機械可読に分類する。

1. 実装済み・人工検証済み。
2. SOURCE上は準備済みだが、live importが必要。
3. Real authority外部条件待ち。
4. 正式科学契約の採択待ち。
5. 新科学計算の実行承認待ち。

実行できない要因を、単なる「未実装」と「未承認」と「外部条件未確立」に分けること。

---

## 13. 最終報告 Q1〜Q10

**Q1.** 最新公開HEAD、固定科学SOURCE、入力、precision、既存Adapter/Bundle/独立受入のSOURCE・SHAを照合したか。

**Q2.** 新しいversioned live wrapper/factory/dispatcherの何を実装したか。実科学moduleをimportせず、どこまで準備できたか。

**Q3.** Canonical 3方式とFull Returnの実API接続経路、Result/Evidence、angle/requery、typed failureは正しく分離されているか。

**Q4.** 512件correctness-onlyの科学条件、trial計画、constructor・angle・requery・bit消費の別予算はどのように固定したか。未確立の内部総call範囲は何か。

**Q5.** Coverage不足INCONCLUSIVE、追加sampling禁止、科学的不一致とtechnical STOPの区別、保存Evidenceとglobal one-shotは成立する契約になったか。

**Q6.** 新preflight/release candidateでSOURCE/input/runtime/seed/authority/budget/STOPを結合できたか。実科学入口は閉鎖されているか。

**Q7.** Trusted store/Real Authorityの技術実装・人工検証はどこまで進み、所有主体・鍵・失効・durability等の外部判断は何が残っているか。

**Q8.** 新規unit/negative/integration/mutationと必要な差分回帰は何件PASSしたか。未解決の技術的不整合はあるか。

**Q9.** ユーザー・GPTが正式SOURCE・correctness契約・authorityを判断するための資料は揃ったか。初回実科学計算の別途実行承認前に必要な条件を、具体的なチェックリストとして提示できるか。

**Q10.** 実G7/F3 Core import/sampling、科学bitstream生成、新科学計算・再解析、実grant/claim発行消費、native処理、旧grant再利用、STOP解除、実correctness、limited pilot、Full-Nがすべて0件であるか。

---

## 14. 完了・公開・Mandatory STOP

次を完了すること。

1. 固定SOURCEと資料の取得・照合。
2. 新branch/worktree作成。
3. 実G7/F3接続用SOURCEの準備。
4. 512件correctness契約候補のversioned固定。
5. 各科学関数・samplingのcall予算整理。
6. Real Authority・trusted storeの技術実装・必要条件整理。
7. Release/preflight/worker bindingの準備。
8. Monitoring・Storage・STOPの接続。
9. 人工unit/negative/integration/mutationテスト。
10. 必要な差分回帰。
11. 未解決事項と必要承認の明示。
12. Evidence Packet作成。
13. Commit/push。
14. Remote HEAD・公開bytes/SHA256照合。
15. 固定SOURCE、raw、grant、ledger、marker、STOP、保護worktreeの保全確認。

技術的な軽微修正は同一工程で解決してよい。

未決の外部authority条件がある場合、それを推測して正式運用可能にしてはいけない。

実科学moduleのimport・samplingが必要になった場合も今回実行せず、その最小必要範囲を報告する。

完了後は**mandatory STOP**すること。

### 次工程へ自動移行しない

次は禁止する。

- 科学計算実行releaseの正式有効化
- 実grant発行
- 実科学correctness512 calls
- Limited pilot16,896 calls
- Full-N
- STOP解除

次の担当は**ユーザー・GPT**である。

最終目標は、既存の独立受入済み人工技術を基点として、実G7/F3 correctness検証の実行SOURCE・契約・authority・監視・保存について、**科学実行前に解消できる技術課題をまとめて解消し、明示承認と外部authority条件だけを残した状態でSTOPすること**である。