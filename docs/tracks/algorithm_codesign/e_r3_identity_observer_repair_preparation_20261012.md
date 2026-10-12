# E-r3 identity-only診断：observer限定修正・source v2準備

2026-10-12 JST。判定：**READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION**。
登録診断は未実行。新authorizationはpending、runs0、science/diagnostic authorized=false。
追加のGPT詳細sourceレビューは不要で、次は新固定sourceに対するユーザーの明示one-shot承認を待つ。
この報告のREADYは実行承認・科学的GOではない。

## 修正の根拠と保護

旧診断source S `8377c1d12439cabab3b97bcc1f57ac030a479785`、公開確認P `0cf991d4371250dc6a2d8b346ed4ed7414f4a198`を保持する。
[最終技術監査](e_r3_identity_final_technical_validation_20261012.md)の公開commit
`101e225a5cc8fe2f385f8fd802682dc87b708c16`から独立準備branch
`track-b-e-r3-identity-observer-repair-preparation-20261012`を作った。
利用者の「次の作業に進んで」は、その監査で提案したobserver限定修正への指示として扱う。
登録診断・新authorization-only commit作成への認可とは扱わない。

issue `E3-ID-TECH-01`はobserverの設置／復元失敗が通常EXCEPTION観測として戻る実装不具合。
今回の修正は既存のtechnical STOP契約を実装するもので、研究仮説・fingerprint受理policyの採択を変更しない。
[共通指示](../../research/research_project_instructions_gpt_codex_review_approval_2026-10-10.md) §2/3/5/14に従い、
この技術修正のためにGPT詳細レビューを反復しない。
旧監査の`BLOCKED_NEEDS_TECHNICAL_REPAIR`記録・再現tests・STOPは履歴として不変。

旧Sの260保護対象のうち、新sourceで意図的に変更したpathはevaluation、launch、
preparation testの3ファイルだけ。残る257は同じbytes。
旧SのGit objectsと旧source worktreeは変更しない。
旧science `41349b73157c7781e0b534cf0fe61e06bf439954`、authorization `a1f537b68f0f22419dff0202c553b7511653caa3`、
旧result28ファイル・marker・terminal・`TECHNICAL_INCONCLUSIVE`は保持する。
共有library Python、worker、registered runner、旧error/resource guardを変更していない。

## observer lifecycleの変更

[evaluation](../../../src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic/evaluation.py)に
`ObserverFailure`とinstallation/restorationのattempt・verification・例外記録を追加した。

- fresh context確認後にobserverを設置し、実際のhookが一致することを検証する。
- 設置後に例外が起きた場合も、以前のhookへのcleanupを一回だけ試み、失敗を併記する。
- 正常／payload例外の両方でrestoreを行い、以前のhookと一致することを検証する。
- preexisting profilerは除去せず、technical failureとして停止する。
- payload生成例外とobserver lifecycle例外を別fieldへ保存する。
- 復元失敗前に取得したpayload・actual hashを保持し、technical prefixとして保存した後にraiseする。
- 設置失敗のpayload実entryは0。取得した場合だけ`recursive_payload_attempts`を1へ増やす。
- cleanupは計算再試行ではない。payload/経路のretryは0、observer cleanupのretryも行わない。

observer failureは`TECHNICAL_OBSERVER_FAILURE`、phase `TECHNICAL_OBSERVER_FAILURE_SAVED`という
診断prefixを持ち、workerの`E_R3_IDENTITY_DIAGNOSTIC_COMPLETE`へ戻らない。
これは新しい科学outcome分類ではない。外側runnerは従来どおりchild failureを
`E_R3_IDENTITY_DIAGNOSTIC_TECHNICAL_INCONCLUSIVE`＋mandatory STOPへ戻す。
prefix保存自体が失敗した場合も例外を伝播し、markerを解除しない。

通常のpayload TypeError/ValueError等は引き続き`EXCEPTION`／actual=Noneとして観測できる。
`shared_None_fallback`はfingerprint生成例外の型だけに対応し、observer例外を混ぜない。
元のpayload hash・expected・matrix bytes・名前差分類は維持する。
`automatic_names_only`はpayload差の属性であり、独立のmatrix-stage比較を併せて読む。

## 診断契約・namespace

新準備pathは
`artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/v2/`。
旧準備ディレクトリ内の既存ファイルは上書きせず、新versionを追加する。
[contract_v1.json](../../../artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/v2/contract_v1.json)は
旧contractと**全bytes同一**：SHA256
`12345d68a1d3d50869f2928d9a441073fce8e9e82f66c73856c6956686a5f488`。
科学条件・error・CPU/RSS/output/log上限を変更していない。

| 固定scope／上限 | 値 |
| --- | --- |
| 入力 | 保存E-r3、3 mode、rank2×3、uncontrolled |
| 順序 | current→legacy、各一回 |
| signs／commutator／rotation | (+1,-1,+1)、comm(Q0,comm(Q1,Q2))、expm(1j*.03125*T/4) |
| worker／numeric threads | 1／1 |
| CPU total／worker | 120 s／90 s |
| aggregate RSS／worker RSS | 768 MiB／512 MiB |
| output／log | 8 MiB／1 MiB、terminal reserve256 KiB |
| review chunk | 512 KiB以下、UTF-8 lossless |
| wall cap／retry | なし／0 |

legacyは旧helperの数値経路を現在の固定runtimeで再構成した参照で、historical runtime再現ではない。
Q/T/U/input構成のAST prefixは旧Sと一致した。
CD、E-r6、controlled、native transpile body、T synthesis、Pauli cost、shots/sampling/trajectory/LP/DF/molecule/NPZ/GPUは対象外。
内部UnitaryGate definition/QSDの診断処理だけを維持する。

新result pathは `artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v2/`。
v1/v2の登録result namespaceはともに未作成。
[launch](../../../src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic/launch.py)の
ART/RESULT/receipt参照だけをv2へ切り替えた。
parent、authorization-only diff、remote SHA、clean、manifest、exclusive marker/child、terminal STOPの規則は不変。

将来のAで変更できるpathは新pending authorizationと任意の
`docs/tracks/algorithm_codesign/e_r3_identity_diagnostic_execution_authorization_v2.md`だけ。
新source S2の直接子Aを要求する。旧S/P、今回のpublication receipt childを実行HEADにしない。
新source SHAは公開確認receiptと最終報告へ記録する。

## focused validation

[結果](../../../artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/v2/focused_tests_v2.json)：
**77 PASS、0 failure/error/skip、旧不具合再現2件はdeselect**。一batchを一回実施した。

| 検証系列 | 件数 | 扱い |
| --- | ---: | --- |
| preparation regressions | 34 | conflict例外fieldのassertionをv2に対応。実scope・保護・synthetic semantics維持 |
| 最終技術監査の回帰controls | 26 | 旧bugを期待する2件を除く同じtests。原tests／原28 PASS記録は保持 |
| 新observer修正controls | 17 | query/install/partial/silent/restore、両例外、payload保持、worker中止、v2namespace |

wall2.180954344 s、CPU(parent/reaped)2.092758 s、sampled aggregate peak RSS405.82421875 MiB。
test-only guard：CPU60 s、aggregate RSS512 MiB、worker448 MiB、output8 MiB、log1 MiB、wall capなし。
CPU32、MemAvailable54,319,224 kBをread-only確認、1 worker・threads1。
旧固定runtime（Python executable、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、mpmath1.3.0とcode/binary tree hashes）を照合。
source bytesはtest前後で一致。full suite、登録diagnostic、science、GPU queryは行っていない。

synthetic frameは従来のanalytic 2-mode/rank1で、登録r3を使用していない。
独立の8×8 diagonal QSD controlは `_define`1／QSD Python entries5／native body0。
このdirect context controlは`observe_path`を経由しないため、そのrecordの
`recursive_payload_attempts=0`はpath入口counterを更新していないという意味である。
test sourceではpayloadを一回直接呼んでおり、QSD probeを0回と主張しない。
registered経路のattempt counterは`observe_path`が所有し、新controlsで0/1を確認した。
新規native transpile bodyは0、入口で止めるsynthetic probe2件。T合成・CD/E登録評価0。
実行時のcap内完了をこれら小さいcontrolsから保証しない。

## 実行承認待ちとSTOP

旧observer blockerは新sourceで解消し、source/contract/manifest/pending authの整合性を静的監査する。
新manifest・pending binding・focused証拠を固定してcommit/pushし、remote bytes/SHA256とcleanを確認する。
技術検証を超える研究判断は今回発生していないため、GPT追加詳細レビューを要求しない。

次に必要なのは、**新固定source S2のE-r3 identity-only one-shotに対するユーザーの別途明示承認**。
承認後だけ、S2から独立実行branchを作りauthorization-only直接子を作成・pushし、新v2 runnerを一回実行できる。
今回はその承認を受けておらず、A・registered marker・registered Q/T/U/payloadを作らない。
実行結果／旧不一致の原因／修正方法の研究上の解釈は、診断後の実測証拠をGPT側へ戻して判断する。
科学条件・fingerprint policy・expected・capsの変更が必要になればその新事項だけGPT判断へSTOPする。

本準備完了時点で**mandatory STOP**。登録診断や旧one-shot再試行へ自動進行しない。
