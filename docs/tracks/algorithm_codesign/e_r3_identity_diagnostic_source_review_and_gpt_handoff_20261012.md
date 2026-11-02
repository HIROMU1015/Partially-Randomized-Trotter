# E-r3 identity-only診断：source準備・限定scope確認

2026-10-12 JST。**READY_FOR_SEPARATE_E_R3_IDENTITY_ONLY_SOURCE_REVIEW / mandatory STOP**。
登録診断は未実施、diagnostic/science execution unauthorized。
今回の作業はGPTレビューを受けたrunner・contract・protectionの準備と登録外テストのみ。

## 固定した入力と来歴

- 最新GPTレビュー原本：[受領Markdown](../../research/track_b_df_correction_fingerprint_diagnostics_review_20261012.md)。
  27,804 bytes、SHA256 `83476eea74d6be02a5c039f325e9384bac5079ff629068fc2e8f66d565717526`。
  intake commit `838768a936ea475d75cae81cb35927a03583832b`からGitHub再取得・bytes一致を確認して実装を開始した。
- 基点公開commit `47bb7e8ef67690846ef4d4769a1f3057e0922061`、prior診断source `b5fd12370eae8fc5064039355ba218d342125d06`。
- 保存input/result基点 `ce0e286430a2a23869cb9417c82113c98064c72c`。
- 旧science S' `41349b73157c7781e0b534cf0fe61e06bf439954`、旧A `a1f537b68f0f22419dff0202c553b7511653caa3`。
- 旧marker SHA256 `dd6b9d5ec96e9d34d7bea0277eadb03f031992e1010cfdb2c940a393b2242701`。
- 準備branch `track-b-e-r3-identity-diagnostic-preparation-20261012`。

保存E-inputsとE-r3 jobのfile identityをcontractへ固定した。
準備時に行ったのはmetadata/bytes照合だけで、登録frameのarray化・Q/T/U計算・payload取得は0。
旧comparison source、共通library、診断adapter、旧authorization/manifest/marker/resultは変更していない。

## 今回準備した入口

[runner](../../../scripts/tracks/algorithm_codesign/run_e_r3_identity_diagnostic.py)はCLI overrideを受けない。
新namespaceの[launch](../../../src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic/launch.py)、
[worker](../../../src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic/worker.py)、
[evaluation](../../../src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic/evaluation.py)を追加した。
full cost adapterを呼ばず、既存array recorder・exception recorder・payload hash・definition index・lossless storageを再利用する。
source/APIや旧数式を修復する変更ではない。

execution対象は**保存r3、3×rank2、uncontrolledのみ**。
旧jobとE-inputsのr3 hexが一致し、旧markerの参照identityも整合することを確認して使う。
RNG、r6 array化、controlled、CD、Pauli table/cost、T inventory、T合成、shots/trajectory/LP/DF/分子/NPZ/GPUは行わない。

固定順序は **current→legacy**。それぞれprimitive signs=(+1,−1,+1)、
`comm(Q0,comm(Q1,Q2))`、`expm(1j*.03125*T/4)`を使う。
legacyは旧sourceのprimitive/JW/commの三関数だけをASTから読み、同じ保存frameと現在の固定runtimeへ適用する。
**legacy-source reconstruction under current runtime**であり、historical旧frame/runtimeを復元したものではない。
currentのassembly/Gram-Schmidt guardsとfloat DAGは維持する。
元のmain、registered recipes、他fixtureは呼ばない。

各経路でinput circuitを一つ作り、recursive payloadを一回だけ取得する。
frame/Q/T/U/target/実gate parameterのdtype・shape・layout・全bytes・SHA256をstageごとに同期保存する。
expected値は旧r3/uncontrolled recordの
`171f321e2f3bcc51b177e7b62735b5fbf9d80e36a5f56639b7f03809efac41b5`のまま。
actual（None含む）、生成例外、全recursive payload、definition index、root circuit名とcounter前後を保存する。
counter reset、名前を消したhash、新expected採択は行わない。

二経路の最初のmatrix bytes差と全payload field差を比較する。
自動名／matrix bytes／phase・parameter／wire／その他という**表示用の差分分類**を保存するが、
元payload/hashや受理guardを変更しない。旧科学結果へactualを書き戻さない。

## definition処理と資源会計

native `transpile()`／native gate数・価格比較は0。
ただしpayload取得は`UnitaryGate._define`と内部`qs_decomposition`を起動し得る。
一回のpayload取得中だけPython call observerを設置し、この二関数の実entry数を記録する。
observerは既存gate定義やname counterを変更せず、終了時にprofile hookを復元する。
fresh childに別profilerがあれば停止する。観測によるtiming overheadは実行会計へ含める。
C/Rust内部演算を個別に数えたとは主張しない。
同scope中の`qiskit.compiler.transpiler.transpile`へのentryは、body開始前に拒否してtechnical STOPする。
これは旧fingerprint guardの変更ではなく、診断scope外APIの誤呼出し保護である。

結果前[contract](../../../artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/contract_v1.json)：

| 項目 | 固定値 |
| --- | --- |
| path evaluations / payload attempts | 2 / 2、各path一回 |
| expm / input circuit | 2 / 2 |
| workers / numeric threads | 1 / 1、phase overlapなし |
| total CPU / worker CPU | 120 s / 90 s |
| aggregate RSS / worker RSS | 768 MiB / 512 MiB |
| output / log | 8 MiB / 1 MiB、terminal用256 KiBを予約 |
| review file | 最大512 KiB、lossless chunk方式 |
| total/phase wall cap | なし。tool polling timeoutとは別 |
| native transpile / T synthesis | 0 / 0 |
| retry | 0、partial failure後も再試行なし |

固定numeric runtime/executable/package/binary treeは旧numeric contractと同一。
precision、old error contract、old resource capsは変更していない。
この小scope用CPU/RSS/outputは旧native/total capの拡大ではない。
内部definition分解がdominant候補であり、serial順序がname/counter観測の前提なのでregistered pathsは並列化しない。
事前read-only CPU32、MemAvailable約51.8 GiBを確認し、BLAS/OpenMP/Rayon等1 threadを選んだ。
GPU query/useとspeedup claimは0。
外側guardはCPU/RSS/output/logを50 msで監視し、検出時にprocess groupをkillする。
workerのRLIMIT_CPU/FSIZEも適用。RSS瞬間上限の無overshootは保証しない。

## one-shot保護・状態の解釈

旧launcher/result pathには接続しない。
新result namespaceは `artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v1/`。
現在、このnamespaceは存在しない。
[新authorization](../../../artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/authorization.json)はpending、runs0、source_commit=null。
別の明示承認まで、registered runnerは入口で拒否する。

未来の実行は、固定source Sの直接子authorization-only Aから開始する。
Aの変更可能pathはこの新authorization JSONと任意の新receipt Markdownだけ。
S/contract/manifest/remote SHA・clean・専用branch・fresh namespaceをlaunch gateで照合する。
runtime checkはmarker前、registered array化はsource/marker/child-exclusive receipt/runtime検証の後。
old authやold markerを新実行へ流用しない。
childは新markerのsource/authorization/contract/manifest/branch/remoteを検証し、exclusive `identity.started.json`で再起動を拒否する。
namespaceが既に存在すればmarkerが未完でもSTOP。

完了分類は `E_R3_IDENTITY_DIAGNOSTIC_COMPLETE` または
`E_R3_IDENTITY_DIAGNOSTIC_TECHNICAL_INCONCLUSIVE`。
正常に保存されたfingerprint `MATCH/MISMATCH/EXCEPTION`は観測値で、比較科学の受理判定ではない。
fingerprint例外も観測記録が完全ならpathは完了できる。
construction/storage/cap/prohibited API/incomplete path failureはtechnical inconclusive。
部分prefixから旧causeやA/Bの優劣を結論しない。
どちらの分類でもmandatory STOP、retry0。

## 登録外検証と来歴

[新tests](../../../tests/tracks/algorithm_codesign/test_e_r3_identity_diagnostic_preparation.py)のみを実行し、
既存104件や本番fixtureを再実行していない。
[focused receipt](../../../artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/focused_tests_v2.json)：
**34 PASS、0 failure/error/skip**。
wall 1.972 s、CPU(parent/reaped)1.900 s、sampled aggregate peak RSS405.637 MiB。
source/test bytesを実行前後で照合し、固定numeric runtime identityも一致した。
初回33 PASSの後、静的確認で既存profilerとの競合を通常のfingerprint例外として完了扱いにできる分岐を確認した。
この実行前提違反を保存後にtechnical STOPへ伝播する保護と専用testを追加し、最終34 PASSを確認した。
初回receiptと凍結前draft source/manifest/audit/authをcheckpointへbytes不変で保存した。
この反復は準備testsであり登録診断のretryではない。

- pending／wrong authorization、direct child／diff／remote／clean／source identityの拒否。
- old marker不変、新marker/terminal/exclusive childの再利用拒否、parent-child provenance一致。
- 小さいanalytic 2-mode rank1×3のcurrent/legacy入力identity観測一組。
- 別のsynthetic 8×8 diagonal入力でdefinition observerを確認：root UnitGate `_define`1 entry、QSD Python5 entries、native transpile0。
- native transpile入口の拒否probe一件。bodyを実行したnative compileは0。
- payload例外3種、no retry、保存失敗時の停止、名前差のdiagnostic-only表示。

QSD controlは登録E-r3 frame/Q/T/Uではない。追加の登録matrix/payload/scienceを取得したと扱わない。
登録diagnostic execution0、registered RNG/arrays/matrix/payload0、native transpile body0、T合成0。
source-bound local testsでありimmutable CIや外部再現ではない。

## GPTへの限定確認とSTOP

確認事項は、source/contract/manifest/launchが本レビューの**E-r3 uncontrolled identity-only**範囲に閉じていること。
新sourceの数学・新規性を再採択するレビューを求めるものではない。
[source manifest](../../../artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/source_manifest_v1.json)、
[静的provenance audit](../../../artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/source_scope_audit_v1.json)と
GitHub取得receiptを公開する。
旧source/result/auth/marker/STOPの不変性を照合する。

scope/source確認後でも実行は別の明示one-shot承認が必要。
レビュー後に、今回固定するsource Sから独立実行branch/worktreeとauthorization-only直接子Aを作る。
公開確認HEAD／receipt commitをsource Sと取り違えない。
本branchでは登録診断・native compile・T合成を行わず、公開と取得確認後**mandatory STOP**。
