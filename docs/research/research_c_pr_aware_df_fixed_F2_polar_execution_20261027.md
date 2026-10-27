# 研究C：F2固定Baseline CAR／polar target限定科学実行

2026-10-27 JST。結果：**F2_STOP_KAPPA_EXACT_HERMITICITY_NOT_MET**。最大7 logical unitsのうち **4 attempts＝3 COMPLETE＋1 FAIL、残り3 NOT_RUN**。CAR reference入口で必要なκの厳密Hermiticity条件が不成立となりSTOPした。Raw/CAR距離は未評価であり、測定されたoperator correctness FAILではない。Retry/resume0、F3未実行。

ユーザーがB0/B1 target採択、新契約固定、F2科学実行、公開・照合・STOPまで一括許可した範囲で実施した。追加承認・承認証明書・承認原文保存・新grantは要求・作成していない。取り消されたP1は再開していない。

## 固定対象・SOURCE・契約

保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF rank8、Baseline／B2 run02 `baseline_00`のみ。Nα=Nβ=2/sector36、system8/full256、last ancilla/full512。元frame順LD2、partial symmetric S2、δ=.1/s4/h=1/40/r16/K0、16 saved events。元g補正κ、scalar、signed atoms、tail identityとhead順を保持し、samplingし直していない。Own-DF ground oracle・状態準備除外は保存条件で、ground/PF/RTE/DFを再計算しない。

P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/qubits_initially_zero=false、MPFR160 declared phase、B2共通Gaussian政策を保持。新しい回路build/compileは0。既存Baseline raw/compiled certificate上界3.4694675824e-12、Proposed3.2092694515e-12は旧結果の引用で、今回再計算していない。

公開起点：`52185764ccda60ddb4be04465cc644ed56deeb98`。独立branch：`codex/research-c-fixed-F2-polar-execution-20261027`。

- 新科学SOURCE commit：`45ae98943492e921c22a0f527d5587cd1e06462f`
- [SOURCE135 manifest](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-polar-preparation/SOURCE.json) SHA256：`9d077123674137b23a9d050f013eb5984f2d57f3ea2e29c4d23f02b16225ca34`
- 新[実行契約](research_c_pr_aware_df_fixed_F2_polar_execution_contract_20261027.json) SHA256：`c310f95ad6f25500284c46edbd82fb31dad1a1b82c7cd5c4980a8c9991716da3`
- 契約・preflight公開／実行commit：`b93a621fb3e154d1e9556fcabfd284976f6561da`
- Fresh output：[F2-polar-run01](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-polar-run01/)

旧[実行不可F2草案](research_c_pr_aware_df_fixed_car_contract_draft_20261027.json)はSHA256 `670d40c62e9d336e8c23b16d2f79f036807280b4888e31c02587209b0f489d7f`、`executable=false`のまま保全した。新契約だけを独立して`executable=true`とした。既存runnerが要求する歴史的`status=UNAPPROVED` tokenは互換のため残し、別fieldで今回の許可scopeを記した。これは承認を証明する仕組みではない。

## 再利用・実行前確認

`build_reference/polar_bridge/compare/compose`、Arb192 gate verifier、CAR algebra、event_unitary、head scheduleは `e7d22a404442234662ef6d117c55dd9868217c9e`からbyte不変で再利用。追加したのはF2専用のtarget／input／fresh output／旧certificate照合 `c_pr_aware_df_fixed_f2.py` と専用CLI、人工scope/bridge testsだけ。Gaussian emitter、matrix exponential、fitter、CAR代数を再実装していない。

[準備記録](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-polar-preparation/method.md)／[テスト](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-polar-preparation/tests.log)：人工・回帰 **107 PASS**。保存数値decode前に、target/arm/index/calls/precision/retry変更拒否、正の対角dyadic Uの既知polar Iに対する解析bridge、既存CAR/native/全controlled基底/位相/不確かさ/null/STOP経路を検証した。人工PASSを実入力のHermiticityや科学的PASSとして扱っていない。

GitHub/local起点identity、SOURCE135/input33/package11、新契約・元入力/JU/B2 trajectory/SOURCE/STOP、既存Baseline compiled/raw certificate hashと同一raw/dimension/phaseがpassive照合PASS。[Preflight](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-polar-preparation/passive_preflight.json)。予定参照block25、全basis column512を既存resource planで並列化する準備を行い、affinity32・空きRAM約56.5GB・memory pressureなしを確認した。CPU/RAM/worker/wallの人工上限、共有venv/cache/OS/他job変更はない。

## 採択した数学的対象と未取得値

B0は保存binary64 J/U・返された元g補正κ・constant・元frame順・signed atoms・16 events/angles・tail identity・forward/reverse headが定義するliteral dyadic CAR map。近似Uがexact dyadicでは厳密unitaryでない可能性を残し、B0をunitaryと仮定しない。

今回に限り採択されたB1は、各保存spatial Uの唯一のright polar因子 `Q=U(U†U)^(-1/2)`をCAR densityのbasisだけへ適用する理想unitary target `CAR_polar_of_exact_saved_U_exact_J_kappa_phase_v1`。J、κ、scalar、angles、events、順序、tail identityは変えない。元Uを書き換えず、raw QPYからtargetを逆算しない。

予定したbridgeはη≥||U†U−I||2<1からdU=η/(1+sqrt(1−η))、density/frame/signed atom/head/event差を伝播し、prod(1+εj)−1で完全trajectory差を上から囲うもの。CAR referenceのball区間はraw/B0差に一度、model bridgeはraw/B1に一度、旧compiled/rawはtriangleに一度だけ加える。全512の位相込みoperator2、Arb192、allowance exact1/100000000、global phase fitなし。数学的条件不成立で自動symmetrization・unitarization変更・fallbackはしない。

今回はκの必要条件で先にSTOPしたため、以下はすべて **未取得/null**。

| 評価対象 | 結果 |
|---|---|
| Raw／B0 literal CAR validated上界 | null／NOT_RUN |
| U†U−I、U→Q、B0→B1 bridge | null／NOT_RUN |
| Raw／ideal CAR-B1 validated上界 | null／NOT_RUN |
| Compiled／ideal CAR-B1 triangle上界 | null／NOT_RUN |
| κ非Hermiticityの大きさ・原因 | null／追加診断未実施 |

## 実行順序とSTOP原因

| Logical call | 状態 |
|---|---|
| input_arrays | COMPLETE：固定元inputとBaseline final J/Uロード |
| arm_replay | COMPLETE：既存prepare_armで元g補正κ・native model record・signed plan再構築 |
| trajectory_load | COMPLETE：保存16 K0 eventsを復元しsigned metadataと照合 |
| CAR_reference | **FAIL：EXACT_HERMITIAN_INPUT_REQUIRED** |
| qpy_decode | NOT_RUN |
| raw_CAR_certificate | NOT_RUN |
| triangle_composition | NOT_RUN |

[Failure原本](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-polar-run01/failure.json)のtracebackは `build_reference → _hermitian_input(kappa)` を指している。登録入力gateはfiniteかつ`np.array_equal(kappa,kappa.conj().T)`を要求する。元の入力・κ・閾値を変更せず、この条件で停止した。

Arb CAR block生成・matrix exponential・polar bridge・raw QPY decode・全basis比較・triangleは開始されていない。したがって「raw回路が理想CARと一致しない」とは判定していない。必要入力条件のFAILと未実行のoperator correctness gateを区別する。κの非Hermiticityの大きさ、その発生が元積分・元g補正・丸めのどこに由来するかは、現在の証拠だけでは特定しない。追加科学callや保存入力の再解析をしていないので、「微小丸めだから無視可能」とも断定しない。

## Call・資源・保全

実績 **4/最大7 attempts、3 COMPLETE、1 FAIL、3 NOT_RUN**。Registered call journal・failure・resources・STOPを保存した。Runner wall **1.0595824290066957秒**、parent peak RSS370624KiB。Block/column workerは入口条件で止まったため未起動で、事前の25/32-worker planを実際の使用数とは扱わない。終了後、自分のjob／worker残存0をread-only確認した。[資源記録](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-polar-run01/resources.json)／[monitor](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2_polar_resource_monitor.jsonl)。

新fit/optimizer/sampling/build/compile/DF-PF-RTE再計算/C1再解析/shots/総資源0、retry/resume0。残枠3を別計算・診断・F3に流用しない。

[Postflight](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2_polar_execution_postflight.json)：SOURCE135/input33・新契約とmanifestが実行前後一致、旧B1 raw25／F1 raw13不変。旧草案・旧SOURCE/raw/result/grant/ledger/marker/STOPを変更していない。今回のraw6ファイルは各SHA256をpostflightへ記録。[要約](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2_polar_execution_summary.json)。

## 科学的範囲・次担当

今回得たのは、固定条件で独立CAR保証へ接続する際に **元g補正κの登録Hermiticity条件が満たされず、既存のpolar-basis補正だけでは実行前提が揃わない** という証拠である。Basis Uのpolar補正はκを変更しないため、このSTOPを回避しない。これはPR-aware DFの科学仮説・CDF fit・総資源削減のNO-GOではない。

元物理積分・DF/PF/RTE誤差・PCG64 IID・未観測trajectory・all-law synthesis・実装CA・保証shots・cost support・母平均/総資源優位は未資格のまま。C1約83%点推定と既存Baseline/Proposedの有限raw/compiled証拠は変更しない。

次は **ユーザー／研究C** が、F3より先にκのHermiticity不成立の大きさと由来を確認する限定診断、またはtarget定義との整合方針を判断する。**Codex** は別途明示された範囲・許可後だけ保存入力の診断／必要な技術準備を行う。κの対称化・新targetへの変更は今回自動採択しない。確率モデル付き資格はこの入力前提を解決しないため、自動移行しない。

新SOURCE・新契約・partial evidence・STOPをcommit/pushし、remote SHA／Git tree／SOURCE/input/成果物照合後に **mandatory STOP**。本7-call枠でretry/resume、F3、追加trajectory、別CAR target実行、sampling、C1再解析、shots/総資源、別分子、GPT詳細科学レビューへ進まない。

要約JSON SHA256：`13320d94133bff474ae0ba7244a41eb74578c2e49d38969d0d3192b4731b4ffb`。Failure原本SHA256：`44f549e07b59f409c901769358ec2b628f0fca7b2d3b07e276a413a5b3d5040d`。
