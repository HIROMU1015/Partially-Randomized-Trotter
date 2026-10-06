# R1 v2: narrowed finite-RTE implementation preregistration / source review

2026-10-06 JST. 利用者の[GPT review指示](inputs/r1_preparation_gpt_instruction_20261006.txt)に従う準備。
`PROCEED_TO_R1_PREPARATION_WITH_NARROWED_CLAIM`、**R1 science未認可**。
R0.5固定commit `61dd534567fda5c7348fdc688814089eb26a3561`から独立branch
`track-b-rte-reallocation-r1-source-preparation-20261006`を作った。
[v1 conditional proposal](rte_reallocation_r1_proposal_for_review_v1.md)の本文は履歴として保持し、
今回の技術仕様は本v2と[machine contract](../../../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/contract_v2.json)へ置く。
v2の数値政策・domain・sourceはGPT source reviewの対象であり、実行承認ではない。

## RQと非claim

Pauli coefficient collectionを要求しないHermitian-involution accessにおいて、
有限meanを保存するAのrepresentation変更によるnormalization改善が、既知ordinary pairing / PTSC-K0に対する
native controlled event資源の差として残るか。restricted classの定理とnative action costの交換だけを問う。
Taylor再配分・identity/common-angleの新原理、世界初、最良LCU、CTS一般への優位、DFでの利益を主張しない。
[R0.5監査](rte_reallocation_r05_equivalence_novelty_audit_v1.md)のscoped method-delta候補を使う。

## 固定domainとevidence role

targetは `P_3(-i sigma x Rhat)`、`K=2`、`Rhat=3Q0/4+Q1/4`。
x={1/8,1/4}、sigma={-1,+1}、native primitive precision={1e-3,1e-4,1e-6}。
旧案のcontext/x/sign/precisionを維持し、eta探索はせずordinary/Aの端点だけを用いる。

| Context | Q0 / Q1 | Arms | Role |
|---|---|---|---|
| Common Pauli commuting | Z0 / Z1 | ordinary, PTSC-K0, A, collected CTS | collection control |
| Common Pauli noncommuting | Z0 / X0X1 | ordinary, PTSC-K0, A, collected CTS | phase/collection control |
| Distinct basis | Z0 / V†Z1V、V=exp(-i pi X0X1/16) | ordinary, PTSC-K0, A | controlled rowsがprimary performance context |

ordinary native circuitとcontrolled native circuitは別記録。ordinaryはdiagnosticで、coherent-taskの主証拠へ混ぜない。
11 context-arm combinations ×2 x ×2 sign ×2 native context ×3 precision = **264予定resource rows**。
controlled rowsは132、comparison groupsは72。新PF/split/geometry/degree/etaのgridはない。

**このtwo-qubit distinct-basis fixtureも原理的にはPauli展開可能である。**
I0 primaryは「このpilotではglobal Pauli collectionを使わない」という比較contractであり、
このfixtureのI1取得が不可能・高コストだという証明ではない。
利用可能な情報による実際の境界やDF scaleの古典費用までこのtoyで立証できるとはしない。
その意味をGPTがsource reviewで確認し、必要ならauthorization前に仕様を改訂する。
CTSにcontrolで負けても、それだけでAのI0仮説failureとは判定しない。

## 同target / 同accessのbaseline

- ordinary: degree0/1、2/3の既知pairing。
- PTSC-K0 finite: identity/order1 pairとpure degree2、3 words。同じtarget、I0生成可能。
  高次Trotter remainder compensationはtargetが違うため入れない。pure degree3の±iもcontrolled phaseとして払う。
- A: R0 closed-form optimumのみ。odd complementの追加Q・角度反転・phaseをすべて実装する。
- CTS: zero-degree identityを別に保つliteral collected `C+I+iS`。Pauli controlsでのみ生成する。

primary samplingは全arm **canonical**。IS/PAIを含めない。ISを将来行うなら別secondary契約で対称に扱い、
今回のrepresentation差へ混ぜない。全event列挙は小pilotのpopulation evaluatorであり、trajectoryはsampleしない。
I0 samplerの係数取得に全word列挙が必要だとはしない。

## Native sourceと結果前synthesis keys

[B専用source](../../../src/trottertracks/algorithm_codesign/rte_reallocation/)と
[runner](../../../scripts/tracks/algorithm_codesign/run_r1_rte_reallocation.py)を追加。
共有trotterlib / 既存RTEEventのeven invariantは変更しない。odd builderは新namespaceだけ。
[phase/basis/error仕様](rte_reallocation_r1_native_semantics_v1.md)を参照。

Pauli contextでは全armのword productを同じPauli規則で簡約し、phaseを保持する。
distinct basisではQ²の隣接相殺だけ。全armに同じnative adjacent-inverse cancellationを適用する。
任意角fusion、異なるeventの結合、post-synthesis whole-string optimizerはない。
metric名は **native IR after shared exact cancellation, additive synthesized-primitive Clifford+T counts**。
actual whole-wrapper compile最適値、hardware gate count、分子DF wrapperとは呼ばない。

non-Clifford magnitudeは各xのatan(x)、atan(x/3)、atan(rho)、CTS controlの二つのatan(Ls)。
ordinaryの±2倍、controlledの±1倍、basisの±pi/8で **42角度**。
3 precisionと組み合わせた **126 planned synthesis keys** を
[inventory](../../../artifacts/track_b_rte_reallocation_r1_source/2026-10-06/synthesis_key_inventory_v1.json)へ列挙した。
旧78 key案からの増加はCTS control追加の結果前変更であり、結果を見た拡張ではない。
literal complement angleは合成せず、exact complement identityを使う。

toolはSP-0.5と同じpygridsynth 2.0.0の固定wheel/source identityを使い、科学sequence cacheは流用しない。
今回の新contractではdps=100、seed=0、**up_to_phase=false**。
strict operator guardを使うため、旧channel用projective option/guardは転用しない。
公式の[angle/phase option仕様](https://github.com/quantum-programming/pygridsynth)と
固定installed sourceのconfig/gridsynth処理を確認した。
登録angleに対する合成・smoke合成はまだ0。runtime照合はmetadata/.py hashだけ。

## Resource vectorとfinite-confidence会計

primaryはB²、E[C_T]、E[C_CX]、E[C_1Q]、workspace、C_classicalの記録。
ideal B² enclosureと、丸め後の実装weight second momentを別fieldで保存する。
C_classicalはdescription entries、expected index draws、CTS collection操作数、evaluator support、setup timeの構造化記録。
操作を任意重みで単一costへ足さず、wall timeをnative resource優位の判定へ使わない。

order/group normを100 dps intervalで囲み、そのmidpointを有理数として固定してから、exact IID word probabilityを掛ける。
これにより丸め後もorder→IID生成を保つ。alpha_mid / B_mid、weight B_midをexactに整合させ、
係数L1変位をbiasへ戻す。実際のfinite-bit RNGは実装・実行しない。
strict native operator誤差の和からjoint event誤差を作り、coherent測定biasへは係数変位に加えて
**2 sum alpha_mid delta_event**を戻す。ancilla-0が完全なidentityを保つという仮定を追加しない。

secondaryはfixed |00> system、一blockのcomplex coherent signalに対する共通sufficient-shot forecast。
同じfinite polynomialをtargetにし、exponential truncation errorを課題へ混ぜない。
epsilon_complex=1/100、axis accuracy=1/200、familywise alpha=1/20、264 axisへalpha_axis=1/5280。
v≤B_mid²、centered range≤2B_midを使うBernstein sufficient shots。
coefficient bias cap=1e-25、shot cap=1e9/axis。biasがaxis budgetを使い切れば不適格。
state preparationのT/CXは0、Hadamard preparation/readoutはRe=2、Im=3 single-qubit gates/shot。
G_T、G_CX、G_1Qを保存するが、測定shots・exact signal・trajectoryは実行しない。

B²で優位でもnative資源にtrade-offがあればそのまま記録する。
ratioはstrict intervalと1との関係だけ。zero baselineをdivision/winに使わず、overlapを未分離として保存。
**単一scalar materialityや自動研究GOを設けない**。成功terminalはresource map complete awaiting GPT review。
これはthresholdを結果後に選ぶ方針ではなく、R1の役割をvector/mechanismの記述へ固定するもの。
method採択・論文claim・次stage必要性は結果後GPTが別判断する。

## Cap / one-shot / source freeze

将来の上限案を結果前固定：one process、one run、retry0、wall20分、CPU900秒、RSS512 MiB、
virtual address1536 MiB、126 synthesis keys、per-key wall30秒/CPU20秒、sequence20,000文字、new output16 MiB。
timeout/error/numeric/cap failureでもmarkerを残してretryしない。grid/precision/比較追加なし。
全結果mandatory STOP、DF/分子/GPU/次wrapper未認可。

本commitがsource Sを固定する。source review後、Sの**直接の子**Aでauthorization JSONと任意receiptだけを変更する。
現authorizationはpending、source_commit=null、science_execution_authorized=false。
runnerはpending時にtool、marker、key/target、native cost取得へ進む前に拒否する。
GPT source review → separate authorization-only child → 利用者の明示実行指示の順を守る。

今回の準備は27 focused tests、static key inventory、runtime hash照合まで。
science resource rows/synthesis/compile/trajectory/分子/DF/NPZ/GPU=0。
sourceをcommit/pushして[GPT source review](rte_reallocation_r1_source_review_request_20261006.md)へmandatory STOP。
