# 誤差・confidence・費用の接続表

2026-10-20 JST。既存保証の適用監査であり、数値評価・新証明・予算算出・算法実装ではない。C1草案のT/q/r/m/ε/αは未採択のまま保持する。固定入力・Gate総合判定は[preflight](README.md)、意味論・fieldは[sidecar仕様](adapter_evidence_draft.md)。

## 1. 誤差ledger：何が既存で、何が未確認か

| 項目 | 使える既存保証・前提 | C1の未確認事項と状態 | 次に必要な作業 |
|---|---|---|---|
| 1 入力・表現 | 指定snapshotのopaque同一性。PR-2保存resultのmodel/state digest、[DFTAIL]のthreshold dropped-L1記録 | 保存DFモデル→basis/component→exact G6 modelの同一性は未認証。数値spectral residualはrigorous operator certificateでない。**CONDITIONAL / REANALYSIS_REQUIRED** | 保存registryの取得。採用モデルと係数/rationalization/basis差の定義・certificate。物理Hamiltonianとの差を別scopeにする |
| 2 outer PF | [PRCONTRACT]の同target規約、[S2]の固定順二次対称PF、既存PF誤差枠組み | C1の実scheduleに結合したrigorous H4 PF error上界は未取得。phase-rate guardや経験的C_useをPF norm上界へ昇格しない。**CONDITIONAL** | SOURCE/input依存の既存保証を選び、必要な保存数値評価を別承認。閉じなければ認証比較を保留 |
| 3 finite Taylor | [DACC] `error_budget`のm5短step剰余と[D3] §3のHermitian-contraction保証 | exact normalized Hermitian R、各x≤1、全tail時間が未確認。**CONDITIONAL / REANALYSIS_REQUIRED** | x/precision条件の認証と既存剰余の適用評価。今回代入なし |
| 4 非unitary積 | [DACC]の`(1+ε)^steps−1`と実装誤差積。独立ordered path、unitaryなexact比較対象 | D1はtoy Rのsteps=1,2。C1 outer PF/全occurrence/deterministic approximationを含む完成bindingはない。**CONDITIONAL / IMPLEMENTATION_REQUIRED** | 既存telescopingの適用条件と合成順を固定。新非自明な拡張が本当に要る場合だけPROOF_REQUIRED |
| 5 RA係数・実proposal | [MATH][BITS][F1]。正有理p、奇数m、短step、positive support、実πとW=c_tilde/π。zeroを含む全trial | 新exact p/λとP0–P5、全law identityをC1で評価していない。F3元canonical非identityは未対応。**CONDITIONAL / IMPLEMENTATION_REQUIRED** | exact入力binding、各armに正当なprecision述語と実law証拠。科学的law/bit条件を変えない |
| 6 root・angle | [F1] root誤差・uniform下界、[F3]の既存half-angle/atan enclosure | root biasはproposal補正で消えない。angle代表値・native角規約のC1接続は未取得。**CONDITIONAL** | H/K/η/ρ/angle精度を事前固定し、F3とproviderのcertificateを結合 |
| 7 strict native | [G8PROOF]の全label unitary strict CQ誤差≤δ、actual adjoint、相対位相を仮定した既存telescoping。RZ誤差は別 | H4の全CQ/basis/中央作用、angle・deterministic部のstrict certificateは未取得。旧δは仮想parameter。**CONDITIONAL** | 物理/論理providerの指定と別承認のnative取得。uniform仮定と実certificateの全support coverageが必要 |
| 8 state/readout | [HAD]のX/Y bit map・S†→Hとstate/backend明示除外、[E2E]の測定契約 | 保存数値stateは物理準備回路でない。bit順・準備誤差/成功率・readout/modelが未取得。**MISSING_SOURCE / IMPLEMENTATION_REQUIRED** | state preparation/論理測定scope・価格を明示。未取得を0にしない |
| 9 moment/range | [G7][W2B] Fullのκ B_ord U、range上界、canonical側のBによる既存certificate。[DACC]の独立path積 | C1全pathのV/R_max、zero・実weight・測定器の結合が未完。**CONDITIONAL / IMPLEMENTATION_REQUIRED** | 全trial accumulatorと全N完了binding。raw/coin/acceptedとtechnical failureを分離 |
| 10 confidence/N | [G7][G8BOUNDS][DACC]の既存Bernstein型、bias控除、ceil、resource-tailとfailure配分 | G7/G8/D1の固定α/familyはC1と異なる。C1 ε/α配分・正s・all-N completion・IID仮定が未採択。**CONDITIONAL** | 全採択armのX/Y・resource/技術failure scopeを事前固定し、別承認で既存式を評価。今回N未算出 |

### 既存保証を過少評価しない

G8既存契約のevent errorは`2 ε_RZ + Σ_i n_ei δ_i`。全labelのuniform δ、短stepのnormalizer上界と`n_e≤m+1`等を使う保守的axis bias式が[G8PROOF][G8BOUNDS]にある。これは全eventを数値列挙しない既存のstrict-error枠組みで、C1用に新しく証明したものではない。価格に使うproposal平均query数を、target-coefficientで重み付けするbias式へ代入してはいけない。direct DF経路やclosed P5を採るときは、その既存前提・呼出し上界・normalizerとprecision policyへの対応も確認する。

D1の非unitary積会計を、unitary誤差の線形和へ置き換えない。C1にはdeterministic blocksもあるため、どの局所誤差・比較norm・段数へ既存の積保証を適用したかを明示する。SOURCEのtoy-input guardを外して分子に実行することは今回行わない。既存部品の通常の適用が可能か未確認であることと、新定理が不可避であることは別である。

### Estimatorとconfidenceの不変条件

- ideal Fullの全trial二次momentはB_ord B_full、accepted canonicalのB_full²とは試行単位が違う。digitalでは実π/Wとκ/U等の認証上界を用い、未知のB_fullや真のsignalをoracleにしない。
- 正当zeroは0、acceptedは実path weight×X/Y測定値。分母は全古典trial N。成功まで再sampleしたりaccepted-onlyへ再規格化したりしない。
- 独立全path draw、全occurrence順、moment/range積、nonunitary bias、state/native誤差を同じtaskへ戻す。各microstepを測定する別taskへ変更しない。
- 十分Nの既存式は、正の残余幅s、全path V/R_max、固定α_axisと外向きlog/ceilが前提。必要最小Nではない。G7固定α、G8の推定/resource failure配分、D系列の多row familyをC1へ混在させない。
- 既存PR-2の保存signalとの差から作ったshot設計や、ordinary RPEのunit-radius/attenuation/固有stateモデルをG6認証Nに移さない。single signalにeigenstateやenergy priorを追加仮定する必要もない。
- technical failure、precision不足、未被覆native key、保存/測定failureはzeroではない。未完走prefix・完走に事後conditioningしたrunを全N confidenceとして採点しない。

## 2. 全event列挙と既存の回避部品

| SOURCE位置 | productionに必要か | 静的な確認と不足 |
|---|---|---|
| [G7] `FullReturnGenerator.sample` (237)、`upper_normalizer` (35) | 必要な局所生成/予算。全event表は不要 | raw word、非reduced reject、parent/child、coin、実weight。Uはrootとraw-reduced DPで得る。局所で全childを作る費用は残る |
| [P5] `P5Closed.__init__`, `sample` | 群normalizer/completionは必要、全event表は不要 | 理想full canonical、root/ordered-pair/first-label群。`reference_events`はsampleから呼ばない |
| [DPIPE] `collect_pending` (49) | 参照・費用取得側 | `list(generator.reference_events() ... else reference_events(generator))`を全側で取得。production budgetは先に作るが、後段費用は全表に依存 |
| 同 `freeze_inventory` (23) | 全keyの参照取得方式 | `row['events']`から全ratio集合。loopを別fileへ移しても列挙は消えない |
| 同 `evaluate` (151) | 参照native/matrix・費用取得方式 | 全eventのnative IR、価格、matrix、π付き平均・worstを求める。C1用の既存非列挙cost oracleではない |
| [G8PIPE] `NativePipeline.step` / `AcquisitionCache.get` | 既存on-demand候補 | 受理eventだけratio取得・strict RZ receiptを保持。未知key/cap/取得failureはSTOP。短traceが全support terminationや期待価格を保証しない |
| [DFBUILD] `estimate_df_rte_structural_size_upper_bound` (28) | 既存の全event列挙前guard | component specsと最大order/回数からraw instruction-sizeを上から覆う。未取得registryを必要とし、T/CX/depth/strict誤差の物理証明ではない |

### Normalizer・メモリ・bit費用

Fullは全B_fullなしにordinary envelope B_ordと認証Uを前処理し、onlineでGreen/parent/全childを求める。P5Closedはm=5の少数群式でB_full区間と群law・completionを前処理する。理想normalizer一致でもdigital proposal、root誤差、trial単位・Nは同一としない。[G7][P5][W2B]

Green前処理のO(Lm²)有理演算・O(Lm)級数保存、単純local coreのO(m³+Lm²)、P5のO(L²)群setupはそれぞれの範囲の記述。selector全走査、sort・検査、Parent.s再sum、F3 law全体hash/serialization、angle時event再query、Evidenceを省かない。rational bit長、root/angleの区間精度、値と実装objectの保存量を別計上する。O(算術回数)をCPU秒・T費用にしない。[F2][A][B]

### 全supportのstrict errorとevent価格：どこまで閉じるか

| 既存回避候補 | 閉じ得る範囲・必要条件 | 現判定 |
|---|---|---|
| 全label provider certificate＋G8既存uniform誤差式 | 各有限labelのstrict CQ/adjointとRZ誤差を保証すれば、語の全列挙なしに保守的event errorを覆う枠組み | **CONDITIONAL**。H4 certificate未取得、完成native対応未実装 |
| P5既存group/ratio式 | root、`pair_scalars(j,k)`、first-label groupでangleが決まる式がある。O(L²+L+1)群のscopeでkey coverageを検討でき、全word列挙と異なる | **CONDITIONAL**。Fullとの対応・実precision/provider/tool bindingを確認する必要。今回key生成・合成なし |
| G8 on-demand strict guard/cache | 実際に遭遇したangleのstrict certificateと取得費用をその場で保持 | **VERIFIEDな実装部品**だが全support/全N保証は**UNRESOLVED**。未観測keyを成功扱いしない |
| 既存構造価格・instruction bound | logical CQ回数、helper骨格、raw size、既知uniform単価がある範囲の保守価格に利用する条件は明確 | **CONDITIONAL**。uniform native単価/T上界が未取得。raw sizeをTへ自動換算しない |
| P5 group別native価格の取得 | 群で同じangleのRZ取得を共有できる可能性 | angle費用coverageだけではword別CQ/basis reuse、π付き期待価格、depthは閉じない。**UNRESOLVED** |
| W2cのliteral価格会計 | `Σ_e π_e C(e)`というtaskの目的量、Full対Ordinaryのchild非依存非負価格についての限定logical比較 | **VERIFIEDな条件整理**。一般child/angle/basis依存native価格・CPUへ拡張しない |

ここでは新しい費用取得算法・DP・サンプリングconfidenceを考案していない。群/構造上界だけで足りるか、期待native価格まで必要かは次のscope判断である。期待価格が必要で、既存構造式・既知保証で覆えない場合は、追加SOURCE/保証が必要。新しい非自明な解析・統計保証を採用するなら**PROOF_REQUIRED**として別承認へ渡す。未観測eventの値を補間・少数標本から無断に認証しない。

## 3. 費用ledgerとstrong baseline

| 層 | 必須計上 | UNKNOWNのまま残すもの |
|---|---|---|
| setup | snapshot/state/registry取得、正規化とmodel certificate、Green/UまたはP5群/completion、law/preflight、provider/angle key取得、hash/監査 | H4の実setup CPU/bit/bytes、native取得全量、償却回数 |
| online | 全trialのraw/coin/accepted、selector・query/root/weight/angle、hash/serialization再query、cache lookup/miss、IPC/I/O | 実分子入力のCPU・母平均・上界。旧Pilotを外挿しない |
| 全trial/量子call | N、全draw/zero/accepted/測定完了、全path選別。期待、tail、hard上限を区別 | 新N/受理率/量子call数。全pathが量子前選別可能か未接続 |
| native | deterministic部・全tail occurrence・basis/central/control/actual-adjoint/相対phase・state/readout、T/CX/1Q/depth/workspace | 未取得provider価格・prep/readout・全回路最適化・FT scheduling。TとTを含む1Qを重複加算しない |
| 保存/終端 | full/compact Evidence、原本/metadata/receipt、ログ、再監査、失敗prefix、STOP/finalization | 全N保存量、serialization・再検証費用。全child復元を監査側に隠さない |

[W2C]の全trial分母を維持する。量子前全path選別・zero後の準備遅延が可能な場合にだけ、prep/readoutをacceptedへ課す。途中まで実行してrejectするarchitectureなら、その量子prefix費用を別に戻す。二次momentの低下、accepted数の低下、総native低下は同じ事実ではない。

| 比較対象 | 既存SOURCEと再利用範囲 | C1実装前提・不足 |
|---|---|---|
| Ordinary RTE | [G7] CanonicalGenerator ordinary、DF ordinary RTE/PR | exact同一入力とfinite target、provider/measurement・誤差予算bindingが必要 |
| Partial Return | [G7] partial_return_tail | 同mのtailと元degree phaseを保持。単独低次数へtarget変更しない |
| Closed P3+tail | [G7] closed_P3_tail | root/conditional lawのcertificate、残りtailと実πを結ぶ |
| General Full | [KERNEL][G7][F3] | F1–F3、prequantum zero・全trial accumulator。未知B_fullを渡さない |
| Closed P5 | [P5] | 群normalizer/completion・positive root/child、実digital lawとbias/全path budget。対照から除外しない |
| 既存DF Ordinary PR | [S2][PRCONTRACT] | C1と同じstate/T/model/provider/error/failureへ合わせる。旧経験的shot/N・固定compiler価格を転記しない |
| deterministic symmetric PF | [S2]等 | 同じexp targetとprecision、合法なstep/order/basis融合、strict native価格が未取得。**UNKNOWN** |
| Pauli CTS/SCU | [NATIVE]の既存toy実装、前回調査の一次資料対応 | g9 provider_polynomials/exact_targetは固定3-qubit専用。分子Pauli取得・cross-term/coherent signalのbridge・precision/価格は**UNKNOWN** |

全側に合法cache/group/selector・basis reuse/cancellation・精度選択を同等に認める。普通法へ全語列挙を課さず、RAだけ高度なproviderを与えない。同一finite固定点の機構比較と、同じexponential精度で各法の正当な最適化を認めるapplication比較を分ける。未取得の強い対照を省いた一般優位は判定しない。

G9/G10の固定辞書/価格下の不利、P_U反例、W2b/W2cの条件、D系列の保証/価格依存性、16,896 Pilotの古典pipeline限定を保持する。C1入力へ旧勝敗・下界を無条件移送せず、旧結果を解除もしない。[前回固定調査][PREV]

**これらは実行前提の整理。新しい値・順位・上界の数値評価は0。公開・保全確認後STOP。**

[DFTAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_tail.py
[PRCONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/pr2_matched_accuracy_resource_contract_v1.md
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_s2.py
[DACC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/tracks/algorithm_codesign/d3_guarantee_price_preregistration_v1_20261013.md
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[BITS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[F1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/handoff.md
[G8PROOF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/tracks/algorithm_codesign/g8_proof_contract_and_on_demand_scope_20261010.md
[G8BOUNDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g8_bounds.py
[HAD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/rpe_hadamard_interrogation.py
[E2E]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/measurement_and_cost_contract.md
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[W2B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed26d1c85cf7ad80625f7abbc662b70abd690a97/docs/research/track_b_g6_w2b_certified_trial_budget_20261018/handoff.md
[W2C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/quantum_event_cost_comparison.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g9_p5.py
[KERNEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/return_aggregation.py
[DPIPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/pipeline.py
[G8PIPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g8_pipeline.py
[DFBUILD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_qiskit.py
[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/df90ed8c8e847f8ea1861d9761b9515e7a895916/docs/research/track_b_partial_sum_query_sampler_audit_20261020/README.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f823e909cb7049a612f1bf2c247f3bbe15fc184d/docs/research/track_b_coefficient_type_canonical_audit_20261020/README.md
[NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g9_native.py
[PREV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/291e21218c1d1fde062c001a1586ba7c1bb9c988/docs/research/track_b_ra_application_feasibility_20261020/README.md
