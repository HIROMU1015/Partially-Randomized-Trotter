# 研究C：固定4-word native/reference pilotの具体化

2026-11-04。ユーザーが固定有限pilotを採択し、word集合・評価方法・精度・call上限の具体化をCodexへ委ねたことを受けた技術準備。起点は `891aa78f47e66184eaf8a6d527ca8773ba6bbdd0` の[B1/B2統合](research_c_b1_b2_reference_bound_integration_20261103.md)。**実H4科学実行はまだ開始していない。**

## 固定する問いと資格水準

同じ保存PR endpoint、4つの事前固定wordで、complete controlled native kernelが独立CAR referenceと数値的に対応し、P0・actual inverse・head/X/Y shellの評価器が働くかを確認する。結果は固定有限集合のbinary64 numerical pilotであり、全law/母平均/rigorous enclosure/actual CAの保証ではない。B2包囲の再計算や精度評価engineの再実装は行わない。

対象：Linear H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、number CDF8、sector36/system256/controlled512。Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、元順LD2・partial ordered S2/s1/r32/K0・h=.1。元g補正κ・scalar一回・signed atom/identity位相を保持する。P0はQiskit1.3/opt1/seed17/rz,sx,x,cx/approximation1/initial-zero false、既存MPFR160 declared phase/Gaussian・relative fusion。Own-DF ground oracle・状態準備除外という研究条件を維持するが、pilotのX/Y検査は保存数値vectorを使い、true eigenstateとの距離は認定しない。

CA=.00159 Ha・DF reserve CA/8、m0..12は親研究の背景条件として保存する。今回評価するのは**1 h-stepのkernelだけ**。13roundのPF/RTE/合成伝播・shots・energy・総費用は対象外であり、4word PASSからStage C READYにしない。

## 固定word集合

Stage A `atoms.json` のframe/support/atom IDだけから選ぶ。係数・費用・過去の残差による最適化や候補順位付けはない。各frameで指定support長のatom IDを辞書順に並べ、その先頭を使う。4wordは各32eventで、RNG・fresh sampling・旧native32の再標本化を使わない。

| ID | 32-event列 | 確認する構造 |
|---|---|---|
| same_atom | F2の最初のZを32回 | 同一atomの繰返しと同basis共有 |
| same_basis_Z_ZZ | F2の最初のZ/ZZを16回交互 | 同basis内の異なるsupport |
| alternating_frames | F2最初のZ/F3最初のZZを16回交互 | 異なる宣言frameのbasis切替 |
| six_frame_cycle | F2..F7の最初のZZを順番に巡回し先頭32event | 6frameを含むorder・basisの接続 |

具体的なatom ID列は[固定契約](research_c_fixed_word_pilot_contract_unapproved_20261104.json)に保存する。「frameが異なる」と「basis_keyが異なる」を同一視せず、既存NativeBuilderのexact basis_keyによる合法共有をそのまま使う。全word・全support・全符号を網羅する設計ではない。

## 既存pipelineの再利用と最小差分

`c_pr_integrated_performance.native` のfull256 CAR、独立signed atom/head cache、StrongBuilder、P0、SharedQiskitCalls、full512 raw/compiled/actual-adjoint/X/Y、QPYとmetricsを再利用する。新しいPF/RTE/native compiler/誤差engineは作らない。

既存関数にopt-in `fixed_words` を追加し、その場合だけ4つの決定済みevent列を使う。既定のnative32 sampling・seed・mean/SE・旧runnerの動作は維持する。有限modeは価格calibrationを呼ばず、母平均・全law supportをnullに保つ。MetadataにもIID sampleではないことを明記する。

新しい `c_fixed_word_pilot.py` は保存endpoint/sector/reference/groundの束縛と、既存K0 event constructorによるevent化、4word dispatch、fresh/Calls/partial STOPを担当する。Stage A/B1/B2のadapter・契約・SOURCE・raw・STOPは変更しない。旧performance実装の科学SOURCE commitは保全し、新しい差分はこの独立branchのSOURCEで固定する。

## Reference・誤差の扱い

1. 元input h/g/scalar、保存J/Uと同じJ/Uのsector CAR exportを照合する。保存DF予算PASSは同一候補の既存判定を引用し、新DF normは計算しない。
2. Full256 CARの9因子を構築し、保存sectorへの制限と比較する。各固定wordのsystem256 referenceは独立CAR signed atom/event作用、forward/reverse head、extracted identity位相から構成する。個々のGaussianをsectorへ射影して全回路検証に代用しない。
3. Complete controlled512に対しraw/reference、raw unitarity、P0/reference、P0/raw、P0 unitarityを比較する。位相fitはしない。
4. Raw/compiledそれぞれのactual inverseを実際の回路操作で作り、reference†と復帰をfull512で確認する。X/Yは保存数値groundと両shellを使ってcomplex signalを比較する。
5. 誤差はbinary64観測残差とFrobenius値。既存component allowance1e-10、X/Y complex allowance2e-10、rtol0を変更しない。数値残差を厳密operator norm上界や全law synthesis boundへ昇格しない。

このnumeric full-Fock referenceとB2のliteral saved-Uを使ったArb数学的mapは、同じ保存入力・符号・frame・head orderという構造を共有するが、今回の計算だけで両者の数値誤差を包囲しない。B2の二項はsector36 **平均reference** の対象限定上界であり、各wordのfull512 numerical referenceにそのまま加えない。保存Uのexact unitarity、generator構築誤差、reference算術、保存stateのoracle距離は未資格のまま。固定4wordで数値的一致が得られてもcategorical unitary前提をtrueにしない。

## 実行順序・call上限

| 区分 | 上限 |
|---|---:|
| Evidence/input/candidate/sector/basis/ground load、binding、native model、ground gate、law/config/gate | 11 |
| K0 event catalog + 固定4word束縛 | 2 |
| 共有native setup（CAR1+因子9+sector1+cache1+law1+bases9+relative4+shell load6+shell gate3） | 35 |
| 各word9種×4（event materialize/build/reference/raw gate/compile/compiled gate/actual inverse pair/X-Y/metrics） | 36 |
| 有限結果summary + 全体summary | 2 |
| **合計** | **86 logical units** |

Actual inverse pairの1単位にはraw/compiled両方の作成・decode・復帰検査が含まれる。科学callはprimitive行列積の回数ではなく、既存instrumentationに従う。新QPYはraw/compiled/raw actual inverse/compiled actual inverseの4種類×4word=16。候補非依存の保存head/X/Y shell6 QPYを構造確認して再利用する。Full256 reference4 NPZ、各検査・metric・SOURCE/契約・journal・STOPを保存する。

Fresh output：`artifacts/research_c_fixed_word_pilot/2026-11-04/run01`。SOURCE/input/runtime/family不一致、signed atom/位相/順序/compile/reference/adjoint/X-Y correctness FAILで新callを止め、開始済みthreadを回収してpartial evidenceを保全する。Retry/resume0、結果を見たword追加・置換・精度変更・fallback0。完了/FAILどちらでも公開照合後mandatory STOP。

## 人工検証と資源

[人工・回帰結果](../../artifacts/research_c_fixed_word_pilot/2026-11-04/technical/tests.json)：85 unique PASS、除外0。発明したJ/U/Hamiltonianに対して4wordをreal native/CAR/P0/full512でend-to-end検査し、86-call上限内/16QPY/fresh拒否/STOPを確認する。人工fixtureでbasis共有によりrelative変換が省略される場合、未消費callを無理に追加しない。既存native32のdispatch・失敗、phase/Gaussian向き/order/compile不一致、B1/B2未資格を保持する検査を再利用する。実H4のNPZ/QPY decodeは人工テストに含めない。

本pilotは4つの独立wordを既存resource_planで並列化する。Build/P0/actual-adjoint作成は共有Qiskit状態のため既存mutexを保持し、reference/Operator/検査は並列。各workerのBLAS1はoversubscription防止のprocess-local設定。CPU/RAM/worker/wallの人工capは設けず、実際の空き資源・quota・pressureを実行直前に再確認する。共有venv/OS/global cache/他jobは変更せずtask-local cache/tmpだけを使う。全512列を覆う既存dense Operatorを一度評価し、同じものをColumnStreamで重複評価しない。

## 実行状態と次の担当

[技術SOURCE](../../artifacts/research_c_fixed_word_pilot/2026-11-04/technical/SOURCE.json)、[passive preflight](../../artifacts/research_c_fixed_word_pilot/2026-11-04/technical/passive_preflight.json)、[静的確認・保全](../../artifacts/research_c_fixed_word_pilot/2026-11-04/technical/static_verification.json)、[技術STOP](../../artifacts/research_c_fixed_word_pilot/2026-11-04/technical/STOP.json)を保存する。今回の具体化では新実H4科学0、B2/Stage A再実行0、Stage C/新shots/総費用/重要科学レビュー0。

前段の非実行設計は、family/target/precision/SOURCE/callsを固定してから別途許可する条件を保持している。今回この固定86-call契約をreview可能な状態まで具体化した。**次はユーザーが固定契約の科学実行を判断し、許可後にCodexがfresh run01を一度実行する。** 有限pilotがPASSしても、fresh IID-grid校正のestimand/N/confidence/failure政策は別途固定・判断が必要。今回の公開照合後はmandatory STOP。
