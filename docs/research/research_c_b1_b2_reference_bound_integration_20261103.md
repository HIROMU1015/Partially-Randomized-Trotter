# 研究C：固定B1/B2証拠の統合・Stage Cの残存条件

依頼識別日2026-11-03。今回の成果は技術統合であり、新H4科学計算・raw数値再解析・Stage C・重要科学レビューは0。**B2の二つのReference上界は接続済み、Stage Cは `NOT_READY_BOUND_INPUTS` を維持する。**

## 固定出典と実施範囲

| 役割 | 固定commit / 状態 |
|---|---|
| 直接起点 | `90a2a7baf07b98108111126504d202294a884af5`、[旧統合準備](research_c_b1_b2_accuracy_scope_integration_20261103.md) |
| B1結果 | `69c9a764aee66cf2ca79ab29bd3145a027385bcc`、[Whole-P0準備](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69c9a764aee66cf2ca79ab29bd3145a027385bcc/docs/research/research_c_p0_uniform_method_readiness_20261103.md)、`P0_UNIFORM_NOT_ESTABLISHED`、実科学0 |
| B1実装SOURCE | `5fc9e6a18e016bfa0725d18edd1c7fc4589a0075` |
| B2結果 | `b7a3db09e395d36965e5b9dc28ae4a299db3b2f0`、[27-call報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b7a3db09e395d36965e5b9dc28ae4a299db3b2f0/docs/research/research_c_reference_bound_execution_20261103.md)、`B2_REFERENCE_OPERATOR_BOUNDS_COMPLETE` |
| B2実装 / 科学SOURCE | `918c63cc2a959f59f2817388bf6be549b905e3e6` / `01baf2b5e400773e00a5ae548941265f85b9882b` |
| Stage A結果 / 科学SOURCE | `ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53` / `ddd8333488e07a3b3af9095b5cb193ef9f90b3c3`、[38-call報告](research_c_iid_grid_stage_a_execution_20261103.md) |

固定GitHub本文を取得し、ローカル固定blobと照合する。B1/B2の実装・出典・数学的対象の技術照合を行ったが、Arb計算を再実行したり、新たな独立証明レビューを実施したりしていない。B2の既存result/契約/SOURCE/STOPとtechnical SOURCEの5 JSONだけを元byteのまま選択収録した。大きなraw・NPZやB2の全branchは複製・mergeしていない。

正式共通指示原本 `research_project_instructions_unified_2026-10-25.md` の§8・§8.4・§10を確認した。元dirty checkoutとB1/B2のworktree/job/lock/venvは変更しない。旧統合module・契約・SOURCE・Stage Aのnull/観測値・全STOPを原本のまま保全する。新しい承認証明・grantや残call転用はない。

## 共通targetとモデル

Endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。Linear H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、number CDF8、sector36、元frame順LD2、partial ordered S2/s1/r32/K0、h=δ=.1、m0..12。CA=.00159 Ha・DF reserve=.00019875 Ha、own-DF ground oracle、状態準備除外。

保存P0はQiskit1.3/opt1/seed17/rz,sx,x,cx/approximation1/initial-zero false、MPFR160 declared phase・既存Gaussian/relative fusion・controlled full512・actual inverse・head/X/Y shell。今回変更していない。

`IID_UNIFORM_53_BIT_WORDS_COUNTERFACTUAL` は各event・coherent occurrence・measurement shot・X/Yで独立な一様53-bitを仮定する。実PCG64 joint streamのIID保証ではない。Inverseを使う箇所は同じ実現回路のactual inverseで、独立に引き直したwordのinverseで代用しない。

## B2の厳密性が覆う対象

B2 SOURCEの `c_reference_bound_contract.py` の `atom_context/atom_mean/head_products/car_tail/kernel_pair/power_tasks/_round_task/records` を静的に確認した。保存Uをexact dyadicとしてCAR密度へ入れ、保存216atom係数/pと保存angle bits、正確なhead半時間1/20・h1/10・tail identity位相を使う。

| 対象 | 定義と包含される演算 |
|---|---|
| `M_atom` | Literal saved Uから構築したCAR Z/ZZ、保存係数/確率/angle bitsの数学的event平均、32event、forward/reverse head、tail identity位相 |
| `C_math` | Exact dyadicの保存G0..8、λ/τ/identityによるK0 corrected polynomialとhead/phase |
| `R_math,m` | `(1+τ²)^(-16·2^m) C_math^(2^m)`。数学的normalizationを含む |
| `R_num,m` | 保存binary64 corrected matrixの共有NumPy反復二乗chainと、各保存binary64 attenuationの乗算 |

`B_m ≥ ||M_atom^(2^m) − R_math,m||₂`、`U_m ≥ ||R_math,m − R_num,m||₂` は異なるtriangle leg。Arb384/python-flint0.8/FLINT3.3.1で指数・積・冪・normalization/phase・差・normの丸めを包含し、outward Frobeniusからoperator2上界へ変換した保存結果である。SVDで観測した距離を上界へ改名したものではない。

任意の正規化sector36状態ψについて `|ψ†Dψ| ≤ ||D||₂` を使い、Re/Imを合わせたraw complex expectation差へ同じ上界を適用する。B2自身がこの換算を出力しており、今回新しいnormや上界を計算していない。保存groundを読み込まず、ground精度を保証しない。二つの異なる状態を比較する場合や、native head/readoutを含む信号への換算には別の状態・作用・shell条件が要る。

保存m12の引用（報告の上向き12桁表示）：

| 保存項 | 上界 |
|---|---:|
| Bridge `B_12` | 6.84027481910e-14 |
| Reference evaluation `U_12` | 2.53822642422e-11 |
| 保存された合計 | 2.54506669904e-11 |

正本は[result JSON](../../artifacts/research_c_reference_bound_contract/2026-11-03/run01/result.json)の各 `upper_exact` / `total_upper_exact`。Adapterは13rowのexact rational文字列をそのままtransportする。既存合計と二項の一致をexact arithmeticで検査するが、新しい科学上界・新しい合計を生成しない。Schema照合は数学証明自体の認証ではない。

**対象外**は、元G/係数生成時の数値誤差、物理積分/DF/model誤差、Stage A binary64 atom演算全体、保存groundのoracle距離、native全word/phase/inverse/shell、実PCG64、actual CAである。二項の丸めは各上界に包含済みなので再加算しない。

## Stage Aとの関係・誤差partition

Stage A保存値はTV約3.4277497476064313e-15、m12 categorical raw上界約8.985640298365403e-10、保存raw再生最大残差約6.021104987719889e-15。このcategorical上界は同じunitary trajectory mapをpからqへ変えたという条件付きのcoupling評価であり、そのunitary前提を新たに証明していない。

Stage Aのm12 ideal binary64 atom対CAR観測約1.0867657355169296e-10と、B2の数学的atom対数学的CAR `B_12` は比較対象が異なる。前者にはbinary64 atom構成・平均・積・冪の算術誤差が混在する。B2の `U_m` はCAR側の数学→binary64 legであり、Stage A atom側の全数値不確かさではない。数値の大小だけから矛盾や全体解決を判断しない。

| 成分 | 現在の証拠 | 会計・資格 |
|---|---|---|
| PF / corrected RTE | 保存sector36 numerical pilot | 既存accuracyのreference項、A倍を一度。B2へ混入しない |
| Whole-P0 synthesis | B1の構造・人工接続、全domain数値上界なし | nullのまま。B2のsector36 boundで埋めない |
| Categorical p→q | Stage Aのexact-grid TVに由来する条件付きraw上界 | 保存rowを再利用、再計算0。Unitary trajectory premise未資格 |
| Atom→数学CAR | B2のrigorous enclosure `B_m` | `atom_to_mathematical_CAR` の一項だけ |
| 数学CAR→binary64 reference | B2のrigorous enclosure `U_m` | `mathematical_CAR_to_binary64_reference` の一項だけ |
| PCG64 stream | IID-gridモデル外 | null、0ではない |
| DF | 保存norm+engineering guard | energy側で一回。raw信号へ加算しない |
| RPE statistics | 既存α/β・RoundEvidence | 今回新shots/energy評価0 |

将来すべての前提が揃う場合の旧式は `A_m(eps_PF+eps_RTE)+eps_synthesis+eps_categorical+B_m+U_m`。四追加項はraw単位なのでAを再乗除しない。**今回この式を実H4保存値で計算していない。** B2合計/直接診断を追加componentにしたり、既包含Arb半径・normalization・TV・PF/RTEを再加算したりすることは拒否する。

## B1との接続と不可欠な残件

1. **全word synthesis**：最終Whole-P0出力を対象とし、raw Gaussian/relative/diagonalからexact targetへの対応、translation/resynthesis/declared phase、actual inverseとshell、成功する全合法word、両mapのnorm growth・coherent反復を覆う必要がある。Event-wiseのcompile残差や有限sample最大値では代用できない。
2. **Trajectory targetの一致**：B1のexact saved-atom trajectoryとB2のliteral-U数学atom mapは現段階で同一のexact unitary mapと認定されていない。保存binary64 Uの直交残差が小さいという観測だけではCAR Z/ZZが厳密なinvolutionとはいえない。Polar/ideal Gaussianなど別のexact targetを選ぶ場合も、その定義差を別途囲い、固定契約にする必要がある。
3. **Sector→native coverage**：B2はsector36の平均reference。Full512 nativeの全入力列、control=0 identity/control=1作用、leakage、relative phase、actual inverse・X/Y shellへのbridgeは別に必要。Sector boundをfull512 operator boundへ昇格しない。有限state-action一致だけでも全operator保証にならない。
4. **反復と信号**：B2の `B_m/U_m` はすでに各2^m回の値で、さらに2^mを掛けない。Native側はfresh独立occurrenceの期待mapと、同一wordを固定反復するmapを区別する。最終raw X/Yへの変換は既存both-norm-growth/shell伝播を再利用し、無根拠にoperator誤差を同値の信号誤差と置かない。

B1は引き続き `P0_UNIFORM_NOT_ESTABLISHED`、B2の `categorical_unitary_trajectory_premise_qualified=false` を保持する。Reference-onlyのrigorousという資格と、全回路・物理系のrigorousを区別する。

## 最小adapter・Stage Cの状態

新規 [c_b1_b2_reference_bound_integration.py](../../src/trottertracks/c_b1_b2_reference_bound_integration.py) は旧 `c_b1_b2_scope_integration.integrate` / `c_iid_grid_accuracy.qualify` / `c_all_law_signal_ca.attach_evidence` を呼ぶ薄い境界層。新accuracy engine・shot式・numerical reference engineはない。旧moduleと旧designの意味は変更しない。

- B2 result/contract/科学SOURCE/実装SOURCE/STOP・Stage A/候補/model/13round/単位/coverageを照合。
- 二つのrigorous rowを元のexact値・included termsで保持。Totalと構成要素の重複、wrong SOURCE/model/round、estimateのrigorous昇格を拒否。
- Stage A categoricalの数値rowは保存するが、資格は `conditional_bound_pending_trajectory_premise`。B2のstate-independent boundを理由にこの前提をtrueへ変更しない。
- Synthesis nullやcategorical未資格がある間は、既存accuracy算術を呼ぶ前に `NOT_READY_BOUND_INPUTS`。仮の合成上界を与えた人工caseでも同様。
- Actual-law `error_evidence`、PCG64 discrepancy、actual/rigorous CAはnull。既存最大16-call Stage Cは0 calls。

## 必須条件とより強い保証の条件

| 目的 | 必要な残件 |
|---|---|
| IID-grid条件付き数値RPE十分条件 | 同じtargetに対する有効なsynthesis/bias evidence、categoricalを使うならunitary premiseまたは有効なnorm-growth置換、sector/native/shell信号coverage、既存PF/RTE/DF pilotの資格を明示、own-DF oracle/alias・独立shotをモデル条件として維持 |
| Rigorousな保存数学モデル全体 | 上記に加えPF/RTE norm/referenceの未包含算術、状態を数値vectorで代用するならそのeigenstate距離、generator生成誤差などのtarget gapを包囲。B2単独では未達 |
| Actual PCG64 / actual CA / 実資源保証 | 実joint stream/shot独立性、実装とモデルの差、必要な物理/DF/state-preparation/noise等の資格。IID-grid仮定の今回とは別scope |

Ground precisionについてB2の二項は任意正規化ψへ成り立つので、この二項の換算だけに保存groundの認証を足す必要はない。一方、保存vectorを真のown-DF eigenstate信号として用いる新校正では状態gapを考慮するか、明示した数値pilotに限定する。Calibration confidenceは確率的案を選んだ場合だけ必要で、決定論的B2包囲にsampling confidenceを付加しない。

## B2完了後の次期scope比較・非実行設計

[旧scope比較](research_c_accuracy_validation_scope_options_20261103.md)の三案を、[追補設計](research_c_b1_b2_reference_bound_design_unapproved_20261103.json)で更新した。旧文書・旧契約は上書きしない。

| 案 | 母集団/対象 | B2再利用と不足 | 新科学計算・技術差分 | 証拠資格・負担 |
|---|---|---|---|---|
| 全domain一様 | 全216 atom・長さ32の合法word、最終Whole-P0/full512/inverse/shell | mean-referenceのB/Uは再利用可。Per-wordのnative target gap・ZSXX/phase一様誤差は覆わない | B1の一様演算/成功coverage/target一致を完成後に固定。現状call/entry未確定 | 成立すればlaw-wide raw上界への十分経路。B2だけでは実行可能にならず、最も大きい技術不足 |
| 固定有限pilot | 事前固定word集合Fと同一候補のraw/compiled/reference/inverse/X/Y | B2はmean-referenceのみ。各wordのreference評価誤差を別途扱う | 既存NativeBuilder/StrongBuilder/full512またはColumnStream/固定CARを再利用。Family・precision・全map coverageを先に固定 | 評価器と位相/target対応を検証する数値pilot。全law保証なし。F/精度未決なのでcalls null |
| Fresh IID-grid確率的 | qからのfresh独立wordのkernel誤差平均/tail、または各roundの期待raw bias | Mean-reference B/Uをそのtargetに合う場合だけ再利用。Sample evaluatorの誤差・同一map・変数範囲は別途必要 | Fresh IIDモデル、固定N/confidence・estimand・反復/shot独立性・per-sample上側量を設計。旧native32をpoolしない | 校正event条件付きの期待bias evidence。未観測tailやfinite最大値を保証へしない。直接round校正は概ねN_m·2^m kernelで高負荷 |

有限pilotの既存logical instrumentationは各wordにつきbuild/compile/reference/raw/compiled/actual-adjoint-pair/X-Yの7種に共通setupを加える案。Streamingなら異なるfull512 map Gに512G列を課金し、dense法と重複させない。Per-word reference enclosureの必要単位は選択した評価器に依存する。**F/N/precision/δcal未採択なので新上限を数字で創作しない。** 旧27/38/16-callは新予算へ転用しない。

推奨は引き続き **有限pilotで評価器とnative/reference target対応を検証し、その後fresh IID-gridの期待bias校正を検討**。B2によりmean-referenceの二項を再計算する必要はなくなったが、per-word evaluator/full512 synthesis/trajectory premiseの作業は残る。Kernel差d≤2の仮定も両mapのexact unitarityが先であり、literal-U mapに無条件適用しない。

直接native→保存CARの総biasを新estimandに選ぶ場合、その比較にはcategorical/bridge/referenceが既に含まれ得る。旧四成分gateへその総biasとB/Uを同時に入れず、包含項を明記した別qualificationを固定する必要がある。B2を使ったからといって誤差分割を自動的に変えない。

確率的校正のconfidence δcalとRPE測定failure α=.01（各軸.01/26）は別。条件付き主張か、α+δcal等の新しいjoint failure政策かを研究C/ユーザーが採択する。旧αを維持して校正失敗を隠さない。今回新保証水準・N・confidence・正式研究方針は採択していない。

## 検証・公開・引渡し

[人工/回帰結果](../../artifacts/research_c_b1_b2_reference_bound_integration/2026-11-03/technical/tests.json)、[passive metadata](../../artifacts/research_c_b1_b2_reference_bound_integration/2026-11-03/technical/passive_preflight.json)、[静的・保全確認](../../artifacts/research_c_b1_b2_reference_bound_integration/2026-11-03/technical/static_verification.json)、[技術SOURCE](../../artifacts/research_c_b1_b2_reference_bound_integration/2026-11-03/technical/SOURCE.json)、[STOP](../../artifacts/research_c_b1_b2_reference_bound_integration/2026-11-03/technical/STOP.json)を保存する。

実H4の科学処理・NPZ decode・新sampling・native build/compile・Stage C・shots/energy・重要レビューは0。固定JSONの読取りとschema/出典/byte検査だけを行った。人工・回帰は329 unique PASS、既知の旧2件を従来と同じ理由でdeselectした。新adapter45件＋既存284件であり、B2科学engineの再実行・独立再証明ではない。人工テストは発明したrational値・既存人工回路に限る。共有venv/cache/OS/他jobを変更せず、人工テストはtask-local cache、CPU/RAM/worker/wallの人工上限なし。新規scientific claimsの採択ではないためEvidence Map/研究概要/旧manifestは書き換えない。

**技術統合GO、Stage C科学実行NO-GO（未資格入力あり）。** 非実行の未承認追補設計を公開する。次担当はユーザー／研究C：有限pilot/確率的estimand・targetと保証水準・confidence政策を選ぶ。Codexはその選択後に既存pipelineで最小のfamily/SOURCE/call上限を閉じ、新科学実行の別途明示許可を受ける。今回の公開照合後はmandatory STOPし、自動実行しない。
