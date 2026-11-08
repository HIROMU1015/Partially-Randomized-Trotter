# 研究C：ENTRY E2＋線形Z再配置＋PR/r16の全law精度保証・統合技術準備

2026-11-08 JST。**CASE B：既存P0の全word一様保証方法は未成立。科学実行前でmandatory STOP。**
候補・SOURCEの静的binding、後続保証成果との照合、最小metadata adapterと人工回帰を完了した。
新しい実H4数値評価、保存科学値の再解析・再集計、reference bridge、回路build/compile/simulation、
追加価格標本、RPE/総資源計算、停止runのretry/resume、重要GPT科学レビューはすべて0。

保存された条件付き総RZ **75,621,395,471** を、実装lawの精度保証付き価格へ昇格できる証拠はまだない。
次の決定的作業は、whole-P0の全到達領域に適用できる誤差・成功条件と、同じtrajectory targetへの対応を成立させること。
Referenceだけの新計算ではこの不足は閉じない。実行可能な全law契約ができたとは表示しない。

## 1. 今回固定した対象と証拠の水準

直鎖H4・隣接1.00 Å/STO-3G、number CDF rank8、元frame順LD2（F0/F1が決定論prefix）、
Nα=Nβ=2/sector36、system256/control込み512。ENTRY E2はC0開始、strength=.03、64更新の保存J/U。
線形Zを既存の元g補正κへ移した同じfull Hamiltonianを使う。Ordered partial S2/s1、δ=h=.1、r16/K0、m0..12。
Own-DF ground oracle、状態準備費用除外。CA=.00159 Ha、DF reserve=CA/8、βtotal=π/6、βstat=π/12、
failure .01、各X/Y軸 .01/26。P0/Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation_degree1、
qubits_initially_zero=false、MPFR160 declared phase、global phase fitなし。Actual raw/compiled inverse、head/X/Y shell、
fenced serialの既存政策を維持する。Kernel全体へのP0の内部にはevent fenceがない。

| identity | 保存値・出典 |
|---|---|
| 候補ID | `H4_100_C0_ENTRY_g003_budget64`。g003はstrength=.03の既存表記 |
| J/U NPZ SHA256 | `0e1119dfb6be9b75fb5e592b7a65c76cda61e348fd7cad65d7f104bf997929f5` |
| J array SHA256 | `69dac3b923847fc85cd403fa38ddf42db31385ca3bbcc973d3646f0567fc6d2d` |
| U array SHA256 | `dfbdd050719f96084d3c5cb3b5c1b9bdd86bef352246fe95ea306aa3934c1310` |
| J/U shape・dtype | どちらも(8,4,4)、`<f8`。Array hashは保存metadataを引用し、新decodeなし |
| 元κ export NPZ SHA256 | `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| 元nuclear scalar array SHA256 | `874b34c93c04576c14050d91ecace903d7b2693560bf77a05b2178fc48fb232b` |
| 元物理h/g/constant NPZ SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 再配置version | `tail_linear_Z_to_existing_kappa_keep_frame_identity_v1:LD2` |
| 科学実装commit | `cfc64b70861a6482dddd042da8a110abebd6ad7a` |
| code SOURCE SHA256 | `6bd88bc86bd1da2876f87b8ae67be509b15727f01517cea9c7219f91c350c4cc` |
| sampling SOURCE SHA256 | `bb102f16c3bf757853f9e9fc98d811dc230592d71cd6bad57bcce0363fa2046e` |
| final r16 summary SHA256 | `97b48bbde0128b244c8ff353bc57758dfe4772c8f8c803a1bcf167fcb424a754` |

[固定契約](research_c_h4_handoff_contract_20261106.json)と
[final64 summary](../../artifacts/research_c_h4_handoff/2026-11-06/science/run01/E2/final_r16/summary.json)の
candidateは上記JU hash＋再配置version＋LD2、r16、13round。`bindings.evidence=null`を維持する。
保存PF/RTE・DF・ground・full512 correctnessは数値pilotの資格であり、厳密包囲・全law証拠ではない。
元契約のphysical record中のtail λは**再配置前**であり、再配置後のlawへ流用しない。

再配置の定義は既存`c_h4_relocated_downstream.prepare/raw_generators`のまま。
Tail t=2..7についてs_t=J_tの行和、Δκ=ΣU_t diag(s_t)U_tᵀ、scalar shift=−Σs_t。
Frame内の元identity/ZZ、prefix、元J/Uを保持し、tailのZ rotationだけを除く。
再配置後のκ/scalar/full Hamiltonianに独立した新array hashを今回生成していない。
それらのidentityは入力hash＋固定コード＋versionによる**来歴binding**であり、新しい数値等価性の証明ではない。

### 公開入力とraw保存の区別

[readiness.json](../../artifacts/research_c_entry_e2_all_law/2026-11-08/technical/readiness.json)は27保存ファイルを固定し、
code SOURCEの64ファイルを含む計91の一意pathをbyte照合した。元16入力も一致した。
科学sourceの変更や元runへの書込みはない。

元E2 reference NPZ（hash `a76f9847d1a3a04ce45b90b24e0b79c01fd93dcf9da3f1ca7b966bd994d8d8ef`）は
**再配置前**のJ/U・generators/H。再配置後のmean kernel保存物として読んではならない。
[raw hash索引](../../artifacts/research_c_h4_handoff/2026-11-06/publication/result_files.json.gz)には次がある。

| raw相対path | SHA256 | 今回の状態 |
|---|---|---|
| `E2/A/ground_candidate.npz` | `6843e56b955a131583950c585287a41f61c08321b48618c1525b83d2361471b4` | hash確認のみ。raw本体を新取得・decodeしていない |
| `E2/selection/r16/reference/law.json` | `91f5f12ce3504d9d7130919f626bacdbc3ad75b16667e611e4616559b1b61019` | hash確認のみ |
| `E2/selection/r16/reference/normalization.json` | `971d6e0da0079eec864a080899d3ca91b55bbaa4d013a78bbf768c809a1000e3` | hash確認のみ |
| `E2/selection/r16/reference/m12.json` | `c9567e18461abdc6a888edcc90862c02ebf9b04bc0279e864ac19fe2a21f0bfb` | hash確認のみ。ほかのround値は保存summaryに存在 |

公開bundleにはraw NPZ/QPY本体がない。さらに`c_integrated_retuning_adapter.reference_r`はcorrected/PF kernelを
メモリ上で計算・cacheするが、行列自体をNPZ exportしていない。保存scalarからその行列を復元した扱いにはできない。
別cacheにあるならbyte inventoryと生成来歴を先に固定し、なければ**新科学計算**が必要。
これをinput不一致CASE Dとは呼ばない。保存metadataに矛盾は見つからず、数値保証の入力が未取得なのである。

## 2. 確認した固定資料と、既に実行済みの保証成果

GitHub branch refで旧readinessのtipを確認し、ローカルの固定Git objectから本文・実APIを読んだ。
各資料とmoduleのcommit/path/SHA256はreadiness.jsonの`source_map`に保存した。
外部の新理論検索や重要科学レビューを実施したという主張ではない。

| 固定資料 | 成立済みのこと | E2/r16へ移せないこと |
|---|---|---|
| [ENTRY下流結果・3015d4fb](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/docs/research/research_c_h4_relocated_downstream_results_20261106.md) | 今回候補の保存PF/RTE、final64 native価格・各sampleのfull512/実逆/X-Y、SOURCE/STOP | 有限sample correctnessを全word誤差上界にしない |
| [CDF0比較・cbc517e7](research_c_cdf0_z_relocation_rerun02_results_20261108.md) | 固定CDF0/r64対E2/r16の条件付き価格比較。正式判定は`UNRESOLVED_PRECISION_TARGET` | 統計精度未達と全law未資格は別問題。追加標本は今回0 |
| [全law準備・3a784888](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3a784888a188db946681a6143c5e18479ddf284a/docs/research/research_c_all_law_signal_ca_readiness_20261103.md) | 誤差分割・scope・13round接続schema | Schemaは証明ではなく、旧candidate/r32固定entryも今回互換ではない |
| [IID Stage A・ae9a4e13](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53/docs/research/research_c_iid_grid_stage_a_execution_20261103.md) | **38-call実行済み**。正確な53-bit grid categorical解析、own-ground bridgeの13round数値pilot | 古い「未実行」記載から再実行しない。旧候補・再配置前r32の値は今回のboundではない |
| [P0方法・3eccab64収録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eccab64a2672813e479c113c2485219f8479885/docs/research/research_c_p0_uniform_method_readiness_20261103.md) | Grammar/rewrite上限、全domain条件付きcomposer、人工hybrid診断 | 全到達domainの数値定数・全word正常終了・native target gapは未成立 |
| [B2/Arb384実行済み](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eccab64a2672813e479c113c2485219f8479885/docs/research/research_c_reference_bound_execution_20261103.md) | **27-call完了**、sector36任意正規化状態に対するbridgeとreference uncertainty各13rowの二項包囲 | Native full512、一様synthesis、literal-Uのunitary trajectory premise、ground精度、PCG64を保証しない |
| [B1/B2統合追補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eccab64a2672813e479c113c2485219f8479885/docs/research/research_c_b1_b2_reference_bound_integration_20261103.md) | 二項の二重計上防止と、categoricalのunitary premise未資格を検出 | B2完了後もStage Cは`NOT_READY_BOUND_INPUTS`。この後続結論を維持する |
| [固定word pilot](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eccab64a2672813e479c113c2485219f8479885/docs/research/research_c_fixed_word_pilot_execution_20261104.md) | **86-call完了**、4固定wordのfull512/P0/実逆/X-Y診断 | Finite-only。同じ4wordを再提案しても一様保証の不足は閉じない |
| [最終総括とEvidence Index](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eccab64a2672813e479c113c2485219f8479885/docs/research/research_c_final_synthesis_20261107/REPORT.md) | DOC01/08/15、POST12/32/08、DOC03、E23/F2/F3の後続資格を区別 | Arb192の固定回路/別target保証を今回の全wordへ転用しない |

同じH4/1Å/STO-3G・CDF8/元順LD2・再配置・s1/h.1/K0/m0..12・own-ground/準備除外/P0の保存比較では、
CDF0/r64の条件付き総RZは231,951,425,282.5、E2/r16は75,621,395,471、保存点差156,330,029,811.5。
ENTRYの点値は67.40%低いが、登録した5%精度に対して半幅/ENTRYは6.09945%で未達。
この引用値を再集計せず、`UNRESOLVED_PRECISION_TARGET`を維持する。統計標本の追加と全law保証の取得は別作業。

上記旧all-law/IID/B2/fixed-word群の候補は`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、
PR-aware別endpoint・Z再配置前r32。B2科学SOURCE `01baf2b5e400773e00a5ae548941265f85b9882b`、
実装`918c63cc2a959f59f2817388bf6be549b905e3e6`、結果`b7a3db09e395d36965e5b9dc28ae4a299db3b2f0`。
Arb384という精度や同じH4という理由では、候補・target・normalization・rの違いを消せない。

## 3. 再利用・不足のgap matrixと実API

| 要素 | 分類 | 使える部分と不足 |
|---|---|---|
| 現E2の保存PF/RTE/DFと条件付き価格 | そのまま再利用可能な保存pilot | 既存の資格と単位を維持。新上界・新shotsを計算しない |
| Raw/reference correspondence | コード再利用＋候補固有入力未取得 | 再配置helperと同じJ/U/κ/scalarでatom/head/identityを固定。旧216 atom監査の固定guardを外すだけで資格が成立するわけではない |
| Reference uncertainty | コード再利用＋候補固有計算が必要 | B2の二つのArb chain/外向きFrobeniusは適合する構成法。E2 relocated/r16の新入力とtargetを要する |
| Finite-bit categorical | 最小routing拡張＋新E2 table | Exact binary64 normalization/cumsum/normalization/searchsorted-rightと53-bit格子の算術を再利用。p/qのtable・atom順・角度bitを今回候補で固定する必要 |
| Actual stream | 根拠未確立 | PCG64 event/occurrence/shot/X-Yのjoint lawと独立性は未資格。IID-gridのTVでは閉じない |
| Whole-kernel P0 | 一様保証の数学的・実装的根拠が未確立 | 下記§4。現最大blocker |
| Actual inverse/phase/shell | 同じSOURCEのcoverageが必要 | 既存builder・検査・伝播は利用可能。有限sample PASSだけでは全wordへのuniform boundなし |
| 13round accuracy | schema/engine再利用、bound入力不足 | `RoundEvidence/assess_round/assess_energy`、全roundとalias/ground/DF/shot独立性。今回評価0 |
| 条件付き75,621,395,471 RZへの接続 | 既存契約との整合が必要 | 精度が得られても既存shotsが十分か検証が必要。law変更時の価格transferも別条件 |
| 新DF最適化、他分子、追加価格統計 | 今回不要 | 一様保証の不足を解消しない |

実APIを確認した結果、次の再利用単位を分離した。

- `c_all_law_signal_ca.attach_evidence`はidentity/5誤差項/13row/premiseを検査するが、
  `categorical_preview`とfixed/preflight/runは旧r32条件。E2へ旧entryを実行しない。
- `c_iid_grid_bridge.prepare_law/kernel_bridge/identity_gate`はr32/216等を固定。
  `heads`（h/2=1/20）、power chain、event-meanの代数は再利用可能。
  `c_pr_aware_df_c2_bridge.atom_audit`にも216固定guardがあり、名称だけでgenericとは扱わない。
- `c_iid_grid_accuracy.qualify/assess`と後続B1/B2統合はIIDとactualを分離する。
  後続のunitary premise gateを省く弱い接続を作らない。
- `c1_h4_full_synthesis.segment/propagate`は直接compiled/reference距離、両側norm growth、head/X/Y、
  n=s2^mを扱う。`c1_subspace_synthesis_bound.ColumnStream/repeat_bound`と
  `c1_h6_synthesis_bound.circuit_identity/column_batch`は全列・位相・入力scopeを保つ。
  ただしrepresented-matrix数値差のboundと、丸めを囲った保証と、全word coverageは別々。
- `c_pr_aware_df_stage_c2`等の古いscopeではs4/h1/40/r16や別候補を使う。
  そのrunner契約・承認は使わず、既存error/accuracy/helperの数式だけ適合範囲で使う。

今回base cbc517e7には後続all-law専用module群が含まれない。調査した別branchの旧candidate専用runner群を
丸ごとcopyして新しいscientific entryを作る必要はない。固定commitで参照を保ち、既存byte checkerとqualificationの拒否規則を再利用した。

## 4. Whole-kernel P0：成立済みの構造と決定的な不足

### 合法wordと変換範囲

再配置tailはF2..F7のsigned ZZのみ。各8 spin modeのpairには最大28 supportがあるので、
構文上は最大6×28=168 atom位置。**実際の非zero catalog数/確率は今回再生成していない**。
K0ではTaylor order0、product wordは空、1 eventは1 signed atom rotation、角度は符号付きatan(τ)、
τは既存r16 normalizationの値。完全kernelは決定論的両half-head＋16 event＋scalar/extracted identityを含む。
Frame選択・basis・符号列に依存するword全体をP0へ渡す。

入力grammarは有限で、同一basis sharing/relative Gaussian等の既存構造を使える。
しかし、有限primitive catalogは最終compiled circuitが小さな閉じたfamilyになる証明ではない。
旧r32のnode数107522や旧r16/s4の数を今回のNに流用しない。現時点でその新数値を計算する必要もない。

| P0箇所 | 今回に関係する事実と未保証事項 |
|---|---|
| Unroll/translation | Official definition、basis+u、空EquivalenceLibrary、通常native grammarという旧前提を使う。Definition/parameter/phaseの数値誤差は別項 |
| 1Q run | RZ/SX/Xを逐次行列積しRust ZSXXへ。Event境界を越えるfusionがあるため、単一event上界の和だけでは最終P0を覆わない |
| CX/inverse cancellation | CX削除後に次iterationで長い1Q runができ得る。Initのparameter-free inverse cancellationを任意RZの許容誤差消去と混同しない |
| Rewrite/termination | Accepted rewriteの構造上限と、全wordでguard内正常終了する証明は別。DoWhile guard1000は全word成功の根拠にならない |
| Euler/phase | Determinant4 cosetと長さ依存near-unitary tube全体、pole、arg cut、mod端点、θ≈0/π/2/π、分岐を覆う必要。MPFR markerは内部Rustの誤差を消さない |
| Controlled map | Control1だけのscalar/global phaseはrelative phase。Phase fitで除けない。実inverseの構築/位相も同じ全word範囲が必要 |
| Shellと反復 | Deterministic half-headはkernel内。伝播APIのshells['head']はancilla H準備。X/Y readoutとn=2^mを一度ずつ加える |
| Full512/leakage | 既存sampleはfull512だが全wordではない。Sector72列への縮小にはreference不変性と全出力leakageの別根拠が必要 |

既存`c_p0_uniform_method`→`c_pr_aware_df_zsxx_uniform.uniform_whole_bound`の条件付き式は存在する：

`eps_P0 <= eps_translation + 6 N p/(1-N p) + N zeta(D_N) + eps_phase`、`N p < 1`。

未取得なのは、全到達domainでのnode評価誤差に基づくp、全branchを覆うphase込みζ、translation/phase累積、
全word成功条件、raw Gaussian/relative/diagonalから同じatom targetまでの誤差、実inverse/shellの包絡。
Rejected candidate計算の誤差は出力を変えないのでaccepted rewriteの誤差へ追加しない。
直接G→Vを評価する経路と、product＋ζ経路を同時に加算しない。

Hybridは構造的一様法の根拠を人工/有限familyで検査する経路として再利用できる。
有限最大残差、captured Rustの有限box、ULP観測、Arb固定回路certificateはその不足した全domain定数を与えない。
Complete-columnは**指定回路の全入力**を覆い、wordの未観測領域を覆うものではない。
Norm-growth伝播も入力boundのscopeを拡張しない。

したがって新H4 probeを増やして小さな一様上界を仮定する実行計画は採択しない。
Eventwise/fenced compilerや別backendを将来比較するなら新政策・新価格となり、保存756億RZを対応価格として使えない。
今回P0政策は変更していない。

## 5. IID-grid・reference・actual streamの誤差会計

各mで既存のraw complex signal（無次元）へ揃える：

`delta_m = A_m (eps_PF,m + eps_corrected_RTE,m)`
`          + eps_synthesis,m + eps_categorical,m + eps_bridge,m + eps_reference,m [+ eps_stream,m]`。

追加5項は既にraw単位。A_m、2^m、rを再び掛けない。
DF誤差はHaで最後のenergy評価へ一度、RPE測定の統計失敗確率は別枠。
`eps_bridge`と`eps_reference`へPF/RTE/DF/RPE統計を混ぜない。
Roundごとに評価済みの二項をkernel boundとして再伝播しない。

### B2の二本の脚と未解決target

既存B2は、literal binary64 Uをexact dyadicとして構成したCAR Z/ZZ atomのmean `M_atom`、
数学的saved-generator reference `R_math,m`、保存数値reference `R_num,m`について、
`B_m >= || M_atom^(2^m) - R_math,m ||_2`、
`U_m >= || R_math,m - R_num,m ||_2`を別々に包囲した。
Arb演算のradiusは各脚に含まれ、別の第三項として足さない。
元結果のnormalization exponentは旧r32に対応するため、そのままr16へ置換引用できない。
E2の新B/U数値は両方null。

**Literal Uの近似直交性はexact involution/unitarityではない。**
`cos(theta) I - i sin(theta) P`というliteral-atom event mapのexact unitary premiseはB2でも未資格だった。
Nativeのexact gate-mapと同じtargetへ結ぶか、有効なnorm-growth付き比較に置き換える根拠が必要。
旧Arb192 F2/F3のpolar/exact-H targetを今回自動採択しない。
これはreferenceの計算精度を上げれば消える問題ではない。

### Categoricalとrandom stream

IID 53-bitの独立格子入力で、同じunitary trajectory mapを共有できるなら、event TVから
`min(2, 2 r 2^m TV)`というraw coupling boundを使う経路がある。
Unitary premiseが未成立なら、この式をliteral-U mapへ無条件に使わない。
Exact probability p、binary64 choiceで実現するgrid q、atom順・符号・角度bits、normalizationを今回候補で固定する必要がある。
旧r32の小さなTVを新E2へ流用していない。

IID-gridは反実モデルであり、NumPy PCG64の実joint streamを保証しない。
Master seed/hashによるnamespace分離はevent/occurrence/shot/X-Yの独立性の証明ではない。
Coherent occurrenceごとのfresh wordならmean kernelのpowerを使えるが、同じwordの繰返しは一般に
`E[U^n] != (E[U])^n`。一方、forwardとその実inverseは**同じ実現word**に結び付く必要があり、独立に再抽選しない。
既存価格標本はclassical calibrationであり、実RPE shotのjoint-law実証ではない。

## 6. 13round packet・75,621,395,471 RZへ接続する条件

機械可読の`bound_request`は5成分ともnull、10のpremiseとも未成立のまま。
Actual向けにはstreamも必要で、IIDモデルではstream項を0としてactual laneに渡すことを拒否する。
各componentには同じcandidate/relocation/physical/κ/scalar/SOURCE/compiler/ground/inverse/shell identity、
raw complex単位、誤差包含項、coverage、13row（m0..12）、資格・provenanceが必要。

| slot | E2/r16現在値 | 必要なcoverage |
|---|---|---|
| synthesis | 全13row null | 全合法word、全coherent occurrence、最終P0、実inverse、phase、head/X/Y、full512/leakage |
| categorical | 全13row null | 同じtrajectory map・p/q・全event/occurrence。Unitaryまたは妥当なnorm-growth条件 |
| reference_bridge | 全13row null | 今回literal-atom meanと今回数学reference。State-independent sector boundか明示したstate scope |
| reference_uncertainty | 全13row null | 上記数学referenceと今回数値referenceの未包含分 |
| stream_model | 全13row null | Actual向けjoint law・occurrence/shot/X-Y条件。IID反実モデルの外側 |

これらを満たした後、既存C1 `RoundEvidence/assess_round/assess_energy`と後続IID/actual分離adapterへ接続する。
Alias branch、own-groundモデル、独立shot条件、DF/数値reference/groundの保証水準も明示する。
Saved numerical vectorを真のgroundとして使う新保証ならそのgapを扱うか、numerical pilotに限定する。
B2のstate-independent二項だけの換算にground認証を重ねる必要はない。

新精度条件が成立しても、保存のideal laneのshots/Wがそのまま十分とは限らない。
まず保存shotsに対する十分条件を確認し、追加shotsが必要なら別の新科学評価・新価格となる。
IID-gridへのlaw変更やexpected-bias校正を選ぶ場合、旧PCG価格proxyからのtransferやconfidence政策も別に必要。
実量子shot未実施の数値的十分条件を`actual_CA_achieved`と呼ばない。

## 7. 最小実装と人工検証

[c_entry_e2_all_law_readiness.py](../../src/trottertracks/c_entry_e2_all_law_readiness.py)と
[passive checker](../../scripts/check_research_c_entry_e2_all_law.py)を追加した。
既存`c1_stage1_execution.read/sha`を使い、NPZ/QPYを開かずbytesを照合する。
科学起動option、numerical backend import、Hamiltonian/PF/RTE/RPE/compileの新engineはない。
このadapterは**metadata検査専用**であり、正のevidenceをattachしない。
人工的に全項を供給してschemaが整っても、`SCHEMA_ONLY_METHOD_NOT_ESTABLISHED`で止まり、
`error_evidence=null`、RPE `UNQUALIFIED`、CA null、scientific_execution_ready=falseを返す。

[新47人工テスト](../../tests/management/test_research_c_entry_e2_all_law.py)は候補/E1/P/CDF0/r32/再配置/SOURCE/κ/scalar/physical/compiler/ground混同、
誤差単位、包含項の重複、finite word、IID→actual昇格、実inverse/phase/shell/full512等の欠損、
null保持、byte drift、準備でのbound差込み拒否、科学前STOPを確認した。
既存accuracy・全列/norm-growth伝播・CDF0接続の人工回帰59件と合わせて**106件PASS**した。結果は
[verification.json](../../artifacts/research_c_entry_e2_all_law/2026-11-08/technical/verification.json)に保存する。
既存`c1_stage3_qualification.synthesis_envelope`の有限残差拒否とも照合した。
これらはlocal技術テストであり、実H4の科学証拠、独立レビュー、CI/外部再現ではない。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/check_research_c_entry_e2_all_law.py
```

## 8. 次の最小作業と、未承認の科学範囲

### 第一優先：新H4計算の前に必要な科学判断・方法確立

既存P0を維持して全word保証を目指すなら、§4の全reachable-domain数値誤差モデルと成功条件、
raw/native→同じtrajectory targetの対応を成立させる。既存条件付きcomposerと構造証明を使い、
新H4有限probeをその代用にしない。成立すれば「全word最終P0の一様誤差を13round raw信号へ伝播できる」
という主張への道が開く。Reference B/UとPCG64はその後も別残件。

これを閉じる有効な方法が現資料から確立できなかったので、**whole-P0科学call数・family数・実行entryはnull/なし**。
今回の実行許可だけを追加しても起動可能な契約にはならない。別compiler、新polar target、fresh IID-gridの期待bias保証等を
選ぶ場合は保証scope/価格/confidence政策を新たに採択する科学判断が先になる。
有限word pilotは既に実行済みであり、その重複を次の最小実験とはしない。

### Referenceだけを先に進める場合の限定案（未承認・非実行）

この案が与えるのはE2固有のreference/categorical接続情報であり、P0/actual-law/CA資格を完成しない。
Whole-P0の最大blockerを放置してこの案を実行する優先性があるかは、次の研究判断である。

| 項目 | 準備した境界 |
|---|---|
| 対象 | E2一候補、同じ再配置LD2、s1/h.1/r16/K0、m0..12のみ。E1/P/CDF0/他rなし |
| SOURCE | 本報告のinput/code/sampling SOURCEを来歴として継承。新adapter/evaluatorを選んだ時点で新科学SOURCEへ固定。旧runの承認やSOURCEを新実行契約として使わない |
| 科学前にbyteで取得できるもの | 索引hashに一致するground、law、normalization、必要なraw記録。存在するものを再計算しない |
| 新計算を避けられない可能性 | 未exportのrelocated generators/head/corrected matrixの再構成、今回p/q table・literal target audit・mean-reference B/U。Retrievalと数値再生成を区別 |
| 計算domain | 有限atom catalogを一度走査、ideal pと53-bit grid qのmean、数学/数値reference chain、13roundの二脚B/U。Word全列挙・native build/compile・価格校正・shots/CA評価は含めない |
| Call範囲 | 上記domainのみ。13×2のbound出力行はcall数ではない。現在のM・raw cache・target・評価器・instrumentationが未固定なので新call上限はnull。旧38/27/86をコピーしない |
| 一回性 | 新固有outputへ一度だけ。未使用pathとhashを実行前照合。旧run01/run02は開かない。Retry/resumeなし |
| 即時STOP | 入力欠損/hash差、target/normalization不一致、literal unitary前提の未解決を隠す要求、数値失敗、範囲外処理の必要、登録対象完了。例外後の精度変更・自動再計算なし |
| 成功 | 同じtarget/units/disjointnessでE2のB/Uと、前提付きcategoricalを取得。新証拠の資格を限定して保存。全law/actual CAとは表示しない |
| 未解決/失敗 | Target premiseや包囲の有用性未達はUNRESOLVED。例外やidentity違反はFAIL/STOP。小さな仮値で先に進まない |
| 資源/効率 | Sector36の小行列を共有し、依存する12回squaring chainは各経路一度だけ。Round出力/独立head等を適切に並列化。Arb contextはprocess分離。実際の空き資源で既存resource_planを使い、恣意的CPU/RAM/worker cap・共有設定変更なし |
| 情報価値 | 新候補でも旧B2型reference誤差が小さいか、literal/native target gapが何を要求するかを分離できる。P0の一様性やPCG64独立性を検査したことにはならない |

Ground新solve、DF fit/物理誤差再評価、exact-H PF/RTE誤差の新評価、実H4回路、RPE shots、総資源、
CDF0追加標本はこの限定案にも含めない。保存値に対する新数値再解析も新科学実行として別許可が必要。
いま実行できる科学runnerがあると誤解させないため、未成立な新実行契約やapproval packetを新設していない。

## 9. 保全・公開・STOP

独立branch `codex/research-c-entry-e2-r16-all-law-readiness-20261108`、
独立worktree `/tmp/research-c-entry-e2-r16-all-law-20261108`、base `cbc517e75ad47054e4246748351f56269dac3a0c`。
正式指示`research_project_instructions_unified_2026-10-25.md`の§8.1/8.2/8.4/10/10.3を実際に確認した。
旧SOURCE/input/raw/result/grant/ledger/marker/STOP、過去のCDF0失敗runと成功run、元dirty checkoutを保持する。
元中断processのwrapper STOP欠落は従来どおりであり、引継ぎrunの成功STOPで補完しない。

研究概要・索引・dated note・manifestには技術準備CASE Bのみを追加する。
旧科学結果の資格・統計判定・研究終了判断は変更しない。完成物をcommitしてGitHubへ公開し、treeと本文・主要blobを照合する。
**Mandatory STOP。新科学計算・追加標本・新科学レビューへ自動継続しない。**
