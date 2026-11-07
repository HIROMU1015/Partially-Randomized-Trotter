# CT-5B：固定SOURCEの構造保証とcompiler条件

2026-11-07 JST。基準 `e3e852a35f24839b42a6d282083583bb611b6a9c`。[REPORT](REPORT.md)。H4/1Å/STO-3G/DF12、Partial S2/LD3/h=.02/K2、q128/r1、strict epsilon1e-11、compiler A/Qiskit1.3.0だけを扱う。以下は著者Codexの静的技術導出であり、独立詳細科学レビュー・正式な新補題採択ではない。

## 1. 資格区分

| 命題 | 判定 | 限定 |
|---|---|---|
| DのSOURCE構造・時間引数・block多重度はevent非依存 | `PROVEN_FROM_FIXED_SOURCE` | 保存preparationとq/rを固定 |
| Dのwhole-native literal RZ価格が固定 | `CONDITIONAL_ON_COMPILER_CONTEXT` | 下記Cと共通record固定の下で全supportへ移送できる。SOURCE固定だけでは認定しない |
| RTE frame pairのSOURCE multisetがpaired arm間で共通 | `PROVEN_FROM_FIXED_SOURCE` | 同一application列、既存basis共有、occurrence単位split、actual SOURCE inverse |
| frameのnative T価格が共通 | `CONDITIONAL_ON_COMPILER_CONTEXT` | Cの下のkey多重度一致。未取得frame wordの取得完了は別 |
| 既存2 trajectoryの数値T削減、2 vacuum価格 | `SAVED_COHORT_ONLY` | CT-4Bの取得済みwordに限る |
| 全supportで実行可能なstrict policy・T優越 | `UNQUALIFIED` | finite frame欠損、event位相、vacuumの資格が残る |

「条件付き」は未観測trajectoryを観測済みと扱う意味ではない。以下に条件のSOURCE根拠と保存contextの照合結果を分けて示す。compilerの版/同値則/対象basisを変えた保証ではない。

## 2. Compiler条件C

固定[compiler A](../../../src/trottertracks/track_c_compiler_a.py)は、初段fixed InverseCancellation、UnrollCustomDefinitions、BasisTranslator、最後のCX/X InverseCancellationを一度ずつ適用する。数値角最適化・近似commutation・RZ cutoff・routing・1-qubit再合成はない。

Cは次を意味する。

1. 固定1.3.0・rz/sx/x/cx・全結合・Targetなし、保存の標準同値則と同名同義gateを使い、SessionEquivalenceLibraryへ追加/変更をしない。
2. SOURCEから得られるpost-unroll gate族集合は、下表の保存contextに入る。gate名/幅/parameter数・同値則を同じに保つ。
3. そのcontextで保存された局所literal templateを使う。元の数値引数を束縛し、signed binary値・SOURCE inverse・joint相対位相を維持する。
4. 初段の削除がD/frameのRZ/XXPlusYYを消さず、最終passはCX/Xだけを削除する。新しいpassを後置しない。

| arm | 常在族 | 任意族 |
|---|---|---|
| B′ | p, xx_plus_yy, rz, rzz, cx, h, crz | cz、sdg(Y) |
| Reference | p, xx_plus_yy, rz, rzz, h, cswap | z、sdg(Y) |

これは保存価格2本だけからの推測ではない。Dに固定非零のframe/対角族があり、B′中央はZ/ZZのどちらもCRZを含む。ReferenceのRZZはDにもある。Referenceは常に2 cascadeのCSWAPとvacuum P、B′は固定scalar Pを置く。SOURCE inverseも同じRZ/RZZ/XXPlusYY族へ閉じる。product-Zのsource消去は任意z族の有無を変え得るが、どちらも表に含む。q/rの多重度やcomponentのwire labelは、Targetのない局所変換mapを増やさない。

### 一次SOURCEと保存contextの照合

Qiskit1.3.0の一次SOURCEを限定確認した。

- [basis_search.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/basis_search.rs)：source gate族が解決するまで同値graphを探索する。価格やtrajectoryの頻度を使う探索ではない。
- [compose_transforms.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/compose_transforms.rs)：gate名/幅とparameter placeholderに変換を構成する。
- [mod.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/mod.rs)：全体mapを作り、各nodeへ引数を束縛する。target-native RZはその値で通す。
- [inverse_cancellation.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/inverse_cancellation.rs)：指定gateのrunだけを消す。RZが指定されていない最終passはRZを消せない。

固定[構造補遺](../pr_reference_bprime_20261107/STRUCTURAL_APPENDIX.md)のcontext閉包を再利用したうえで、保存[B′ catalogue](../../../artifacts/pr_reference_bprime/2026-11-07/Bprime_local_catalogue.tar.xz)の4 contextと、保存[Reference catalogue](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/local_catalogue_boundary.tar.xz)の4 contextを読取照合した。

- 各armのD/frame recordの**保存lowering配列全体**はcontext間で一致した。
- 同じbasis/basis_inverse名の保存loweringはB′とReference間でも一致した。
- p_symbolicは全contextで `RZ(symbolic_correction)` 1個、引数を変えない。
- H/Sdg、存在するZ/CZ、Reference CSWAPの各保存loweringも対応context間で一致した。
- [V5補強inverse D archive](../../../artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/stage_A_raw.tar.xz)の実SOURCE inverse D loweringも保存4 context間で一致した。

これは既存配列の同一性確認で、新しいhistogram価格集計、compile、guard、局所operator計算ではない。同値則変更・別gate族・新passの任意context不変性を証明したものではない。

## 3. Stage A：D価格の固定性

[pilot.symmetric_fused](../../../src/trottertracks/pr_reference_interference_pilot.py)は、qで決まるleft_all/left_no0、directional right、block0 boundary controlを配置する。eventを受け取るtail内部をDの構成判断に使わない。[transfer.assemble](../../../src/trottertracks/pr_reference_transfer_qualification.py)のReferenceはq≥2でoccurrence列をq/2ずつに分け、固定boundary_optimized D列を作り、F側にはactual SOURCE inverseを使う。

[V3 dparts/deterministic_profiles/diagonal](../../../src/trottertracks/pr_reference_rotation_v3.py)の記号的台帳は、固定qについて

\[
\begin{aligned}
H^D_B&=q(L_{all}+R_{all})
 +(q-1)(L_{no0}+R_{no0}+B_0-L_{all}-R_{all}),\\
H^D_R&=q\sum_{k=0}^3\sum_{t\in\{+.01,-.01\}}H_{k,t}
 +(q/2-1)\sum_{t\in\{+.01,-.01\}}(H_{0,2t}-2H_{0,t})
\end{aligned}
\]

と表せる。ここでHは**signed key多重度の台帳**、B_0は既存controlled block0 full-step、L/Rは既存plain/directional D profile。symbolic式を書き換えただけで、新しいq128の価格を算出していない。

SOURCEのD frameはRZ/XXPlusYYで初段fixed cancellationの対象外。directional内部CXは削除され得るが、RZのliteral値や出現を変えない。最終CX/X取消しもRZを削除しない。Cのcontext閉包と局所template一致を合わせると、event列が違ってもD由来のliteral RZ台帳は同じになる。したがって同じword recordを使うD価格も固定になる。

CX数やdepthの加法性にはfenceが必要だったが、**RZ単位のT価格には、CX/X取消しが一致することまでは必要ない**。既存fenceはSOURCE境界の保護を裏付ける証拠として保持する。CX fenceだけから一般のT/depth保証を移したのではない。

actual SOURCE inverseを正方向native列のinverseと同一視しない。V5のinverse閉包・20保存loweringと今回のcontext配列照合を使い、負時間側/逆方向も別の固定templateとして含める。正負でD0 half/fullのRZ順序が変わっても、event非依存性は失われない。

**数値anchorは再計算しない。** CT-4BのD未融合/融合補正の保存値をD_B^star/D_R^starと記す。Cの下で、そのanchorを同じq/r/recordを使う全supportのD項へ移す条件が揃う。recordの選び直しや別precisionの価格へは移せない。signed融合補正は負の物理gateでも負のnorm上界でもない。

## 4. Stage B：frame共通性

[DF RTE builder](../../../src/trotterlib/df_rte_qiskit.py)は、各occurrenceのapplication列をproducts→rotationの順に処理する。basis共有は `(basis_id,basis_hash)` の連続runで決まり、controlは対角作用だけに掛ける。controlledか否かはbasis選択/共有の判定に入らない。各occurrenceの終端ではactive basisを閉じる。

basis bのSOURCE列をA_b、actual SOURCE inverseをA_b†とする。1 runは両方を1回ずつ持つ。同じevent列のB′と、ReferenceのF/G分割前のuncontrolled列では、このpair multisetが等しい。splitはoccurrenceの間なのでrunを途中で切らない。F†は順序を反転し各SOURCE操作を反転するが、各pairには再びA_bとA_b†が1個ずつ現れる。V5保存のdouble-inverse binary parameter閉包がこれを支える。

よって、run数N_b(omega)を記号として

\[
\mathcal F_B(\omega)=\mathcal F_R(\omega)
=\biguplus_b N_b(\omega)\{A_b,A_b^\dagger\}
\]

がSOURCE multisetの命題として成立する。これは別々にcompileしたA_b†のT数がA_bと同じという仮定を使わない。

Cの下では、対応するnative frame recordも両armで同じliteral key multisetを持つ。SOURCEのproduct-Z消去はZを消すがRZ/XXPlusYY frameそのものは消さず、最終passもframeのRZを消さない。そのためframe由来keyのpaired多重度差は0であり、**任意の共通価格記号の代入に対してframe差は0**。

ただし実際のF_B=F_Rという価格の等式には、共通keyすべてでwordを返すpolicyが必要である。CT-4Bの299 keyは全m7 frame域を覆っていない。保存CT-0 role_rowsのm7/RTE_basis_framesにある `0x1.08ba32d420f2cp+2` は、同CT-0 mappingでNUMERIC_KEY_MISSING、CT-4B inventoryにもない。価格差の記号的取消しは、そのwordの取得・strict誤差資格を省略する許可ではない。

## 5. 残差とSTOP

Cを満たさないcompiler/contextでは、D固定性やframe差0を認定せず、CT-5AのA_T(omega,x)とframe差を残す。Cの下では、product-Z消去がRZ histogramの残差を生じても、そのZ templateの採用T=0を用いて**T加重残差だけを0**にできる。全literal histogram残差まで0という主張とは区別する。[有限key](FINITE_KEY_AND_UNKNOWN_PRICE.md)を参照。

条件Cと共通partial cacheの技術的適合性を整理したのであって、全support policyやRPE成功を認定したものではない。新科学計算なしの静的項目を完了し、**mandatory STOP**。
