# M1〜M12：数学的主張のEvidence Matrix

**本書は既存命題の正確な再掲と証拠の接続であり、新しい証明・科学的受理ではない。** [機械索引](claim_matrix.json)は各IDと証拠・API・既存test名を結ぶ。[SOURCE inventory](source_inventory.json)に固定commit・bytes/SHA256・各class/methodの開始/終了行がある。

## 共通の対象・証拠ID

有限L≥1、明示された正の有理pᵢ（Σpᵢ=1）、有理xで0<x≤1、奇数m≥1、σ=±1。Qᵢ†=Qᵢ、Qᵢ²=Iを満たす同じHilbert空間上のHermitian involution。異なるlabel間の追加関係は使わず、Q(u)=Qᵢ₁…Qᵢₗの順序を保持する。R=ΣpᵢQᵢ、対象M=Pₘ(−iσxR)。任意の密度行列ρについてcoherentなfirst operator momentを扱い、channel一致・無条件のexponential精度・energy保証を主張しない。分子geometry/basis/DF rank/L_Dはこの一般形式命題には非該当。

x=0はidentityの別経路。零質量labelは事前除外。SOURCEの固定H/Kがすべての数学的入力で成功するという量化はしない。以下の「証明」は既存監査内の一般論証の所在、「test」は有限fixtureの既済記録であり別の証拠層である。

| ID | 固定資料／役割 |
|---|---|
| [P] | [G6独立数学監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md) |
| [F] | [G6 finite-bit/access監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md) |
| [R] | [既存GPT G6レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md)、特に§5〜7/§9 |
| [K] | [G6 ReturnKernel SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py) |
| [G] | [G7 Generator SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py) |
| [C] | [G7 controlled provider IR](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py) |
| [T6] | [G6 test SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/tests/tracks/algorithm_codesign/test_g6_return_aggregation.py)／[既済16件記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/artifacts/track_b_g6_return_generator_audit/2026-10-10/focused_tests.json) |
| [T7] | [G7 test SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/tests/tracks/algorithm_codesign/test_g7_budget_and_control.py)／[既済18件記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/artifacts/track_b_g7_budget_control_preparation/2026-10-10/focused_tests.json) |

## 対応の一覧

| ID | 支持される既存の内容 | 証拠の型／主な留保 |
|---|---|---|
| M1 | 同じ有限Taylor targetへの形式return集約 | [P]§2の一般論証＋[K]＋形式test。channel保証ではない |
| M2 | 自由積Green型係数再帰 | [P]§5、既知式の特殊化。新random-walk定理ではない |
| M3 | 全語表なしの局所係数query | [F]§2、[K]。全量子回路の安価な生成とは別 |
| M4 | 0<x≤1での正値性 | [P]§3–4。x>1・符号付きpへの拡張なし |
| M5 | parent/child pairingのmean保存 | [P]§6。単一Qiだけを回転する |
| M6 | ordinary envelopeからの事前zero-fill | [P]§7–8、[G]。G6単体はpacket API、乱択interfaceはG7 |
| M7 | idealの受理率・moment・range | [P]§7–8。条件付きA_uと全試行Z、NとNZを区別 |
| M8 | global normalizer不要の上界U | [R]§7、[G]。G6当初監査からの後続拡張で、元certificateを改訂しない |
| M9 | 実dyadic確率による有理補正weight | [F]§3/5、[K/G]。meanは近似係数に対するもの |
| M10 | 有限bitのbias・range・精度取得条件 | [F]§4–6、[R]§7.4。自動precision selector未実装 |
| M11 | 算術count・係数bit長の多項式上界 | [F]§2/5。全online/runtime/量子costとの分離が必要 |
| M12 | controlled位相・角度・順序の条件 | [P]§1/6、[F]§6、[R]§9、[C]。general physical providerは未取得 |

<a id="m1"></a>

## M1：有限Taylor operatorとreturn集約

**命題・仮定／量化**：共通条件のすべてのp,x,m,Q,σと長さl≤mのreduced uについて、Pₙ(u)をn回IID語のreduce先がuとなる確率とする。a_u=Σₙ(−1)^((n−l)/2)xⁿPₙ(u)/n!（n=l,l+2,…≤m）と置くと、M=Σᵤ(−iσ)^l a_uQ(u)。有限和の等式で、追加のQ間関係があっても写像として有効。

**証明・SOURCE・既済検証**：[P]§2。[K] `reduce_word`、`ReturnKernel.word_series/_mass/mass`。[T6] `test_general_series_insertion_positivity_parent_envelope` はraw語oracleと係数を、σ±の形式平均も照合した既済test。

**既知性／独立性**：同一operatorを収集する原理と高次寄与吸収は既知。全formal returnを同じfinite targetに対して局所生成へ接続する結合の独立性は未確定。**未実装・未検証**：無条件のexponential精度、channel一致、量子backendの実行は含まれない。**利用位置案**：本文の対象定義・構成保証、収集の詳細は付録。

<a id="m2"></a>

## M2：Green再帰

**命題・仮定／量化**：共通条件でFᵢ(0)=0、Fᵢ=zpᵢ/(1−zΣⱼ≠ᵢpⱼFⱼ)、G=1/(1−zΣpᵢFᵢ)、ΣₙPₙ(u)zⁿ=G∏ᵢ∈ᵤFᵢ。形式級数を次数mまで切断する。係数の積の可換性でoperator語を並べ替えない。L=1のrecurrent caseを含み、analytic transience・数値resolvent branchは仮定しない。

**証明・SOURCE・既済検証**：[P]§5。[K] `ReturnKernel.__init__`、`word_series`、`convolution`。[T6] general-series test、`test_single_label_recurrent_walk`。

**既知性／独立性**：Aomoto–KatoのZ2特殊化で代数対応済み。再帰自体を新成果にしない。**未実装・未検証**：無限長samplerや新しいスペクトル保証はない。**利用位置案**：本文の既知lemmaと引用、特殊化・切断の詳細を付録。

<a id="m3"></a>

## M3：局所係数取得

**命題・仮定／量化**：明示p tableと共通条件の任意の指定reduced uについて、切断F/Gからa_uを、偶数uについて全childのa_iuも局所取得できる。全reduced-word辞書、全費用表、B_RA=Σ_even reduced u d_uを入力・前計算しない。input readのΩ(L)を除外しない。

**証明・SOURCE・既済検証**：[F]§2、[P]§5–6。[K] `_word/word_series/_mass/parent`。[T6] general-series、invalid-query、`test_no_enumeration_or_science_import_in_prototype`。[T7] `test_enumeration_separated_from_production` はG7経路を点検。

**既知性／独立性**：既知Green級数の切断・畳み込み。ordinaryも非列挙IID、CTSもlayer samplingを持つため、この性質だけでは優位・新規性を示さない。**未実装・未検証**：cost tableを要求しないことと、可変角のnative価格取得が安価であることは別。**利用位置案**：本文のaccessと生成アルゴリズム、計算量表。

<a id="m4"></a>

## M4：short-step正値性

**命題・仮定／量化**：共通条件、l=|u|、t_l=xˡ/l!、p(u)=∏pᵢ、χ=Σpᵢ²。既存の挿入cover上界 Pₙ₊₂(u)≤(n+1)χPₙ(u) により、t_lp(u)(1−χx²/(l+2))≤a_u≤t_lp(u)、従ってa_u>0。すべての該当uへ量化し、末尾一項・L=1も含む。

**証明・SOURCE・既済検証**：[P]§3–4の有限交代和論証。[K] `_mass` は係数を計算するが、挿入coverを証明するコードではない。[T6] general-series、`test_insertion_overcount_is_not_equality`、invalid-domain、x=0別扱い。

**既知性／独立性**：cover・交代和の既存論証と基本不等式の適用。原論文に本保証全体が同じ形であるか、構成lemmaとしての価値は別途判断。**未証明の拡張**：x>1、負p、無限Taylor、一般signed collectionでの正値性は主張しない。**利用位置案**：本文の適用域lemmaと証明付録。

<a id="m5"></a>

## M5：parent/child pairing

**命題・仮定／量化**：偶数reduced parent u、空語なら全i、非空ならi≠first(u)をchild iuとする。s_u=Σa_iu、d_u=√(a_u²+s_u²)、φ_u=atan(s_u/a_u)、q(i|u)=a_iu/s_u。V_ui=(−iσ)^l exp(−iσφ_uQᵢ)Q(u) に対し d_u EᵢV_ui はparentと全childの元係数を再現し、全parent和でMとなる。s=0は純parentの別扱い。

**証明・SOURCE・既済検証**：[P]§6。[K] `Parent.s/d2/angle_tangent/child_law`、`ReturnKernel.parent`、`digital_parent`のphase/time-order記述。[T6] general-seriesのsigned mean、`test_odd_word_need_not_be_involution`。後続[C]は条件付きcontrolled実装記述。

**既知性／独立性**：単一involutionのEuler式はWBC/PTSCと近接。return集約後のeven suffix pairingとword依存角を含む全lawの方法同値は未確定。**留保**：Q(u)全体をinvolutionとして回転しない。**利用位置案**：本文のideal構成、phase/ordered-product規約を付録にも置く。

<a id="m6"></a>

## M6：envelopeとnormalizer不要生成

**命題・仮定／量化**：b_l=√(t_l²+t_(l+1)²)、B_o=Σ_even l≤m−1 b_l。d_u≤b_lp(u)、B_RA≤B_o。lをb_l/B_o、raw語をIID pから提案し、非reducedならzero、そうであればA_u=d_u/(b_lp(u))で受理しchildを選ぶ。全語辞書・B_RAを計算せずideal event記述を生成する。

**証明・SOURCE・既済検証**：[P]§7–8。[K] `envelope2/order_intervals/digital_parent`は局所packet。[G] `FullReturnGenerator.__init__/sample`がorder→raw→事前zero→局所coin→childを接続。[T6] `test_reducing_raw_proposal_would_double_count`、[T7] forced raw-pair zeroと非列挙性test。

**既知性／独立性**：ordinary IIDとrejection/importance weightingは既知。全結合の同情報・同保証での還元は未確定。**実装限界**：G6単体はRNG・測定を含まない。G7はbits APIからeventを返す。fresh uniform bitsを仮定するideal lawと決定的diagnostic streamを区別する。B_oの次数和とnormalizationは必要であり「一切のnormalizer不要」ではない。**利用位置案**：本文の生成手順・oracleモデル。

<a id="m7"></a>

## M7：受理率・二次moment・range

**命題・仮定／量化**：ideal一試行、任意ρ・X/Y軸で、受理後だけHadamard Y=±1を測定しX=B_oY、拒否はX=0。全試行分母で E[X]=Re/Im Tr(ρM)、E[X²]=B_oB_RA、絶対値range≤B_o。Z=B_RA/B_o≥1/(2e)。A_u≥1/(2√2) は**有効reduced parentが提案された条件付き**の下界であり、Zとは別。

**証明・SOURCE・既済検証**：[P]§7–8。[G] `FullReturnGenerator.sample`のNoneとevent weight、[K]補正API。[T6] `test_accept_only_average_changes_mean`は誤分母の有限negative診断で、一般moment等式の証明は[P]。実際の測定器は未実装。

**既知性／独立性**：全試行weight会計は既知。IS論文の量子error-detection後ZeroFillと、G6の量子前の古典棄却を同じ物理処理としない。**費用・精度留保**：二次momentを総量子費用としない。一試行の量子呼出し期待値はZ、N試行ではNZ（既存[P]§8）。拒否にも古典費用はある。未知Zを実際の十分Nへ無料代入せず、期待呼出しをhard capにしない。digital lawへideal等式を移送しない。**利用位置案**：本文の統計保証・費用の解釈。

<a id="m8"></a>

## M8：B_RAを計算しない上界U

**命題・仮定／量化**：共通条件でv₁ᵢ=pᵢ、r_l=Σv_lᵢ、v_(l+1),i=pᵢ(r_l−v_lᵢ)、r₀=1。r_lは「raw語が既にreduced」の質量。U=d_empty+Σ_even l≥2 b_l r_l は B_RA≤U≤B_o を満たす。reduction後の長さ分布ではない。m=1、L=1の等号caseを含む。

**証明・SOURCE・既済検証**：Uは当初の[P]ではなく、後続の[R]§7.1–7.4にある論証。[G] `raw_reduced_masses/upper_normalizer`、`FullReturnGenerator.U`、`budget`。[T7] `test_raw_reduced_recursion_independent_oracle`、`test_U_contains_normalizer_without_budget_oracle`。小さいtest内の全語和は検証oracleで、production入力ではない。

**保証と量化の範囲**：idealの一般式ではlog(2/α)=ℓ、残余幅s>0を用いた十分全試行数 ceil[ℓ(2B_oU/s²+4B_o/(3s))] が[R]にある。SOURCEはUの外向き区間を構築し、そのhiをdigital budgetで用いる。実装G7のconfidenceは固定α=1/320で、任意αの一般budget APIではない。

**既知性／独立性・留保**：既存envelopeとroot局所係数の認証上界。独立新規性未認定。UはB_RA自体でもZの実測推定でもなく、point値や区間midpointを認証上界と呼ばない。ideal U≤B_oを数値区間endpointの任意の相互順序保証へ拡張しない。**利用位置案**：本文のnormalizer不要の統計保証、interval処理は付録。

<a id="m9"></a>

## M9：finite-bit proposalとweight

**命題・仮定／量化**：既存enclosureと十分な固定bit数でfull-support dyadic masses h_l,h_i,h_A,h_(i|u)を構成できた各eventについて、実π_ui=h_l(∏h_i)h_Ah_(i|u)、近似係数c̃_ui=d̃_u a_iu/s_u、W̃=c̃_ui/π_ui。π_ui W̃=c̃_uiは正確な有理等式。meanはM̃=Σc̃_uiV_uiで、元のMと無条件にexact一致しない。

**証明・SOURCE・既済検証**：[F]§3/5。[K] `root_interval/dyadic_distribution/dyadic_bernoulli/dyadic_index/digital_parent/rational_event_weight`、[G]実proposalとevent記述。[T6] `test_dyadic_law_support_and_exact_bit_map`、digital-packet、insufficient-precision。[T7] digital moment/range/support test。

**既知性／独立性**：実proposal補正は既知importance weighting。局所root精度・support・符号を同じ有限targetへ接続した全構成の非自明性は未確定。**留保**：確率丸めの補正と係数近似biasを区別する。symbolic angleのままの保証で、物理RNGやnative列を作ったものではない。**利用位置案**：本文の有限bit構成保証、丸めアルゴリズムを付録。

<a id="m10"></a>

## M10：bias・range・必要精度

**命題・仮定／量化**：存在する各確率因子の相対誤差≤η（0<η<1）、abs(d̃−d)≤ρd（ρ>0）とsupportを満たすとする。既存上界は π_ui≥(1−η)^(l+3)c_ui/B_o、abs(W̃)≤(1+ρ)B_o/(1−η)^(m+2)、∥M̃−M∥≤Σabs(c̃−c)≤ρB_RA≤ρB_o。B_o≤eˣ<3を用いる保守的版もある。

**証明・SOURCE・既済検証**：[F]§4–6。digital moment上界≤κB_oU、κ=(1+ρ)²/(1−η)^(m+2)は後続[R]§7.4と[G] `budget`に接続。[K]はisqrt外向き根、largest remainders、support/relative-errorの検査、不足時fail closed。[T6] root、insufficient-precision、digital全L1-budget既済test。[T7] budget/moment test。

**必要精度と未実装**：[F]§5にはp_min、t_(m−1)、A_u、child確率の既存下界と、入力bit長・log(1/η),log(1/ρ)に多項式の精度で構成可能という論証がある。任意入力へK/Hを自動選択するSOURCEはない。固定256/160bitが全入力を覆うとはしない。角度近似・native誤差は別予算。固定bitでr→∞の保証もない。

**既知性／独立性**：標準的な補正・誤差会計と本構成固有のenvelope/supportを接続した候補。tight lower boundやplatform certificateがあるとはいわず、上下界一致・普遍的資源改善を論文化の必須条件にも加えない。**利用位置案**：本文のbounded finite-precision保証＋完全な条件を付録。

<a id="m11"></a>

## M11：算術count・bit complexity

**命題・仮定／量化**：[F]§2の既存countは、次数mの系列前計算O(Lm²)有理演算・O(Lm)係数、一つのreduced uの係数O(|u|m²)、偶数parentと全child O(m³+Lm²)。p/xの入力bit長H_p/H_xに対し係数bit長O(m(H_p+H_x+log m))。D=lcm den(pᵢ)を用いた既存分母上界に基づく多項式処理で、単位費用の有理演算を実秒としない。

**証明・SOURCE・既済検証**：[F]§2/5。[K] `ReturnKernel.__init__/word_series/parent/convolution`。testは算術結果と無列挙境界を確認した有限fixtureで、漸近scaling benchmarkでもlower boundの証明でもない。

**静的対応上の注意**：raw語のreduced判定O(m)と、label drawを含むwhole online費用は別。[K] `dyadic_index`はlawの検査・線形走査、`dyadic_distribution`はsortingを含む。現[G]はそれらを呼ぶため「IID labelへのO(m) query」を「実装全体がO(m)時間」と呼べない。P5の群数/群式のO(L²)も、selector・丸め・bit・root・全online費用を一括したcountではない。

**既知性／独立性・未取得**：通常の切断級数/DP演算。全表構築を避ける性質は支持されるが、同情報CTS/ordinaryに対する新しい計算量分離は未確定。新しいtight boundやnative oracle費用を補完しない。**利用位置案**：本文の限定計算量表、bit/selector/modelの詳細を付録。

<a id="m12"></a>

## M12：controlled位相・回転角・順序

**命題・仮定／量化**：idealの全σ・ordered uで、回転軸は単一Qi、Q(u)は固定順。controlled evolutionをdiag(I,V_ui)として実装するとき、parent phase (−iσ)^lをancilla相対位相として保持し、時間順はreversed(u)。一般のsystem-only Q oracleから安価なcontrolled-Qを得られるとは仮定しない。

**証明・SOURCE・既済検証**：[P]§1/6、[F]§6、[R]§9。[K]symbolic tangent/phase/time order、[G] `_event`、[C] `controlled_event_ir/provider_contract/bind_synthesized_provider_ir`。[T6]signed形式平均・順序、[T7] `test_strict_controlled_semantics_and_helper_reset`（off-domain・σ±）、actual-adjoint/scalar保持test。G9の物理3qubit fixtureは補助例で一般provider保証にはしない。

**角度・有限precisionの条件**：[F]にはtanφ=s/a≤2x/(l+1)、角度enclosureの構成可能性、角度誤差τからoperator誤差≤τへの接続がある。G6実装はtangentを渡すだけでatan/native列を生成しない。G7はexact controlled-Q・actual adjointを条件とするIR。provider誤差0はそのqueryモデルの仮定で、物理測定済み値ではない。

**既知性／独立性・未実装**：Euler/phase kickback・既知controlled操作との組合せ。一般involutionの物理費用・角度取得・workspace・native誤差の取得は未完。**利用位置案**：本文にoracle条件、付録にphase/order・conditional providerの検証、native総資源主張は補助資料へ限定。

## Matrixの利用境界

M1〜M12が論理的に一つの構成候補を形成することと、その結合が独立原著として十分であることは別である。ここに挙げた未確定事項を新定理で補わず、[章立て・不足事項](paper_structure_and_gaps.md)と[方法差](prior_art_and_method_delta.md)へ返す。前回A/Bの技術分類や旧結果を上書きしない。
