# RA-D0 v4：構造保存・丸め余裕付き数値実装設計

**設計日：2026-10-08**  
**状態：GPTによる数学設計案。独立監査前。登録最適化・v4実行の認可ではない。**

## 要旨

T0.2までの保存証拠を踏まえ、v4の第一案を「失敗後に許容幅を緩める修復」ではなく、**丸め後に既存certificateを満たす候補を、最初から構成する数値層**とする。

採用を提案する部品は次の三つ。

1. 保存された非正規化Taylor列と有理数norm midpointを使い、B2 membershipとB3 degree matchingを有理数の構造等式で表す。
2. groupの整数総countsを先に決め、その内部をprecisionへ配る階層的量子化を使う。
3. mean・confidence・resource capに対する丸め損失を上から評価し、その余裕をLPへ事前に入れる。

すべての生成解は旧と同じ最終certificateで認証する。比較対象の数値クラス、候補回路、精度、Taylor target、信頼度、shot数をこの設計によって都合よく変更しない。

**重要な射程：** 任意のnominal候補を必ず修復できるとは言わない。以下は、明示した内側LPにexact feasible pointが得られた場合の構成保証である。内側LPのinfeasibilityは、元の数値クラスのinfeasibility証明ではない。

---

## 1. 固定証拠と変更しない対象

### 1.1 固定identity

| 項目 | commit |
|---|---|
| v3 source S | `45cffb2aa10f9219b6cad929c3ade49fe7d36ca8` |
| 旧authorization A | `2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9` |
| 旧v3 result | `35f8b949079f15d0348bc082b916324870da7246` |
| T0 | `72192b3475d59f5c56370cb4068d5659789f0ef4` |
| T0.1 | `5cf56e4a5949d64c24eac127bac0c223d431df87` |
| T0.2 | `d3a7cbb239487ddedf44699378f6c182c1fe5993` |

T0.2では、同一ordinary/O2の既存precision二列間でcountsを移し、単一の保存点においてsampler・membership・mean・confidence・workspaceのすべてが認証された。この事後的構成は、一般的な修復成功、最適性、B3/B2比較、新規性の証拠ではない。[R1, R2]

### 1.2 不変条件

- 保存済みsingle-block finite P3、distinct-basis controlledの対象。
- 既存のcandidate table、logical events、precision variants、cost/error values。
- 原比較のB1_num、B2_num、B3_numという最終認証クラス。
- canonical samplingと補正weight。
- common count denominator `N = 2^60`。
- `e = 1/200`、`alpha_axis = 1/5280`、`delta_num = 10^-12`。
- 既存のoutward `ell_upper` / `kappa_upper`規則。
- B2/B3の同一budget条件、原クラスを包含するouter lowerに対するstrict witness。
- peak workspace判定。期待workspaceへの変更は禁止。
- 旧result、消費済みmarker、source、authorization、T0系列の分類。

この設計は新しいprimal生成器の提案である。旧v3を上書き・retryしない。

---

## 2. 今回修正する考え方

### 2.1 「任意候補を必ず認証する」は不可能

元のconfidence条件とresource capsが両立しない場合、どの修復も成功しない。候補表内に低errorなimplementationが存在しない場合もある。生成失敗と数学的infeasibilityを分ける。

### 2.2 membership保存だけではresource capsを保存しない

同じlogical eventのprecision変更はDqとgroup massを保存しても、T/CX/1Q費用を保存するとは限らない。特にepsilon-constraintの非目的座標が既に上限にあると、微小な増分でも不適格になる。

### 2.3 B2だけ直しても不十分

B3にもsimplex、mean、confidence、dyadic roundingの問題がある。B2専用projectionを全体の修復法と呼ばない。共通の最終認証器と、両クラスを扱える生成器を用意する。

### 2.4 数値最適化法自体を新規量子アルゴリズムと呼ばない

有理数LP、exact reconstruction、余裕を設けた制約、整数配分は既知の数値的部品である。研究上の主張は引き続きdegree-local representation freedomの資源価値に置く。[E1, E2]

---

## 3. 記号と入力契約

- `g`：logical event group。同じ理想eventを表すprecision variantsをまとめる。
- `p = 1,...,P_g`：保存済みimplementation variant。現登録表では原則3 precision。
- `N = 2^60`：整数sampler denominator。次数行列Dとは区別する。
- `t`：固定Taylor係数vector。
- `D_g`：groupの理想正規化degree column。保存区間 `D_g^- <= D_g <= D_g^+` を使用。
- `v_g`：非正規化degree column。保存された `saved_ideal_ab_exact` を該当次数へ配置した有理数vector。
- `c_g`：理想norm。`D_g = v_g/c_g`。
- `[c_g^-, c_g^+]`：保存norm区間。
- `cbar_g = (c_g^-+c_g^+)/2`、`rad_g = (c_g^+-c_g^-)/2`。
- `d_gp`：保存されたphase-preserving実装bias係数の上界。
- `C_gp,Q`：Q∈{T,CX,1Q}の保存条件付き期待費用。
- `W_gp`：peak workspace。
- `y = 1/B`、`q_gp`：補正付きcanonical samplerの確率。

### 3.1 Preflight

各group内でD intervals・logical event・条件付き内部sampling・phaseの同一性を確認する。angle表記が同じというだけで合併しない。

workspace上限超過variantは最初から使用不可とする。ただし今回の固定表に想定外の欠落や除外が生じた場合は、別候補を追加せずreviewへ戻す。B2で必要なgroupが空になるのにbaselineだけ消すことはしない。

norm区間は既存tableから読み、再合成・新angle・新operator evaluationを行わない。cbarを真のnormと同一視しない。差は以下のmean boundへ計上する。[R3, R4]

---

## 4. 構造を保つ有理数パラメータ化

### 4.1 共通変数

非負の有理数変数 `u_gp` を使い、

    U_g = sum_p u_gp
    q*_gp = cbar_g u_gp

と定義する。星印は丸め前の連続設計点を表す。

正規化は、

    sum_g,p cbar_g u_gp = 1

というexact rational equalityで課す。

### 4.2 B3：degree-local class

    sum_g v_g U_g = y t
    u_gp >= 0

を課す。

理想norm c_gを使う場合は、w_g=c_g s_g、s_g=U_g/yと置けば通常のdegree coefficient matchingと対応する。
実際の生成器は保存区間のmidpoint cbar_gを使うので、元の理想等式と完全同値であるとは言わない。midpoint差を認証済みmean誤差へ戻す内側生成器である。

### 4.3 B2：whole-representation mixture

representation rに属するgroupを別名付きで保持する。共有O0もmembership確認が終わるまではr別aliasとして扱う。

    sum_p u_rgp = z_r     (各r,g)
    sum_r z_r = y
    z_r >= 0

各rについて保存された非正規化列が `sum_g v_rg = t` を満たすので、

    sum_r,g v_rg (sum_p u_rgp) = y t

がexactに成立する。

B1は一つのrだけを許し、z_r=yとした特殊ケース。

### 4.4 Zero massは自然に処理できる

z_r=0なら非負性と等式から、そのrepresentationのuとq*はすべて0。
U_g=0の場合も、そのgroupの全uは0である。precision shareを0で割って定義する必要はない。

### 4.5 Midpointに由来するmean bound

    rho_g = sum_k max(
        abs(cbar_g D^-_kg - v_kg),
        abs(cbar_g D^+_kg - v_kg)
    )
    Xi(u) = sum_g rho_g U_g

と置く。

非負性と構造等式から、丸め前のinterval mean residualは

    xi(q*,y) <= Xi(u)

で抑えられる。全係数は保存値から作った有理数であり、Xiもuの線形式である。

---

## 5. 階層的dyadic rounding

### 5.1 group massを先に丸める

    M_g = sum_p q*_gp

にlargest-remainder法を一度適用し、整数K_gを得る。

    K_g >= 0
    sum_g K_g = N
    |K_g/N - M_g| <= 1/N

tieは固定group ID順。M_g=0のgroupへcountsを与えない。

### 5.2 group内部のprecisionへ配る

M_g>0なら

    pi_gp = q*_gp/M_g = u_gp/U_g

とし、K_gをpiに従ってlargest-remainder配分する。

    K_gp >= 0
    sum_p K_gp = K_g
    q^N_gp = K_gp/N

M_g=0なら全K_gp=0。K_g=0なら内部配分も全0でよい。

### 5.3 yとz

    y^N = nearest_half_up(N y)/N

B2ではtheta_r=z_r/yをexact rationalで計算し、

    z^N_r = theta_r y^N

とする。zはlatent membership witnessであり、physical sampling probabilityではない。

y>0と丸め後y^N>0を保証するため、生成LPでは `1/N <= y <= Ymax` を課す。
Ymaxは元の保存モデルの有理数上界を再利用し、生成LPとcertificateで明示的に確認する。

### 5.4 実行時のsampler

階層的なのはcounts生成手順である。完成したjoint integer countsを平坦化し、N通りの一つの一様整数からsampleできる。量子回路を追加したり、測定回数を二重化したりする変更ではない。

---

## 6. Membershipが保存される条件

B2丸め前では、M_rg=cbar_rg z_r。
階層的配分によりgroup massの丸め誤差は1/N以下であり、|z^N_r-z_r|<=1/(2N)。
よって両区間端点に対して、

    |M^N_rg - z^N_r c_rg|
    <= 1/N + c^+_rg/(2N) + Ymax rad_rg.

既存の

    tau_rg = (3+c^+_rg/2)/N

を変更せずに使うには、preflightで

    Ymax rad_rg <= 2/N

を確認すれば十分である。

この条件が成立しない場合にtauを広げてはならない。生成法の前提が満たされないものとして返す。

これは通常のflat per-column roundingよりも、group-totalの誤差を直接制御する点が重要である。元nominal solverのmembership residualをtauへ混ぜ込むのではなく、exactな構造等式からcountsを生成する。

B3にB2のmembership制約は追加しない。

---

## 7. Mean・confidence・資源の丸め損失

以下は安全だが保守的な上界である。今回の登録結果を評価してtightnessや改善率を算出したわけではない。

### 7.1 Meanの丸め上界

    L_g = sum_k max(abs(D^-_kg),abs(D^+_kg))
    Gamma_xi = (sum_g L_g + ||t||_1/2)/N

group内でDが同じなので、Dqの変化はgroup massの変化だけに依存する。
したがって、

    xi(q^N,y^N) <= Xi(u) + Gamma_xi.

### 7.2 Biasの丸め上界

同じgroupで

    q^N_gp - q*_gp = pi_gp (M^N_g-M_g) + eps_gp/N
    |eps_gp| <= 1

と書けるため、保存d>=0に対して

    Gamma_d = sum_g( max_p d_gp + sum_p d_gp )/N

と置けば、

    |d·q^N - d·q*| <= Gamma_d.

nominal precisionが厳しいほどdが小さい、と仮定しない。保存dをそのまま使用する。

### 7.3 各resourceの丸め上界

    Gamma_Q = sum_g( max_p C_gp,Q + sum_p C_gp,Q )/N

により、

    |C_Q·q^N - C_Q·q*| <= Gamma_Q.

同じN、同じcandidate tableでもB2のalias展開とB3のgroup数が異なるため、第一版では両クラスについて計算した上界の最大を共通reserveとして使う。

    Gamma_bar = max(Gamma_B2, Gamma_B3)

各種類のGammaについてこのmaxを取る。B3だけ小さいreserveを使うことを初期版の利益にしない。

### 7.4 Confidenceの丸め上界

    H(u,y) = e y - sum_g,p cbar_g d_gp u_gp - Xi(u)
    Gamma_h = e/(2N) + Gamma_bar_d + Gamma_bar_xi

とすると、

    e y^N - d·q^N - xi(q^N,y^N)
    >= H(u,y) - Gamma_h.

このGamma_hは、T0.2で観測した特定marginから決める値ではない。candidate table、N、丸め手順から事前に導く。

---

## 8. v4の丸め余裕付き内側LP

固定x、固定axis shots n、固定objective Qと残りresource caps b_Rを入力とする。

### 8.1 目的

    minimize 2n ( sum_g,p cbar_g C_gp,Q u_gp + h_Q )

h_T=h_CX=0、h_1Q=5/2は元会計のまま。
必要ならobjective upperへ定数2n Gamma_bar_Qを付けられるが、これはminimizerを変えない。

### 8.2 制約

1. 非負性、exact normalizer、クラス別構造等式。
2. `1/N <= y <= Ymax`。
3. mean：

       Xi(u) + Gamma_bar_xi + delta_num/(2N) <= delta_num y

4. confidence：

       H(u,y) >= kappa_n,up + Gamma_h

5. 非目的resource R：

       2n( sum cbar_g C_gp,R u_gp + h_R + Gamma_bar_R ) <= b_R

6. すべての使用可能columnについてpeak workspace cap。

**すべて線形であり、coefficientsは保存intervalに基づく有理数である。**

### 8.3 構成保証

前提を満たすexact feasible (u,y,z)が得られ、上記階層丸めを適用すれば、

- q^Nの非負性、総和1、denominator N
- B2では元tauを使ったmembership
- mean residual cap
- 元kappaを使ったconfidence
- 指定されたresource caps
- peak workspace

を同時に満たす。

ただし、実装は定理だけを信頼してPASSとせず、出力lawを旧定義の独立certificateで再計算する。

### 8.4 「同じ有限平均」の意味

理想の研究targetは変えていない。一方、midpoint係数とdyadic lawでは厳密なoperator equalityを一般に保てないため、その差を元のnumerical mean budgetへ戻す。

mean residualを都合のよいoptimization budgetへ広げたり、true time evolutionへtargetをすり替えたりしない。

---

## 9. Exact candidate acquisition

### 9.1 推奨方針

既存のdouble solver statusだけをprimal acceptanceに使わない。
小型のv4内側LPについては、**既存のexact-rational LP機能を用いてexact feasible pointを取得することを推奨**する。新しいLP solverの自作は研究の主題にしない。

既存SoPlexにはrational inputに対するexact solving、refinement、rational reconstructionの機能がある。[E2, E3]
これは実装候補の能力根拠であり、この環境での導入済み・性能・動作確認を意味しない。

入力Fractionを一度floatへ落としてexactとして読み直すことは禁止。exact入力・出力のround tripと独立row substitutionを必須にする。

### 9.2 Floating candidateを使う実装を選ぶ場合

nominal resultからのrational basis reconstructionを使うこと自体は可能だが、

- 再構築する基底の選択手順
- exact linear system solve
- 全row/非負性の独立確認
- 失敗時の分類と計算上限

を結果前に固定する。単にdoubleをFractionに包むだけでは解を正確にしない。

第一版でregistered結果を見ながらexact backendとdouble backendを切り替えることはしない。backend選択はoff-domain benchmarkとsource reviewで一度固定する。

### 9.3 計算上限

旧sourceのcap内に新backendが必ず収まるとは言わない。
exact LP呼出し、内部refinement、rational elimination、certificate byte数を記録し、既存gridのworst-case実行数を新しいbackend構成で再計数する。

本設計は旧2秒/LPや旧total callsを自動継承する実行認可ではない。新contractで必要最小範囲を固定するまでregistered solveを行わない。

---

## 10. Outer lowerと公平性

元の認証済み数値クラスをK2、K3、原クラスを包含するouter relaxationをO2と書く。
新しい丸め余裕付き生成領域I2、I3は、**候補生成用の内側領域**である。

    decode(I2) subset K2
    decode(I3) subset K3
    K2 subset K3 subset (共通の基本条件)
    K2 subset O2

B2 lowerは元K2を包含するO2から取得する。新しい保守的I2の最小値を、元B2全体のlower boundと呼ばない。

primary witnessは引き続き、

    U3_certified < L2_outer_certified

だけ。

この式が成立すれば、primal生成の保守性や修復方法の違いによってB2を人工的に弱くして勝った、という偽陽性を避けられる。ただし生成器の保守性による偽陰性は残り得る。

### 10.1 B2 feasible seedをB3へ失わず渡す

budgetを作ったcertified B2 lawは、保存identityが等しいaliasだけを合算してB3へ埋め込める。再量子化しない。

B3用inner LPに余裕がない場合でも、そのseedが指定capsを満たすならB3の実行可能upperとして残せる。
seedを使っただけでは改善witnessにはならない。原則としてB2 lower以下の優位をseedから主張しない。

B0_savedは引き続きK2のsubsetとは仮定しない。

### 10.2 最適性のgap

取得したUとouter Lのgapを保存する。
丸め前後の同一設計のresource差は、

    |G_Q^N - G_Q^*| <= 2n Gamma_bar_Q

で抑えられる。

この丸めboundだけで、元の全数値クラスに対する近似最適性を保証しない。構造的内側化・margin reserve・solver gapは別にある。

---

## 11. Infeasibilityの規約を必ず区別する

v4では新たな内側化があるため、従来の「LP infeasible→そのB2点をskip」という規則を無条件に移植できない。

| 状態 | 意味 | 扱い |
|---|---|---|
| CERTIFIED_PRIMAL | 元certificateとcapsを満たすlaw取得 | budget/upperに使用可 |
| CERTIFIED_ORIGINAL_CLASS_INFEASIBLE | 元クラスを包含するouter problemのinfeasibilityを認証 | 正常skip可 |
| INNER_MODEL_INFEASIBLE_ONLY | 丸め余裕付き生成領域だけが空 | 元クラスの不存在を言わない |
| EXACT_ACQUISITION_FAILED | exact point/証明未取得 | numerical/technical inconclusive |
| RESOURCE_LIMIT | 新計算guard到達 | technical inconclusive |

inner infeasibleだが既存seedがある場合はseedを使用できる。ない場合、勝敗は未判定。

特に、丸めに使う余裕のせいで境界解を除外したことを、物理的・数学的なB2のinfeasibilityと混同しない。

---

## 12. T0.2の移動式を一般化すると何が言えるか

これは主runの無制限fallbackにはしないが、理論的な比較・回帰testとして残す。

同groupのvariant s→t、同一D intervals、d_s>d_t、固定Nに対してk countsを移すと、

    delta_h = (k/N)(d_s-d_t)
    delta_G_R = 2n(k/N)(C_t,R-C_s,R)

であり、group mass、y/z、Dqとxiは保存される。

現在margin m、目標margin sigma>=0なら、

    k_min = max(0, ceil(N(sigma-m)/(d_s-d_t)))

である。受理条件が「>=0」である場合は、ceil後に無条件に1を足す必要はない。

移動元counts K_sとresource上限を考慮し、初期capsを満たす点に対して

    k_max = min(
        K_s,
        floor(N(b_R-G_R)/(2n(C_t,R-C_s,R))) for positive cost increments
    )

を課す。k_min<=k_maxのときだけ、そのpairでの修復が可能である。

**このpairで不可能**は、他pair・他representationを含む問題全体で不可能という意味ではない。
複数pairをgreedyに選ぶ手順は一般に多資源の最小overheadを保証しない。

このためv4の第一案では、precision配分をmargin付きLP内で決める。T0.2の特定deltaや特定O2 pairをproductionへhard-codeしない。

---

## 13. Budget freezeと実行順序

研究上の順序は維持する。

1. 固定input/table/backendからB2 certified primalとその完全resource vectorsを得る。
2. budget値、sampler law、certificate、queryをfreezeする。
3. 同じqueryで元B2 outer lowerとB3 certified upperを得る。
4. B3の結果からbudget、candidate、margin式、backendを変更しない。

v4のbuilderがB2 minimumをexactに最適化しても、丸め後lawまで最適であるとは限らない。artifact名は「certified B2 operating point」等として、nominal minimumと区別する。

anchor-first等の旧sequential protocolを採る場合も、新builder/outer-infeasibility規約・新コストを含めてsource reviewし直す。旧authorizationは流用しない。

---

## 14. Codexへ渡す統合監査の内容

追加の単一点診断T0.3、T0.4を先に増やさない。次の一件で、数学監査とsynthetic/off-domain確認をまとめる。

### 14.1 数学監査

- 保存v_g/c_gとD intervalsの関係。
- B2構造等式からdegree matchingが従うこと。
- midpoint residual Xi(u)の正当性。
- 階層LRMのgroup error、内部error、support保存。
- 元tauのままのmembership保証。
- Gamma_xi、Gamma_d、Gamma_Qの丸めbound。
- mean/confidence/resource capの同時保証。
- すべてのLP係数が有理数かつ線形であること。
- original outerと新innerの包含方向。
- inner infeasibleをoriginal infeasibleにしないこと。

### 14.2 必須のoff-domain cases

- inactive representation、active groupの0 counts、単一precision support。
- nominal doubleでは0に近い負成分を含む場合の拒否。
- 有効なexact structureと、mean residualを誤って0にしたcandidateの区別。
- confidence境界、resource cap境界、両方が同時にactive。
- precision nominal値と保存dの大小が逆の人工table。
- workspaceを満たさないdestination。
- exact continuous inner feasible→rounding後全certificate PASS。
- original feasibleだがmargin付きinnerが空になる人工反例。
- certified B2 seedのB3へのalias合算。
- precision移動がconfidenceを改善してresource capを破る反例。

### 14.3 Backend検証

既存exact solverを利用する場合はrational入力のparser、primal/dual/Farkasの出力と独立再検証を確認する。
small artificial rational LPと、保存R1 tableから独立した係数を使う。

T0.2の保存点は既知regression fixtureとしてのみ扱い、その成功を未観測条件への成功として数えない。

### 14.4 次の境界

この段階はv4 design/source preparationまで。registered optimization、B2/B3 comparison、新合成、DF/分子計算、authorization、one-shotは認可しない。

数学と数値取得の両方が成立し、compute budgetを固定できた場合に限り、GPT source review後に新execution contractを判断する。

---

## 15. 研究の着地点とGO/STOP

### 継続する理由

T0.2により、元の数値certificateを緩めず実行可能な一点が構成できることは確認された。数値インターフェースを構造的に設計し直す動機がある。

### 継続の成功条件

この数値層は、最終的にB2/B3の比較を可能にするための共通基盤である。数値補修の成功自体をRA-RTEの主成果としない。

### 今回のGO

`PROCEED_TO_V4_INTEGRATED_MATHEMATICAL_AND_NUMERICAL_AUDIT`

### 今回認可しないもの

- 新one-shot。
- 新angle、precision、molecule、degreeへの拡張。
- T0.2のPASSを使ったB3優位・新規性成立の主張。
- 旧marker削除・authorization再利用。
- 登録結果を見たmargin、denominator、backendの再調整。

### 停止・再評価

固定した統合設計でexact certificateと現実的な計算範囲の両立に見通しが立たない場合、数値診断を細分化して延長するのではなく、このRA-D0実現方式の優先順位を再評価する。

この判断はPRアルゴリズム改善全般の否定ではない。Hamiltonian前処理の別研究経路もここへ混ぜない。

---

## 16. この回答で実施した自己検算

登録データを読み込まない人工有理数例で、次を検算した。

- 120 cases：階層丸め、group error、mean/bias/costの上界、confidenceの伝播。
- 120 cases：inactive representationを含むB2 membershipと元tauの保証。
- 合計240 cases、すべてPASS。

LP solver、Hamiltonian、circuit、synthesis、登録samplerの実行は0。これらは導出の自己検算であり、独立したCodex監査・production source検証・資源改善の証拠ではない。

---

## 17. 参照情報

### リポジトリ内の証拠（すべて固定commit d3a7cbb...で読んだもの）

- [R1] `docs/tracks/algorithm_codesign/ra_d0_t02_gpt_handoff_20261008.md`
- [R2] `artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/verification_v1.json`
- [R3] `src/trottertracks/algorithm_codesign/ra_d0/table.py`
- [R4] `src/trottertracks/algorithm_codesign/ra_d0/exact.py`
- [R5] `src/trottertracks/algorithm_codesign/ra_d0/numerical.py`
- [R6] `src/trottertracks/algorithm_codesign/ra_d0/lp.py`

Repository / ref:

```text
HIROMU1015/Partially-Randomized-Trotter
d3a7cbb239487ddedf44699378f6c182c1fe5993
```

### 外部一次資料（数値方法の位置づけだけに使用）

[E1] SciPy v1.16.2, `linprog(method='highs-ds')` documentation. Feasibility tolerances、nominal residuals、HiGHS wrapperの仕様。

```text
https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.linprog-highs-ds.html
```

[E2] SoPlex official documentation. Rational LP、iterative refinement、exact rational LU/reconstruction。

```text
https://soplex.zib.de/
```

[E3] SoPlex 7.0 exact-mode documentation. Exact rational solutionを得るための機能と設定。

```text
https://soplex.zib.de/doc-7.0.0/html/EXACT.php
```

これらの文献は本設計の量子アルゴリズム新規性を裏付けるものではない。新規性は今後のRA-RTE本体の比較で別に評価する。
