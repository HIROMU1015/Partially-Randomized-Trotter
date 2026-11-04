# 局所座標事象からbranch・energyへの十分条件

2026-11-04 JST。**証明・実装の技術準備。実H4の新しい数値判定は未実行。**
対象はlinear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D=3、
δ=.02、m0..17、保存q/r/K/N/α/β/ε、保存physical stateのnormalized ray、4電子sector70。
Full、実測定、費用は対象外。本書は条件付きの数学的十分条件であり、研究方向の採択や独立科学レビューではない。

## 1. 使用する事象と入力の意味

各軸a∈{c,s}のideal categorical ±1測定の真の平均をν_m,a、
保存binary64平均を**exactな中心**μ₀_m,a、標本平均をh_m,aとする。
保存包含I_m,aと既存の局所strict条件は

    ν_m,a ∈ I_m,a
    U_cat,m,a ≥ sup_{ν∈I_m,a}|ν−μ₀_m,a|
    U_cat,m,a + sqrt(2 log(2/α_m,a)/N_m,a) < ε_m,a.

したがって、各軸内の独立な[-1,1]有界shotという仮定の下で

    Pr(|h_m,a−μ₀_m,a| ≥ ε_m,a) ≤ α_m,a.

事象G_m={両軸の|h−μ₀|<ε}とG=∩G_mを用いる。
下記はG上の**全ての座標値**に対する条件である。
標本格子の一部、保存中心だけ、観測branchが正しかった例だけを保証するものではない。
U_catはεの中で消費済みであり、下記の位相誤差やPF/RTE予算へ再加算しない。
既存36軸の判定を再利用し、局所包含も局所条件も再計算しない。

共通referenceは全roundで一つのE_refを固定し、φ_ref=−E_ref δとする。
roundごとに位相をfitしたりbranchをE_refで選んだりしない。
元SOURCEの符号はU(t)=exp(−itH)、z=〈ψ|U|ψ〉、sine軸はIm(z)。
identity/scalar位相は既にsignalに含まれる。energy shiftの追加補正はしない。

## 2. 半径・円周位相への幾何学的接続

円周距離をd_T(x,y)=min_{k∈Z}|x−y+2πk|とする。
ρ₀_m=|μ₀_m,c+iμ₀_m,s|、D_m=sqrt(ε_c²+ε_s²)。
まず[既存ギャップ整理](../rpe_18round_ideal_enclosure_preparation_20261104/RPE_GLOBAL_EVIDENCE_GAPS.md)
の条件を用いる。

    ρ₀_m > 0,  D_m < ρ₀_m
    s_m ≥ d_T(arg μ₀_m, q_m φ_ref)
    b_disk,m ≥ s_m + asin(D_m/ρ₀_m).

証明：G_m上のh_c+ih_sは中心μ₀の半径Dの円板内にあり、原点を含まない。
原点から円板への接線の角度がasin(D/ρ₀)。円周距離の三角不等式で
d_T(arg(h_c+ih_s),q_m φ_ref)≤b_disk,mとなる。
任意の円周位相からの距離はπ以下だが、上式の和がπを超えた場合に小さい予算内と扱わない。

これは**保存中心の正の半径**である。ν_catの半径、PF unitary信号の半径、
attenuationを除いたmodel半径、PF固有状態controlの半径とは異なる。
ρ₀≈1という観測をρ_PF≥1へ昇格しない。必要ならG上の標本半径にもρ₀−D>0が従う。
ν_cat自体の正半径が別の主張で必要なら、保存包含矩形の原点からの距離を別に証明する。
このSOURCEはその新評価を要求せず、局所事象と保存中心による上記経路を用いる。

### 事前固定した最小の代替法

円板は矩形を大きく包むため、十分条件が不成立でも失敗を意味しない。
最小の代替は同じεの閉矩形の四隅

    w_στ = (μ₀_c+σε_c+i(μ₀_s+τε_s)) exp(−iq_m φ_ref), σ,τ∈{−1,+1}

を使う。全四隅でRe(w)>0を保証できれば矩形全体が同じ開右半平面にある。
その半平面では偏角の最大・最小は頂点で達成される：
tan(arg w)=Im(w)/Re(w)の比は、各辺上で単調または一定だからである。
従ってb_rectangle=max|arg w_στ|は矩形全体の有効な上界である。
原点、負実軸のbranch cut、Re=0に触れる場合はこの代替ではUNKNOWN。
adapterは両方を事前に定めて評価し、有効な上界の小さい方を使う。
これは同じ条件・同じ矩形に対する証明の鋭化であり、予算や入力の変更ではない。

両経路でも保存β_RPEへの接続ができなければUNKNOWN/STOP。
必要なら次の別計画として、固定Nの**全座標成功格子**での厳密な決定論的含意確認を検討できる。
それを今回または次の限定SOURCEへ無断追加しない。
区間全体の二項確率sup、Nやεの再選択、保存点のV2確率の代入は行わない。

## 3. PF/RTE・状態・referenceとの対応

直接評価するs_mは、保存physical-state平均と共通E_refの差であり、
PF・finite RTE・stateのspectral mixture等が保存中心に及ぼした位相差をまとめて含む。
これに保存β_PF、β_RTEを再び加えない。
必要な判定は直接求めたb_mが保存β_RPEにstrictに収まるかである。

別の分解証明を採るなら、同じphysical state・normalization・parameterの下で
PF系統差≤β_PF、RTE系統差≤β_RTE、座標→位相≤β_stat、和≤β_RPEを
それぞれ保証する必要がある。既存accountingはunit unattenuated radiusとPF有効固有状態を
仮定し、epsilon_coordinate=rho_lower sin(β_stat)/sqrt(2)を使う。
その仮定を物理H基底stateへ自動移送しない。経験的C_useや保存PF残差は保証付き上界ではない。
本adapterはaccountingのcertified gateを変更せず、cost再会計も呼ばない。

保存stateのnormalized ray、厳密な物理H基底固有state、PF有効固有state、実際に準備したstateは別物。
本幾何経路は保存rayの信号についての条件付き結果を作れるが、実state preparation、
真の基底state fidelity/gap、DFから真の分子energyへの誤差を証明しない。
保存E_refも数値referenceであり、真のsector基底energy E_*との差|E_ref−E_*|≤η_refは別条件である。
η_refがなければphysical energyの保証はUNKNOWNのまま。

## 4. nearest-branchの定理とFloat64実装

理想decoder：θ_m=arg(h_m,c+ih_m,s)から
{wrap((θ_m+2πk)/q_m):k=0..q_m−1}を作り、前roundの推定に円周距離で最も近い候補を選ぶ。
q₀=1、q_m=2q_(m−1)、各b_m<π、全roundが同じφ_refに対して
d_T(θ_m,q_m φ_ref)≤b_mであることを仮定する。

帰納法により、前推定の誤差はb_(m−1)/q_(m−1)。現在の正解rootの誤差はb_m/q_m。
正解rootと前推定の距離は高々(2b_(m−1)+b_m)/q_m。
候補の間隔は2π/q_mなので、

    M_m = π − 2b_(m−1) − b_m > 0

なら正解rootはstrictに一意である。runner-upとの距離差は少なくとも2M_m/q_m。
同一bならb<π/3。b_mが異なる場合、全roundのβ<π/3だけを引用せず各隣接対を判定する。
等号はtieを許す。qの欠落・重複・非倍増、referenceの不一致にはこの証明を適用しない。
最終円周誤差はb_17/q_17。

### 既存SOURCEの実際の演算

`reconstruct_rpe_phase`はfloat化、`_wrap_phase`の加算・剰余、π64、候補の乗算・加算・除算・wrap、
`_phase_distance`の差・complex exp・angle・absを経て`np.argmin`で選択する。
最初のangleは呼出側が計算する。零信号、非有限値、subnormal、branch cut、
sample meanの2k/N−1の丸めも検討範囲に含む。
`np.partition`のrunner-up値は診断欄であり、argminの選択処理ではない。
tieでargminが一つ返ることは一意性の証明にならない。

実装へ移すには、座標成功事象の**全域**・全候補上で次の上界が必要：

- a_m：sample meanの浮動小数点化、angle、principal-wrapを含む位相入力の円周誤差。
- c_m：理想rootから実装candidateへのbase-phase円周誤差。m0の最後のwrapも含む。
- d_m：float candidateとfloat前推定の真の円周距離から、計算された各距離への絶対誤差。

候補と理想rootの全単射対応、全距離の有限性、欠落・重複候補がないこともこのenvelopeの前提。
q倍した誤差をB_m=b_m+a_mとすると、前推定誤差≤B_(m−1)/q_(m−1)+c_(m−1)。
候補生成と距離計算の両側誤差を含め、十分条件は

    π − [2B_(m−1)+B_m+q_m(c_(m−1)+c_m+d_m)] > 0.

実装の最終円周誤差はB_17/q_17+c_17。
`argmin`は有限な距離配列にstrictな差があればその順序を選べる。
adapterは与えられた一様envelopeに対するこの含意だけを評価し、envelopeの出所を認証しない。

**一様a/c/dは現時点で未取得。** NumPy/C libmのangle・exp等へ、
IEEE754 binary64という理由だけで正しく丸められるという契約を仮定しない。
人工入力・少数の高精度照合や保存μ₀のbest/runner-up gapは一様上界ではない。
そのため次の限定科学SOURCEでも実装decoder保証はUNKNOWN固定。
別途必要なのは対象runtime全域の妥当な数値誤差証明、または固定有限測定域の網羅的な証明であり、
任意の1e−12を設定してPASSにしたりdecoderを変更したりしない。

## 5. Alias、実数energy、reference誤差

q₀=1は円周上でrootが一つだが、EとE+2πk/δの区別はできない。
外部の初期energy priorでE∈(−π/δ,π/δ]を選ぶ必要がある。
strictな十分条件として|φ_ref|<πを使い、A=π−|φ_ref|>0を評価する。
これは保存referenceの位置確認であって、そのpriorが物理的に正しいという証明ではない。

decoderは中間roundも円周距離を使うため、中間代表元が±πをまたぐこと自体は誤branchではない。
最終円周誤差をe_phiとしたとき、**e_phi<A**があれば最終代表元とφ_refの差を
同じ実区間へliftできる。これがない場合、円周誤差が小さくてもenergy差は2π/δ近くなり得る。

    E_hat = −φ_hat/δ
    |E_hat−E_ref| ≤ e_phi/δ + e_conversion
    |E_hat−E_*| ≤ e_phi/δ + e_conversion + η_ref.

e_conversionはfloatによるenergy出力を要求する場合の除算・丸め誤差であり、
exact/Arbの数学的変換では0。今回の後続runnerはexact/Arb変換だけを扱う。
δは元SOURCEのbinary64 .02が表すexact rational。表示decimal1/50へ置換しない。
φ_refもexactなbinary64 E_refとδの積で定義し、保存された丸め済みround referenceを流用しない。

η_refを上の最終三角不等式へ一度加える方式を採る。
q_m δ η_refを各b_mへ既に加えた別方式と混在させて二重計上しない。
物理E_*も同じalias window内と主張する場合、δ η_ref<Aも別に必要。
共通referenceの精度、ground/sectorの同定、外部alias priorは相互に代替しない。

保存暫定目標は`.00015936001019904` Ha、保存β_RPEは`.4`。
これらの判定値も元SOURCEのbinary64が表すexact rationalとして保全する。
nominalなβ_RPE/(q_max δ)だけでは成功保証にならない。
本SOURCEは同じ保存目標とのstrict比較を用い、CA/10を新しく採択しない。
reference誤差が欠けたまま「真の物理基底energyまで目標内」とは判定しない。

## 6. 全roundの確率

各軸の局所事象から、union boundによって

    Pr(G) ≥ max(0, 1−Σ_(m,a) α_m,a).

union boundにはround間・軸間の独立性は不要。
各軸のHoeffdingには軸内shotの独立性と有界性が必要で、fresh IID RTEとideal測定モデルは条件として残る。
二項PMFのcos/sin productなら軸間独立も必要だが、本経路はそのproductを用いない。
固定PRNG seed・非重複列・有限samplingテストから数学的IIDを推論しない。
決定論的なArb包含へ架空のconfidence ηを追加しない。

geometry→branch→alias/lift→energyがG上で決定論的に成立するときだけ、
同じ下界を該当する**条件付きenergy成功確率**へ移せる。
実装の一様丸め証明またはη_refがなければ、その段階への移送はUNKNOWN。
必要なのはGの含意であり、decoderの厳密joint失敗確率そのものではない。

保存α文字列をexact decimalとして36項加算する。これは既存Arb局所条件のalpha解釈に一致する。
exact和と外向きbinary64表示を併記し、保存alpha_total=.05とのexact比較を行う。
元accountingのbinary64和も別の明示した診断として確認できるが、decimal経路が不通過のときに
別の解釈を選んでPASSへ変えない。和が僅かに予算を超えても.05へ丸めない。
その場合、実際の1−Σα下界と「保存.95条件は未確認」を区別する。
V2の保存中心点のfloat64二項確率・成功率は、新ν_cat包含の全域へ移送しない。

## 7. 判定の限界

adapterのSUFFICIENTは入力された数学的条件に対する局所の結論。
UNKNOWN/STOPは十分条件で判定できないことを表し、物理的失敗の証明ではない。
不明なphysical fidelity、reference誤差、一様float誤差を0として補完しない。
技術準備・人工PASS・次のideal条件付き結果のいずれも、既存global statusを昇格させない。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
元仮想測定bridgeの`U_virtual_IID_upper=null`、`U_exec_upper=null`を維持する。
