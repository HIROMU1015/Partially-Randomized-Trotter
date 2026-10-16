# M1-S Gate 0：tail局所係数誤差・演算経路の条件付き証明（2026-10-16）

**著者による条件付き導出。旧浮動小数点coverage仮定を前提とする技術的上界であり、プラットフォーム数値certificateでも独立科学レビューでもない。**
旧v4のTECHNICAL_STOP、W/v/rho=null、消費済みone-shotを変更しない。新規数値検証・Gate 0実行は0。

対象：linear H4、1.00 Å、STO-3G、8 spin orbitals、保存DF rank12、HD=保存順G3/G6/G9、HR=残りのDF tail。
非冗長signed Pauli集約、実部射影、HR identityをHDへ移動、cutoff=0、保存数値sector oracle。
delta window・finite-step評価なし。exact入力は保存binary64 bit値であり、化学Hamiltonianや真の固有状態への誤差ではない。
基点a058ef14d4ea162547ec73b707a4662ec8fcd7d6、対象科学SOURCE606036db52da2f080cf846e48824e5e418430762。

## 1. 条件付き算術とノルム

u=2^-53、gamma_k=ku/(1-ku)、ku<1とする。次のv4条件を維持する。

- round-to-nearest binary64、未会計overflow/underflow/FTZなし。
- scalar複素積は高々8 real-operationの標準誤差包絡。BLAS/dot/matvecは既存gamma_(8k+32)包絡。
- 実数fsumの絶対誤差はgamma_(m+32) sum_j |x_j|以内。正しく丸められることは仮定しない。
- raw g_lとone_bodyはexact Hermitian、lambda_lは実数。入力修復・再対称化なし。
- SOURCE、入力、dtype、runtime/build/thread、資源上限を維持し、正常な算術経路以外はtechnical STOP。

zのboxノルムをb(z)=|Re z|+|Im z|、係数mapのノルムを||a||_box=sum_P b(a_P)とする。
b(zw)<=b(z)b(w)、Pauli位相±1/±iはexact targetにおいてboxノルムのisometryである。
各複素積の通常の実部・虚部の積和を個別に囲えば

    b(fl(zw)-zw) <= gamma_8 b(z)b(w)

が得られる。これはv4で使用した「複素積<=8 real-operation」の具体化であり、新しい確率的丸めモデルではない。
compilerやscalar/BLASの実経路が包絡内という未検証coverageは残る。

`_sum`はreal/imagを別にfsumするので、m項のgroupを合算すると

    ||fl(sum x_j)-sum x_j||_box <= gamma_(m+32) sum_j ||x_j||_box.

空groupは出力されず、dictのkey操作、real部取り出し、符号取り出し、represented exact zeroの削除は係数値へ丸めを加えない。
消えた真の微小係数も、exactとcomputed supportのunion上の差として扱う。
途中の相殺で入力絶対値質量を小さく見積もらない。

合成には

    (1+gamma_a)(1+gamma_b) <= 1+gamma_(a+b)

を繰り返し用いる。証明は(1-au)(1-bu)=1-(a+b)u+abu² >= 1-(a+b)u。
相関した誤差や同一mapの二乗でもこの包絡は有効であり、独立性は要求しない。

## 2. 出力係数ごとのgroup長

Pauli語を位相を除いて二値対(x,z)で表せば、積の語は(x_a xor x_b,z_a xor z_b)。
出力語cと左語aを固定すると、右語bは一意に定まる。したがってsupport s_A,s_Bの積で
**一つの出力語へ合算される項数は高々min(s_A,s_B)**。
生成された全寄与数s_A s_Bとは異なる。非可換性は位相へ残し、順序を入れ替えない。

8-modeのa_p^dagger a_qは、p=qならIとZ_p、p!=qなら端点p,qがX/Yでその間がZの4語以内。
全p,qの語のunionは

    s_one <= 1+8+4 binom(8,2) = 121.

これは入力の実測疎性・浮動小数点相殺によらない文字列支持の上界。
近似bilinearも同じ生成語の中にあり、新しい語を丸め誤差が作ることはない。
`_one_body_pauli`では各(p,q)のbilinear dictに同じkeyは一つだけなので、出力語ごとの項数は高々64。
`_multiply(A,A)`では上の一意性から高々121。
tailの`_combine`では各block dictがkeyごとに一係数なので、r個のblockならgroup長は高々r。
実測counter27,781もresource cap1,000,000も、このgroup長の導出には使わない。

## 3. 係数生成DAGの累積gamma index

`_multiply`内の位相計算は8文字で最大8回の複素積、続く`phase*ca*cb`は2回の複素積。
位相が実際にはexactに計算される場合にも依存せず、leafのgamma indexを8*8+2*8=80で包む。
JW ladder自体の±1/2係数はexactで、各ladderのbox-L1は1。

|段階|出力語ごとの和・新規演算|累積index|基準絶対質量|
|---|---|---:|---|
|bilinear B_pq|multiply leaf80、group<=4なのでfsum36|116|1|
|A_l=sum_pq g_lpq B_pq|input scalar複素積8、group<=64なのでfsum96|220|G_l=sum_pq b(g_lpq)|
|A_l A_l|両入力220+220、multiply leaf80、group<=121なのでfsum153|673|G_l²|
|lambda_l A_l²|実lambdaの複素scaleを8で包む|681|S_l=|lambda_l|G_l² [Ha]|
|r-block tail aggregate|group<=rなのでfsum(r+32)|713+r|S_R=sum_(l in tail) S_l [Ha]|

例えばA_lについて、bilinear error<=gamma_116、その入力scaleとsumを合成し

    ||Ahat_l-A_l||_box <= gamma_220 G_l,
    ||Ahat_l||_box <= (1+gamma_220) G_l.

二乗ではexact bilinear convolutionを*として

    ||Ahat_l*Ahat_l-A_l*A_l||_box
      <= [(1+gamma_220)^2-1] G_l².

computed-input convolutionの演算誤差をさらに(1+gamma_233)で包めば
(1+gamma_233)(1+gamma_220)^2-1 <= gamma_673。
実lambda scale後はgamma_681 S_l。
tail aggregateのerrorは

    sum_l gamma_681 S_l
      + gamma_(r+32) sum_l (1+gamma_681) S_l
      <= gamma_(713+r) S_R.

この証明ではshared bilinearの誤差を各blockの質量へ伝播し、同じAを二度使う誤差も含めた。
他のblockやHDの計算回数を独立な丸め誤差として足す必要はなく、入力依存DAGに沿って包む。
途中係数がzeroになってdictから消える場合も、mapをゼロ拡張したL1差で上の式が成立する。

## 4. 実部射影・符号付き集約・identity移動

exact g_lがHermitianで実lambda_lなら各exact DF blockはHermitianで、exact Pauli係数は実数である。
signed集約した後のcomplex errorに上の包絡を適用する。Re射影はbox-L1に対してcontractiveなので、
computed mapを実部へ射影した後もerror<=gamma_(713+r) S_R。
imaginary L1の実測値をthresholdや係数errorの代用品にしない。

exact HR identityをexact HDへ移し、computed HR identityをcomputed HDへ移すtargetを対応させる。
`rreal.pop(identity)`は非identityの値を変えない。従ってnonidentity HRの係数errorは増えない。
HD scalarへのfsum誤差はHD/reconstruction/phase側の旧v4 global ledgerへ残す。
その誤差をHR lawへ再度加える必要はない。HD global基準をこの局所式で置き換えない。

よって本候補の係数補題は、exactとcomputedのnonidentity supportのunion上で

    alpha_R := gamma_(713+r) S_R,
    sum_(P!=I) |c_hat_R,P - c_exact_R,P| <= alpha_R.

符号反転を含み、任意の近似cutoffを導入しない。G3/G6/G9ではr=9/6/3、index=722/719/716。
rank12全体でもr<=12なのでindex<=725。これはreal instruction実測数ではなく、証明で合成したgamma indexである。

## 5. 正規化とW/vの条件付き伝播

L=sum |c_hat_R,P|は将来Fractionでexactに求め、lambda_hatは既存fsumの出力とする。
s=gamma_(M+32)L、delta_lambda=alpha_R+s、lambda_lo=lambda_hat-delta_lambda。
lambda_lo<=0は従来通りprecision unresolved。lambda_exact=sum |c_exact_R,P|について

    |lambda_hat-lambda_exact| <= delta_lambda,
    ||c_exact/lambda_exact-c_hat/lambda_hat||_1
      <= alpha_R/lambda_lo + delta_lambda L/(lambda_lo lambda_hat) = ec.

実係数の正・負channelへevent空間をliftすると、同じ符号でも符号反転でもそのL1差は実係数差と一致する。
従ってこのecはsigned-event lawにも適用できる。
probability divisionを加えたep=ec+gamma_8 L/lambda_hat、mass defect、Pauli expectation ea、mean error em、
centered variance error evは既存v4式を維持する。

Wではoperator norm<=coefficient L1を使い、normalized HRの差<=ec。
既存matrix assembly・matvec・division errorを加えたyに対し、norm-square差<=2y+y²。
dot・norm quotient・最後のsubtractionは既存v4の項をそのまま加える。

vではexact signed expectationの|a|<=1、exact meanの|mu|<=1から、law変動に4ep、
expectation/meanのずれdelta=ea+emに(1+defect)(4delta+delta²)、
最後に既存centered-variance算術項を加える。このため状態のexact固有状態性は必要ない。
対象は保存vectorで定まる数学的oracleであり、真の固有状態距離の誤差は含まない。

## 6. 従来2^-24の精度要求を満たし得ることの解析的十分条件

ここでは実入力で式を評価せず、一般のratio領域に対する2冪包絡を証明する。
d=256、M=65536、r<=12、lambda_hat>0として

    S_R/lambda_hat <= 8,  L/lambda_hat <= 2

を仮定する。これは**解析的な十分条件の領域**であり、新しい実行gate・科学的閾値・scope制限ではない。
局所式はこの領域外でも有効で、実際の停止は従来通り算定予算と採択ceilingの比較で行う設計。
L/lambda_hat<=2は、L>0と既存fsum条件からlambda_hat>=(1-gamma_(M+32))L>L/2でも得られる。

ku<=1/2ならgamma_k<=2ku。これより

    gamma_(713+r) <= 2^-42
    h=gamma_(M+32) <= 2^-35
    j=gamma_(8M+32) <= 2^-32
    g=gamma_(8d+32) <= 2^-40
    t=gamma_8 <= 2^-49
    gm=gamma_(4M+32) <= 2^-33
    gh=gamma_(16M+32) <= 2^-31.

a=alpha_R/lambda_hat、b=s/lambda_hat、d0=a+bとおくと

    a<=2^-39, b<=2^-34, d0<2^-33, 1/(1-d0)<2,
    ec <= 2a+4d0 < 2^-30.

v4のyの各追加項はassembly<=2^-27、matvec<2^-34、division<2^-42なのでy<2^-26。
A=(1+y)^2<2、quotient項<2^-36、Qup<16、subtraction項<2^-48より

    eW < 2^-25 + 2^-52 + 2^-36 + 2^-48 < 2^-24.

またep<2^-29、defect<2^-33、ea<2^-37。
am=(1+gm)(1+defect)(1+ea)<(9/8)^3<2、
delta=ea+em=ep+(2+defect)ea+gm(1+defect)(1+ea)
<2^-29+3*2^-37+4*2^-33<2^-28。
従って

    ev < 2^-27 + (2^-25+2^-55) + 2^-26 < 2^-24.

最後の不等式は7*2^-27+2^-55<8*2^-27。
2^-24はbinary64で表現可能なので、有理数予算を上向きbinary64へ変換してもeW/evの値は2^-24以下になる。
従来の精度要求を緩和せずに、同じ条件付きモデルで満たせる領域があることを示した。

過去の原因調査で報告済みのS_R/lambda_hatは約5.10/4.62/5.34。
本作業ではこれらへ新しい式を数値代入しておらず、新精度予算の実入力評価・数値correctnessは未実施。
この補題からrhoの分母条件、全17品質gate、runtime成功、科学的GOを推論しない。

## 7. SOURCEとの対応、候補設計と未解決事項

対応SOURCEは固定v4 `science.py`の`_sum/_combine/_multiply/_one_body_pauli/prepare/split`。
[機械可読count台帳](../../../artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1/tail_local_model_candidate.json)と
[v5精度設計候補](m1s_gate0_tail_local_precision_contract_v5_design_candidate_20261016.md)を参照。
新しいruntime resolverを既存SOURCEへ接続していない。旧SOURCEやadopted JSONを書き換えていない。
global residual/reconstruction等の6式・8 scalar・eps_rho・runtime/capsを保持し、
law側のalpha参照のみを新局所モデルへ変更する設計である。

未解決は、将来の独立namespaceへのSOURCE接続、proofとSOURCEの再照合、platform coverageの未検証条件、
実入力上の初回数値correctness、契約採択、別の実施承認。
係数上界そのものには条件付き証明を得たため、今回は厳密有理数Pauli再構成へ切り替える必要はない。
その代替案を実行したものではなく、後続BLAS等まで数値certificateにする場合は別の設計・検証を要する。

## 8. 一次資料と証拠区分

gammaと通常の丸めモデルは[Higham–Mary, §1, (1.1)–(1.2)](https://eprints.maths.manchester.ac.uk/2673/1/paper.pdf)の古典的な部分を使用。
確率的sqrt(k)モデルは使用しない。
[Python math.fsum](https://docs.python.org/3.11/library/math.html#math.fsum)のrounding条件やdouble-roundingの注意、
[NumPy matmul](https://numpy.org/doc/1.26/reference/generated/numpy.matmul.html)のBLAS依存は引き続き未認証条件として残る。
静的検査はcount・式の整合と2冪不等式を扱い、Py/NumPy/BLASの実誤差を実測・認証していない。
証拠区分CONDITIONAL_ENGINEERING_NOT_CERTIFIED、著者条件付き導出、候補未採択、mandatory STOP。
