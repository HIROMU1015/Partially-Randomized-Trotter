# 現行経路を覆う丸め誤差上界

2026-11-04 JST。[実行経路](KERNEL_EXECUTION_PATH.md)で確認した限定profileに対する数式。
実H4のnorm、上界値、平均、予算を評価した資料ではない。

## 基本モデルと適用域

u=2^-53、η=2^-1075（最小binary64 subnormalの半分）。
有限でoverflowしない実数基本演算は、nearest-even/gradual underflowの下で

    fl(x op y) = (x op y)(1+δ)+e,  |δ|≤u, |e|≤η

で覆う。FMAもexact積和を一度丸めた同じ形。通常値だけの相対モデルと区別する。
FTZ/DAZを許容する式ではない。使用経路のSSE/AVX命令とMXCSRの確認は前掲資料。
一般のBLAS規格だけからこのモデルやgraphを仮定していない。
[LAPACKの加法underflowを含む解析](https://www.netlib.org/lapack/lawnspdf/lawn149.pdf)も参照。

κの導出条件：C-contiguous complex128 matrix、contiguous正stride vector、m/n≤70、
GEMV alpha=1+0i/beta=0、確認済みPrescott GEMVT/DOT・NumPy ufunc・libm経路。
配置の異なるGEMVN/strided fallbackは今回のprofile外。
FP modeは全参加threadで安定、intermediateのoverflowなし、NaN/Inf入力なしを必要とする。

## Reductionのforest上界

実binaryのGEMVT/DOTでは、実部・虚部それぞれ2n個のsigned productを一度ずつ計算する。
複数のaccumulatorへ分けてから合算しても、各productから出力への丸めdepthと総siteを

    D = 2n+16,   J = 4n+16,   γ_D = Du/(1−Du)

で上から覆える。Dの余裕はhorizontal reduction、alpha作用、yへの加算、NumPy wrapperを含む。
Jは2n乗算＋高々2n加減算と短い後処理を含む。
beta=0の初期出力は実kernelのzero-storeであり、未初期化値×0とは扱わない。
FMAはsite/depthを増やさないので、非FMAも含むこのboundで覆える。
巨大block反復、thread間の部分内積合算、任意の別アルゴリズムへD/Jを転用しない。

forestにおける各rounding誤差は、残りの加減算で高々(1+u)^D倍。
(1+u)^D≤1/(1−Du)なので、実数出力の誤差は

    γ_D Σ |real_product| + Jη/(1−Du)

以下。加法項を葉の値へ掛けず、そのまま各siteから伝播する。
このgraphには誤差を任意のデータで再乗算する箇所はなく、後処理のalphaはexact signed-unit/zero。
reduction順の特定や標本誤差の最大値を必要としない。

## KernelBoundsへの係数

複素積の各座標の絶対積和は|a||b|以下。両座標の合成で√2を使い、さらに安全に2で上から覆う。
行ごとのCauchy–Schwarzと行の二乗和から、下記のmatrix/vector norm形式を得る。

| 処理 | 相対係数 | 加法項（保守的なrational上端） |
|---|---|---|
| m×n GEMVT | κ_mv=2γ_D | a_mv=2m Jη/(1−Du) |
| n要素のcomplex DOT | κ_dot=2γ_D | a_dot=2Jη/(1−Du) |
| m要素complex scale | κ_scale=2γ_2 | a_scale=8mη/(1−2u) |
| m要素complex subtract | κ_sub=u | a_sub=2mη |

GEMVTは||fl(Av)−Av||₂≤κ_mv||A||_F||v||₂+a_mv、
DOTは|fl(ψ†v)−ψ†v|≤κ_dot||ψ||₂||v||₂+a_dot。
scaleはκ_scale|c|||v||₂+a_scale、subtractはκ_sub(||x||₂+||y||₂)+a_sub。
√(2m)を2m、√2を2で覆うため整数の保守的加法係数になる。

scaleの各座標は2乗算＋1加減算。各productからのdepth≤2、座標ごとのsite≤3。
FMA版の1乗算＋FMAも同じ上界に入る。subtractは各実座標1回だけ。
dotの共役は符号変更であり追加の丸めを要しない。
`kernel_bounds.path_bounds()`はこの式だけで既存のKernelBoundsを生成する。
実測誤差から係数をfitしたり、underflow項をfloat変換して0へ潰したりしない。

## Overflowを避ける十分条件

各matrixのFrobenius norm、initial ψのnorm、全supportの**全state boundary**のnorm、
scalar modulusがそれぞれB=2^200以下なら、
matvecの実product/reductionはO(B²)、rotation内部のscale/subtractはO(B³)で、
保守的に64B³<2^610で全intermediateを覆える。DOTもO(B²)。binary64最大値から十分離れる。
これはsource graphの有限性に必要なscreenで、恣意的な計算資源上限ではない。

`finite_intermediate_envelope()`はこの十分条件の判定関数。
境界normは既存Growthのuniform prefix上端等から得る必要があり、平均normでは代用しない。
gain≥1/additive≥0のprimitive chainはprefix上端が最終上端以下になるが、
rotation内部の非unitaryな一時vectorも上のO(B³)へ含める。
今回実H4にこのscreenを適用していない。大きい値でscreen不通過でも実overflowの証明ではない。

## Complex abs：実glibc hypotの保守的上界

CPython _Py_c_absは有限成分をhypotへ渡す。確認したlibmは非FMAの補正多項式版。
有限|z|≤2^400、同じFP modeの範囲で

    |hypot_fl(Re z, Im z) − |z|| ≤ 1024u |z| + η

を使える。glibcの「expected ULP」を保証値として引用したものではない。
以下の粗いforward error解析を使い、実行時のscaling・subnormal出力も含める。
[glibc2.35 SOURCE](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/ieee754/dbl-64/e_hypot.c)。

1. a≥b≥0へ並べる。支配項分岐b≤2^-54 aはa+bを返し、誤差≤2uρ+η、ρ=√(a²+b²)。
   zero軸もここに含む。比較に使うpower-of-two scalingはこの分岐域でexact。
2. 通常kernelに入る値は必要なら2^600倍される。有限入力の最小非zeroは2^-1074なので、
   scaled a/bは2^-474以上。今回の|z|上限ではhuge-input分岐を使わず、a≤2^400。
   最初のa²、b²、その和、sqrtはnormalかつfiniteで、hに|h−ρ|≤4uρを置ける。
3. 補正の両分岐をexact演算で書くと、どちらもr=h²−a²−b²となる。
   (i) d=h−b: r=a(2d−a)+(d−2(a−b))d。
   (ii) d=h−a: r=2d(a−2b)+(4d−b)b+d²。
4. aでnormalizeし、a=1、b∈[0,1]、h∈[1/2,2]として、各ノードの値の区間とforward errorを別に伝播する。
   degree1/2のノードでは、underflow floorをaまたはa²で割った値はu²以下
   （η/a²≤2^-127<u²）。加減算のincoming errorはe_x+e_y、
   積はM_x e_y+M_y e_x+e_x e_y、各丸めでu(M_out+e_in)+u²を足す。
   [exact rationalの証明budget計算](kernel_bounds.py)で両分岐ともe_poly<256uが成立する。
   これは入力samplingの精度テストではなく、全区間の誤差伝播不等式の算術確認。
5. |h²−ρ²|≤9uρ²、2h≥2(1−4u)ρより、補正r/(2h)の大きさを
   c=[(9u+e_poly)(1+u)/(2(1−4u))+u²]ρで覆う。
   最後のsubtractまでの相対誤差は4u+c/ρ+u(1+4u+c/ρ)+u²<1024u。
6. tiny-inputの最後の2^-600倍はnormalならexact、subnormalなら最終丸め≤η。
   これを明示した加法項として残す。多項式の中間値もa²の小さい定数倍以下でoverflowしない。

`complex_abs_error(M)`はこの上端を返す。Mは実際のsignalの全support上端である必要がある。
finite envelopeのscreenと、このhypotの|z|範囲条件は別々に確認する。

## 全support guardとRPEへの境界

coordinate guardはfloatで作る1+1e-12、magnitude guardは1+1e-10。
uniformな|z_fp|上端Mに対して、M≤coordinate_limitかつ
M+1024uM+η≤magnitude_limitなら、既存guard_completionへ接続できる。
平均majorantだけではこの判定を行わない。clipping前の検査を省略しない。

実H4のM、matrix norm、state norm、η_action、U_virtual/予算は今回未評価。
qが大きいとgainの保守性・Frobenius化・worst component/order・一様guardが支配する可能性がある。
係数の導出は、固定N/α/εで十分条件を認定できることや、guardが実際に成立することを意味しない。
不通過ならARITHMETIC_BOUND_OBTAINED_BUT_COMPLETION_UNRESOLVED等を保持し、上界未確立項を0にしない。
