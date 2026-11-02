# Angle enclosure：既存half-angle手順の有限実装

固定[finite-bit audit §6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)と[F2 §6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md)にある公知手順を[precision module](../../../src/trottertracks/algorithm_codesign/g6_f3_precision.py)に実装した。native rotation・物理provider・回路は作成しない。

## 1. 入力と固定precision

`enclose_atan(ratio,*,tau,root_bits,terms,sigma=1,context=None)`はexact有理0≤r≤2、τ>0、positive int K_phi/J、σ=±1を受け付ける。root_bitsとtermsはcaller指定値だけを1回使う。rootや級数を結果に応じて延長しない。

wrapperは旧eventのr=s/aとr≤2x/(l+1)を確認し、source/input/policy/preflight/event/parent/child/σをcontextへ結ぶ。standalone helperではparent未bindingと明記する。zero rは正確[0,0]を返しrootも呼ばない。合法identity x=0のdispatcherはこのhelper自体を呼ばない。

## 2. exact enclosure

旧root_intervalでq=√(1+r²)の外向き[lo,hi]を1回生成する。1≤lo≤hiかつlo²≤1+r²≤hi²を検査する。half-angleの引数区間は、

```text
z_lo = r/(1+hi), z_hi = r/(1+lo)
atan(r) = 2 atan(r/(1+sqrt(1+r*r)))
```

とする。旧監査の理想引数は2/3未満。実装では提供されたroot区間から0≤z_lo≤z_hi<1をexactに確認し、交代級数を利用できることを検査する。

端点zごとにJ項のS_J=Σ_(k=0)^(J−1)(−1)^k z^(2k+1)/(2k+1)と次項a_J=(−1)^J z^(2J+1)/(2J+1)を作る。既存の交代級数remainderで`[S_J+min(0,a_J), S_J+max(0,a_J)]`がatan(z)を囲む。atanの単調性に従い、phi_loはz_lo側lowerの2倍、phi_hiはz_hi側upperの2倍。無認証float atan、BLAS/trigや数値固有値を使わない。

φ区間は[phi_lo,phi_hi]、代表値はmidpoint、代表値の最大誤差半径は(hi−lo)/2。**区間幅と代表値誤差を区別**し、後者≤τのときだけ返す。σ=−1ならsigned区間[-phi_hi,-phi_lo]とsigned midpointを別に保持する。half-argument、root区間、両端点series/remainder、τの左右もexact evidenceへ保存する。

不足なら`ANGLE_PRECISION_INSUFFICIENT`とcandidate証拠を返し、sessionはSTOP。入力ratio/精度/contextの不正はC分類。root自体が外向き条件を破ればcontract inconsistency。

## 3. 人工検証と限界

testではr=0、1/3、1、2、σ±、τ不足、非法型/domain、rootの固定1回呼出しを確認した。r=1のπ/4はMachin identityの**別のdirect rational atan series**で囲み、r=2はπ/2−atan(1/2)で照合する。direct referenceはhalf-angleもF3 enclosureも使わない。小さい有限人工例の独立算術経路であり、無限入力の機械証明ではない。

固定監査が述べるinvolution回転のstrict差≤τは、exact Qと正しい回路意味論を仮定した既存の数学的帰結である。今回実際に取得したのは角度evidenceまで。native RZ列、T数、CX/depth、controlled relative phase、actual adjoint、state preparation、backend errorは未取得。event係数丸めとangle誤差、物理実装誤差を合算して消さない。

費用には旧event再計算のpacket/roots、1 rootと2端点のJ項series、正確大整数/有理pow/加算/除算、evidenceの保持とhex JSON/hashが加わる。τに応じた最適J/K選択やruntime性能を主張しない。**mandatory STOP。**
