# RPE m17：列挙不要な包絡は人工収縮系で成立、H4保証への根拠は未確立

2026-11-04 JST。**状態・trajectoryを列挙しないH=Iの球／box包絡と局所rounding-cellを実装し、
人工dense/non-Hermitian収縮系の全support guardを条件付きで証明した。
一方、厳密unitaryなdense人工例では汎用radial包絡による不変球を認定できなかった。
H4への適用を裏付ける根拠が得られていないため、科学SOURCEは準備せず、この人工adapter系列はいったん終了する。**

これは現行kernelでのあらゆる保証が不可能という定理ではない。
今回の方法からH4の全support保証が得られる見込みを示せなかった、という技術的結論である。
H4の新しい評価・保存数値の再解析は0。両U=null、全体UNDETERMINEDを保持する。

## 対象・再利用・変更範囲

基点は`f7f588d92c8a101acce25f33cc29a6b93bc2faaf`の
[人工adapter](../rpe_m17_row_energy_adapter_preparation_20261104/IMPLEMENTATION_REPORT.md)。
固定研究対象はlinear H4 / 隣接1.00 Å / STO-3G / 保存DF12 / B physical state / 4電子sector70、
Partial L_D3 / δ=.02 / m17 q131072 r32 K2。
保存state・N/α/β/ε・現行NumPy/OpenBLAS kernel/FP条件・guard閾値を変更しない。
今回の小さい人工行列・trig値は技術fixtureであり、このH4条件を変更した科学実験ではない。

[invariant_adapter.py](invariant_adapter.py)を追加し、旧row_energy_adapterのexact dyadic演算・
包含sqrt・array検査、energy_prototypeのnorm/prefix合成、KernelBoundsのforest係数、
finite screen、complex abs、guard、独立したstrict判定を再利用した。
旧SOURCEは変更しない。新しいfile loader、RNG、科学CLI、grant/承認証明機構もない。
Fraction/ball演算は証明計算だけに用い、実際のFPプログラムや演算順を置換しない。

保存norm²は、前段でidentityと保証scopeを確認した96 bit区間を再利用する方針を保持する。
新adapterはnorm²区間を入力として受け取り、state自体をロード・正規化・再計算しない。
今回その実H4区間を新しい比較へ代入していない。

## 列挙不要な球・boxとrow誤差

集合はbinary64状態と実数の球／boxとの共通部分で考え、連続な実数包絡上の不等式を証明する。
各数値行列AについてGram A†Aをexact dyadicで形成し、Hermitian Gershgorin上界から
`ν² >= ||A||₂²`を得る。非HermitianなAにも適用でき、H=I以外のmetricは使わない。

複素Aの実表現Bの各rowに対して、旧forestの

    |e_i(x)| <= γ_D (|B||x|)_i + τ,
    D=2n+16, J=4n+16, τ=J·2^-1075/(1−D·2^-53)

を保持する。zero row、単一signed power-of-twoの**上方scale**／swapは、有限性とgradual underflowの下でexact。
そのrowだけ誤差を0と証明できる。下方scaleはunderflowし得るため同じ扱いにしない。

残るrowの絶対係数行列Wから、`||W||₂ <= sqrt(||W||₁ ||W||∞)`を用いて

    ||e(x)||₂ <= η||x||₂+a,
    η=γ_D sqrt(||W||₁||W||∞),  a=τ sqrt(非exact実row数)

を得る。行情報を残してからnormへ移し、Frobenius化を最初から適用しない。
Frobenius normはno-overflow screenだけに使う。underflowの加法項はexact rationalで保持する。
box像は各rowのinterval積和とforest残差を足して覆い、像の包含をそのまま検査する。

## Energyとsigned radialの扱い

N=||v||²について

    N_next = N + drift + 2·radial + ||e||²,
    drift <= (ν²−1)N,
    radial = Re((Av)†e)

を分ける。一般領域では符号を証明できないため

    −νR(ηR+a) <= radial <= νR(ηR+a),  ||v||<=R

へ戻す。負のdrift係数を単に独立したintervalへ切り離さず、Nと組み合わせたまま

    N_next <= ((ν+η)sqrt(N)+a)²

へまとめる。これが収縮を許すnorm map `h(R)=(ν+η)R+a`の根拠である。
「radialが0／非正」「誤差が独立／平均ゼロ」という仮定は入れない。

局所cellは、実rowの有効productが高々一つの場合だけ扱う。
候補binary64 yの隣接値からRNE cellをexactに形成し、積のinterval全体がcellに入るとき、
実出力の数値はyに固定され、residualと`radial=t(y−t)`を包含できる。
端点のties-to-even、最小subnormal、符号を含める。非emptyなbinary64入力witnessのある人工例で、
負と正のradialをそれぞれ証明できた。

**複数productのGEMV行を、exact内積一回のcorrect roundingだとは仮定しない。**
そのcell判定はUNKNOWN_MULTIPRODUCT_CELLへ戻す。局所有利なcellの符号を球全体へ流用しない。
cellを細分化して全binary64状態を列挙する処理や、際限のない領域探索は実装していない。

## 全support・作用順序・prefix・runtime

componentとorderの正のmassを全て含める。product familyを共通norm/prefix包絡で覆うため、
component列・event列・全stateの列挙は不要。
rotationのexact数値作用T=cI−isPを証明上だけ形成し、そのGramを使う。
実プログラムのmatvec、二つのscale、subtractの丸めと非Hermitian項は保持する。
c/sが0や±1でも、内部P@vなどの演算は実行される前提でprefixへ含める。

初期半径R0を保存norm²上端から作り、Fの像をRFで覆う。
全event familyについて`h_event(RF)<=RF`、さらに`h_R(RF)<=R0`を確認する。
これで`F→product→rotation→phase→R`のq反復、各tailのr反復すべてを帰納的に覆える。
prefix最大値はF・各event内部・Rを分けて評価し、最終半径だけで有限性を判断しない。
q/rの大きさに応じてFractionの巨大なq乗を生成せず、同じ不変性検査で任意長を扱う。

この上界は旧D/J forestに入る**全演算経路**を包含する。runtimeの決定性は要求しない。
ただし、binary/dispatch/profile、全threadの安定したRNE/gradual-underflow、
許されるgraphが導出済みforest族に入ることは引き続き必要。
それが不明ならUNKNOWN_RUNTIMEでSTOPする。人工testの仮定はlive runtime検査の代替ではない。

全prefixのfinite screenと、最後のvdot丸め、complex abs誤差、両座標guardを旧コードへ接続する。
平均の作用誤差は独立に保持し、不明ならstrict判定用Uはnullのまま。
人工guardの成立でも、トップレベル両U=null・RPE_tolerance_certified=false・UNDETERMINEDを変えない。
固定PCG64のIID性や全18round／compiled回路／費用の保証は得られない。

## 人工結果と限界

新14＋再利用30、計44 tests PASS。[VALIDATION.txt](VALIDATION.txt)に記録した。
点での旧adapter比較は包含式の実装参照であり、集合全体の証明は上記の解析不等式・包含検査による。

- dense/non-Hermitianな収縮generatorではbox・球の不変性が成立。
  Fの中間増幅もprefixへ残し、全support guardを条件付きで示せた。
- q/rを1から10^12へ変更しても、generator証明のcall数と包絡が同じ。
  state/trajectory列挙は0で、性能の倍率は主張しない。
- 厳密dyadicのdense unitary `((1+i)/2)[[1,1],[1,−1]]`では、Gram normは1。
  しかし汎用forestのη/aが正なのでg=1+η>1となり、この球包絡を閉じられない。
  boxも絶対値化のwrappingで不変性を示せない。

最後の例は実FP状態が発散する証明ではなく、今回の包絡方式の限界である。
同じuniform norm法で毎回正の余裕を足すなら、前段の長時間guard障害が残る。
unitaryという性質だけでは、任意の正定値Hでも全方向のstrict収縮は導けない。
有利なHとobservableへの戻しの根拠が得られていないため、metric開発を追加しない。

## H4適用判断と次の担当

**現行kernelでH4の全support guardを保証できる見込みは、今回も示せなかった。**
H4の全域に必要な収縮余裕やsigned radial条件は未証明で、実H4行列・保存stateを調べて推測していない。
人工収縮系の成功をunitaryに近い研究対象へ外挿せず、H4科学SOURCE・実行計画は準備しない。
同様の人工adapterを次々に増やすことも、次段として提案しない。

次は**ユーザー**が研究上の選択を判断する段階である。
選択肢は、既存ideal categoricalの保証scopeと未解決float bridgeを明示して現状を保持すること、
または保証対象・数値実装／測定規則を含む別方針を検討すること。
後者は現行kernel・guardを今回変更する提案の自動採用ではなく、新しい作業範囲の判断が必要。
**Codexは新しい方針の明示指示まで停止する。**
将来新しい科学評価を行うなら、適用可能なSOURCE・入力/hash・精度・call範囲・runtime・判定・STOPを固定公開し、
ユーザーの別途明示承認を受ける。重要な独立科学レビューも別の許可範囲で、今回未実施。

旧SOURCE/raw/grant/ledger/marker/STOP、他branch・SERVER2 H8・共有環境を保全した。
CPU/RAM/worker/wallの恣意的な上限は導入せず、将来重い評価には共有immutable入力・並列row/matrix証明を準備する方針を保持する。
今回のproofはgenerator数・行列次元に依存し、q/r trajectory数には依存しない。
専用branchへのcommit/push・remote SHA/tree/公開fileの照合後、**mandatory STOP**する。
