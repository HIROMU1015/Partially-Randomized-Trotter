# Target保存と数値誤差の保証方式

今回成立したのは、実装に対応する保守的な全入力誤差評価の方法と、その人工correctnessである。実H4の上界は未算出。新しい科学的allowanceは採択していない。旧Exact Fraction engine certificateを新gadgetへ移していない。

## 理想target・angle・inverse

旧0edc0fa5の導出を再利用する。任意psiに対しL/Rが真空を既知位相で保てば、CSWAP→L†/R→CSWAP→P(ell+r)は元の複素振幅、X/Yのbinary分布を保つ。L/R可換性・path固有stateは不要。fresh-state RPEへの接続とcoherent QPE全体の追加条件は従来通り分離する。

保存SOURCEが生成したbinary64角を正確な実数として読むfinite gate modelを比較対象にする。元化学係数・PF/RTE理論への近似誤差はこの接続誤差とは別。SOURCEのatan／pi／積や丸めを勝手に理想化せず、実際のgate角とancilla phase項を保持する。

scalar-free L0/R0の位相と全scalar termの和PhiをFractionで取得し、Pへはfloat(Phi)を一回渡す。ledger計算誤差は正確に0、補正角差

\[
\epsilon_\phi=|\operatorname{Fraction}(\operatorname{float}\Phi)-\Phi|
\]

を正確に記録する。これによる座標signal差はepsilon_phi以下。RZ/RZZ inverseは角の符号反転、XXPlusYYはtheta反転とbeta保持、Z†=Zであり、有限角の符号反転はexact。scalarをlane global phaseへ移してmod 2piする処理は導入しない。未対応inverseは拒否する。

## 局所gateと演算丸め

既存a1_engine_connection.certify_gateは、実際の小さいbinary64行列と、保存角の理想gateのinterval行列との差にFrobenius上界deltaを付ける。既存100dps interval contextとoutward dyadic serializationを再利用する。新CSWAP／CX／SWAP／X／Zは厳密な置換/符号行列、SXは半整数複素行列として比較し、実matrixが一致するときだけdelta=0とする。未知gateのdeltaを0にしない。

新数値コードは物理state/Born engineではなく、全system入力を列とするisometryの検証器。BLAS/dotの内部演算を仮定せず、numpyの実数multiply/add/subtractを明示して小gateを作用させる。binary64 round-to-nearestとgradual underflowを前提とし、実行時に基本条件を検査、overflow/nonfiniteはSTOP。標準gamma_kモデルの根拠は [Higham, 1993, §1–2](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf)。以下の具体的な保守化は本実装の演算列に対応する導出であり、文献から実H4保証を借りたものではない。

u=2^-53、eta=2^-1074、局所dimension d、k=4d+4、

\[
\gamma_k=\frac{ku}{1-ku},\quad
a_g=\delta_g+2d\,\gamma_k(1+\delta_g)
\]

とする。実2d表現の絶対値行列normをsqrt(2d)(1+delta)以下で抑え、さらにsqrt(2d)≤2dへ保守化する。明示複素dotの各termは実乗算2回・和差・蓄積からなり、各経路の演算数をkで覆う。underflowは相対誤差だけへ押し込まず加法etaを残す。

isometryはrows=2^w、columns=c=2^nで、入力Frobenius norm sqrt(c)≤c。最大a、最大k、gate数sについて

\[
P=(1+a)^s,\quad
E_W\le c(P-1)+s\,\frac{2\,rows\,c\,k\,\eta}{1-ku}P
\]

をinterval上向き丸めで計算する。global phaseのsin/cos生成にも独立interval差と同じ明示乗算を使い、出力phase fittingはしない。各gateの支持wire順を保持し、matrix hashも確認する。P−1≥1ならSTOP。

## なぜcompiled unitary一致ではなくPOVM差か

Qiskit level2は、最終Z読出し直前のRZを省くことがある。測定分布を保ちながらunmeasured output isometryが変わるので、全isometry差だけを受理条件にすると正当な最適化を拒否してしまう。phaseを出力にfitしたり、compiler最適化を弱めて回避しない。

信号bit0に射影した全入力isometryをW0とし、binary測定effectを

\[
F_0=W_0^\dagger W_0,\qquad
p(0|\psi)=\langle\psi|F_0|\psi\rangle
\]

で定義する。psiや実分子Born値を計算せず、effect operator全体を求める。任意の正規化psiに対し、確率差≤||F_raw−F_compiled||2、±1座標差≤2||F_raw−F_compiled||2。global phaseや読出し後の未観測stateだけの違いは、定義に従って自然に相殺される。相対Pの誤りは相殺されず人工negative testで検出される。

effect積は実multiply/addと固定pairwise sumで計算する。M個の出力行に対してk'=ceil(log2 M)+4とする。isometry誤差eからeffect誤差は2e+e²以下、計算した積の丸めには

\[
2c\gamma_{k'}(1+e)^2+
\frac{2c^2 M k'\eta}{1-k'u}
\]

を加える。差行列は各成分の浮動小数点減算誤差も含め、max real/imag component×ceil(sqrt(2c²))でFrobenius normを上から抑える。rawとcompiledのeffect計算上界を足し、operator norm以下の保証を得る。1以上の情報不足なeffect enclosureは技術STOP。これは科学的精度受理値の採択ではない。

initial layoutで入力columnを埋めるwireを移し、final layoutでsignalとreference wireを追跡する。追加compiler ancilla、未追跡permutation、動的制御、reset位置/個数の変化、signal以外の測定、測定後に同wireを操作する回路は拒否する。compiled reference leakageも全入力isometryから保守的上界を記録する。rawのreference復帰は理想gate代数で保証される。

## 誤差項と今回の到達点

| 項目 | 扱い |
|---|---|
| Gate representation/synthesis | 局所interval差をisometry計算へ加算。native compilation後のgateも再評価。FT合成errorは未定義 |
| Binary64 angle | 元SOURCEの保存角を同じexact実数として保持。新補正への丸めepsilon_phiを別計上 |
| Vacuum ledger | Fractionによる正確和。gatewise真空phaseから取得。Hのvacuum energyは使わない |
| Relative phase | epsilon_phi。新allowanceなし |
| L†/R | scalar-free対応gateのexact inverse意味論。matrix生成/作用は上記bound |
| CSWAP分解 | native回路の全入力effect差に含む。実際の全wrapperで評価し、固定CSWAP単価を加算する方式ではない |
| Compiler phase/routing | raw/compiled両effect、initial/final layout、readout/resetを照合。phase fittingなし |
| Born/readout/grid | 今回は実state/Born/bit生成0。ideal Z読出しを定義。将来bit-gridを使うなら独立uniform53-bitの条件付き2/2^53項が別途必要 |
| Vacuum initialization | 理想resetを定義・価格計上。hardware trace distance D_refなら座標2D_ref。D_ref実測/校正はUNKNOWN |
| State | 全正規化stateに一様。旧N2 state値・state certificateをロードしない。実groundへのstate誤差接続は将来の測定契約 |

候補の元finite targetに対するcompiled座標差は

\[
b_{\mathrm{coord,path}}\le\epsilon_\phi+2\epsilon_{F,\mathrm{compile}}
\]

となる。controlled側にも独立にcompile effect差を付ける。将来hardware/FT/measurement noiseを入れる場合の未知項は0と仮定しない。今回出力は理想gate・保存binary64角モデルの技術的資格に限る。

## 人工証拠と実入力に残る評価

新18 tests PASS、level2で24行、全2-system入力columnを使うeffectの一様評価にPASS。明示演算とeffect積をexact Fraction参照で検査した。人工の最大combined座標上界は約1.383e−10、compiled reference leakage上界は約7.250e−22。これらは人工入力の実際の上界で、実H4の数値や全supportの固定上界ではない。

実H4では2 pathのgateとcompiled列を一度取得し、各unique小gateのcertificateと、最大16回のraw/compiled effect比較を行う必要がある。n=8では17-qubit×256入力columnを使う有限operator評価であり、N2 stateや新しいBorn信号の計算ではない。basis-columnの全線形空間を覆うため、少数の代表state一致を一様保証へ拡張していない。全RTE supportへの一様誤差やQ期待費用はこの2 pathだけでは保証しない。
