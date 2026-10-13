# D3：Taylor保証と保存実装の対応確認

2026-10-13 JST。GPT統合レビュー第7・11節への独立数式確認と静的source照合。
登録資源の再会計、信号・行列・回路生成、合成、pair評価は行っていない。
旧D1/D2は旧保証policyで有効な証拠として維持する。

## 1. 必要な仮定と積分剰余

実数 \(x\)、Hermitian \(R\)、作用素ノルムを用いる。一般には

\[
e^{-ixR}-P_5(-ixR)
=\frac{(-ix)^6}{5!}\int_0^1(1-s)^5 R^6e^{-isxR}\,ds.
\]

これは \(s\mapsto e^{-isxR}\) のTaylor積分剰余であり、6階微分が

\[
(-ix)^6R^6e^{-isxR}
\]

になることから得る。Hermitian性と実数時間により指数のノルムは1。
従って

\[
\|e^{-ixR}-P_5(-ixR)\|
\le\frac{|x|^6\|R\|^6}{5!}\int_0^1(1-s)^5ds
=\frac{|x|^6\|R\|^6}{6!}.
\]

**最後の \(|x|^6/6!\) への置換には \(\|R\|\le1\) が必要**。
Hermitianというだけではこのnorm係数を落とせない。非Hermitian、複素時間にも無条件で移せない。
一般証明は全実数時間で成立するが、D3の登録domainは既存の \(x=\tau/r\)、\(r=1,2\) のまま。

D1の理想providerは \(Q_i=V_i^\dagger Z_iV_i\)、従って \(Q_i^\dagger=Q_i\)、

\[
Q_i^2=I,\quad p_i>0,\quad\sum_i p_i=1,
\quad R=\sum_i p_iQ_i,
\quad\|R\|\le\sum_i p_i\|Q_i\|=1.
\]

これは入力契約とideal unitary構成からの解析的根拠であり、生成したmatrixの固有値や保存残差を使うnorm認証ではない。
丸めた平均演算子自身をこのHermitian contraction \(R\) と取り違えない。

## 2. 非unitaryな有限平均と反復

\(U=e^{-ixR}\)、\(M=P_5(-ixR)\)、\(\delta=|x|^6/720\) とする。

\[
\|M-U\|\le\delta,\qquad\|M\|\le1+\delta.
\]

任意の二演算子についてのtelescoping

\[
A^r-B^r=\sum_{j=0}^{r-1}A^{r-1-j}(A-B)B^j
\]

を使えば、\(M\)をunitaryと仮定せず

\[
\|M^r-U^r\|\le\delta\sum_{j=0}^{r-1}(1+\delta)^j
=(1+\delta)^r-1.
\]

実装平均 \(\widetilde M\) に \(\|\widetilde M-M\|\le e\) があれば

\[
\|\widetilde M^r-M^r\|
\le e\sum_{j=0}^{r-1}(1+\delta+e)^{r-1-j}(1+\delta)^j
=(1+\delta+e)^r-(1+\delta)^r.
\]

独立stepのfirst momentを積へ戻す既存仕様を継承する。相関lawは追加しない。
旧factor2を維持する各axisの保守的biasは

\[
b_g=2\left[((1+\delta_g)^r-1)
 +((1+\delta_g+e)^r-(1+\delta_g)^r)\right]
=2[(1+\delta_g+e)^r-1].
\]

ここで \(\delta_0=3|x|^6/720\)、\(\delta_1=|x|^6/720\)。最後の簡約式だけで保存せず、Taylorと実装の二項も別fieldへ保存する計画。
signalへの係数2を軽くする変更はD3に含めない。

## 3. finite-bit、native/controlとの接続

旧 `rte_development/accounting.py` は

\[
e=8\rho+16(1+\rho)\epsilon_{\rm primitive,account}
\]

を全方式の共通一step上界として使用する。
係数・root/CTS interval丸めの共通上界が \(8\rho\)。各eventは最多2個の合成primitiveを含み、
strict errorのtelescopingが \(2\epsilon_{\rm primitive,account}\)。
既存の係数norm envelope \(B<8\) を保守的に戻した項が \(16(1+\rho)\epsilon_{\rm primitive,account}\)。
このenvelope検査・旧係数認証を新保証で省略しない。

保存eventの正確な有理数weightは \(w_e=a_e/q_e\)、dyadic \(q_e>0\)。
従って \(q_ew_e=a_e\) で、確率丸めそのものをもう一つの平均biasとして二重加算しない。
丸めはmoment・range・acceptanceの旧envelopeへ戻されている。
将来weightを別の有限bit量へ再丸めする実装はこの等式を変えるため、D3に含めない。

正負のRには新しく取得したsequenceとその**actual adjoint**を使用し、
`up_to_phase=false`、Wによる位相、controlled相対位相を保持する。
価格axisは取得精度だけを \(10^{-7}\) に変える。
上式の会計精度は四ledgerすべてで旧 \(10^{-6}\) のまま。新精度によるnative余裕をshotへ返さない。
新sequenceがstrict error \(\le10^{-7}\) なら旧 \(10^{-6}\) 上界にも適合する。

統計残余 \(s_g=\epsilon_{\rm axis}-b_g\) に、同じproduction moment・rangeと旧log上界を用いる。
exact signal、reference normalizer、実測の小matrix残差によるbudget削減はしない。

## 4. 結果前の方向予測と限界

固定native会計 \(e\) の下で \(\delta_1<\delta_0\)、従って \(b_1\le b_0\)、\(s_1\ge s_0\)。
固定moment/rangeのBernstein十分shot数は保証改善だけでは増えない。
整数ceilで同じNになる場合はある。方式間の新N・順位はまだ計算していない。

時間分割では \(x=\tau/r\) なのでleading Taylor寄与は \(r(\tau/r)^6\)。
保証改善の影響はr=1側の方が大きいと予測できる一方、finite-bit/native寄与は反復とともに増える。
この方向だけから最適r、P5/partialの順位を決定しない。

新合成のT価格は単調増加とも一定差とも仮定しない。seedを同じにしてもsequenceは新条件の出力であり、
安い方だけを選ばず、全登録keyの結果を保存する。価格最小性・独立providerでの再現は未検証。

## 5. D2の相殺量を再価格付けへ引き継ぐ根拠

[静的証明記録](../../../artifacts/track_b_d3_preparation/2026-10-13/v1/static_boundary_price_certificate_v1.json)は、
保存r=2の六rowのliteral IRを読み、先頭word集合 \(P\) とinverseを反転順に並べた末尾word集合 \(S\) を作った。
Rを含むwordだけを保持する。長さnのliteral相殺が起きるなら必ずそのwordは \(P\cap S\) にある。
六rowすべてでRを含む交差集合は空。従ってこの保存domainの任意の二step連結において**Rを含む相殺は不可能**。

費用・確率を用いないfinite-word証明であり、219,303 pair、相殺深さ、削減費用を再評価していない。
旧D2に欠けていたdepth×costのjoint traceを回復したとは主張しない。
同じwordとinverse規則を維持すれば、削減は固定literal gateに限られ、新sequence価格には依存しない。
従って旧per-whole-trial savingsを新価格へ適用できる。新Nが変わる場合は**旧総費用の差ではなくper-trial savings**を戻す。

scopeはsymbolic R tokenまでの旧pass。新sequence内部・複数primitiveを展開した後の追加gate相殺、角度融合、
commutation、whole-circuit compilerは認可範囲外であり、その利益を加算しない。

## 6. sourceと検証の限界

静的照合対象：`rte_development/accounting.py`、`pipeline.py`、`boundary.py`、
`rte_reallocation/numeric.py`、`g9_native.py`、`return_aggregation.py` と固定D1/D2契約。
全original bytes identityは[入力照合記録](../../../artifacts/track_b_d3_preparation/2026-10-13/v1/saved_inputs_identity_v1.json)で参照。

一般の積分剰余は上記の解析的証明。人工scalar testsは積分係数・telescoping・native項を落とさない会計の実装確認であり、
登録科学計算や一般定理の代替ではない。入力domain拡大、最終総資源優位、新アルゴリズム採択は行わない。

**mandatory STOP。新保証による登録shot/resource再会計と新合成は別明示認可まで開始しない。**
