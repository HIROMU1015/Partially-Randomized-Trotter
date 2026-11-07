# CT-3A：停止性・strict誤差・実出力T上界の条件

固定参照 `6b0b6669bbf58ef22e789bfef134a1a6da9f2343`、2026-11-07 JST。[REPORT](REPORT.md)、[候補比較](FALLBACK_POLICY_CANDIDATES.md)。以下の数式は静的な条件整理であり、word生成・数値実験・価格評価ではない。文献で成立する性質、SOURCEで確認した性質、Codexの導出を明示する。

## 1. 入出力と現backendの限界

入力thetaはcompiler後のliteralな**有限binary64**を正確な有理数として解釈する。targetは `RZ(theta)=diag(exp(-i theta/2),exp(i theta/2))`。precisionは既存の十進指数等、有限に記述される正の計算可能数。非計算可能な実数を機械入力とする議論には広げない。NaN/Infは対象外で、到達SOURCEの有限性は旧資格を参照する。

必要な出力は、Wを含む実word s、T/T†数、phase表現、strict guard上界、科学的な取得provenance。保証は `||s-RZ(theta)||2≤epsilon_G`、全入力で取得完了、選ばれたsに対する `T(s)≤B(epsilon_G)`。存在する別wordの価格では足りない。

[track_c_h4_synthesis.acquire/strict_guard](../../../src/trottertracks/track_c_h4_synthesis.py)を静的に再確認した。固定backend7 SOURCEの同一性は[CT-2既存照合](../pr_reference_ct2_20261107/STRICT_SYNTHESIS_BOUNDS.md)を再利用し、再hash監査していない。

| 箇所 | 保証される範囲／不足 |
|---|---|
| key/cache | abs(theta).hex・10^exponent・backend2.0.0・strict・seed17。cache hitはJSON読取後に即returnするため、その経路には新規guard/長さ拒否の再実行がない |
| 新規取得 | exact binary角、seed17、up_to_phase=False、backend要求epsilon_G/4 |
| backend探索 | [固定gridsynth.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/gridsynth.py#L245)は分母kを増やす外側探索。全入力の有限終了上限は確認できない |
| 内側norm処理 | 有限attemptがNO_SOLUTIONとなる経路は数学的不可解の証明ではない。内側予算だけで外側完了を保証しない |
| timeout | 既定未指定。仮に内側timeoutを設定しても、全体への有限遷移証明は別 |
| guard | exact cyclotomic wordとexact binary targetを外向きFrobenius区間で比較。phase fitはしない |
| 再guard | 初回失敗時に同じwordを1回高精度で確認する。新しい取得や必ず成功する再帰fallbackではない |
| 拒否 | nonfinite/guard不成立/長さ>20,000なら失敗保存と例外。backendが返る前には検査できない |
| 価格 | 新規の受理後ならT≤len≤20,000。未取得wordの存在・取得時間・全域価格にはならない。cacheにも個別前提が要る |
| phase出力 | strict経路の明示Wをexact行列に含める。backendのprojective diagnosticや別phase fieldはguardの代用品ではない |

したがって「pygridsynthが失敗したらfallback」は、例外にならず返答もしない入力を扱えていない。**返却時correctness、guard完了、探索完了は別々の命題。** 有限に成功した旧CT-1の結果を否定するものではない。

## 2. strict SU(2)生成集合：Codexの記号的導出

SKの前提をprojective普遍性だけで満たしたことにしない。[Selinger v2 Remark26](https://arxiv.org/pdf/1212.6253v2)のstrict近似可能性を使い、以下の有限指数kernelの通常の書換えを適合性証明として整理する。実角度への合成や生成word catalogueの構築ではない。

omega=exp(i*pi/4)、W=omega I、物理gate alphabetを `Sigma={H,T,T†,W,W†}` とする。detの指数dをmod8で記すと

\[
d(H)=4,\ d(T)=1,\ d(T^\dagger)=7,\ d(W)=2,\ d(W^\dagger)=6.
\]

coset代表t_k=T^k（0≤k≤7）を用い、kとa∈Sigmaに対し

\[
r=(k+d(a))\bmod8,\qquad
g_{k,a}=t_k\,a\,t_r^{-1}.
\]

各gはdet=1。これらとその**actual adjoint**を生成集合Gammaとする。元のdet=1 wordを左から読み、prefixのdet指数をkとして各因子を置き換えると、隣接t_k^{-1}t_kが消え、末端t_0=Iとなる。よってGammaは元Clifford+T群のdet=1 kernelを生成する。

strict近似可能性から、任意U∈SU(2)へ近づく元gate word列が存在する。det値は有限離散集合なので、十分近い列のdetは1であり、kernelはSU(2)に密。したがってGammaはstrict SU(2)の有限・inverse閉包・密な集合というSKの前提を満たせる。Wを除いてSO(3)へ落とした推論ではない。

各macroはT/T†を高々 `7+1+7=15` 個使って元alphabetへ展開でき、adjointも同数。これは**記号展開の粗い計数上界**で、新AのT価格の数値評価ではない。H/WをT数0としても、位相のW自体は消さない。将来、基底netのword長がGammaに関して高々ell0と証明されれば、`t0≤15 ell0`という上界を得られる。今回はell0を探索・取得していない。

## 3. exact-real SK定数：Codexの十分条件の導出

[Dawson–Nielsen v2](https://arxiv.org/pdf/quant-ph/0505030v2) §4.1のbalanced commutatorと§4.2 Lemma1／式(14)を用いる。以下の**c=12、eta0≤1/256は今回の代数的十分条件**であり、文献の定理として引用した値でも、採択した実装precisionでもない。独立科学レビューは未実施。

Delta∈SU(2)、||Delta-I||2=e_actual≤e≤1/16とする。e_actual=0ではV=W=I。e_actual>0では、直交軸の等角phiの回転V/WのcommutatorをDeltaに共役させる。小角branchでcommutatorの半traceは `1-2sin^4(phi/2)` なので、

\[
e_{\rm actual}=2\sin^2(\phi/2),\qquad
\delta^2:=\|V-I\|_2^2=\|W-I\|_2^2
=\frac{e_{\rm actual}}{1+\sqrt{1-e_{\rm actual}/2}}\le e_{\rm actual}\le e.
\]

共役はこの距離を変えない。V/Wの近似誤差が各e以下ならLemma1から（eは実残差と再帰近似誤差の共通上界）

\[
\begin{aligned}
\|[V,W]-[\widetilde V,\widetilde W]\|_2
&\le8e\delta+4e\delta^2+8e^2+4e^3+e^4\\
&\le8e^{3/2}+12e^2+4e^3+e^4\\
&<12e^{3/2}\quad(0<e\le1/16).
\end{aligned}
\]

最後はe^{3/2}で割った右辺の上界が `8+3+1/16+1/1024<12` となることによる。これはexactな不等式の整理で、サンプル数値や格子の最大値ではない。残差Delta=U U_prev†にこの構成を使えばidealなSK更新の誤差は高々12e^{3/2}。

**条件付き上界**：実wordから成るstrict全SU(2)基底netが半径0<eta0≤1/256を保証し、最大T数t0が既知で、すべての副演算をexact-realとして行える場合、

\[
\eta_n=12^{-2}(12^2\eta_0)^{(3/2)^n},\qquad
n=\min\{j\ge0:\eta_j<\eta_{\rm aim}\},\qquad
T(s_n)\le5^n t_0.
\]

144 eta0<1なので有限nが存在する。wordは `V_prev W_prev V_prev† W_prev† U_prev` の5因子であり、実際にこのwordを構成する場合のT上界である。最適wordの存在上界を別policyの返却価格に貼る推論ではない。

最小nを浮動小数log式の無条件ceilだけで決めず、境界では十分条件を厳密に確認できる有限記述が要る。現状はeta0/t0の実証拠と有限精度再帰がないので、式は**実装された全support capではない**。保存の10^-9や約10^-16域へn/Tを数値代入していない。

## 4. 初期netと有限精度で残る証明

### 初期net

必要なのは、実word、exactなSU(2)行列、全域coverage、eta0、最大T=t0が対応する有限表である。保存RZ-only wordのguardだけで代替できない理由は[候補文書](FALLBACK_POLICY_CANDIDATES.md)。有限成功例・最近傍距離の経験的最大・SO(3) coverageはstrict SU(2) certificateではない。

例えばSU(2)を `aI+i(bX+cY+dZ)`、q=(a,b,c,d)∈S³として表せば、strict spectral距離は

\[
\|U(q)-U(q_j)\|_2^2=2-2q\cdot q_j.
\]

したがって必要なcoverage命題は `forall q∈S³, exists j: q·q_j≥1-eta0²/2`。これは資格すべき有限表の数学的条件例で、実際のcertificateを生成したものではない。±qを同一視しない。各q_jのexact wordとT価格を失わず、境界・丸めも覆う証明が必要である。

### 有限精度再帰とguard

exact-realのc=12は、有限精度のcommutator分解、軸/平方根のbranch、near-identityの処理、数値最近傍、残差演算の誤差を含まない。副手順の全入力停止と、追加誤差を含めても縮小する再帰式を別途証明する必要がある。単にmpmathのdpsを増やす設定や、固定回数のテストでは埋まらない。

repo guardは2×2 Frobenius上界なので、理論spectral目標etaとguard許容epsilon_Gに `sqrt(2)eta<epsilon_G` の余裕を設ける必要がある。さらに有限精度再帰の誤差余裕も残す。丸めを含む最終wordがguard許容より**strictに内側**にある証明と、区間評価がその余裕を有限精度で識別する保証があれば、guardの適応精度検証を停止論証へ組み込める。境界で等しいwordを永久に再guardする方法はtotal policyではない。

現在のexact_word_matrixとstrict_guardは返却wordの事後検証に適合するが、「何桁で必ずPASSとなるか」や「現行1回retryで全入力PASS」は未証明。大角thetaのexpを計算する場合もbinary角のexact入力を維持し、無条件の2pi縮約・piへのsnapはしない。実装が有限precision/word長の限界で拒否するなら、その受理領域に限定した`PARTIAL_POLICY`とし、全support保証とは呼ばない。

## 5. 複合policy全体の停止性

以下は設計条件の比較で、timeout秒数・iteration値・CPU制限・実行方式を採択/実装したものではない。仮にfallback Fが全対象で有限に返り、guardを満たし、実出力T≤B(epsilon_G)と証明できたとする。Pは固定pygridsynthの未保証の取得経路。

| 構成 | 完了するための条件 | T上界を保つ選択 |
|---|---|---|
| Fを先に実行 | F完了後、そのwordを直ちに返せる。Pの改善を無期限に待たない | Fを採用。Pも使うならguardを通りT(P)≤T(F)等のcapを満たす場合だけ |
| Pを有限探索してからF | Pへ与える有限の論理work量だけでなく、その内側呼出しから必ず戻る制御が必要。上位kループだけの上限では足りない | P成功でもguard＋T(P)≤Bを要求。超過/失敗/打切りならF |
| 独立にFとPを扱う | Fが必ず進行し、P待ちに依存しない。Pが返らなくてもFの返却経路・後始末が塞がれない | 先着というだけでは不可。Pにguard＋capを課し、それ以外はFを待つ |

thread timeoutでC/内部呼出しの完了まで保証したことにはならない。別process等を検討する場合も、起動・progress・IPC・停止/隔離・cleanupの前提を明示し、自分のchildに限定する必要がある。OSや他ユーザーのprocess制御を変更して保証を作らない。返却後に無期限joinするなら、APIの返答とwhole processの停止は別であり、全体停止の命題には未解決を残す。今回は実際のbackend起動・強制停止・raceを行っていない。

[CT-1 stage_b](../../../src/trottertracks/pr_reference_ct1.py)の `next(as_completed(running))` は完了まで待ち、例外経路も `pool.shutdown(wait=True, cancel_futures=True)` を使う。このままのworker controllerは、別workerのbackendが返らない場合の有限遷移を保証しない。CT-1の有限成功結果は維持しつつ、total fallback controllerとしてはそのまま流用できないと区別する。

### 実際の採用wordにcapを課す

guard、spec/入力、phase規約、T数を資格した候補だけを選ぶ。FとPの両方を持つ場合にT最小の正しい候補を選べば、`T(selected)≤T(F)≤B`。Fがまだ完成していない間にPを返す場合にも、PがB以下である証拠が要る。**短いfallbackの存在だけでは、長いPを採用するpolicyのcapにならない。**

cache hitも同じ条件。保存wordの採用規約に適合するguard記録・key・precision・実word hash/phase・T数を利用し、guard方式未記載のlegacy recordを自動資格しない。既存の20,000長さ拒否を新fallbackへ無条件に移すと、証明付きでもそれより長いwordを拒否してtotalityを失う可能性がある。変更を要する場合は新policyの固定条件として判断を戻し、旧SOURCE/記録は変更しない。

## 6. emitter・符号・全supportへの移送

[lower_native](../../../src/trottertracks/track_c2_b_verify.py)はwritten matrix productを逆順にemitし、Wをglobal phaseに加え、負角では**得たwordのactual adjoint**を取る。SX=HSH、CX/Xも旧規約。局所RZ(-theta)=RZ(theta)†なので、負角の誤差は同じ。ただしforward SOURCEをcompileしたnativeの逆と、inverse SOURCEを別compileしたnativeが一致するとは仮定しない。各literal native targetを覆う。

WはT metricでは0でもexact演算子の一部。joint回路へ入った後の全体共通phaseと、branch/control間の相対phaseを混同せず、新しいcontrolを合成wordへ後付けしない。Qiskitの浮動global phase診断だけでexact Wの証明を置き換えない。現lower_nativeだけではresetやreadoutを扱わないため、[CT-1のwhole置換/測定規約](../../../src/trottertracks/pr_reference_ct1.py)と合わせて再利用条件を確認する。

保存F/G、逐次vacuum_phase、event pi加算、compiler AによるP→RZのliteral角規約は[CT-2全support対応](../pr_reference_ct2_20261107/FULL_SUPPORT_PRICE_APPLICABILITY.md)を継承する。Referenceのvacuum Pは1shotに1個。全実角に一様なstrict F/Bが成立すれば、その到達subsetにも同じ保証が成立し、直接全trajectoryを列挙する必要はない。現状はそのF/Bが未資格。

B′/Referenceでprecision・価格取得/選択機会を揃え、他roleの未知価格を落とさない。各trajectoryでの局所誤差のtelescoping後に

\[
\epsilon_{CT}\le\sum_j\epsilon_j,\quad
d_{syn}\le2\sqrt2\epsilon_{CT},\quad
d_{total}=d_{old}+d_{syn}
\]

として、同じjoint/reset/測定のraw diskへ一度だけ加える。既存compiler/vacuum算術diskを重複させず、`rho_lower=rho0-d_total`、`beta_stat=.4-beta_PF-asin(d_total/rho0)`と保存の統計/alias条件へ接続する。期待gate数をpointwise誤差上界にしない。今回は新しいshot数/precision/alpha/価格を評価しない。

## 7. 上界の有用性と未資格事項

資格済みBが存在すれば、unknown roleの各出現にpointwiseでBを適用し、期待multiplicityでT上界を会計する余地がある。1個のvacuum Pだけなら保存Nを用いた `sum_(m,x) N_(m,x) B(epsilon_(m,x))` がその寄与の上界となる。ただしReference全体価格ではない。新しい融合を入れない物理出現と採用wordの対応が必要。

SKの5^n増幅は保守的で、基底netも未知であるため、上界が実費比較に有用かは未判定。存在/停止証明に役立つことと、新Aの優越を判別できる価格になることを分ける。今回は価格の桁も数値化していない。Reference上界とB′上界の大小では実費順序を確定せず、資格済み下界/実価格と比較する。期待Tは有理会計で、整数ceil/gcd下界を無条件に移さない。

必要な証拠は、**初期netの有限coverage証明、実word/eta0/t0との対応、有限精度再帰の停止/誤差証明、guardの余裕と完了、controllerの有限遷移、実採用wordへのcap**。これは科学的correctnessの証拠であり、ユーザーの承認を認証する仕組みではない。新規net・合成・実装が必要となる地点で**mandatory STOP**。
