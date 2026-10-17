# 有限step相対形式保証：独立数学的技術監査

2026-10-17 JST。対象は④の未採択の保証理論候補。紙上の独立検算であり、形式証明器・数値・symbolic checkerによる検証ではない。

**判定：`VALID_WITH_UNRESOLVED_NOVELTY`。** 下記の有限次元・PSD・正stage対称S2・解析的log・exact ground・読出し条件の下で、RF0、RF1、EW、固定unitary QPEへの接続は成立する。定数32と512、K0の行和定数の再利用を独立に追跡した。中心命題に重大な破綻は見つからなかった。原レビューの補助比較式(29)には閾値の適用域の補足が必要である（§7）。新定理の正式受理・新規性・実用優位は認定しない。

入口は[判断用ハンドオフ](decision_handoff.md)、文献・費用は[対応表](prior_art_and_practical_conditions.md)、原本identityは[証拠索引](evidence_index.json)。[受領レビュー原本](inputs/qc_track_iv_research_question_redesign_review_20261017.md)は38,606 bytes、SHA256 `954bd40d1da7dce70a4b97336c71d1cd55031314df9c339a2de95b82fabf8527`。§4–10・12を含め全文を確認した。[今回の指示](inputs/user_instruction_relative_form_audit_20261017.txt)もbyte保存する。

## 1. 対象と共通仮定

有限次元の**共通不変sector**で

\[
 H=LI+G,\qquad G=\sum_{j=1}^M F_j,\quad
 0\preceq F_j\preceq f_jI,\quad \Lambda=\sum_j f_j.
\]

同じ分割について認証されたscalar Lと上界Uを用い、`e=E0−L≥0`、`W=U−L≥e` とする。norm・順序・groundはすべてこのsectorのもの。各logical fragmentがsectorを保存する必要がある。native gate一個ずつのsector保存までは要求しない。単に全Hが対称性を持つだけでは不十分である。

表示順の積

\[
 S_2(z)=e^{-izF_1/2}\cdots e^{-izF_{M-1}/2}
 e^{-izF_M}e^{-izF_{M-1}/2}\cdots e^{-izF_1/2}
       =E_J(z)\cdots E_1(z)
\]

を定義する。`E_l=exp(−izA_l)`、`A_l≥0`、`A_l≤a_l I`、`ΣA_l=G`、`Σa_l=Λ`、`J=2M−1`。半stageのcapも半分とする。零fragmentを含めてよい。以下は負stageを含む高次PFへの保証ではない。

`Λ=0` なら全Fj=0、H=LI、S2=I、K=0であり、`1/Λ` を使う議論から除外する。h=0ではKは連続延長でGだが、QPEのenergy復元でhで割らない。

## 2. RF0：複素logと二端telescopingからの再導出

### 2.1 logの定義域と対称性

有限次元の積の展開から

\[
 \|S_2(z)-I\|\le e^{|z|\Lambda}-1.
\]

従って `|z|Λ<log 2` で `X(z)=S2(z)−I` のlog級数

\[
 \operatorname{Log}(I+X)=\sum_{k\ge1}(-1)^{k+1}X^k/k
\]

がnorm収束する。これは原点から連続な解析的branchであり、他の2π branchを任意に選ぶ自由はない。`K(z)=i Log S2(z)/z` はz=0で可除、K(0)=G。

S2のpalindromeを逆転して `S2(−z)=S2(z)^{-1}`。原点近傍でlogの符号が反転し、解析接続によりこの円板全体で `Log S2(−z)=−Log S2(z)`。従ってKは偶関数である。実hではS2はunitaryであり、このlogはanti-HermitianなのでK(h)はHermitianである。

以下ではCauchy円 `r=1/(4Λ)` を使う。`x=rΛ=1/4`、`e^x≤1/(1−x)=4/3` より `q=e^x−1≤1/3<1`。円周を含めlogは解析的である。この1/4は最適収束半径の主張ではない。

### 2.2 任意vectorに対するPSDの使い方

任意の単位vector vに対し `E_v=〈v,Gv〉` とする。`A_l²≤a_l A_l` とCauchy–Schwarzにより

\[
 \sum_l\|A_lv\|\le
 \sqrt{\Big(\sum_l a_l\Big)\Big(\sum_l\langle A_l\rangle_v\Big)}
 =\sqrt{\Lambda E_v}.
\tag{A1}
\]

`B_l=E_l−I=−iz∫_0^1 exp(−itzA_l)A_l dt` なので、`|z|=r` で

\[
 \|B_lv\|,\ \|B_l^\dagger v\|\le r e^{ra_l}\|A_lv\|,
 \qquad |\langle B_l\rangle_v|\le r e^{ra_l}\langle A_l\rangle_v.
\tag{A2}
\]

最後の期待値評価はA_lの非負スペクトル測度を積分して得る。単なる `||v|| ||A_l v||` ではこの一次のenergy依存は得られない。adjointには共役z・符号の違いがあるが同じnorm評価が使える。

一端telescopingは正確に

\[
 X=\sum_j E_J\cdots E_{j+1}B_j.
\tag{A3}
\]

であり、右端は未回転のvへ作用する。逆順adjointにも同じ恒等式を使うと

\[
 \|Xv\|,\ \|X^\dagger v\|\le r e^x\sqrt{\Lambda E_v}.
\tag{A4}
\]

### 2.3 非可換な中間作用素を消さない二端恒等式

必要な恒等式は

\[
 X=\sum_jB_j+
 \sum_{i>j}B_iE_{i-1}\cdots E_{j+1}B_j.
\tag{A5}
\]

独立な確認：`∏(I+B_l)−I` の非空の語を展開する。singletonは第一和。二個以上のBを含む語には最大添字iと最小添字jが一意に存在する。内部のBの有無をすべて足すと、順序を保った `E_(i−1)…E_(j+1)` になる。よって各語をちょうど一回数える。全stageの可換性も、回転されたvectorを元のvへ置換する仮定もない。

各pairに(A2)を両端から使い、内部積は `exp(rΣ_(j<l<i)a_l)` で抑える。端点を含め指数はe^x以下である。従って

\[
 |\langle X\rangle_v|
 \le r e^x E_v+r^2e^x\sum_{i>j}\|A_iv\|\|A_jv\|
 \le r e^x(1+x/2)E_v.
\tag{A6}
\]

最後は `Σ_(i>j)t_i t_j≤(Σt_i)²/2` と(A1)。またk≥2に対し

\[
 |\langle X^k\rangle_v|
 \le\|X^\dagger v\|\|X\|^{k-2}\|Xv\|
 \le r^2e^{2x}\Lambda E_v q^{k-2}.
\tag{A7}
\]

### 2.4 Cauchy定数の検算

log級数へ(A6),(A7)を入れ、zで割ると

\[
 |\langle K(z)\rangle_v|
 \le\left[e^x(1+x/2)+xe^{2x}
 \sum_{k\ge2}\frac{q^{k-2}}k\right]E_v.
\]

`Σ_(k≥2)q^(k−2)/k≤1/[2(1−q)]≤3/4`。`x=1/4` と有理数上界だけで括弧は

\[
 \frac43\frac98+\frac14\frac{16}9\frac34
 =\frac32+\frac13=\frac{11}6<2.
\tag{A8}
\]

これは数値代入実行・数値テストではなく紙上の定数評価である。偶解析関数を

\[
 \langle K(z)\rangle_v=E_v+\sum_{n\ge1}c_{2n}(v)z^{2n}
\]

と書くと、Cauchyにより `|c_(2n)(v)|≤2E_v/r^(2n)`。実 `|h|<r` で

\[
 |\langle K(h)-G\rangle_v|
 \le\frac{2(h/r)^2}{1-(h/r)^2}E_v
 =\frac{32(h\Lambda)^2}{1-16(h\Lambda)^2}E_v.
\tag{RF0}
\]

これは全単位vに一様で、K(h)−GはHermitian。従って
`−η0 G≼K(h)−G≼η0 G` となる。Gの逆行列は使わない。E_v=0なら全A_l v=0、S2 v=v、K v=0であり、零空間も含む。縮退の有無は関係しない。η0<1を必要とするEWにはさらに `|h|Λ<1/√48` が十分であり、単なる1/4領域と区別する。

## 3. RF1：K0の何を再利用できるか

[固定K0 §4 Eqs.(5)–(9)](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md)の辞書・行和定義そのものを使う。ground状態方程式を要するK0二分割の符号評価や、`〈[H,A]〉=0` は、以下の全vector評価に持ち込まない。

### 3.1 三次係数と符号の独立な照合

`exp(X/2)exp(Y)exp(X/2)` の次数1はX+Y、次数2は(X+Y)²/2。積の次数3ではX³,Y³の係数1/6、X²Y,YX²は1/8、XYXは1/4、XY²,Y²Xは1/4、YXYは0である。logの三次係数はここから `(X+Y)³/6` を引いたものなので

\[
 \log(e^{X/2}e^Ye^{X/2})
 =X+Y-\frac1{24}[X,[X,Y]]+\frac1{12}[Y,[Y,X]]+O(5).
\]

X=−ihA、Y=−ihBを入れてi/hを掛けると
`D_ABA=[A,[A,B]]/24−[B,[B,A]]/12`。外側wrapperの再帰では内側Dがこの次数で加算され、交差は高次へ入る。従って

\[
 D=\sum_{j<M}\left\{[F_j,[F_j,R_j]]/24-[R_j,[R_j,F_j]]/12\right\},
 \quad R_j=\sum_{k>j}F_k.
\tag{A9}
\]

raw辞書は `κ_jjk+=1/24 (k>j)`、`κ_klj+=−1/12 (k,l>j)`。実κとHermitian fragmentの二重交換子なのでDはHermitian。Gの単位をenergyとするとDはenergy³、K_SOSはenergy²であり、h²K_SOSは無次元である。

### 3.2 ground固有方程式なしの全vector評価

任意単位ψに対し `φ_j=〈F_j〉`、`σ_j²=〈F_j²〉−φ_j²≤f_jφ_j`、`δF_j=F_j−φ_jI`、`w_j=||δF_j||≤f_j`。定数shiftは交換子を変えない。四つの語を展開し、両端のδFをvectorへ当てると

\[
 |\langle[F_a,[F_b,F_c]]\rangle|
 \le2(\sigma_a\sigma_c w_b+\sigma_a\sigma_b w_c)
 \le2f_b\sqrt{f_af_c}\sqrt{\phi_a\phi_c}
       +2f_c\sqrt{f_af_b}\sqrt{\phi_a\phi_b}.
\tag{A10}
\]

K0と同じく、各raw項について対称で成分ごとに非負の行列Bへ（B自体がPSDという意味ではない）

\[
 |\kappa_{abc}|\{f_b\sqrt{f_af_c}(e_ae_c^T+e_ce_a^T)
                 +f_c\sqrt{f_af_b}(e_ae_b^T+e_be_a^T)\}
\tag{A11}
\]

を加える。添字一致時も表示どおり加算する。`x_j=√φ_j` とおけば

\[
 |\langle D\rangle|\le x^TBx\le\|B\|_2\|x\|^2
 \le K_{\rm SOS}\langle G\rangle,
 \qquad K_{\rm SOS}=\max_a\sum_bB_{ab}.
\tag{A12}
\]

実対称Bは `||B||2≤√(||B||1||B||∞)=maxrowsum(B)`。φ取得や固有値計算は不要である。全ψで成立しDがHermitianなので `−K_SOS G≼D≼K_SOS G`。これは旧K0のgroundに書かれた評価を、今回任意vectorで検算したもの。旧資料にLoewner形式が明記済みだったという主張ではない。

### 3.3 定数512と共通領域

K(h)=G+h²D+… のn≥2だけを§2.4のCauchy評価で足すと

\[
 |\langle K-G-h^2D\rangle_v|
 \le\frac{2(h/r)^4}{1-(h/r)^2}\langle G\rangle_v
 =\rho_4(h)\langle G\rangle_v,
 \quad\rho_4(h)=\frac{512(h\Lambda)^4}{1-16(h\Lambda)^2}.
\]

従って `η1=h²K_SOS+ρ4` でRF0と同じ相対形式が成立する。共通領域は `|h|Λ<1/4`。両ηは同じG・同じS2・認証済みの同じcapに対するscalar上界なので `η=min(η0,η1)` も使える。これは異なる作用素の「最小値」を取る操作ではない。両ηはこの領域で|h|に単調である。

旧K0 §5.2の有限剰余はscalar normの `r4 I` であって `ρ4 G` ではない。後者は今回のRF0証明を使って初めて閉じる。K0の非縮退固有枝展開・resolvent・gap依存overlapの主張を、そのままgap-free化したわけでもない。

## 4. EW：個別固有枝でなくスペクトル出力の窓

`−ηG≼K−G≼ηG`、`0≤η<1` を仮定する。eはGの最小固有値であり `K≥(1−η)eI`。Gのground空間に支持を持つ任意の純粋・混合state ρでは

\[
 \mathbb EY=\operatorname{Tr}(\rho K)\le(1+\eta)e,
 \qquad Y\ge(1-\eta)e.
\]

YはKの**スペクトル測定値**である。`Z=Y−(1−η)e≥0`、`EZ≤2ηe` なので、d>0に対するMarkov不等式は

\[
 \Pr(Y>e+d)\le\frac{2\eta e}{d+\eta e}.
\tag{EW}
\]

下側tailは別に、`d≥ηe` なら `Pr(Y<e−d)=0`。上側だけの式をそのまま無条件の両側集中と呼ばない。

`0<α_spec<1`、e≤Wと

\[
 \eta W\le\frac{\alpha_{\rm spec}}{2-\alpha_{\rm spec}}d
\tag{A13}
\]

を仮定すると、ηe<dかつEW右辺≤α_spec。従って `Pr(|Y−e|>d)≤α_spec`。η<1は別途必要である。Wだけに関する(A13)をもって省略してはいけない。原レビュー§8はη_target<1を明示しており、この点は欠陥ではない。

ground縮退を許す。特定のK固有stateへの高overlapは要求しない。一方、一般の近似stateに同じeの平均上界を無条件で使えない。ground空間のstateからのtrace distance `τ=||ρ−ρ_g||1/2` が証明済みなら測定失敗確率へτを加えられるが、低い変分energyだけではgapなしにそのτを得られない。e=0は全Fjがgroundを消しKも消す自明なcase。W=0ではE0=Lが既知である。

## 5. 固定S2のQPE：branch・読出し・絶対原点

η<1のとき `0≤K≤(1+η)ΛI`。RF0/RF1の領域と合わせれば `|h| ||K||<1/2` rad。log branchは定まるが、0/2π境界で丸めたQPE出力をunsigned energyへ機械的に直すことまでは正当化しない。

明示的な十分な処理の一例として、既知scalar `c=(1+η)Λ/2` を使う。h>0を固定し、controlled scalar phaseを含む `exp[−ih(K−cI)]` の位相θ=−h(y−c)を読む。支持は `|θ|≤a=h(1+η)Λ/2<1/4`。一様な円周読出し誤差 `δθ=h ε_read<π−a` の成功時にはprincipal branchが切断されず `yhat=c−θhat/h` と復元できる。これは紙上のbranch条件の具体化であり、実装・実行仕様を新設したものではない。

**同じ固定unitaryの整数powers**を実装する。roundごとに別h、別PF、独立乱択kernelへ置換して同じスペクトルを測るとは主張しない。理想QPEはK固有basisのスペクトル混合である。各固有値に一様に `Pr(|yhat−y|>ε_read)≤α_read` が成り立つなら、EWとの和事象評価から

\[
 \Pr(|\widehat E-E_0|>d+\varepsilon_{\rm read})
 \le\alpha_{\rm spec}+\alpha_{\rm read},\qquad
 \widehat E=L+\widehat y.
\tag{A14}
\]

独立性は不要である。状態が複数のK固有空間へ広がっていること自体は障害にならない。Lは一度だけ復元する。既にcontrolled exp(−ihL)を含めた実装なら、その既知shiftのunwrapをし、二重加算しない。

(A14)は理想条件付きの保証。別途認証された状態trace distanceτ、全controlled powersにわたる実装の出力確率誤差τ_impl、証拠誤り確率α_certがあれば失敗予算へ加える。係数誤差・モデル差・合成誤差はその根拠となるnorm/diamond等の台帳を別に持つ必要がある。これらの数値や配分は今回決めていない。多round RPEやM1-Sの平均複素信号への自動拡張はしない。

## 6. stepとqueryの式の検算

Λ>0、`0<η_target<1` としRF0を反転すると

\[
 0<h\le\Lambda^{-1}\sqrt{\frac{\eta_{\rm target}}{32+16\eta_{\rm target}}}
\]

が十分であり、自動的にhΛ<1/4。RF1の単調上界でも(A13)を満たす十分stepを選べる。真のeやground vectorはstep式の入力ではないが、exact ground copiesの用意はQPEの別前提であり無料実装ではない。

一定の読出し失敗確率における通常QPEのbase-S2 queryは `O(1/(h ε_read))`。powersの呼出しを一回と数えるoracleモデルではなく、その中のbase適用数を数える。W,Λ>0固定、dとε_readを総精度εに比例配分し、小さいε/Wの漸近域で上記十分stepを取れば

\[
 h=\Theta(\Lambda^{-1}\sqrt{\varepsilon/W}),\qquad
 Q=O(\Lambda\sqrt W\,\varepsilon^{-3/2}).
\]

単位も整合する。これは十分設計の依存性であり、最適性、強い既知保証に対する改善、native gate削減ではない。ΛやWが系サイズと変わる場合、固定量と呼んで消してはいけない。

## 7. 発見した補足事項と依存関係

原レビュー§9.3式(29)は、既存vector残差から

\[
4\,\mathbb E\sin^2[h(Y-e)/2]
=\|(S_2(h)-e^{-ihe})|g\rangle\|^2
\]

を用いてtailを比較する。sineを閾値で下から抑えるには、非自明なeventの範囲で **`0<|h|d≤π`** と支持上の `|h(Y−e)|≤π` が必要である。原記述の「d>0」だけでは、例えば `|h|d=2π` で分母が0となり式をそのまま読めない。閾値が支持差全体より大きいならeventは空であり確率0として別扱いする。h=0も除外する。

これは**補助比較式の適用域の書き足し**である。RF0、RF1、EW、(A14)、step反転はこの式を使わないため影響されない。同式や旧粗い残差を強対照すべてに対する優位の証拠にはしない。数値反例・別定理の探索・救済実装は行っていない。

依存関係は `PSD＋正stage＋palindrome → RF0 → 相対四次剰余`、別に `K0辞書＋全vector分散 → Dの相対形式`、両者からRF1、`相対形式＋η<1＋ground支持 → EW`、`EW＋固定powers＋branch/readout/実装台帳 → QPE`。大域的な新規性・実費用の主張はこの依存列だけからは出ない。

**Mandatory STOP。** 本監査でA1 CLOSED、A2 NOT_ADOPTED、K1-F未承認・未実行は変更しない。新規科学計算、symbolic checker、科学SOURCE、実行契約、authorization、grant、markerは0件である。
