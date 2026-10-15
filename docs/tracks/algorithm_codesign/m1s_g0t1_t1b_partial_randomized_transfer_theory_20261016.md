# M1-S G0-T1／T1b：部分乱択への解析的移送・適用域・共鳴の限界

2026-10-16 JST。**`T1B_ANALYTIC_TRANSFER_SUBMITTED_WITH_DOMAIN_CHECKPOINT2_PENDING_MANDATORY_STOP`**。

**固定有限次元・固定r・小step・有限減衰予算という正式契約の系列で、全Hに基づくC-sec複素信号への一様上界を示す。** 単一pair-sum対象blockなら先導減衰とordinary参照との係数比も導ける。一方、孤立H固有stateだけではscalar減衰を保証できず、解析的な共鳴例を示す。

本書はユーザーが明示承認した**T1bの著者による解析的証明の提出**である。Checkpoint 2の独立科学的受理・新規性認定・全M1-SのGOではない。新科学数値計算、数値行列評価・対角化、sampling、simulation、回路、compile、費用計算は全て0件。[Checkpoint 2資料][CP2]と共に提出して停止する。

## 1. 承認・固定資料・既存成果の扱い

ユーザー提供「Track B／M1-S G0-T1／T1b部分乱択への理論的移送検証」はT1bの新規数学的導出・証明・反例・資料公開を明示承認した。[今回の承認記録][AUTH]と原本を別namespaceに保存する。[Checkpoint 1 GPT原本][GPT1]はbyte copyで固定し、T1aの限定数学的受理とT1b推奨を記録する。T1b実施許可の主体は今回のユーザー指示であり、GPTの推奨そのものではない。

| 固定資料 | commit | 用途 |
|---|---|---|
| [T1a全証明][T1A]／[Checkpoint 1資料][CP1] | `4e7b18dd2aa14c127bce9e248c63996c62a055cc` | 散逸性・平均化構造の基点。定数18は転用しない |
| [S0正式契約][S0] | `373b8e4b9831f4d00bdc00ffc741ac61e03f0841` | §§8–14、17のlaw・joint path・停止条件 |
| [Checkpoint 1 GPT原本][GPT1] | `4899608ea222777e64e706fda2b7e54b84c86247` | 紙上受理の範囲、未認証事項、T1bへ渡す問い |
| [旧G0候補量の記録][G0] | `0f0952614dcf5c06d3d63c21b5d49091b8ecce37` | §10のrho^(H)候補を取り違えない |

固定GitHubの必須3資料は再取得してlocal/Git blob/bytesを照合した。[証拠索引][INDEX]。旧T1a・S0・原本レビュー・結果・classification・authorization・marker・SOURCEは書き換えない。以下の新しい結果は**今回の著者主張**であり、過去のpendingを遡及して消さない。

## 2. 規約・正確な平均写像

有限次元 `n=dim H`、`H=H_D+H_R`、`H_R=lambda h`、`h=E P_j`、lambda=lambda_R>0、`P_j=P_j^dagger`、`P_j^2=I`、固定確率lawを使う。全Hの正規化された孤立固有stateを `H|k>=E_k|k>` とする。hの固有stateとは仮定しない。

δ=delta_rel、固定整数r≥1、

\[
c=\frac{\lambda}{2r},\quad\theta=c\delta,\quad\alpha=\delta/4,
\quad\beta=rc^2=\frac{\lambda^2}{4r},
\quad u=s\delta^2,\quad D_{\rm damp}=\beta u=sr\theta^2\le D_{\rm cap}.
\]

δ→0、sは整数、u有界。非自明な減衰ではβu→正の有限値。相対時間はT_rel=sδで、各枝の理想時間絶対値はT_rel/2。

`D=exp(i delta H_D/4)`、`A_j=exp(i theta P_j)`、右端先作用の `R=A_jr ... A_j1`、`R_-=A_jr^dagger ... A_j1^dagger` とする。`G_+=DRD`、`G_-=D^dagger R_- D^dagger`。同じ位置の二枝がlabelを共有し、位置間・step間はIID。

\[
G_-^\dagger=D A_{j_1}\cdots A_{j_r}D,
\qquad\Psi_\delta(X)=\mathbb E[G_-^\dagger XG_+]
=D\Phi_{c\delta}^{\,r}(DXD)D.
\tag{1}
\]

内側のj_rから順に期待値を取るとΦがr回現れ、これが(1)の導出である。rやstepのlawを変更していない。負枝は同順sign reversalで、full-word adjointではない。`DXD`を`DXD^dagger`へ変えない。対象信号は `z_s=<k|Psi_delta^s(I)|k>`。

## 3. HS生成子と散逸性

演算子空間にHS内積を入れ、以下のsuperoperator normはinduced 2→2 normとする。

\[
J_j(X)=P_jX+XP_j,\quad L_R(X)=hX+Xh,
\quad J_D(X)=H_DX+XH_D,
\]
\[
B=J_D/4,\quad C=\lambda L_R/2,
\quad A=2B+C=J_H/2.
\]

J_j,L_R,B,C,AはHS-self-adjoint。Aの周波数は **omega=(E_a+E_b)/2**。以下のΠ_ωはこの全Hのpair-sumの直交射影であり、hの射影を流用しない。

\[
D_R=-\tfrac12(\mathbb E J_j^2-L_R^2)
=-\tfrac12\mathbb E(J_j-L_R)^2\preceq0,
\qquad Q=\beta D_R.
\tag{2}
\]

E J_j=L_Rなので非可換交差項も平均後にL_R²となる。T1aと同じvariance恒等式を、**stepの係数βを付けて**使う。`-2 beta Id<=Q<=0`、`q=norm(Q)<=2 beta`。

ΨはunitaryなHS作用素 `e^{iδB} e^{icδJ_jr} ... e^{icδJ_j1} e^{iδB}` の平均として表せるため収縮。ここでのHS productは写像(1)の表示であり、一つの物理branch wordと取り違えない。`F_delta=exp(i delta A+delta^2 Q)` も散逸性から収縮する。CPTP性、Ψの正規性、Φの固有射影の条件数を使わない。

## 4. 一step展開と新しい局所剰余

### 4.1 二次までの係数

`M_ell=E J_j^ell` と置く。ΦのTaylor展開から

\[
\Phi_{c\delta}^{\,r}
=I+i\delta C+\delta^2(Q-C^2/2)+O(\delta^3).
\]

実際の二次係数は `-r c² M_2/2-r(r−1)c² L_R²/2`。これを(2)でまとめた表示が上式である。左右の `exp(iδB)` を掛けると

\[
\boxed{\Psi_\delta
=I+i\delta A+\delta^2(Q-A^2/2)+R_\Psi(\delta).}
\tag{3}
\]

対称配置により、二次log生成子にHD/HRのcommutatorは残らない。Qは乱択の二次散逸、A²/2は理想coherent exponentialの二次項。非可換な分割・位相補正は三次以降にあり、§6で分離する。

### 4.2 全δで使えるTaylor剰余

`d=norm(H_D)`、`m=d+lambda`、`a=norm(A)<=norm(H)<=m` とする。各HS unitary productのgenerator normの和は

`2 norm(B)+r c max norm(J_j)<=d+lambda=m`。

δに関する三階微分をLeibnizで展開すると、各項の残りのfactorはunitaryで、multinomialの和からnormはm³以下。平均後も同じであり、積分形Taylor剰余から

\[
\|R_\Psi(\delta)\|\le v|\delta|^3,\qquad v=m^3/6.
\tag{4}
\]

これは(3)を長時間へ延ばす前に得た明示評価で、数値Taylor fittingではない。

任意の固定δ_0>0を解析上限として、`abs(delta)<=delta_0` の範囲で

\[
K=\frac{m^3}{6}+\frac{(a+\delta_0q)^3}{6}
+a q+\frac{\delta_0q^2}{2}.
\tag{5}
\]

G=iδA+δ²Qのexpの積分剰余は `(a+delta_0 q)^3 abs(delta)^3/6` 以下。`G²/2`の追加項 `iδ³(AQ+QA)/2+δ⁴Q²/2` は `(a q+delta_0 q²/2)abs(delta)^3` 以下。二次までが(3)と一致するため

\[
\boxed{\|\Psi_\delta-F_\delta\|_{2\to2}\le K|\delta|^3.}
\tag{6}
\]

T1aの定数18は使っていない。KはHD、lambda、r、δ_0、normへの依存を持つ。δは時間の単位を持つため、無根拠にδ≤1を実験閾値として固定しない。

## 5. 主定理：全Hに基づくC-secへの長時間一様移送

\[
Q_0=\sum_\omega\Pi_\omega Q\Pi_\omega,\qquad q_0=\|Q_0\|\le q.
\]

Q_0≤0、[A,Q_0]=0。T1bで定義するC-sec参照は

\[
z_s^{H,\rm sec}=\langle k|e^{is\delta A}e^{uQ_0}(I)|k\rangle.
\tag{7}
\]

全Hのcoherent生成子にtailの二次補正をsecularizeしたもの。T1aの主比較をC-logへ変更する行為ではなく、今回のT1bにおける移送先を明示した定義である。

η≥0について

\[
Q_n=\sum_{0<|\omega-\nu|\le\eta}\Pi_\omega Q\Pi_\nu,
\quad q_n=\|Q_n\|,
\quad\kappa_f=\sum_{|\omega-\nu|>\eta}
\frac{\|\Pi_\omega Q\Pi_\nu\|}{|\omega-\nu|}.
\tag{8}
\]

ordered pairの和を使い、空和は零。exact resonanceはQ_0へ保持する。ηは解析用で、参照blockを変更しない。

**定理1。** 固定モデル、固定r、実abs(δ)≤δ_0、全整数s≥0、`beta s delta²<=D_cap` で

\[
\boxed{|z_s-z_s^{H,\rm sec}|\le\sqrt n\left[
Ku|\delta|+u q_n
+2|\delta|\kappa_f\{1+u(q+q_0)\}\right].}
\tag{9}
\]

特にη=0ならq_n=0、κ_f有限で、`u<=U_cap=D_cap/beta` により右辺は**全sに一様なO(abs(delta))**となる。βや次元・gapを同時に変える一様性は主張しない。

**証明。** Ψ,Fが収縮なのでtelescopingで `norm(Psi^s−exp(sG))<=sK abs(delta)^3=Ku abs(delta)`。

`Y(t)=exp(−itδA)exp(tG)`、`Z(t)=exp(tδ²Q_0)`、`Q(t)=exp(−itδA)Qexp(itδA)` とすると、Y'=δ²Q(t)Y、Y,Zともnorm≤1。Duhamelのnear部分はtδ²q_n以下。

far部分のprimitiveは

\[
V(t)=\sum_{|\omega-\nu|>\eta}
\frac{e^{i\delta t(\nu-\omega)}-1}{i\delta(\nu-\omega)}
\Pi_\omega Q\Pi_\nu,\quad\|V(t)\|\le2\kappa_f/|\delta|.
\]

V(0)=0から

\[
\begin{aligned}
\int_0^t Z(t-v)V'(v)Y(v)dv={}&V(t)Y(t)\\
&+\delta^2\int_0^t Z(t-v)Q_0V(v)Y(v)dv\\
&-\delta^2\int_0^t Z(t-v)V(v)Q(v)Y(v)dv.
\end{aligned}
\]

最後はnearを含む全Q(v)。従って平均化誤差は `tδ²q_n+2abs(delta)κ_f[1+tδ²(q+q_0)]` 以下。t=sとしlocal誤差を足す。入力IのHS norm√n、functional E_kkのHS norm1で(9)。δ=0またはs=0は厳密一致。□

### 5.1 Weightと適用限界

全演算子空間のΠ_ω、global block norm、入力√nを開示した。bare energy gapをpair-sum gapの代わりに入れていない。共通のHS直交射影PがA,Qと可換でP(E_kk)=E_kkなら、平均化部分はP空間へ制限し、√nをnorm(P I)へ変更できる。ただしlocal誤差は一般にΨがPを保存しないので√nが残る。Ψ自体もPを保存すると証明できる場合だけlocal部分も制限できる。

小gapのfamily、巨大n、極小βではこの保証は保守的・実用的に空になり得る。固定lambda/rを変える資源最適化には使っていない。

## 6. Outer splitting・有限step位相・交差項の分離

### 6.1 決定論的対称参照

`S_delta=exp(iδB)exp(iδC)exp(iδB)` と置く。物理的にはexactなHD halfとexactなHR中央を置いた参照の二側作用で、random tailの代替結果を新たに実装したものではない。二次まで `exp(iδA)` と一致し、§4.2と同じTaylor評価で

\[
\|S_\delta-e^{i\delta A}\|\le K_{\rm out}|\delta|^3,
\quad K_{\rm out}=(m^3+a^3)/6,
\quad\|S_\delta^s-e^{is\delta A}\|\le K_{\rm out}u|\delta|.
\tag{10}
\]

従ってexact HDでもouter biasは残り、有限減衰系列でO(u abs(delta))に累積する。randomのO(uQ)という有限減衰と同じ項ではなく、平均化誤差と同じδ次数の誤差である。(9)のKにはこのlocal効果も含むため、(10)を(9)へさらに加算して二重計上しない。

### 6.2 三次log係数の解析的同定

`norm(Psi_delta−I)<=m abs(delta)` なので `m abs(delta)<1` ではIから連続なlogを級数で定義できる。以下は固定モデルの解析級数の係数同定であり、(9)の明示剰余は(4)–(6)から独立に保証する。

\[
T_R=-M_3/6+(L_RM_2+M_2L_R)/4-L_R^3/3,
\]
\[
K_{\rm sp}=\{[B,[B,C]]+[C,[B,C]]\}/6,
\quad T=rc^3T_R+K_{\rm sp}.
\]

全てHS-self-adjoint。`log(I+Z)=Z−Z²/2+Z³/3+...` にΦの係数を入れると `log Phi_theta=iθL_R+θ²D_R+iθ³T_R+O(θ⁴)`。左右halfを含むproductの三次係数を掛け合わせ、logの三次を引くと

\[
\boxed{\log\Psi_\delta=i\delta A+\delta^2Q+i\delta^3T+O(\delta^4).}
\tag{11}
\]

K_spは `S_delta−exp(iδA)=iδ³K_sp+O(δ⁴)` の係数でもある。対称halfによりB,Qの単一commutatorに比例する三次log項は相殺し、二次にも残らない。**HDとの非可換性の効果が全て消えたわけではなく**、K_spと、全H射影に対するQの結合・mode変化に残る。四次以上の交差項も(6)の剰余に含まれる。(11)だけを根拠に長時間保証を宣言しない。

## 7. 有効mode・左右weight・縮むgapの明示評価

ここからは追加の十分条件 **(SP)：`(E_a+E_b)/2=E_k` となるpairは(k,k)だけ** を課す。主定理1には不要であり、scalar rateを抽出する際だけ使う。全Hilbert空間の非縮退最低/最高固有stateなら満たす。sector内groundの通常gapだけでは認定できない。

`e_0=E_kk`をHS norm1のvector、`P_0=|e_0)(e_0|`、`g=min_{omega!=E_k}abs(omega−E_k)>0` とする。Ωが一点のみで全HS空間が一次元の場合はP_delta=P_0=Idとしてmodeの議論を直接処理する。

### 7.1 Riesz射影を使う有限δ評価

δ≠0で `M_delta=(Psi_delta−I)/delta`。式(3)–(4)から

\[
M_\delta=iA+\delta(Q-A^2/2)+\delta^2R_\delta,
\quad\|R_\delta\|\le v.
\]

`t_*=q+a²/2`、`T_*=t_*+delta_0 v`、`epsilon_delta=abs(delta) T_*` とする。`epsilon_delta<g/8` の範囲で、iE_kの周りの半径g/2の円Γを使う。iAはnormalでΓ上のresolvent norm≤2/g。Neumann級数から摂動後≤4/g、従って

\[
P_\delta=\frac{1}{2\pi i}\oint_\Gamma(z-M_\delta)^{-1}dz,
\quad\|P_\delta-P_0\|\le p_\delta:=4\epsilon_\delta/g<1/2.
\tag{12}
\]

contourを保った連続摂動なのでrankは1。P_deltaはΨと可換な**非直交**射影で、左右modeの積を表す。`norm(P_delta)<=1+p_delta` で条件の悪さを隠していない。`v_delta=P_delta e_0/(e_0,P_delta e_0)` は右mode、`(e_0,v_delta)=1`、`norm(v_delta−e_0)<=4p_delta<=16epsilon_delta/g`。

対応固有値をλ_deltaとすると、右modeの固有方程式に(e_0,·)を取って

\[
\lambda_\delta=1+i\delta E_k+
\delta^2\{(e_0,Qe_0)-E_k^2/2\}+r_\lambda,
\quad |r_\lambda|\le K_\lambda|\delta|^3,
\]
\[
K_\lambda=v+16T_*^2/g.
\tag{13}
\]

実際、M_deltaの残りVのnorm≤epsilon_delta、正規化右modeの差≤16epsilon_delta/gなので、固有値の一次補正からの誤差は `delta² v+16epsilon_delta²/g` 以下。λへ戻すと(13)。gapの縮みを無視したO(δ²)摂動論ではない。

### 7.2 実信号に必要なweightと他mode

`c_delta=(e_0,P_delta I)` とすると

\[
|c_\delta-1|\le\sqrt n p_\delta,
\qquad |z_s-c_\delta\lambda_\delta^s|\le\sqrt n p_\delta
\quad(\hbox{全}s\ge0).
\tag{14}
\]

後式は `(e_0,(I−P_0)X)=0`、P_deltaとΨの可換性、Ψ^sの収縮から従う。他modeを固有値だけで捨てず、左右射影weightと全s一様のresidualを明示した。固有値の成立だけから信号のrateを結論しない。

## 8. 全H stateでの先導減衰・位相補正

\[
\mu_k=\langle k|h|k\rangle,
\quad W_k=1-\langle k|h^2|k\rangle,
\quad v_k=\operatorname{Var}_j\langle k|P_j|k\rangle.
\tag{15}
\]

`W_k=E norm((P_j−h)|k>)²`、`v_k=E abs(<k|(P_j−h)|k>)²` なので **0≤v_k≤W_k**。この不等式はh固有stateを仮定せず成立する。

\[
(e_0,Qe_0)=-\beta(W_k+v_k)=:-\gamma_{\rm sh}.
\tag{16}
\]

(SP)ならQ_0はこの対象blockでscalarなので

\[
\boxed{z_s^{H,\rm sec}=e^{iT_{\rm rel}E_k}e^{-u\gamma_{\rm sh}}
=e^{iT_{\rm rel}E_k}e^{-D_{\rm damp}(W_k+v_k)}.}
\tag{17}
\]

ここで初めて全H stateの先導減衰を主信号へ結び付けた。h_R固有stateの係数の単純代入ではない。

### 8.1 有限δのrate・位相の誤差

さらに `m abs(delta)<=1/2` とする。`abs(lambda_delta−1)<=m abs(delta)`、logの三次以降の和≤`2m³abs(delta)^3/3`、(13)を使って

\[
\left|\log\lambda_\delta-i\delta E_k+\delta^2\gamma_{\rm sh}\right|
\le C_{\log}|\delta|^3,
\]
\[
C_{\log}=K_\lambda+m(t_*+\delta_0K_\lambda)+2m^3/3.
\tag{18}
\]

従ってper-step normalized decay `gamma_sh(delta)=−log abs(lambda_delta)/delta²` はγ_shとの絶対差≤C_log abs(delta)。相対差はγ_sh>0かつ下界がある場合だけ `C_log abs(delta)/gamma_sh`。s倍したlog位相誤差は≤C_log u abs(delta)。c_deltaやresidualも(14)で会計する必要があり、これだけでsignal floorを保証しない。

### 8.2 三次の位相係数とouter項

固定モデル・(SP)の十分小さいδではlog Ψ/δはiAからの解析的摂動である。contour resolventのNeumann展開により対象の固有値・射影も解析的。式(11)の一次補正modeを代入すると

\[
\log\lambda_\delta=i\delta E_k-\delta^2\gamma_{\rm sh}
+i\delta^3\chi_k+O(\delta^4),
\]
\[
\chi_k=(e_0,Te_0)+
\sum_{\omega\ne E_k}\frac{\|\Pi_\omega Qe_0\|_2^2}{\omega-E_k}.
\tag{19}
\]

導出は `v_1=sum_{omega!=E_k}Pi_omega Qe_0/[i(E_k−omega)]`、次次数の固有方程式で `i(e_0,Te_0)+(e_0,Qv_1)` を取ることによる。χは実数で、`abs(sum)<=q²/g`。T内のK_spがouter splitting、rc³T_Rがtailの有限角、最後の和がQによるmode mixingの位相補正である。三者を「dressing」の一語で同一視しない。O(δ⁴)の解析級数係数に明示定数を与えたとはせず、一様有限δ保証には(9),(18)を使う。

## 9. Dressing：状態・mode・係数・ratioを分ける

`a_D=<k|H_D|k>`、α=δ/4。全Hの固有stateでは `(H_D−a_D)|k>=−(H_R−<H_R>)|k>` なのでσ_k(HD)=σ_k(HR)。unitary積分から

\[
\|D|k\rangle-e^{i\alpha a_D}|k\rangle\|
\le|\alpha|\sigma_k(H_R).
\tag{20}
\]

これは状態差。対応projectorのHS差≤その2倍。仮にこのdressed projectorでQのRayleigh quotientを取るなら、その**補助的な係数差**は≤`4q abs(alpha) sigma(H_R)`。理由は両projectorのHS norm1と二次形式の差の評価である。ただしそのRayleigh quotientはΨの真の減衰係数ではない。

真のmode差は(12)、真のrateの絶対差は(18)。どちらもgapやnormを含み、(20)の小ささだけでは置き換えられない。rateの相対差にはγ_shの下界、ratioの差にはordinary分母の下界が別に必要。HDは有限δ補正とfull-Hのstate/射影を通じて効き、採択済みδ→0ではαによる追加のO(1)先導rate補正を(16)に付け加える必要はない。固定αでr→∞という別極限にはこの結論を移さない。

## 10. 同じ理想taskのordinary／独立±参照とrho

これは**係数比の定義に必要な解析的対照**であり、新しい回路費用baselineの採択やPR-RTEとの完全同値の主張ではない。全て同じH、分割、law、state、相対時間sδ、固定rとする。

### 10.1 独立±：各枝r個・角θ

二枝のlabel列を独立drawすると、`M_theta=E A_j=cos(theta)I+i sin(theta)h`、`V_delta=D M_theta^r D` により平均写像は `Psi_ind(X)=V_delta X V_delta`。列のiidによるfactorizationであり、共有のΦではない。その二次log項は

\[
Q_{\rm ind}(X)=-\tfrac\beta2\{I-h^2,X\},\qquad\|Q_{\rm ind}\|\le\beta.
\tag{21}
\]

一次生成子は同じA。§4のunitary-product Taylor評価は独立な左右labelを平均してもaggregate norm≤mで成立するため、QをQ_indへ替えたKと(9),(12)–(18)が成立する。(SP)で `gamma_ind=beta W_k`。**同じper-branch rではshared rateはind rate以上**であり、独立±に対して同じrで減衰を半分にしたという結論ではない。独立±は2r個の独立drawを持つが、物理gate価格はこのdraw数だけで決まらない。

### 10.2 Ordinary：片枝r個・角2θ

ordinaryの一step平均振幅operatorを

\[
O_\delta=e^{i\delta H_D/2}M_{2\theta}^{\,r}e^{i\delta H_D/2}
\]

と明記する。位相信号は `<k|O_delta^s|k>`、理想は `exp(iT_rel E_k)`。物理Hilbert空間のoperator normで

\[
O_\delta=I+i\delta H+\delta^2(Q_o-H^2/2)+R_o,
\quad Q_o=-2\beta(I-h^2),\quad\|R_o\|\le m^3|\delta|^3/6.
\tag{22}
\]

各物理wordのgenerator normの和もd+lambda=mなのでこの剰余が従う。Q_o≤0であり、A→H、q→norm(Q_o)≤2βとした§§4–5の証明を物理Hilbert空間で行える。ここでの周波数はHの固有値、入力/出力は|k>でnorm1、√nは不要。非縮退E_kの一次元blockで `gamma_ord=2beta W_k`。単なる名称のqDRIFTからこの係数を推測したのではなく、角2θとmeanを定義して導いた。

### 10.3 候補式の判定域と誤差

(SP)、W_k>0、上記ordinary対照の下で先導減衰係数の比は

\[
\boxed{\rho_\infty^{(H)}=\frac{\gamma_{\rm sh}}{\gamma_{\rm ord}}
=\frac{W_k+v_k}{2W_k}
=\frac12+\frac{v_k}{2(1-\langle h^2\rangle_k)}\in[1/2,1].}
\tag{23}
\]

これは[G0記録][G0]§10の実際の候補式と同じであり、**定義した十分条件下で`HOLDS`**。都合よく候補を置き換えたものではない。全Hのstateでもv≤Wとなるvariance恒等式が区間を支える。

finite-step eigenmode ratioについて、ordinaryのenergy gapを用いた(18)の定数をC_o、sharedをC_shとする。`C_o abs(delta)<=gamma_ord/2` なら

\[
\left|\frac{\gamma_{\rm sh}(\delta)}{\gamma_{\rm ord}(\delta)}
-\rho_\infty^{(H)}\right|
\le\frac{2(C_{\rm sh}+\rho_\infty^{(H)} C_o)|\delta|}{\gamma_{\rm ord}}.
\tag{24}
\]

これは**mode rateのratio**であり、finite-sの実信号のlog半径比ではない。実信号には(14)、floorと相対誤差が必要。イベント数・必要shots・RPE RMS・native総費用の比とは主張しない。

W=0ならv=0で(23)は0/0、`NOT_APPLICABLE`。Wが極小なら(24)は悪条件。pair-sum共鳴でscalar先導rateに還元できない場合、候補を普遍的なratioとは認定せず `NOT_APPLICABLE_AS_A_SINGLE_RATE` とする。block保証(9)は別に成立する。任意の共鳴blockに有用な代替ratioを定める研究は未解決。

## 11. 非可換二準位例：単純代入の誤りと非空域

全て**手計算の解析値**で、数値行列評価ではない。`P_1=Z,P_2=X,p=1/2,lambda=1`、`H_R=(Z+X)/2`、`H_D=(Z−X)/2`、従ってH=Z。HDとHRは非可換。対象はZの±固有state、固定r任意。分子geometry/basis/DF rank/L_Dは該当しない。

`h²=I/2`、`mu_k=±1/2`、W=1/2、v=1/4、`sigma_k(h)^2=1/4`。全Hの対象pair-sum blockは一次元。

\[
\gamma_{\rm sh}=3\beta/4,\quad
\gamma_{\rm ind}=\beta/2,\quad
\gamma_{\rm ord}=\beta,\quad\rho_\infty^{(H)}=3/4.
\tag{25}
\]

`1−mu_k²=3/4` をWの代わりに単純代入すると2/3となり、誤る。**実際の旧候補式はh²の期待値を使うので、この例は旧候補への反例ではない。** h固有stateでないことに伴う分散補正の必要性を示す。

一般に `1−mu_k²=W_k+sigma_k(h)²`。この違いはαによる有限step補正とは別で、full-H stateを用いる先導次数ですでに必要。

この例では(17)から全u≤U_capに参照floor `r_*=exp(−3D_cap/4)>0`。η=0の(9)の係数をC_sigとすれば、十分小さい非零δで `C_sig abs(delta)<r_*`、(12),(18)の条件も同時に満たせる。全sに非空の一様位相保証域がある。実用gate数や分子への有用性は認定しない。

## 12. 解析的共鳴例：孤立stateだけではscalar式は成立しない

許可されたHermitian involution classでの三準位例を構成する。basisを `|−>,|0>,|+>`、`H=diag(−1,0,1)`、対象|k>=|0>とする。`|v>=(|−>+|+>)/sqrt(2)`、`|w>=(|−>−|+>)/sqrt(2)`、

\[
P=|v\rangle\langle0|+|0\rangle\langle v|+|w\rangle\langle w|,
\quad P_1=P,\ P_2=-P,\quad
p_1=(1+\xi)/2,\ p_2=(1-\xi)/2,\quad0<\xi<1.
\]

P=P†、P²=I、h=ξP、H_R=lambda ξP、H_D=H−lambda ξP。lambda>0とξは固定し、HR≠0、[HD,HR]≠0である。これは指定されたHermitian involution class内の例で、Pauliのみのsubclassの反例と主張しない。異符号の同じinvolutionを含む固定sampling lawを使い、別の分解へ統合してlawを変えていない。異符号を統合した非冗長なPauli表だけに限定する別classへの反例は、この構成だけでは示していない。

`W=1−xi²>0`、v=0、`D_R(X)=W(−X−PXP)`。Hの対象stateは孤立しているが、ω=0にはE_00,E_−+,E_+−がある。`S=(E_−++E_+−)/sqrt(2)` と置くと、その対称二次元subspaceでD_sec/Wは

\[
\begin{pmatrix}-1&-1/\sqrt2\\-1/\sqrt2&-3/2\end{pmatrix}
\quad\hbox{（basis }E_{00},S\hbox{）}.
\]

例えば `Pi_0(P E_00 P)=(E_−++E_+−)/2`、`Pi_0(P E_−+ P)=E_00/2+(E_−++E_+−)/4` を直接掛けるとこの表示になる。Iのω=0成分はE_00。対角成分を厳密に指数化すると

\[
\boxed{z_s^{H,\rm sec}=\tfrac23e^{-W D_{\rm damp}/2}
+\tfrac13e^{-2W D_{\rm damp}}\ne e^{-W D_{\rm damp}}.}
\tag{26}
\]

(26)の不等号はD_damp>0に対するもの。零では一致する。二つのD_damp当たりの固有rateはW/2,2W、E_00へのweightは2/3,1/3で、上の2×2の特性多項式・固有vectorから得る解析値。single scalar γ_sh=βWは初期微分だけであり、有限減衰域の指数ではない。定理1により真の信号は(26)へ一様に近づく。

ordinaryはI−h²=W Iなので先導 `exp(−2W D_damp)`。従ってshared/ordinaryに一つの時間独立の先導rate比1/2を割り当てる一般claimは成立しない。**(SP)は結果を救うために後付けした普遍仮定ではなく、scalar系の独立した十分条件**であり、主block定理はこの例をそのまま含む。

## 13. Pair-sum・alias・sectorの区別

通常のgap `abs(E_a−E_k)`、HS gap `abs((E_a+E_b)/2−E_k)`、実際のcoupling `Pi_omega Q Pi_nu` は異なる。exact resonanceはQ_0内のmatrixとして保持する。近共鳴は(8)のq_n、farはcoupling加重κ_f。零couplingの小gap項はκに寄与しない。

finite-step unitary参照 `U_delta=exp(iδA)` の異周波数固有位相gapは `2abs(sin(delta(omega−nu)/2))`。`abs(delta)*diam(Omega)<=pi` なら

\[
|e^{i\delta\omega}-e^{i\delta\nu}|
\ge(2/\pi)|\delta|\,|\omega-\nu|.
\]

またDuhamelと(6)で `norm(Psi_delta−U_delta)<=delta²(q+K abs(delta))`。対象の非alias分離に対する摂動比は高々 `(pi/2)abs(delta)(q+K abs(delta))/g`。gapもδとともに縮むが固定g>0なら比は零へ進む。(12)は別途この小分母を明示してmodeを制御した。固定有限angleやδとmodel/gを同時変更する経路に、無条件なalias除外はない。

Pauli-tailが粒子数/spin sectorを出る場合は全spaceでΠを取る。小さいsectorに制限するには、そのspaceを全eventとHDが保存すること、対象stateの端点性・pair-sum条件を個別に証明する必要がある。粒子数保存やsector内gapという名称だけで認定しない。

## 14. Signal floor付き位相保証

(9)の右辺をE_sigとし、参照 `abs(z_s^{H,sec})>=r_*>0`、E_sig<r_*なら

\[
|z_s|\ge r_*-E_{\rm sig},\qquad
|\operatorname{Arg}(z_s/z_s^{H,\rm sec})|
\le\arcsin(E_{\rm sig}/r_*).
\tag{27}
\]

参照で割った信号が中心1・半径E_sig/r_*<1の円板にあることによる。(SP)なら(17)から `r_*=exp(−U_cap gamma_sh)>0` と選べ、η=0の誤差が十分小さいδの非空域がある。共鳴blockの一般の場合はfloorを別に与える必要がある。単一信号の主値であり、phase unwrapping、RPE統計、shot数、RMS energy保証を含まない。

## 15. T1aから移送できた内容と残る範囲

| 項目 | 著者が示した内容 | 残る限界 |
|---|---|---|
| 一step | exact Ψから(3)、β付きvariance散逸 | 一般DF/Hermitian eventには拡張していない |
| 長時間 | 定理1、全sでβsδ²≤D_cap、一様O(abs(delta)) | 固定model/rのみ。size/gap一様でない |
| Mode | (SP)と明示gap条件下の非直交Riesz射影・weight・residual | 共鳴blockのscalar化を一般には認めない |
| 減衰・位相 | W+v、有限δ bound、三次位相係数の分離 | 一般guiding state/RPEはscope外 |
| Dressing | 状態差、mode差、rate絶対/相対、ratioを分離 | 相対誤差には分母下界が必要 |
| Outer | local O(δ³)、累積O(u abs(delta))、commutator係数 | 実装PFの内部誤差は未実施 |
| rho候補 | (SP),W>0,指定ordinaryでHOLDS | 一般共鳴single-rateは適用外、有限総費用ではない |
| 反例 | 二準位の誤ったV代入、非零HR/非可換HDを持つ三準位のscalar共鳴claim不成立 | Pauliのみのsubclassをこの例で反証したとはしない |

著者提出区分は主定理・条件付きmode/ratioについて **`PROVED_WITH_DOMAIN`**、定義した共鳴classのscalar化について **`COUNTEREXAMPLE_IN_DEFINED_CLASS`**。適用域外のratioはNOT_APPLICABLE、任意共鳴blockの有用な代替ratio、分子の実用的な定数・floorは未解決/未検証。これらを「一般のT1bが全て成立」または「全M1-Sのno-go」へ拡張しない。

## 16. 既知理論との対応と文献確認の範囲

証明の道具はTaylor積分剰余、HS散逸semigroup、Duhamelと振動平均、有限次元resolventのNeumann級数・Riesz射影であり、一般手法の発明を主張しない。Kato型の解析的摂動を使う箇所は(12)と(19)に自己完結した計算を記した。Daviesの無限heat-bath/open-system定理を、この離散coherent写像に無条件に適用したわけではない。

今回[Campbell][CAMP]、[KGW][KGW]、[Davies][DAVIES]の一次公開ページの書誌・abstractを限定確認した。全文の新規精読、全定理の再現、網羅的新規性調査ではない。G0-pre/P0/P0.5の既存監査とCheckpoint 1のlaw/target区別を再利用する。

| 先行対象 | 今回との対応・差 |
|---|---|
| T1a | 同じ散逸/平均化構造。ただし全HのA、β、新K、HD/HR分割と全H weightを導出し直した |
| 通常qDRIFT | random指数eventとmeanは既知。本書のordinary係数は(22)の明示lawから紙上導出、既知diamond係数を輸入していない |
| PRの独立±RTE | 独立枝・normalization・Taylor event lawの既存条件を保持。本書の独立±Pauli exponentialとPR-RTEの完全同値・価格同値は未証明 |
| Directional control／random PF | 構成自体は既知。無条件coherent first moment、有限減衰の保証とsingle-trajectory集中を区別する |
| KGW randomized Hamiltonian phase estimation | 状態依存分散・gapを扱う近接対象。law、観測量、sδ²系列が同一との主張はしていない |
| Kato/Davies型 | 標準的な摂動・平均化の考え方。今回の量化・係数・共鳴制約をそのまま既知定理として引用していない |

今回の**比較材料**は、固定joint pathにおけるQの規格化、full-HのWとv、ordinary/独立±との先導係数、outer/finite-angle/mode-mixing位相の分離、共鳴時のscalar不能例。これらが既知の直接系以上の独立学術的価値を持つかはCheckpoint 2の判断であり、Codexが認定しない。

## 17. 未証明・未実施と停止

未証明は、size/gap一様な分子保証、任意共鳴blockのscalar rho、無料の実用predictor、固定δでr→∞・rと系サイズの同時極限、一般DF event、HD内部PF、RPEのRMS/shot・native総費用改善。有限減衰係数比をイベント半減・費用半減へ格上げしない。

新しい理論導出・解析的反例は実施した。**新しい科学数値計算、H4/H6保存値からの新科学量、Hamiltonian/行列の数値評価・対角化、simulation、sampling、科学runner/test、回路生成、compile/synthesis、資源forecastは0件**。記載した小次元行列・数値分数は全て紙上の厳密解析式である。文書/hash/公開bytesの検査を科学的validationや独立受理と呼ばない。

成果と限界は[Checkpoint 2 Decision Packet][CP2]へ渡す。**Checkpoint 2のGPT重要科学レビューは未実施・開始未承認。ここでmandatory STOP。** Gate 0/E0、Q4/Q5、DF拡張には進まない。

[T1A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md
[CP1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_checkpoint1_decision_packet_20261016.md
[S0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/373b8e4b9831f4d00bdc00ffc741ac61e03f0841/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md
[GPT1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4899608ea222777e64e706fda2b7e54b84c86247/docs/research/track_b_m1s_g0t1_checkpoint1_gpt_independent_scientific_review_20261016.md
[G0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f0952614dcf5c06d3d63c21b5d49091b8ecce37/docs/tracks/algorithm_codesign/m1s_g0_proof_and_gate0_precontract_20261015.md
[AUTH]: ../../../artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/user_authorization_record.json
[INDEX]: ../../../artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/evidence_index.json
[CP2]: m1s_g0t1_t1b_checkpoint2_decision_packet_20261016.md
[CAMP]: https://arxiv.org/abs/1811.08017v2
[KGW]: https://arxiv.org/abs/1907.10070v1
[DAVIES]: https://link.springer.com/article/10.1007/BF01608389
