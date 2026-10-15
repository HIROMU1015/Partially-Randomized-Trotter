# M1-S G0-T1／T1a：C-sec複素信号の一様上界と解析的証明

2026-10-16 JST。**`T1A_ANALYTIC_PROOF_SUBMITTED_WITH_DOMAIN_CHECKPOINT1_PENDING_MANDATORY_STOP`**。

[正式S0契約][S0]のT1aを、ユーザーの「次の作業に進んで大丈夫」という直後の実施承認に基づいて行った。[承認記録][AUTH]。本書は**著者による解析的な証明の提出**であり、独立した重要科学レビュー、独立新規性の認定、Checkpoint 1のGOではない。T1b・分子数値計算・RPE・compileは実施していない。

## 1. 結果と適用域

固定有限次元 `n=dim H`、Pauli/Hermitian involution、同位置共有±、位置間IID、共通線形角というS0のlawを維持する。対象は固定hの孤立した固有state|k>、厳密信号 `z_N=⟨k¦Phi_theta^N(I)¦k⟩` と、採択済みC-secの `z_N^sec` の差である。

以下の上界を、**全ての整数N≥0、実theta、`abs(theta)<=1`、`tau=N theta^2<=D_cap`**について示す。theta=0は厳密一致として扱う。

\[
\boxed{
|z_N-z_N^{\rm sec}|
\le \sqrt n\left[
18\tau|\theta|+\tau b_{\rm near}(\eta)
+2|\theta|\kappa_{\rm far}(\eta)
\{1+\tau(b+b_0)\}\right].}
\tag{T1a}
\]

全ての定数・normは§2で定義し、`0<=b_0<=b<=2`。ηは非負の分析用cutoffで、C-secの厳密同一周波数blockを変更しない。低周波結合はb_near、分離した周波数間の逆gap重みはκ_farで明示する。18 tau abs(theta)は**局所三次以上・有限角度誤差の累積**であり、他の漏れが零でも消してはいけない。

固定モデルの相異なる周波数間の最小非零gapよりηを小さくすればb_near=0。特にη=0とすると、

\[
|z_N-z_N^{\rm sec}|
\le \sqrt n|\theta|
\left[18D_{\rm cap}+2\kappa_{\rm far}(0)(1+4D_{\rm cap})\right].
\tag{U}
\]

従って、固定モデル・有限D_capで、対象となる**全Nに一様なO(abs(theta))**の複素信号評価が得られる。この上界は次元非依存でも、小gap familyに一様でもない。κや√nが大きければ実用的に空になり得る。§9は、非可換な二準位例で非空の明示域を与える。

## 2. 記号・norm・spectral weight

### 2.1 正式契約と同じ写像

\[
P_j=P_j^\dagger,\quad P_j^2=I,\quad p_j\ge0,\quad \sum_jp_j=1,
\quad h=\mathbb E P_j,\quad A_j=e^{i\theta P_j},
\]
\[
\Phi_\theta(X)=\mathbb E[A_jXA_j],\quad
\mathcal L(X)=hX+Xh,\quad
\mathcal C(X)=\mathbb E[(P_j-h)X(P_j-h)],
\]
\[
\mathcal D(X)=-\tfrac12\{I-h^2,X\}-\mathcal C(X).
\]

ΦはCPTP channelの写像ではない。有限次元の演算子空間 `B(H)` をHilbert–Schmidt内積 `⟨X,Y⟩_2=tr(X^dagger Y)` のHilbert空間として使う。以下のsuperoperator normは**induced 2→2 norm**。matrix/operator spectral normやdiamond normと混ぜない。

Lの相異なる固有周波数の集合をΩとする。hの固有値がe_aなら周波数は `omega=e_a+e_b`、直交射影がΠ_ωである。同一周波数の全固有operatorを一つのblockへまとめる。

\[
\mathcal D_0=\mathcal D_{\rm sec}
=\sum_{\omega\in\Omega}\Pi_\omega\mathcal D\Pi_\omega,
\quad b=\|\mathcal D\|_{2\to2},\quad b_0=\|\mathcal D_0\|_{2\to2}.
\]

C-secはS0と同じ

\[
z_N^{\rm sec}=\langle k|
e^{iN\theta\mathcal L}e^{\tau\mathcal D_0}(I)|k\rangle.
\]

本証明はΦの固有値・左右固有modeやlog Φを使わない。C-logへ主targetを変更していない。

### 2.2 Near/far分解

η≥0に対して、厳密に異なる周波数だけを分ける。

\[
\mathcal D_{\rm near}(\eta)
=\sum_{0<|\omega-\nu|\le\eta}\Pi_\omega\mathcal D\Pi_\nu,
\quad b_{\rm near}(\eta)=\|\mathcal D_{\rm near}(\eta)\|_{2\to2},
\]
\[
\mathcal D_{\rm far}(\eta)
=\sum_{|\omega-\nu|>\eta}\Pi_\omega\mathcal D\Pi_\nu,
\quad
\kappa_{\rm far}(\eta)
=\sum_{|\omega-\nu|>\eta}
\frac{\|\Pi_\omega\mathcal D\Pi_\nu\|_{2\to2}}{|\omega-\nu|}.
\tag{K}
\]

和はordered pairで取る。空和は零。`D=D_0+D_near+D_far`。厳密共鳴ω=νはD_0に保持し、near誤差へ捨てない。κは各couplingが零ならその項も零で、単なる最小gapの逆数とは異なる。ただし全spaceのblock normを使うため、状態に不要なcouplingを含む保守性がある。

## 3. 散逸性と安定性：非CPTPでも使える構造

### 補題1：二次補正はHS空間で負半定値

`J_j(X)=P_jX+XP_j` と置く。J_jとL=E J_jはHS空間でself-adjoint、`norm(J_j)<=2`、`norm(L)<=2`。

直接展開すると、

\[
\mathcal D
=-\tfrac12\left(\mathbb E\mathcal J_j^2-\mathcal L^2\right)
=-\tfrac12\mathbb E(\mathcal J_j-\mathcal L)^2.
\tag{V}
\]

実際、`E J_j^2(X)=2X+2 E[P_j X P_j]`、`L^2(X)=h^2X+2hXh+Xh^2`。この差は§2のDと一致する。E J_j=Lなので、中央の二つの交差項もL^2へ平均される。

従って任意のXに対し

\[
\langle X,\mathcal DX\rangle_2
=-\tfrac12\mathbb E\|(\mathcal J_j-\mathcal L)X\|_2^2\le0.
\]

また `0<=E J_j^2-L^2<=E J_j^2<=4 Id` より

\[
-2\,\mathrm{Id}\le\mathcal D\le0,\qquad b\le2.
\]

直交pinchingによりD_0も負半定値、`b_0<=b`。LとD_0は可換である。□

### 補題2：必要なpropagatorはHS収縮

Φはunitaryな二側乗算の平均なので `norm(Phi_theta)<=1`。`G_theta=i theta L+theta^2 D` のHermitian部分はtheta^2 D≤0。従って任意t≥0で、

\[
\|e^{t\mathcal G_\theta}\|_{2\to2}\le1,
\quad\|e^{t\theta^2\mathcal D_0}\|_{2\to2}\le1,
\quad\|e^{it\theta\mathcal L}\|_{2\to2}=1.
\]

第一式は `d norm(e^{tG}X)^2/dt=2 theta^2 ⟨e^{tG}X,D e^{tG}X⟩<=0` から従う。これはΦやDをCPTPとした証明ではなく、Hilbert空間上の散逸性である。Φが非正規でもその固有射影の条件数を仮定せず、この安定性を使える。□

## 4. 一stepの局所高次誤差：定数18

### 補題3

実theta、`abs(theta)<=1` に対し、

\[
\|\Phi_\theta-e^{\mathcal G_\theta}\|_{2\to2}
\le18|\theta|^3.
\tag{L}
\]

**証明。** `exp(i theta J_j)(X)=A_j X A_j`。unitary groupの積分形Taylor remainderを用いると、

\[
\Phi_\theta=\mathrm{Id}+i\theta\mathcal L
-\tfrac{\theta^2}{2}\mathbb E\mathcal J_j^2+R_\Phi,
\quad \|R_\Phi\|\le\tfrac{8}{6}|\theta|^3.
\]

一方G=G_thetaについて、補題2の `norm(e^{uG})<=1` と積分形remainderから

\[
e^G=\mathrm{Id}+G+\tfrac12G^2+R_G,
\quad\|R_G\|\le\tfrac16\|G\|^3
\le\tfrac{64}{6}|\theta|^3.
\]

最後は `norm(G)<=2 abs(theta)+2 theta^2<=4 abs(theta)` を使用した。二次までの共通部分は

\[
\mathrm{Id}+i\theta\mathcal L
+\theta^2(\mathcal D-\tfrac12\mathcal L^2)
=\mathrm{Id}+i\theta\mathcal L-\tfrac{\theta^2}{2}\mathbb E\mathcal J_j^2.
\]

G^2に含まれる追加項は `i theta^3(LD+DL)/2 +theta^4 D^2/2` で、そのnormは `4 abs(theta)^3+2 abs(theta)^4<=6 abs(theta)^3`。ゆえに `8/6+64/6+6=18` を得る。□

この補題に現れる1は規格化されたJ_jの解析的評価域であり、数値探索・実験閾値の採択ではない。定数18は鋭さを主張しない。

### 系：N反復の局所誤差

ΦとF=e^Gがともに収縮なので、非可換なtelescoping identityにより

\[
\Phi^N-F^N=\sum_{m=0}^{N-1}\Phi^{N-1-m}(\Phi-F)F^m,
\]
\[
\|\Phi_\theta^N-e^{N\mathcal G_\theta}\|
\le18N|\theta|^3=18\tau|\theta|.
\tag{A}
\]

固定Nの展開を長時間へ無条件に延長せず、収縮性と明示remainderから累積を制御した。有限角度の位相歪みはこの局所誤差に含まれ、別に重ねて加算しない。

## 5. 平均化誤差：near/farを分けた積分評価

theta≠0とし、実連続変数t≥0で

\[
Y(t)=e^{-it\theta\mathcal L}e^{t\mathcal G_\theta},\quad
\mathcal D(t)=e^{-it\theta\mathcal L}\mathcal De^{it\theta\mathcal L},
\quad Z(t)=e^{t\theta^2\mathcal D_0}
\]

と置く。Yの方程式は `Y'=theta^2 D(t)Y`、Y(0)=Id。補題2よりY,Zのnormは1以下。D(t)のnormはb、D_0は時間不変。

Duhamel公式から

\[
Y(t)-Z(t)=\theta^2\int_0^t
Z(t-u)[\mathcal D_{\rm near}(u)+\mathcal D_{\rm far}(u)]Y(u)\,du.
\tag{D}
\]

near部分のnormは `theta^2 t b_near` 以下。ここは振動による相殺を仮定せず、低周波の結合そのものを会計に残す。

far部分のprimitiveを

\[
B(t)=\int_0^t\mathcal D_{\rm far}(u)\,du
=\sum_{|\omega-\nu|>\eta}
\frac{e^{i\theta t(\nu-\omega)}-1}{i\theta(\nu-\omega)}
\Pi_\omega\mathcal D\Pi_\nu
\]

とすると

\[
\sup_{t\ge0}\|B(t)\|\le2\kappa_{\rm far}(\eta)/|\theta|.
\tag{B}
\]

B(0)=0であり、far積分をpartsで積分する。

\[
\begin{aligned}
\int_0^t Z(t-u)B'(u)Y(u)du
={}&B(t)Y(t)\\
&+\theta^2\int_0^t Z(t-u)\mathcal D_0 B(u)Y(u)du\\
&-\theta^2\int_0^t Z(t-u)B(u)\mathcal D(u)Y(u)du.
\end{aligned}
\]

従って、tau_t=t theta^2とおくと、

\[
\|Y(t)-Z(t)\|
\le \tau_t b_{\rm near}
+2|\theta|\kappa_{\rm far}[1+\tau_t(b+b_0)].
\tag{S}
\]

Y(t)の微分には**nearを含む全D(t)**を使った。near結合がある場合にfar誤差を誤って独立扱いしていない。この連続変数は解析用で、物理時間の変更でも科学simulationでもない。

## 6. 主定理の完成とstate/mode weight

t=Nで§4の(A)と§5の(S)を結び、左のunitary factorを戻すと、

\[
\|\Phi_\theta^N-e^{iN\theta\mathcal L}e^{\tau\mathcal D_0}\|_{2\to2}
\le18\tau|\theta|+\tau b_{\rm near}
+2|\theta|\kappa_{\rm far}[1+\tau(b+b_0)].
\tag{O}
\]

matrix elementのfunctional `X↦⟨k|X|k⟩` のHS dual normは `norm(|k><k|)_2=1`、入力IのHS normは√n。よって(T1a)が従う。theta=0またはN=0は両側写像が同じであり別途整合する。η=0では有限集合の異なる周波数の差は非零なのでκ_far(0)は有限で、(U)も従う。□

### 6.1 Weightの扱いと限界

主上界は**全演算子空間のnorm**を使い、入力Iのweightを√n、対象stateのfunctionalを1として開示した。量子化学サイズやdense spectrumに安いstate-specific predictorを得たとは主張しない。Π_ω D Π_νが零のcouplingをκから落とせるが、残るblock normはstateに不要な方向を含み得る。

補助的に、HS空間の直交射影PがL,Dの両方と可換で、`P(|k><k|)=|k><k|` を満たす場合、**平均化部分だけ**はP空間に制限できる。このとき(S)に由来する項の√nを `norm(P I)_2` に替え、b,b_0,b_near,κを制限空間の値にできる。理由はPが各Π_ω、D_0、Y,Zとも可換で、対象functionalが(1−P)成分を消すためである。

局所誤差(A)は一般にΦの高次項がPを保存するとは限らないため、そのまま√nを保持する。これを全誤差のstate-weight改善と誤認しない。P=Idなら主定理に戻る。実際の分子に有用なPやweightの存在は未検証である。

### 6.2 共鳴・非正規性をどこで扱ったか

厳密共鳴はD_0内のblockとして保持し、近共鳴はb_nearとして残した。LはHS-self-adjointなのでΠ_ωは直交射影である。ΦやD_0を別途対角化してnon-normalなmodeの条件数を隠す証明は使っていない。

ただし共鳴block内のC-sec信号は一般に単一の正の減衰係数ではない。(T1a)が成立しても、rhoの単純式やsignal floorは別の条件を要する。hの孤立固有stateだけでpair-sumの単純性は保証されない。

## 7. Signal floor付きの単一信号位相保証

ある対象点で(T1a)の右辺をEとし、`abs(z_N^sec)>=r_*>0`、`E<r_*` とする。すると

\[
|z_N|\ge r_*-E>0,\qquad
\left|\operatorname{Arg}\frac{z_N}{z_N^{\rm sec}}\right|
\le\arcsin(E/r_*).
\tag{P}
\]

**証明。** `w=z_N/z_N^sec` は `abs(w−1)<=q=E/r_*<1` を満たし、Re w>0。w=t exp(iφ)と書くと `abs(w−1)^2=(t−cosφ)^2+sin^2φ`。従って `abs(sinφ)<=q`、`abs(φ)<pi/2` から主値の上界が従う。半径の下界は三角不等式である。□

floorは**C-sec参照信号**についての条件である。これは一つの信号の相対位相に関する系で、RPEのphase unwrapping・複数round・RMS error・required shotsを保証しない。

## 8. 単一pair-sum blockでの先導減衰係数

追加の十分条件として、`2e_k=e_a+e_b` を満たすpairが(k,k)だけとする。するとΠ_{2e_k}のrangeはE_kk=|k><k|の一次元で、

\[
\langle E_{kk},\mathcal D E_{kk}\rangle_2
=-(1-e_k^2)-\mathbb E(\langle k|P_j|k\rangle-e_k)^2
=-(V_k+v_{kk}).
\]

D_0は周波数を跨がず、Iのこのblock成分はE_kkだから、

\[
\boxed{z_N^{\rm sec}
=\exp(2iN\theta e_k)\exp[-\tau(V_k+v_{kk})].}
\tag{C}
\]

これは採択済みC-secの中の厳密な評価であり、(T1a)と併用して真の信号への誤差を得る。単一pair条件は**十分条件**で、一般の共鳴blockへこのscalar式を流用しない。結合が零なら緩和できる場合があるが、その一般的な最適条件は本書で主張しない。

V_k>0のとき、以前のレビューが記録した係数比 `(V_k+v_kk)/(2V_k)` は、このscalar C-secの減衰係数と整合する。しかし有限イベント数でのratio=その値、同RPE精度、native費用改善までを示したわけではない。V_k=0はratio未定義である。

## 9. 非可換な二準位参照：非空の保証域

過去レビューと同じ `P_1=Z, P_2=X, p_1=p_2=1/2` を**解析的な例**として用いる。新しい数値行列評価・sampling・反例探索ではない。

h=(X+Z)/2の固有basisで `h=Z_tilde/sqrt(2)`、`P_1=(Z_tilde+X_tilde)/sqrt(2)`、`P_2=(Z_tilde−X_tilde)/sqrt(2)` と書ける。従って

\[
\mathcal C(A)=\tfrac12\widetilde X A\widetilde X,\qquad
\mathcal D(A)=-\tfrac12 A-\tfrac12\widetilde X A\widetilde X.
\]

Ω={sqrt(2),0,−sqrt(2)}。DはE_++とE_--を係数−1/2で結び、E_+-とE_-+を同じ0周波数block内で結ぶ。よって

\[
b=b_0=1,\qquad
\kappa_{\rm far}(0)=\frac{1}{2\sqrt2},\qquad
V_\pm=\frac12,\quad v_{\pm\pm}=0.
\]

これらは上の表示を手計算で作用させた解析値で、外部数値出力ではない。主定理よりtau≤D_capで

\[
z_N^{\rm sec}=e^{\pm i\sqrt2 N\theta}e^{-\tau/2},\qquad
|z_N-z_N^{\rm sec}|
\le |\theta|[18\sqrt2 D_{\rm cap}+1+2D_{\rm cap}].
\tag{X}
\]

`C_*=18 sqrt(2)D_cap+1+2D_cap` とおく。任意の有限D_capに対して

\[
0<|\theta|<\min\{1,e^{-D_{\rm cap}/2}/C_*\}
\]

は非空であり、**全対象N**でfloor `r_*=e^{-D_cap/2}` と(P)を使える。有限減衰を保ったまま位相誤差はO(abs(theta))となる。

さらにtau→tau_*>0とする系列では、理想の単位半径信号 `u_N=e^{±i sqrt(2)N theta}` に対する真の複素誤差は、

\[
|z_N-u_N|\longrightarrow 1-e^{-\tau_*/2}>0,
\]

である一方、相対位相誤差は零へ近づく。これは(X)と三角不等式の上下評価による。**有限の半径損失と小さい位相誤差を区別する非可換例**になっている。

比較targetが異なるnorm boundを数値的に打ち負かしたという主張ではない。理想unitaryへの複素/norm距離をそのまま位相誤差と読む保守性を、この例では明示できるという結果である。一般分子、部分乱択、有限Nのイベント半減、総資源改善を意味しない。

### 9.1 同じlawのchannel保証と何が違うか

ancillaを加えたunitaryを `U_j=diag(A_j^dagger,A_j)`、CPTP写像を `T_theta(rho)=E[U_j rho U_j^dagger]` とする。その10 blockは厳密にΦ_thetaである。一方、通常の片枝channel `E[A_j X A_j^dagger]` の写像そのものはΦではない。この区別を保てば、同じdirectional lawを保ったchannelとcoherent信号を対応させられる。

初期密度行列 `rho_in=|+><+| tensor |k><k|` をT_thetaでN回進めた後のancilla coherenceはz_N/2である。これは `tr[Phi(X)Y]=tr[X Phi(Y)]` という二側乗算のtrace恒等式から、`tr[Phi^N(E_kk)]=tr[E_kk Phi^N(I)]=z_N` となるためである。理想の `diag(exp(−iN theta h),exp(iN theta h))` によるcoherenceは `u_N/2`。systemをtraceしたancillaの差は対角成分が零、off-diagonal成分が `(z_N−u_N)/2` なので、そのtrace normは `abs(z_N−u_N)`。traceの収縮性から、**この同じlawを理想unitary channelと比較するdiamond norm**は少なくとも `abs(z_N−u_N)` である（diamond distanceを1/2倍で定義する場合は右辺も1/2倍）。

従って§9の非可換例・有限tau_*では、このideal-channel距離は正の下界を持つ。同時に、本書はC-secへのO(abs(theta))の複素誤差と、floor付きのideal位相誤差を示している。既知のideal-channel保証が誤りという意味ではなく、**有限の半径損失を許した位相評価とは保証targetが異なる**。既存qDRIFT定理の数値係数・適用域をここで輸入したわけでも、一般のchannel-based位相解析を排除したわけでもない。

## 10. 契約との対応・未達事項

| S0の義務 | 本書の提出内容 | 残る限界 |
|---|---|---|
| C-sec主比較 | 同じL,D_0,z_secを固定し(T1a)を証明 | C-log targetへ変更していない |
| 長時間一様性 | 全Nでtau≤D_cap、明示定数18とκ,b_near | 固定モデルのみ。size/gap familyに一様でない |
| 局所高次・有限角度 | 補題3とtelescopingで18 tau abs(theta) | 定数は保守的、最適化していない |
| 非共鳴・低周波 | κ_farとb_nearを分け、exact resonanceはD_0へ保持 | 高次元のstate-specific weightは未解決 |
| state/mode/norm | HS norm、√n、対象functional norm1、補助射影の条件 | global normの次元負担が残る |
| signal floorと位相 | 参照floorの下で(P)、非可換例の非空域 | RPE/shot/多roundは未実施 |
| 先導係数 | 単一pair-sum条件で(C)、二準位で(X) | 共鳴blockで一般scalar rhoを認定しない |
| T1bへの可搬性材料 | 散逸性、明示local remainder、interaction-picture近共鳴分解 | **T1bの移送は一切証明していない** |

著者の技術的提出区分は **`PROVED_WITH_DOMAIN`（この文書の自己完結した証明に基づく著者主張）**。独立重要科学レビューは未実施で、Checkpoint 1の受理・研究価値・新規性・T1b GOは判断していない。証明の誤りが独立検討で見つかれば、訂正を新しい版・履歴へ残す。

## 11. 先行研究の境界と確認範囲

この証明の道具は有限次元の散逸semigroup、Taylor積分剰余、Duhamel、振動積分のpartsであり、既知の平均化の考え方に沿う。**これら一般手法の発明や独立新規性を主張しない**。採択レビューが挙げたDaviesの[出版社ページ][DAVIES]で書誌とabstractを確認した。同abstractの対象は弱結合のopen-system master equationであり、本書の離散coherent写像へ定理を適用した証拠にはしていない。全文の定理・証明は今回取得していない。

本書の具体的な上界は外部定理の未確認の適用に依存せず、§§3–6で証明を全記した。Project Euclidの経路はiframeのみで本文を取得できず、全文読了扱いにしていない。PR/Campbell/Gu等との詳細対応はS0と既存監査を参照し、今回網羅的新規性調査を繰り返していない。既知の直接系以上の価値があるかはCheckpoint 1の科学判断へ残す。

## 12. 停止位置

T1aの解析的証明と[Checkpoint 1 Decision Packet][CP1]を提出する。数値的validation・CI再現・分子の性能結果はない。T1b、Gate 0/E0、Q4/Q5、一般DF event、回路・compile・資源forecastには進まない。

**Checkpoint 1での重要科学判断は未実施。次の研究段階への実行許可はなく、ここでmandatory STOP。**

[S0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/373b8e4b9831f4d00bdc00ffc741ac61e03f0841/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md
[AUTH]: ../../../artifacts/track_b_m1s_g0t1_t1a_theory/2026-10-16/v1/user_authorization_record.json
[CP1]: m1s_g0t1_t1a_checkpoint1_decision_packet_20261016.md
[DAVIES]: https://link.springer.com/article/10.1007/BF01608389
