# 「研究方法・解析手順」補足Q&A

## この資料の位置付け

本資料は、「研究方法・解析手順.md」のうち、外側・内側最適化の関係、$C_{\mathrm{gs}}$ proxy、$\lambda_R$、$\eta$、信号半径 $\rho_m$ および簡単な最適化例を補足するものである。

## Q1. なぜ $r_m$ と $K_m$ の最適化より先に $L_D$ と $\delta_{\mathrm{time}}$ を決めるのか

$L_D$ と $\delta_{\mathrm{time}}$ を最初に最終決定するわけではない。外側候補 $(L_D,\delta_{\mathrm{time}})$ を1つ仮固定し、その条件下で内側変数を最適化する。

$$
G_{\mathrm{opt}}(L_D,\delta_{\mathrm{time}})
=
\min_{\{r_m,K_m,\overline{\beta}_{j,m},\alpha_{m,b}\}}
G_{\mathrm{total}}.
$$

その後、すべての外側候補について得た $G_{\mathrm{opt}}$ を比較する。したがって、処理の意味は

$$
\text{外側候補を仮固定}
\longrightarrow
\text{内側最適化}
\longrightarrow
\text{外側候補間の比較}
$$

である。

## Q2. $L_D$ と $\delta_{\mathrm{time}}$ を外側変数として分ける理由は何か

両者が、内側問題を定義する別の要素だからである。

| 外側変数 | 主に決める量 |
|---|---|
| $L_D$ | $H_D$、$H_R$、$\lambda_R$、決定論回路、RTE event分布 |
| $\delta_{\mathrm{time}}$ | PF誤差、RPE round数 $M$、発展時間 $t_m$、RTE無次元時間 $\tau_m$ |

ただし、$(L_D,\delta_{\mathrm{time}})$ を決めても、$r_m$、$K_m$、誤差配分および失敗確率配分は一意には決まらない。これらにはコストと誤差のtrade-offがあるため、内側で比較する必要がある。

## Q3. $C_{\mathrm{gs}}$ ではランダム部分を無視しているのか

Hamiltonian $H_R$ 自体を無視しているわけではない。部分ランダム化した外側Product Formulaを、例えば二次の場合、

$$
S_2^{(L_D)}(\delta)
=
e^{-i\delta H_1/2}\cdots
e^{-i\delta H_{L_D}/2}
e^{-i\delta H_R}
e^{-i\delta H_{L_D}/2}\cdots
e^{-i\delta H_1/2}
$$

と考え、$H_R$ を1つの厳密なHamiltonian blockとして含める。この外側分割によるenergy biasが

$$
\left|E_0-\widetilde E_0^{(L_D)}(\delta)\right|
\leq
C_{\mathrm{gs}}(L_D)\delta^p
$$

である。

一方、この $e^{-i\delta H_R}$ を有限RTE回路でどのように実装するかは、$C_{\mathrm{gs}}$ に含めない。したがって、$C_{\mathrm{gs}}$ の評価ではRTEのランダム回路を標本化しないという意味では、ランダム実装の詳細を切り離している。

本研究では、

- 外側Product Formulaのenergy bias：$C_{\mathrm{gs}}\delta^p$
- 有限RTEの信号誤差：$\epsilon_{Z,m}$
- RTE normalizationによるattenuation：$A_m^{\mathrm{att}}$

を別々に評価して、最後に総コストへ接続する。

## Q4. 分割ごとの真の $C_{\mathrm{gs}}(L_D)$ はどのように求めるのか

古典計算可能な小規模系であれば、各 $L_D$ について次を行う。

1. $H_R(L_D)$ を構成する。
2. $e^{-i\gamma H_R(L_D)}$ を厳密に含む $S_p^{(L_D)}(\delta_j)$ を複数の $\delta_j$ で構成する。
3. 実効Hamiltonianの基底エネルギー $\widetilde E_0^{(L_D)}(\delta_j)$ を求める。
4.

   $$
   \Delta E^{(L_D)}(\delta_j)
   =
   \left|E_0-\widetilde E_0^{(L_D)}(\delta_j)\right|
   \approx
   C_{\mathrm{gs}}(L_D)\delta_j^p
   $$

   をfitする。

この計算には行列指数、固有値計算または状態ベクトル計算が必要になるが、RTEのランダム回路を標本化する必要はない。

大規模系ではこれを全 $L_D$ について行うことが難しいため、PR論文は

$$
C_{\mathrm{gs}}(L_D)
\approx
C_{\mathrm{gs}}^{\mathrm{full-det}}
$$

という共通proxyを用いる。これは厳密な上界ではなく、PR論文が小規模系で妥当性を確認したheuristicである。本研究でもこれを基準評価に用い、可能な範囲で代表的な $L_D$ のみ検証する。

## Q5. 有限RTEを含む誤差を直接調べるには、ランダム回路のシミュレーションが必要か

直接的な数値評価を行うなら必要になる。有限RTEのevent平均から得られる複素信号を計算し、基準信号との差を求める。

ただし、本研究の大規模系評価では、直接シミュレーションを必須とせず、normalization補正後のround演算子に対する上界

$$
\epsilon_{Z,m}
\leq
\left\|
\widetilde U_{K_m,m}^{\mathrm{RTE}}
-
U_m^{\mathrm{PF}}
\right\|_{\mathrm{op}}
$$

を有限Taylor残差から評価する。この演算子上界が得られれば、任意の正規化状態について複素信号誤差も同じ値以下になる。

したがって、使い分けは次のとおりである。

| 目的 | 方法 |
|---|---|
| 小規模系で近似を検証する | event平均または状態ベクトルシミュレーション |
| 大規模系で候補を最適化する | 有限Taylor残差による解析的上界 |

## Q6. 「$\lambda_R$ を最小化する」とは何を意味するのか

ランダム部分を

$$
H_R(L_D)=\sum_j h_j^{(L_D)}P_j
$$

としたとき、

$$
\lambda_R(L_D)=\sum_j\left|h_j^{(L_D)}\right|
$$

である。$L_D$ を増やすと一般にランダム部分が小さくなり、$\lambda_R$ も減るため、RTEの回転数やattenuationは改善しやすい。

しかし、$\lambda_R$ だけを最小化すると、すべてを決定論部分へ入れた $\lambda_R=0$ が自明に選ばれる。これは本研究の目的ではない。実際には、

$$
\text{ランダム部分のコスト低下}
\quad\text{と}\quad
\text{決定論部分のコスト増加}
$$

を合わせたcompiled総コストで $L_D$ を比較する。

## Q7. $\delta_{\mathrm{time}}$ のalias条件は最適化手順に必要か

本研究では、energy shift・scaleと探索範囲を事前に調整し、候補とする $\delta_{\mathrm{time}}$ ではaliasが起こらないと仮定する。このため、alias条件を外側最適化の独立な制約として毎回評価しない。

$\delta_{\mathrm{time}}$ の候補は主に、

- Product Formula誤差
- RPE round数 $M$ の切替点
- 有限RTEの誤差・attenuation
- 最終的なcompiled総コスト

によって比較する。

## Q8. 以前の $\eta_m$ の式はどこから来たのか

以前の

$$
\eta_m
=
\frac{\rho_m^{\mathrm{obs}}}{\sqrt{2}}
\sin\!\left(\overline{\beta}_{\mathrm{stat},m}\right)
$$

は、cosine・sineそれぞれに許容する統計的な座標誤差を表していた。

各座標の誤差を $\epsilon_{\mathrm{coord},m}$ 以下とすると、複素信号全体の誤差は

$$
\left|\widehat Z_m-Z_m\right|
\leq
\sqrt{2}\,\epsilon_{\mathrm{coord},m}
$$

である。半径 $\rho_m^{\mathrm{obs}}$ の信号の角度ずれを $\overline{\beta}_{\mathrm{stat},m}$ 以下にする十分条件

$$
\sqrt{2}\,\epsilon_{\mathrm{coord},m}
\leq
\rho_m^{\mathrm{obs}}
\sin\!\left(\overline{\beta}_{\mathrm{stat},m}\right)
$$

から、この式が得られる。

ただし、PR論文では $\eta$ が基底状態populationの下界 $c_0\geq\eta$ を表す。本資料では混同を避けるため、座標誤差を $\eta_m$ ではなく

$$
\epsilon_{\mathrm{coord},m}
$$

と書き直した。厳密基底状態を仮定する場合、PR論文の意味での $\eta$ は1である。

## Q9. 大規模系で $\rho_m$ を求める必要はないのか

厳密基底状態と厳密時間発展に対しては、

$$
Z_m^{\mathrm{ideal}}
=
\langle\psi_0|e^{-iHt_m}|\psi_0\rangle
=
e^{-iE_0t_m}
$$

なので、基準信号半径は厳密に1である。

本研究の基準評価ではProduct Formulaの影響をenergy biasだけで表し、$Z_m^{\mathrm{PF}}$ も単位半径の信号として扱う。したがって、有限RTE誤差は

$$
\beta_{\mathrm{RTE},m}^{\mathrm{ub}}
\leq
\arcsin(\epsilon_{Z,m})
$$

と評価でき、未知の $\rho_m$ を計算する必要はない。

さらに、

$$
\left|\widetilde Z_m^{\mathrm{RTE}}-Z_m^{\mathrm{PF}}\right|
\leq
\epsilon_{Z,m}
$$

なら、attenuation後の観測信号半径には

$$
\rho_{m,\mathrm{lb}}^{\mathrm{obs}}
=
A_m^{\mathrm{att}}(1-\epsilon_{Z,m})
$$

という下界を使える。

注意点は、厳密基底状態 $|\psi_0\rangle$ がProduct Formulaの実効Hamiltonianの厳密固有状態であるとは限らないことである。したがって、単位半径はProduct Formulaによる固有ベクトル変化を無視する基準モデルであり、数学的に厳密な等式ではない。小規模系では

$$
\left|
\langle\psi_0|S_p(\delta)^{q_m}|\psi_0\rangle
\right|
$$

を計算し、この近似の影響を検証する。

## Q10. 簡単な数値例では、最適化はどのように進むか

以下の数値は、変数間の依存関係を示すための無次元化された架空の値であり、実Hamiltonianの計算結果ではない。詳細な算術計算ではなく、「何を選び、そこから何を計算するか」を順に示す。

### Step 0. 最適化前に与える値

この例では、次を入力として固定する。

| 入力量 | 例の値 | 決め方 |
|---|---:|---|
| Product Formula次数 $p$ | $2$ | 手法として事前に選択 |
| PF誤差proxy $C_{\mathrm{gs}}^{\mathrm{full-det}}$ | $0.02$ | 完全決定論的PFの $\Delta E(\delta)\approx C_{\mathrm{gs}}\delta^p$ からfit |
| RPE目標エネルギー精度 $\epsilon_{E,\mathrm{RPE}}$ | $0.50$ | 研究上の要求精度として指定 |
| RPE位相予算 $\beta_{\mathrm{RPE}}$ | $0.40$ | $\pi/3$ より小さい設計値として指定 |
| 総失敗確率 $\alpha_{\mathrm{tot}}$ | $0.10$ | 成功確率の要求から指定 |
| 入力状態 | -| $\ket{\psi_{0}}$ | 厳密基底状態を仮定 |
| ideal／PF信号半径 | $1$ | 本研究の単位半径基準モデル |
| compiler条件 | 固定 | backend、basis gate、最適化levelを指定 |

$C_{\mathrm{gs}}^{\mathrm{full-det}}$、$\epsilon_{E,\mathrm{RPE}}$、$\beta_{\mathrm{RPE}}$ および $\alpha_{\mathrm{tot}}$ は、この例の内側最適化では変更しない。

### Step 1. 外側候補 $L_D$ を1つ仮固定する

固定されたDF fragment列に対して、候補

$$
L_D=3
$$

を選ぶ。ここから

$$
H_D(L_D)=\sum_{\ell=1}^{L_D}H_\ell,
\qquad
H_R(L_D)=H-H_D(L_D)
$$

を構成し、ランダム部分の係数から

$$
\lambda_R(L_D)
=
\sum_j\left|h_j^{(L_D)}\right|
=1.2
$$

を計算する。同時に、決定論blockの回路構造、ランダム部分のcomponent確率

$$
p_j^{(L_D)}
=
\frac{|h_j^{(L_D)}|}{\lambda_R(L_D)}
$$

および各componentの回路情報を準備する。

ここで選ぶ値は $L_D$ であり、$H_D$、$H_R$、$\lambda_R$ および $p_j$ は $L_D$ から計算される。

### Step 2. 外側候補 $\delta_{\mathrm{time}}$ を1つ仮固定する

次に

$$
\delta_{\mathrm{time}}=0.10
$$

を選ぶ。共通proxyからPFのenergy bias上界を

$$
\epsilon_{\mathrm{PF}}
=
C_{\mathrm{gs}}^{\mathrm{full-det}}
\delta_{\mathrm{time}}^p
=0.0002
$$

と計算する。

現在用いているRPE精度モデルでは、round数を

$$
M
=
\left\lceil
\log_2
\frac{\beta_{\mathrm{RPE}}}
{\delta_{\mathrm{time}}\epsilon_{E,\mathrm{RPE}}}
\right\rceil
=3
$$

と求める。各roundの反復数と発展時間は

$$
q_m=2^m,
\qquad
t_m=q_m\delta_{\mathrm{time}}
$$

から計算する。

以下ではround $m=2$ を具体的に追う。このroundでは

$$
q_2=4,
\qquad
t_2=0.40
$$

であり、PFによる位相誤差上界は

$$
\beta_{\mathrm{PF},2}^{\mathrm{ub}}
=
t_2\epsilon_{\mathrm{PF}}
=0.00008
$$

となる。

### Step 3. roundの位相誤差と失敗確率を配分する

この例ではPFに対する許容量を実際の上界と等しく置き、RTEへの配分候補を

$$
\overline{\beta}_{\mathrm{PF},2}
=
\beta_{\mathrm{PF},2}^{\mathrm{ub}},
\qquad
\overline{\beta}_{\mathrm{RTE},2}=0.020
$$

と選ぶ。統計誤差の許容量は残余として

$$
\overline{\beta}_{\mathrm{stat},2}
=
\beta_{\mathrm{RPE}}
-
\overline{\beta}_{\mathrm{PF},2}
-
\overline{\beta}_{\mathrm{RTE},2}
=0.37992
$$

と計算する。

したがって、有限RTEに許される複素信号誤差は

$$
\epsilon_{Z,2}^{\mathrm{budget}}
=
\sin\!\left(
\overline{\beta}_{\mathrm{RTE},2}
\right)
\approx0.019999
$$

となる。ここで、$\overline{\beta}_{\mathrm{RTE},2}$ は配分候補として選ぶ値であり、$\epsilon_{Z,2}^{\mathrm{budget}}$ はそこから計算される値である。

失敗確率について、この例では全 $2(M+1)$ 個のcosine・sine系列へ等配分し、

$$
\alpha_{m,b}
=
\frac{\alpha_{\mathrm{tot}}}{2(M+1)}
=0.0125,
\qquad
b\in\{c,s\}
$$

とする。これはbaselineの配分であり、最終最適化では非一様配分も比較する。

### Step 4. 内側候補 $(r_2,K_2)$ から有限RTE誤差を計算する

各整数候補 $(r_2,K_2)$ に対して、1 short stepの無次元時間を

$$
\tau_2(r_2)
=
\frac{\lambda_R\delta_{\mathrm{time}}}{r_2}
$$

から計算する。paired RTEの1 short stepに用いるTaylor残差上界を

$$
\overline{R}_{K_2}(\tau_2)
=
\sum_{j=K_2+2}^{\infty}
\frac{|\tau_2|^j}{j!}
$$

であり、round全体のnormalization補正後の信号誤差上界を

$$
\epsilon_{Z,2}^{\mathrm{calc}}(r_2,K_2)
=
\left[
1+\overline{R}_{K_2}(\tau_2)
\right]^{q_2r_2}-1
$$

と計算する。この $\epsilon_{Z,2}^{\mathrm{calc}}$ は $(r_2,K_2)$ から導かれる値であり、自由に指定する値ではない。

そこから有限RTEによる実際の位相誤差上界を

$$
\beta_{\mathrm{RTE},2}^{\mathrm{ub}}(r_2,K_2)
=
\arcsin\!\left(
\epsilon_{Z,2}^{\mathrm{calc}}(r_2,K_2)
\right)
$$

と計算し、配分値 $\overline{\beta}_{\mathrm{RTE},2}=0.020$ 以下かを確認する。これは、$\epsilon_{Z,2}^{\mathrm{calc}}\leq\epsilon_{Z,2}^{\mathrm{budget}}$ と同値である。

| $(r_2,K_2)$ | $\tau_2$ | $\overline{R}_{K_2}(\tau_2)$ | $\epsilon_{Z,2}^{\mathrm{calc}}$ | $\beta_{\mathrm{RTE},2}^{\mathrm{ub}}$ | $\beta_{\mathrm{RTE},2}^{\mathrm{ub}}\leq0.020$ |
|---|---:|---:|---:|---:|---|
| $(1,0)$ | 0.120 | 0.007497 | 0.030326 | 0.030331 | 不可 |
| $(2,0)$ | 0.060 | 0.001837 | 0.014787 | 0.014788 | 可 |
| $(1,2)$ | 0.120 | $8.85\times10^{-6}$ | $3.54\times10^{-5}$ | $3.54\times10^{-5}$ | 可 |
| $(3,0)$ | 0.040 | 0.000811 | 0.009773 | 0.009773 | 可 |

$(1,0)$ は回路が短くてもRTE誤差予算を満たさないため、この時点で総コスト比較から除外する。

### Step 5. event分布とattenuationを計算する

実行可能な各 $(r_2,K_2)$ について、偶数Taylor次数 $n$ の係数、有限normalizationおよびorder確率を

$$
a_n(\tau_2)
=
\frac{|\tau_2|^n}{n!}
\sqrt{1+\frac{\tau_2^2}{(n+1)^2}},
$$

$$
B_{K_2}(\tau_2)
=
\sum_{\substack{0\leq n\leq K_2\\n\ \mathrm{even}}}
a_n(\tau_2),
\qquad
q_{K_2}(n)
=
\frac{a_n(\tau_2)}{B_{K_2}(\tau_2)}
$$

から計算する。1 short stepのeventを

$$
\omega=(n,\ell_1,\ldots,\ell_n,\ell_{\mathrm{rot}})
$$

と書けば、その確率は

$$
p_2(\omega)
=
q_{K_2}(n)
\left(\prod_{a=1}^{n}p_{\ell_a}\right)
p_{\ell_{\mathrm{rot}}}
$$

である。round全体のevent列の確率は、各short stepのevent確率の積から構成する。

round全体のattenuationは

$$
A_2^{\mathrm{att}}
=
B_{K_2}(\tau_2)^{-q_2r_2}
$$

である。

| $(r_2,K_2)$ | $B_{K_2}(\tau_2)$ | 主なorder確率 | $A_2^{\mathrm{att}}$ |
|---|---:|---|---:|
| $(2,0)$ | 1.001798 | $q_0(0)=1$ | 0.985729 |
| $(1,2)$ | 1.014380 | $q_2(0)=0.992896$, $q_2(2)=0.007104$ | 0.944490 |
| $(3,0)$ | 1.000800 | $q_0(0)=1$ | 0.990454 |

$B_{K_2}$ と $A_2^{\mathrm{att}}$ は有限RTEの定義式から計算される量であり、誤差配分として選ぶ量ではない。

### Step 6. 信号半径、座標誤差許容量、必要標本数を計算する

単位半径の基準モデルと有限RTE誤差上界から、観測信号半径の下界を

$$
\rho_{2,\mathrm{lb}}^{\mathrm{obs}}
=
A_2^{\mathrm{att}}
\left(
1-\epsilon_{Z,2}^{\mathrm{calc}}
\right)
$$

と計算する。次に、cosine・sine各座標に許される統計誤差を

$$
\epsilon_{\mathrm{coord},2}
=
\frac{\rho_{2,\mathrm{lb}}^{\mathrm{obs}}}{\sqrt{2}}
\sin\!\left(
\overline{\beta}_{\mathrm{stat},2}
\right)
$$

と求める。最後に、Hoeffding上界から各系列の量子回路実行回数を

$$
N_{2,b}
=
\left\lceil
\frac{2}{\epsilon_{\mathrm{coord},2}^2}
\log\!\left(
\frac{2}{\alpha_{2,b}}
\right)
\right\rceil
$$

と計算する。

| $(r_2,K_2)$ | $\rho_{2,\mathrm{lb}}^{\mathrm{obs}}$ | $\epsilon_{\mathrm{coord},2}$ | $N_{2,c}=N_{2,s}$ |
|---|---:|---:|---:|
| $(2,0)$ | 0.971153 | 0.254663 | 157 |
| $(1,2)$ | 0.944456 | 0.247663 | 166 |
| $(3,0)$ | 0.980774 | 0.257186 | 154 |

この順序から、$\epsilon_{\mathrm{coord},2}$ は直接選ぶ値ではなく、

$$
(r_2,K_2)
\rightarrow
\epsilon_{Z,2}^{\mathrm{calc}},B_{K_2}
\rightarrow
A_2^{\mathrm{att}},\rho_{2,\mathrm{lb}}^{\mathrm{obs}}
\rightarrow
\epsilon_{\mathrm{coord},2}
\rightarrow
N_{2,b}
$$

の順に計算される派生量であることが分かる。

### Step 7. full-circuitの期待compiled costを計算する

各候補のevent確率 $p_2(\omega)$ に従ってevent列を生成し、決定論block、RTE blockおよびHadamard-test回路を結合してcompileする。各event列のコストを

$$
C_{2,b}^{\mathrm{full}}(\omega)
$$

とし、exact列挙できない場合には、古典標本数 $S_{\mathrm{MC}}$ を使って

$$
\widehat C_{2,b}
=
\frac{1}{S_{\mathrm{MC}}}
\sum_{s=1}^{S_{\mathrm{MC}}}
C_{2,b}^{\mathrm{full}}(\omega_s),
\qquad
\widehat{\mathrm{SE}}(\widehat C_{2,b})
=
\frac{s_{C,2,b}}{\sqrt{S_{\mathrm{MC}}}}
$$

を計算する。$S_{\mathrm{MC}}$ は期待コストの推定精度が停止条件を満たすまで増やす値であり、量子回路実行回数 $N_{2,b}$ とは別である。

以下は、このcompileとMonte Carloから得られたと仮定する架空の期待コストである。

| $(r_2,K_2)$ | $\widehat C_{2,c}$ | $\widehat C_{2,s}$ | round総コスト $G_2=N_{2,c}\widehat C_{2,c}+N_{2,s}\widehat C_{2,s}$ |
|---|---:|---:|---:|
| $(2,0)$ | 64 | 66 | 20,410 |
| $(1,2)$ | 54 | 56 | 18,260 |
| $(3,0)$ | 80 | 84 | 25,256 |

したがって、このroundと誤差配分では

$$
(r_2^*,K_2^*)=(1,2)
$$

となる。$N_{2,b}$ が最小の候補ではなく、$N_{2,b}$ と期待compiled costを掛けた $G_2$ が最小の候補を選んでいる。

### Step 8. 全roundを合計し、配分も比較する

同じ計算を $m=0,1,\ldots,M$ に対して行い、

$$
G_m
=
\sum_{b\in\{c,s\}}
N_{m,b}\widehat C_{m,b},
\qquad
G_{\mathrm{total}}
=
\sum_{m=0}^{M}G_m
$$

を計算する。

ここまでの計算は1つの誤差・失敗確率配分に対する結果である。$\overline{\beta}_{\mathrm{RTE},m}$、$\overline{\beta}_{\mathrm{stat},m}$ および $\alpha_{m,b}$ の配分候補を変えると、実行可能な $(r_m,K_m)$ と $N_{m,b}$ が変わる。そのため、外側候補に対する内側最適コストは

$$
G_{\mathrm{opt}}(L_D,\delta_{\mathrm{time}})
=
\min_{\{r_m,K_m,\overline{\beta}_{j,m},\alpha_{m,b}\}}
G_{\mathrm{total}}
$$

として求める。

### Step 9. 外側候補を比較する

各 $(L_D,\delta_{\mathrm{time}})$ についてStep 1からStep 8までを繰り返す。以下の総コストは、各外側候補で内側最適化まで完了したと仮定する架空の値である。

| $L_D$ | $\delta_{\mathrm{time}}$ | $\lambda_R$ | $M$ | $G_{\mathrm{opt}}(L_D,\delta_{\mathrm{time}})$ |
|---:|---:|---:|---:|---:|
| 2 | 0.10 | 2.0 | 3 | 64,800 |
| 3 | 0.10 | 1.2 | 3 | 59,360 |
| 4 | 0.10 | 0.5 | 3 | 61,200 |
| 3 | 0.05 | 1.2 | 4 | 60,500 |

この例では、外側の最適解は

$$
(L_D^*,\delta_{\mathrm{time}}^*)=(3,0.10)
$$

となる。$\lambda_R$ が最小の候補や、$\delta_{\mathrm{time}}$ が最小の候補が自動的に最適になるわけではない。

### Step 10. 値の依存関係のまとめ

| 値 | 種別 | 何から決まるか |
|---|---|---|
| $L_D$, $\delta_{\mathrm{time}}$ | 外側探索変数 | 候補として仮固定し、最後に比較 |
| $H_D$, $H_R$, $\lambda_R$, $p_j$ | 派生量 | $L_D$ とHamiltonian係数 |
| $\epsilon_{\mathrm{PF}}$ | 派生量 | $C_{\mathrm{gs}}^{\mathrm{full-det}}\delta_{\mathrm{time}}^p$ |
| $M$ | 派生量 | $\beta_{\mathrm{RPE}}$, $\delta_{\mathrm{time}}$, $\epsilon_{E,\mathrm{RPE}}$ |
| $q_m$, $t_m$ | 派生量 | $q_m=2^m$, $t_m=q_m\delta_{\mathrm{time}}$ |
| $\beta_{\mathrm{PF},m}^{\mathrm{ub}}$ | 派生量 | $t_m\epsilon_{\mathrm{PF}}$ |
| $\overline{\beta}_{\mathrm{RTE},m}$, $\alpha_{m,b}$ | 配分変数 | 配分候補として選び、総コストで比較 |
| $\overline{\beta}_{\mathrm{stat},m}$ | 配分変数または残余 | この例では $\beta_{\mathrm{RPE}}-\overline{\beta}_{\mathrm{PF},m}-\overline{\beta}_{\mathrm{RTE},m}$ |
| $\epsilon_{Z,m}^{\mathrm{budget}}$ | 派生量 | $\sin(\overline{\beta}_{\mathrm{RTE},m})$ |
| $r_m$, $K_m$ | 内側探索変数 | 整数候補として列挙 |
| $\tau_m$ | 派生量 | $\lambda_R\delta_{\mathrm{time}}/r_m$ |
| $\overline{R}_{K_m}(\tau_m)$ | 派生量 | Taylor remainder上界 |
| $\epsilon_{Z,m}^{\mathrm{calc}}$ | 派生量 | $[1+\overline{R}_{K_m}(\tau_m)]^{q_mr_m}-1$ |
| $\beta_{\mathrm{RTE},m}^{\mathrm{ub}}$ | 派生量 | $\arcsin(\epsilon_{Z,m}^{\mathrm{calc}})$ |
| $a_n$, $B_{K_m}$, $q_{K_m}(n)$, $p_m(\omega)$ | 派生量 | 有限RTE定義とHamiltonian component確率 |
| $A_m^{\mathrm{att}}$ | 派生量 | $B_{K_m}(\tau_m)^{-q_mr_m}$ |
| $\rho_{m,\mathrm{lb}}^{\mathrm{obs}}$ | 派生量 | $A_m^{\mathrm{att}}(1-\epsilon_{Z,m}^{\mathrm{calc}})$ |
| $\epsilon_{\mathrm{coord},m}$ | 派生量 | $\rho_{m,\mathrm{lb}}^{\mathrm{obs}}\sin(\overline{\beta}_{\mathrm{stat},m})/\sqrt{2}$ |
| $N_{m,b}$ | 派生量 | $\epsilon_{\mathrm{coord},m}$ と $\alpha_{m,b}$ をHoeffding式へ代入 |
| $C_{m,b}^{\mathrm{full}}(\omega)$ | compile結果 | event列ごとのfull circuitをcompile |
| $\widehat C_{m,b}$, $\widehat{\mathrm{SE}}$ | 古典推定量 | $S_{\mathrm{MC}}$ 個のcompiled cost標本 |
| $G_m$, $G_{\mathrm{total}}$ | 派生量 | $N_{m,b}$ と期待compiled costを掛けて全系列・roundで合計 |
| $G_{\mathrm{opt}}$ | 最適化結果 | 誤差・確率制約を満たす内側候補の最小総コスト |

## 参考箇所

- [PR論文 Sec. V：部分ランダム化Product Formulaと $C_{\mathrm{gs}}$](https://arxiv.org/pdf/2503.05647v2#page=8)
- [PR論文 Sec. VI.B：完全決定論的 $C_{\mathrm{gs}}$ を用いるheuristic](https://arxiv.org/pdf/2503.05647v2#page=11)
- [PR論文 Appendix D：$C_{\mathrm{gs}}$ の数値評価と分割依存性](https://arxiv.org/pdf/2503.05647v2#page=34)
- [PR論文 Sec. II：厳密基底状態とground-state overlap $\eta$](https://arxiv.org/pdf/2503.05647v2#page=3)
