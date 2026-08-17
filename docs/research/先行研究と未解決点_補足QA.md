# 「先行研究と未解決点」補足Q&A

## この資料の位置付け

本資料は、「先行研究と未解決点.md」を読んだ際に生じた疑問を補足するものである。証明を詳細に再掲するのではなく、各式が表す量、式が必要になる理由、似た記号の区別をQ&A形式で整理する。

記号は原則として次の意味で用いる。

| 記号 | 意味 |
|---|---|
| $q_m=2^m$ | RPE round $m$ におけるpartial-S2の反復数 |
| $\delta$ | 1回のpartial-S2に含まれるtail occurrenceの物理時間 |
| $\lambda_R$ | ランダム部分 $H_R$ の1-norm |
| $t_{R,m}$ | round $m$ 全体におけるランダム部分の総無次元時間 |
| $r_m$ | 1回のtail occurrence内のRTE short-step数 |
| $R_m^{\mathrm{PR}}$ | PR論文でround $m$ 全体に用いるRTE event数／rotation component数 |
| $\tau_m$ | 1 RTE short step当たりの無次元時間 |
| $K_m$ | round $m$ の有限偶数Taylor cutoff |

---

## Q1. なぜ総無次元時間に $\lambda_R$ が入るのか

ランダム部分を

$$
H_R=\sum_\ell h_\ell P_\ell
$$

とし、

$$
\lambda_R=\sum_\ell |h_\ell|,
\qquad
p_\ell=\frac{|h_\ell|}{\lambda_R}
$$

と定義する。係数の符号をPauli演算子へ吸収すれば、

$$
H_R
=
\lambda_R\overline H_R,
\qquad
\overline H_R=\sum_\ell p_\ell P_\ell
$$

と書ける。ここで $\overline H_R$ は確率分布 $p_\ell$ で正規化されたHamiltonianであり、$\lambda_R$ がランダム部分全体の強さを表す。

RPE round $m$ の物理的な総発展時間は

$$
T_m=q_m\delta
$$

なので、

$$
e^{-iT_mH_R}
=
e^{-i(\lambda_RT_m)\overline H_R}
$$

となる。したがって、正規化Hamiltonian $\overline H_R$ に対する自然な時間変数は

$$
\boxed{
t_{R,m}
=
\lambda_R T_m
=
\lambda_R\delta q_m
=
\lambda_R\delta2^m
}
$$

である。

$\hbar=1$ では、$\lambda_R$ はenergy、$T_m$ はinverse energyの次元を持つため、$t_{R,m}$ は無次元である。

役割を分けると、

- $p_\ell$：どのPauli componentを選ぶか
- $t_{R,m}=\lambda_R\delta q_m$：ランダム時間発展全体の強さ

を決める。RTEのTaylor係数に現れるのも物理時間 $T_m$ 単独ではなく、この無次元時間である。

---

## Q2. round誤り確率を「幾何学的に緩める」とは何か

PR論文のRPEでは、早いroundのbranch選択を誤ると、その後のroundで誤ったbranchを高精度化してしまい、最終エネルギー誤差への影響が大きくなる。一方、遅いroundの誤りは、より細かい桁の決定にだけ影響する。

そのため、早いroundほど誤り確率を小さくし、roundが進むにつれて許容誤り確率を一定倍率で大きくする。PR論文のRMSE解析では、概念的に

$$
P_{\mathrm{err},m}
\leq
\xi^2 4^{-a(M-m)},
\qquad a>1
$$

のような配分を用いる。

これは、

> 影響の大きい序盤のbranch判定を強く保護し、影響が小さい後半では必要以上に標本を使わない

ための配分である。各roundの誤り確率を単純に等しくするよりも、RPEの階層構造と最終RMSEへの影響を反映している。

なお、この確率はPR論文がRMSEを評価するために内部で用いるround誤り確率であり、本研究で明示的に配分する総失敗確率 $\alpha_{\mathrm{tot}}$ やbranch別の $\alpha_{m,b}$ と同一ではない。

---

## Q3. なぜ $N_m$ は $M-m$ に対して線形になるのか

Hadamard testの標本平均の標準偏差は $N_m^{-1/2}$ で減少する。一方、固定された判定marginを超えてround判定を誤る確率は、集中不等式により標本数 $N_m$ に対して指数的に小さくなる。

PR論文では、round誤り確率を $M-m$ に対して幾何学的に変化させる。指数的に減る誤り確率を逆算すると、必要標本数はその確率の対数に比例するため、$M-m$ に対して線形になる。

この考え方から、attenuationがない場合の標本scheduleを

$$
N_m^{(0)}
=
11+4(M-m)
$$

としている。$11$ と $4$ は、RPEの既存解析とPR論文の数値的な定数評価に基づく。

RTE normalizationによる信号減衰 $B_m^{-1}$ がある場合は、同じ精度を得るために標本数を $B_m^2$ 倍し、

$$
N_m
=
B_m^2N_m^{(0)}
$$

とする。PR論文の $B_m\leq e^{1/\kappa}$ を使うと、

$$
N_m
=
e^{2/\kappa}
\left[11+4(M-m)\right]
$$

となる。

ここで「指数的に減る」のは標準偏差そのものではなく、固定marginを越える大偏差確率、すなわちroundの誤判定確率である。

---

## Q4. $\kappa$ はどこから導入され、$\tau_m$ とどう関係するのか

$\kappa$ はHamiltonianから決まる物理定数ではなく、1回路の長さとRTE normalizationによる信号減衰の交換関係を調整する無次元の設計パラメータである。

round $m$ 全体の総無次元時間 $t_{R,m}$ を $R_m^{\mathrm{PR}}$ 個のRTE short stepsへ分けると、1 short step当たりの無次元時間は

$$
\tau_m
=
\frac{t_{R,m}}{R_m^{\mathrm{PR}}}
$$

である。また、PR論文のnormalization上界は

$$
B_m
\leq
\exp\!\left(
\frac{t_{R,m}^2}{R_m^{\mathrm{PR}}}
\right)
$$

である。そこで、

$$
\boxed{
R_m^{\mathrm{PR}}
=
\kappa t_{R,m}^2
}
$$

と置くと、

$$
B_m\leq e^{1/\kappa}
$$

となり、normalization上界をroundに依存しない形へ整理できる。

同時に、

$$
\boxed{
\tau_m
=
\frac{1}{\kappa t_{R,m}}
=
\frac{1}{\kappa\lambda_R\delta q_m}
}
$$

となる。したがって、$\kappa$ を大きくすると、

$$
R_m^{\mathrm{PR}}\uparrow,
\qquad
\tau_m\downarrow,
\qquad
B_m\downarrow
$$

となる。つまり、回路は長くなるが信号減衰は小さくなる。

本研究の $r_m$ を1 tail occurrence当たりのshort-step数とすると、

$$
R_m^{\mathrm{PR}}=q_mr_m
$$

より、PR論文のbaselineに対応する値は

$$
\boxed{
r_m^{\mathrm{PR,occ}}
=
\kappa\lambda_R^2\delta^2q_m
}
$$

である。PR論文は全roundで共通の $\kappa$ を用いるが、本研究では有限 $K_m$ とcompiled costを含めて、整数 $r_m$ をroundごとに比較する。

---

## Q5. なぜTaylor打切り誤差が $(1+R_K)^n-1$ で評価されるのか

1 RTE short stepの理想時間発展を $U_j$、有限Taylor近似を $\widetilde U_j$ とし、

$$
\widetilde U_j=U_j+E_j,
\qquad
\|E_j\|
\leq
R_{K_m}(\tau_m)
$$

とする。

round全体では複数の近似演算子を積として用いるため、積を展開すると、誤差項 $E_j$ が1個含まれる項だけでなく、2個以上同時に含まれる高次項も現れる。triangle inequalityとoperator normのsubmultiplicativityを用いて、これらをすべてworst caseで足し合わせると、$n$ short stepsに対して

$$
\left|
\widetilde U_n\cdots\widetilde U_1
-
U_n\cdots U_1
\right|
\leq
\left(
1+R_{K_m}(\tau_m)
\right)^n-1
$$

となる。

round $m$ では

$$
n=r_mq_m
$$

なので、

$$
\boxed{
\epsilon_{T,m}
\leq
\left(
1+R_{K_m}(\tau_m)
\right)^{r_mq_m}-1
}
$$

を得る。

これは確率的な「stepの失敗確率」を合成した式ではなく、operator normによるdeterministicなworst-case誤差上界である。$R_{K_m}(\tau_m)$ が十分小さい場合は、一次近似として

$$
\epsilon_{T,m}
\approx
r_mq_mR_{K_m}(\tau_m)
$$

と理解できるが、安全側の評価では高次項を含む上の式を使う。

---

## Q6. 「ランダム回転数」と「RTE short-step数」は同じなのか

概念としては異なる。ただし、本研究で採用しているpaired RTEでは両者が1対1に対応する。

Taylor orderが $n$ の1 RTE eventは、概念的に

$$
U_\omega
=
(\text{Taylor phase})
\,P_{\ell_n}\cdots P_{\ell_1}
e^{-i\theta_nP_{\ell_{\mathrm{rot}}}}
$$

という構造を持つ。各eventには、

- $n$ 個のproduct component
- 1個のrotation component

が含まれる。

したがって、paired RTEでは

$$
\boxed{
N_{\mathrm{short\ step},m}
=
N_{\mathrm{rotation\ component},m}
=
q_mr_m
}
$$

となる。この1対1対応を前提とすれば、PR論文の「Pauli rotation数」を本研究のRTE short-step数へ対応させられる。

ただし、$q_mr_m$ は次の量とは一致しない。

1. ランダムに選ばれるcomponentの総数

   Taylor orderが $n$ のeventには $n+1$ 個のcomponentがあるため、round全体のcomponent数はevent列に依存する。

2. コンパイル後のRZ、CX、depth、size

   DF basis transform、Pauli演算、controlled化、隣接block間のcancellationなどがあるため、short-step数からnative gate countを直接求めることはできない。

したがって、まとめmdの

$$
R_m^{\mathrm{PR}}=q_mr_m
$$

は、より正確には次のように読む必要がある。

> $q_mr_m$ はround全体のRTE short-step数である。paired RTEでは各short stepが1個のrotation componentを持つため、rotation component数も同じになる。ただし、これはランダムcomponent総数やcompiled gate countを意味しない。

---

## Q7. これらの量はどの順番で決まるのか

round $m$ における主な依存関係は次のとおりである。

$$
\begin{aligned}
H_R
&\longrightarrow
(\lambda_R,p_\ell),
\\
(\lambda_R,\delta,q_m)
&\longrightarrow
t_{R,m}=\lambda_R\delta q_m,
\\
r_m
&\longrightarrow
\tau_m=\frac{\lambda_R\delta}{r_m},
\\
(\tau_m,K_m)
&\longrightarrow
B_{K_m}(\tau_m),\ R_{K_m}(\tau_m),\ p_\omega,
\\
(B_{K_m},R_{K_m},r_m,q_m)
&\longrightarrow
\text{attenuationとTaylor誤差},
\\
\text{attenuationと誤差予算}
&\longrightarrow
N_{m,b},
\\
p_\omega
&\longrightarrow
\mathbb E[C_{m,b}^{\mathrm{full}}],
\\
(N_{m,b},\mathbb E[C_{m,b}^{\mathrm{full}}])
&\longrightarrow
G_{\mathrm{total}}.
\end{aligned}
$$

したがって、$r_m$ は単にPR論文の式から固定する量ではない。PR論文の値をbaseline候補としながら、有限 $K_m$、attenuation、Taylor誤差、compiled期待コストを再計算し、roundごとの総コストで比較する必要がある。

## 参照

- 本編：「先行研究と未解決点.md」
- J. Günther et al., [Phase estimation with partially randomized time evolution, arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2)
- K. Wan, M. Berta, and E. T. Campbell, [A randomized quantum algorithm for statistical phase estimation, arXiv:2110.12071](https://arxiv.org/abs/2110.12071)
