# 「研究目的・研究課題」補足Q&A

## この資料の位置付け

本資料は、[研究目的・研究課題](研究目的・研究課題.md)のうち、RTEのnormalizationとattenuation、および誤差をRPEの位相誤差へ換算する考え方を補足するものである。

## Q1. RTEのnormalizationとは何か

RTEでは、Taylor展開から得られる正係数を、実際にサンプリングできる確率分布へ変換する。

有限cutoff $K$ のRTEを

$$
\widetilde U_K(\tau)
=
\sum_{\omega}c_\omega U_\omega
$$

と書く。係数の総和

$$
B_K(\tau)=\sum_\omega c_\omega
$$

で割り、

$$
p_K(\omega)=\frac{c_\omega}{B_K(\tau)},
\qquad
\sum_\omega p_K(\omega)=1
$$

とすれば、$p_K(\omega)$ をevent $\omega$ のサンプリング確率として使える。このとき

$$
\widetilde U_K(\tau)
=
B_K(\tau)
\sum_\omega p_K(\omega)U_\omega
$$

であり、$B_K(\tau)$ がRTEのnormalization factorである。

例えば係数が $0.7$ と $0.5$ なら、$B=1.2$ であり、実行確率は $0.7/1.2$ と $0.5/1.2$ になる。

なお、これは

$$
H_R=\lambda_R\overline H_R
$$

と書くHamiltonianのnormalizationとは別である。$\lambda_R$ はランダムHamiltonianの強さを取り出す量、$B_K$ はRTE event係数を確率分布へ直す量である。

## Q2. attenuationとは何か

確率 $p_K(\omega)$ でunitary $U_\omega$ を選ぶと、その平均は $\widetilde U_K/B_K$ になる。無限次数RTEでは $\widetilde U_\infty=e^{-i\tau H_R}$ が成り立つため、1 short stepの平均信号は理想信号の $1/B_\infty$ 倍になる。

同じshort stepを $R$ 回用いる場合、全体のnormalizationとattenuation factorは

$$
B_{\mathrm{tot}}
=
B_\infty(\tau)^R,
\qquad
a=B_{\mathrm{tot}}^{-1}
$$

である。したがって、理想的な複素信号を $Z$ とすると、観測信号の期待値は

$$
\mathbb E[\widehat Z]
=
aZ
$$

となる。$a$ は正の実数なので、

$$
|Z|\longrightarrow a|Z|,
\qquad
\arg Z\longrightarrow\arg Z
$$

となる。この半径だけが小さくなる現象がattenuationである。

本研究の記法では、RPE round $m$ に含まれるshort-step総数は $R=q_mr_m$ である。

PR論文では、full Taylor expansionを確率分布へ正規化したLCUと、$B(\tau)^{-r}$ 倍されたHadamard-test信号を式 (27)--(29) で示している。

## Q3. なぜattenuationによって標本数が増えるのか

複素信号の統計的なずれを $\Delta Z$ とすると、小さい誤差に対する角度のずれは概略

$$
|\Delta\phi|
\sim
\frac{|\Delta Z|}{|Z|}
$$

である。attenuationにより信号半径が $a$ 倍になると、同じ標本数では角度誤差が約 $1/a$ 倍になる。

標本平均の揺らぎは $N^{-1/2}$ で減少するため、元と同じ角度精度を保つには

$$
N
\propto
a^{-2}
=
B_{\mathrm{tot}}^2
$$

の標本数が必要になる。したがって、attenuationは位相biasではなくsampling overheadとして総コストへ入る。

## Q4. 有限RTEの打切り誤差はattenuationと何が違うか

無限次数RTEでは、normalizationを除けば理想時間発展信号の不偏推定量が得られる。一方、有限cutoff $K$ では

$$
\widetilde U_K(\tau)
\neq
e^{-i\tau H_R}
$$

であり、normalizationを補正しても、得られるのは有限Taylor展開の信号である。理想信号を $Z$ とすれば、

$$
\widetilde Z_K
=
Z+\Delta Z_K
$$

となり、$\Delta Z_K$ は一般に $Z$ と平行とは限らない。そのため、有限RTEの打切りは信号半径だけでなく位相角も変え得る。

例えば、$Z=1$ に対して $\widetilde Z_K=1+0.1i$ なら、位相は $0$ から約 $0.10\,\mathrm{rad}$ へずれる。これは既知の正の実数倍であるattenuationとは異なる。

## Q5. 「各誤差をRPEの位相誤差へ換算する」とは何か

RPE round $m$ でbranch選択に直接関係するのは、推定位相 $\phi_m$ と真のround位相との差である。PR論文の無次元化された記法では、Lemma B.1において各roundで

$$
d(\phi_m,2^mE)<\frac{\pi}{3}
$$

を満たすことが、正しいbranchを帰納的に選び続けるための十分条件として用いられている。

そこで、元の形が異なる誤差を、最終的に $\phi_m$ を最大何radずらすかへ変換する。

| 誤差源 | 元の評価量 | 位相誤差上界への変換 |
|---|---|---|
| Product Formula | $\lvert\Delta E_{\mathrm{PF}}\rvert\leq\epsilon_{\mathrm{PF}}$ | $\beta_{\mathrm{PF},m}^{\mathrm{ub}}\leq t_m\epsilon_{\mathrm{PF}}$ |
| finite-RTE | $\lvert\Delta Z_{\mathrm{RTE},m}\rvert\leq\epsilon_{Z,m}$ | $\beta_{\mathrm{RTE},m}^{\mathrm{ub}}\leq\arcsin\!\left(\epsilon_{Z,m}/\rho_{\star,m,\mathrm{lb}}\right)$ |
| finite標本 | $\lvert\Delta X_m\rvert\leq\epsilon_{\mathrm{coord},m,c}$、$\lvert\Delta Y_m\rvert\leq\epsilon_{\mathrm{coord},m,s}$ | 観測半径に対する合成座標誤差から換算 |

統計誤差の保守的な一般形は

$$
\beta_{\mathrm{stat},m}^{\mathrm{ub}}
\leq
\arcsin\!\left(
\frac{
\sqrt{
\epsilon_{\mathrm{coord},m,c}^{2}
+
\epsilon_{\mathrm{coord},m,s}^{2}
}
}{
A_m^{\mathrm{att}}
\left(
\rho_{\star,m,\mathrm{lb}}-\epsilon_{Z,m}
\right)
}
\right)
$$

である。finite-RTE誤差が位相だけでなく半径も低下させ得るため、分母に $\rho_{\star,m,\mathrm{lb}}-\epsilon_{Z,m}$ が現れる。主解析の簡略モデルでは、この半径低下を省略して分母を $A_m^{\mathrm{att}}\rho_{\star,m}$ とするが、これは一般的な厳密下界ではない。

各座標の確率事象は

$$
\Pr\!\left(
\lvert\Delta X_m\rvert
>
\epsilon_{\mathrm{coord},m,c}
\right)
\leq
\alpha_{m,c},
$$

$$
\Pr\!\left(
\lvert\Delta Y_m\rvert
>
\epsilon_{\mathrm{coord},m,s}
\right)
\leq
\alpha_{m,s}
$$

と管理する。共通誤差を使う場合は $\epsilon_{\mathrm{coord},m,c}=\epsilon_{\mathrm{coord},m,s}=\epsilon_{\mathrm{coord},m}$ とする。

各誤差源に配分する位相誤差の許容量を

$$
\overline\beta_{\mathrm{PF},m},
\qquad
\overline\beta_{\mathrm{RTE},m},
\qquad
\overline\beta_{\mathrm{stat},m}
$$

とする。換算した誤差上界は、それぞれ

$$
\beta_{j,m}^{\mathrm{ub}}
\leq
\overline\beta_{j,m}
$$

を満たす必要がある。さらに、round $m$ の誤差予算について、

$$
\overline\beta_{\mathrm{PF},m}
+
\overline\beta_{\mathrm{RTE},m}
+
\overline\beta_{\mathrm{stat},m}
\leq
\beta_{\mathrm{RPE}}
$$

を課す。この線形和は、各誤差による位相角のずれを最悪方向に合成した、保守的な十分条件である。

## Q6. 位相誤差への換算を数値例で示すとどうなるか

Product Formulaのenergy biasが

$$
|\Delta E_{\mathrm{PF}}|
\leq
10^{-3}
$$

で、roundの発展時間が $t_m=100$ なら、

$$
\beta_{\mathrm{PF},m}^{\mathrm{actual}}
\leq
100\times10^{-3}
=
0.10\,\mathrm{rad}
$$

である。

また、normalization補正後の信号半径が $\rho_m=0.8$、有限RTEによる複素信号誤差が $0.04$ 以下なら、

$$
\beta_{\mathrm{RTE},m}^{\mathrm{actual}}
\leq
\arcsin\!\left(\frac{0.04}{0.8}\right)
\simeq
0.05\,\mathrm{rad}
$$

である。さらに統計誤差へ $0.15\,\mathrm{rad}$ を許容すると、位相ずれの最悪値の和は

$$
0.10+0.05+0.15
=
0.30\,\mathrm{rad}
<
\frac{\pi}{3}
$$

となる。

## Q7. $\beta_{\mathrm{PF},m}+\beta_{\mathrm{RTE},m}+\beta_{\mathrm{stat},m}\leq\beta_{\mathrm{RPE}}$ は何を表すか

加えているのはenergy errorや複素信号誤差そのものではなく、Q5で位相角へ換算した許容幅である。実際のPF・finite-RTE上界は対応する配分値以下でなければならず、統計誤差は

$$
\Pr\!\left(
\beta_{\mathrm{stat},m}^{\mathrm{actual}}
>
\overline\beta_{\mathrm{stat},m}
\right)
\leq
\alpha_{m,c}+\alpha_{m,s}
$$

となるように $N_{m,b}$ を選ぶ。右辺はcosine・sineの二事象をunion boundで合成した値であり、単独の $\alpha_{m,b}$ ではない。位相許容幅の線形和は、三角不等式で最悪方向のずれを合成した保守的な十分条件である。

## Q8. 系統誤差が増えると統計誤差marginは自動的に小さくなるか

自動的には小さくならない。

統計誤差の許容量を残余予算として

$$
\overline\beta_{\mathrm{stat},m}
=
\beta_{\mathrm{RPE}}
-
\overline\beta_{\mathrm{PF},m}
-
\overline\beta_{\mathrm{RTE},m}
$$

と定義する場合には、系統誤差側へ多く配分するほど統計誤差の許容量が小さくなる。しかし配分値を固定している場合、実際の系統誤差が増えても統計誤差の許容量は変わらず、系統誤差の制約余裕が減るか、制約違反になる。

したがって本研究で調べるのは、「系統誤差が増えれば統計marginが必ず変わる」という関係ではなく、配分を変えたときに回路設定、必要実行回数および総コストがどう変化するかである。

## Q9. PR論文と本研究では何が異なるか

| 項目 | PR論文 | 本研究 |
|---|---|---|
| RTE | full Taylor expansionに基づく不偏信号 | 有限cutoff $K_m$ と打切り誤差を導入 |
| attenuation | normalization上界からsampling overheadを評価 | 有限 $B_{K_m}$ と実際のround別attenuationを評価 |
| RPE標本数 | $N_m=e^{2/\kappa}[11+4(M-m)]$ を解析的十分条件から設定 | $\overline\beta_{\mathrm{stat},m}$、$\alpha_{m,b}$、attenuationから決定 |
| 誤差管理 | Product Formula biasとRPEのRMSEを主に評価 | PF・有限RTE・統計誤差をround位相への影響として整理し、配分方法を比較 |
| 回路コスト | 平均primitive costに基づく解析式 | time-evolution subcircuitから状態準備なしRPE interrogationへ段階的に接続するcompiled期待コスト |

本研究の違いは、特定の誤差配分を新しい固定則として採用することではない。有限RTEとcompiled costを含むモデルの中で、誤差・失敗確率の配分が総コストへ与える影響を明示的に比較する点にある。

## 参考箇所

- [PR論文 Sec. IV.B, Eqs. (27)--(29)：RTE normalizationとHadamard-test信号](https://arxiv.org/pdf/2503.05647v2#page=7)
- [PR論文 Appendix B, Lemma B.1：RPEの $\pi/3$ 条件](https://arxiv.org/pdf/2503.05647v2#page=26)
- [PR論文 Appendix E, Eqs. (E20)--(E22)：部分ランダム化のattenuationと標本数](https://arxiv.org/pdf/2503.05647v2#page=42)
