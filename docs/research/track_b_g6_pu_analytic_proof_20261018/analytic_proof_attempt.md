# P_Uの解析的反証：定義から厳密な符号まで

2026-10-18 JST。結果は`DISPROVED`。以下の既存定義と今回の紙上導出を区別する。数値プログラム・科学SOURCE・数式処理ソフトは実行していない。

## 1. 命題の分離と既存の定義

[固定W2aの命題][S02]と[定義][S03]の全量化域で

\[
P_{U,3}: U_m\le B_{3,m},\qquad
P_{U,5}: U_m\le B^{\rm form}_{5\oplus o,m}
\]

の両方を主張するのがP_Uである。一方の厳密な反例で全体を否定できる。今回扱うのはP_{U,5}だけ。

既存記法は`t_n=x^n/n!`、`b_l=√(t_l²+t_(l+1)²)`、

\[
U_m=d_{\emptyset,m}+
\sum_{\substack{2\le l\le m-1\\l\ {
m even}}}b_lr_l,
\qquad
B^{\rm form}_{5\oplus o,m}=B_{5,5}+
\sum_{\substack{6\le l\le m-1\\l\ {
m even}}}b_l.
\]

r_lはraw IID wordが最初からreducedである質量。[M] §§2,6、[GC]のU節、[U] §7、[G] 19–41により

\[
d_{\emptyset,m}=\sqrt{a_{\emptyset,m}^2+s_{\emptyset,m}^2},
\]
\[
a_{\emptyset,m}=\sum_{\substack{n\le m\\n\ {\rm even}}}(-1)^{n/2}t_nP_n(\emptyset),
\quad
s_{\emptyset,m}=\sum_i\sum_{\substack{n\le m\\n\ {\rm odd}}}(-1)^{(n-1)/2}t_nP_n(i).
\]

ここでP_n(u)はraw wordのreduce先がuとなる確率。B_{5,5}は固定P5の既存group平方根和であり、近似root midpointやdigital normalizerではない。[P5] 19–45,50–74、[PC]の「P5の独立再導出」を用いる。

## 2. 今回の反例入力と有限wordの解析的計数

\[
L=2,\quad(p_1,p_2)=(\tfrac12,\tfrac12),\quad x=1,\quad m=7.
\]

すべて指定量化域内。scalar定義はQ間の追加関係を利用しないため、同じ空間上のHermitian involution Q_iという条件はそのまま保持する。

**今回の導出。** 二つのinvolution labelが生成する形式Cayley graphは双方向の無限道である。各頂点で異なる二本の辺を確率1/2ずつ選ぶので、方向の各列は等確率。従って長さ2kの歩行が根に戻る数は`binom(2k,k)`、長さ2k+1で指定された隣接頂点に着く数は`binom(2k+1,k)`。後者の隣接頂点は二つである。この有限計数から

| n | P_n(empty)（even n）またはΣ_iP_n(i)（odd n） |
|---:|---:|
| 0 | 1 |
| 1 | 1 |
| 2 | 2/4=1/2 |
| 3 | 2·3/8=3/4 |
| 4 | 6/16=3/8 |
| 5 | 2·10/32=5/8 |
| 6 | 20/64=5/16 |
| 7 | 2·35/128=35/64 |

となる。これは列挙プログラムやsamplingの出力ではなく、上記binomial計数の紙上適用である。

また長さl≥1のraw wordがreducedなのは、二つの交互label列だけ。各列の質量が2^(−l)なので

\[
r_2=\tfrac12,\qquad r_4=\tfrac18,\qquad r_6=\tfrac1{32}.
\]

「reduce後の長さ」の確率をここへ代入していない。

## 3. Empty parent：次数5と7を混同しない

上の有限係数を既存の交代和へ代入する。以下は今回の反例入力についての新しいexact有理式である。

\[
A_5:=a_{\emptyset,5}=1-\tfrac14+\tfrac1{64}=\tfrac{49}{64},
\quad
S_5:=s_{\emptyset,5}=1-\tfrac18+\tfrac1{192}=\tfrac{169}{192},
\]
\[
A_7:=a_{\emptyset,7}=A_5-\tfrac1{2304}=\tfrac{1763}{2304},
\quad
S_7:=s_{\emptyset,7}=S_5-\tfrac1{9216}=\tfrac{8111}{9216}.
\]

`D_5:=√(A_5²+S_5²)`、`D_7:=√(A_7²+S_7²)`とする。特にU_7のrootはD_7であり、P5 rootのD_5へ置き換えない。

ユークリッドノルムの逆三角不等式と`√(u²+v²)≤u+v`（u,v≥0）から

\[
D_7-D_5\ge-\sqrt{(1/2304)^2+(1/9216)^2}
\ge-\frac5{9216}>-\frac1{1024}.
\tag{R}
\]

最後は`5·1024=5120<9216`という整数の不等式である。rootの変化は負になり得るため、省略せず下から評価した。

## 4. P5の非root群：既存SOURCE式の特殊化

今回のpで`χ=1/2, μ_3=1/4, μ_4=1/8`。既存の`pair_scalars`式

\[
A_{jk}=t_2-t_4(3\chi-p_j^2-p_k^2),
\]
\[
S_{jk}=(1-p_j)t_3-t_5\{(1-p_j)(4\chi-p_j^2-p_k^2)-(\mu_3-p_j^3)\}
\]

を転記してj≠kへ代入すると

\[
A_{jk}=\tfrac12-\tfrac1{24}=\tfrac{11}{24},\qquad
S_{jk}=\tfrac1{12}-\tfrac1{192}=\tfrac5{64}.
\]

二つのordered pairは各p_jp_k=1/4。`H:=√[(11/24)²+(5/64)²]`とすれば、そのnormalizer寄与はH/2。

長さ4 parentの群は、[P5]の係数`t_4p(u)`、child和`t_5p(u)(1-p_first)`により、

\[
\frac18C_4,\qquad
C_4:=\sqrt{t_4^2+(t_5/2)^2}.
\]

従って固定W2aの量そのものが

\[
B_{5,5}=D_5+\tfrac12H+\tfrac18C_4,
\quad B^{\rm form}_{5\oplus o,7}=D_5+\tfrac12H+\tfrac18C_4+b_6,
\]
\[
U_7=D_7+\tfrac12b_2+\tfrac18b_4+\tfrac1{32}b_6
\]

となる。P5のtail係数をr_6付きへ変更しない。Closed P5＋tailは定義通りordinary b_6を丸ごと含む。

## 5. 厳密な符号：有理数だけによる上下評価

差を項別に書くと

\[
\Delta:=U_7-B^{\rm form}_{5\oplus o,7}
=(D_7-D_5)+\tfrac12(b_2-H)+\tfrac18(b_4-C_4)-\tfrac{31}{32}b_6.
\tag{D}
\]

pair群について、平方根の中身を同じ有理分母にすると

\[
H^2=\frac{7969}{36864}
<\frac{8100}{36864}=(\tfrac{15}{32})^2.
\]

両辺は非負なので`H<15/32`。また`b_2≥t_2=1/2`であり

\[
\tfrac12(b_2-H)>\tfrac12(\tfrac12-\tfrac{15}{32})=\tfrac1{64}.
\tag{P}
\]

長さ4群は`t_5≥t_5/2≥0`から`b_4≥C_4`。次数6/7のordinary tailは

\[
b_6\le t_6+t_7=\tfrac1{720}+\tfrac1{5040}=\tfrac1{630}<\tfrac1{512},
\]

なので`−(31/32)b_6>−1/512`。この**負のtail差**も省略しない。

(R)、(P)を(D)へ戻すと

\[
\boxed{\Delta>
-\tfrac1{1024}+\tfrac1{64}+0-\tfrac1{512}
=\tfrac{13}{1024}>0.}
\]

平方根の符号・有理入力・finite cutoff・ordinary tailをすべて固定した厳密な反証である。近似小数、O記号、漸近剰余、数式ソフトの判定を根拠にしない。

## 6. 結論と未検討部分

P_{U,5}はこの入力で偽であり、したがってP_Uは`DISPROVED`。P_Uが全量化域で成立するとの命題は決着した。P_{U,3}の全域の真偽、各入力での順序、厳密な成立域・等号条件は今回扱わない。これらが未検討でも、全体のstatusを`PARTIALLY_RESOLVED`へ下げる理由にはならない。

今回用いた補題は有限道上のbinomial計数、ノルムの逆三角不等式、非負平方根の単調性・和による上界で、本文に適用根拠を示した。未証明の比較lemmaやW1未完成事項は前提にしていない。反例確定後、別入力・別次数・P_{U,3}・新しい上界は探索しない。

<!-- FIXED_REFERENCES -->

[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[GC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[PC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md
[S02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/43d6533970e0eb4c430fe1cff2e798d681b707d4/docs/research/track_b_g6_w2a_general_order_normalizer_20261018/general_order_comparison_question.md
[S03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/43d6533970e0eb4c430fe1cff2e798d681b707d4/docs/research/track_b_g6_w2a_general_order_normalizer_20261018/baseline_normalizer_definition_matrix.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
