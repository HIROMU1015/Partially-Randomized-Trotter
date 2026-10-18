# P_Uの厳密な反例certificate

**Status：`DISPROVED`。証拠の型：紙上の解析的反証。** 管理コードによる数学の機械検証、科学SOURCE/testの実行、外部の独立レビューを意味しない。

## 入力と対象

\[
\boxed{L=2,\quad p_1=p_2=\tfrac12,\quad x=1,\quad m=7.}
\]

[固定W2aの命題][S02]と[定義][S03]の量化域内である。対象は同じ有限P7。σ±、同じHilbert空間上のHermitian involution Q_iという前提を保持する。これらnormalizerのscalar定義はσや追加のQ間関係に依存しない。

`t_n=1/n!`、`b_l=√(t_l²+t_(l+1)²)`とする。以下は正本の係数・tailの定義を保持したexact表現である。

\[
\begin{aligned}
A_5&=\tfrac{49}{64},&S_5&=\tfrac{169}{192},&D_5&=\sqrt{A_5^2+S_5^2},\\
A_7&=A_5-\tfrac1{2304},&S_7&=S_5-\tfrac1{9216},&D_7&=\sqrt{A_7^2+S_7^2},\\
H&=\sqrt{(11/24)^2+(5/64)^2},&&
C_4&=\sqrt{(1/24)^2+(1/240)^2}.
\end{aligned}
\]

既存定義のこの入力への適用根拠は次の通り。

- 二つのuniform labelの形式Cayley graphは無限道。根への2k歩の確率は`binom(2k,k)/2^(2k)`、二つの隣接頂点への合計2k+1歩の確率は`2 binom(2k+1,k)/2^(2k+1)`。既存の交代和[M]に代入すると上のA_5/S_5/A_7/S_7となる。
- 最初からreducedであるraw語は各長さで二つの交互列なので、`r_2=1/2,r_4=1/8,r_6=1/32`。
- 既存P5のroot、二つのordered-pair群、長さ4のfirst-label群[P5][PC]の質量はそれぞれ`D_5,H/2,C_4/8`。χ=1/2、μ_3=1/4、μ_4=1/8を既存pair係数へ代入して`A_jk=11/24,S_jk=5/64`を得る。係数導出の全過程は[解析記録](analytic_proof_attempt.md) §§2–4。

したがって

\[
U_7=D_7+\tfrac12b_2+\tfrac18b_4+\tfrac1{32}b_6,
\quad
B^{\rm form}_{5\oplus o,7}=D_5+\tfrac12H+\tfrac18C_4+b_6.
\tag{1}
\]

Closed側のtailはb_6であり、r_6b_6に変えていない。Uのrootも次数7のD_7である。

## 有理数で向きを確定する証明

ノルムの逆三角不等式から

\[
D_7-D_5\ge-\sqrt{(1/2304)^2+(1/9216)^2}
\ge-\tfrac5{9216}>-\tfrac1{1024}.
\tag{2}
\]

また

\[
H^2=\tfrac{7969}{36864}<\tfrac{8100}{36864}=(\tfrac{15}{32})^2,
\quad b_2\ge\tfrac12,
\]

なので

\[
\tfrac12(b_2-H)>\tfrac1{64}.
\tag{3}
\]

非負平方根の単調性で`b_4≥C_4`。最後に

\[
b_6\le t_6+t_7=\tfrac1{630}<\tfrac1{512}
\quad\Longrightarrow\quad
-\tfrac{31}{32}b_6>-\tfrac1{512}.
\tag{4}
\]

(1)〜(4)より

\[
\boxed{
U_7-B^{\rm form}_{5\oplus o,7}
>(-\tfrac1{1024})+\tfrac1{64}+0-\tfrac1{512}
=\tfrac{13}{1024}>0.
}
\]

よってこの入力でP_{U,5}が破れ、全量化命題P_Uは偽。P_{U,3}の真偽はこの結論に不要であり、本作業では判定していない。形式P5＋tailの統合samplerを仮定せず、scalar benchmarkの定義そのものを比較した。

反例はこの1件で確定したため、追加の入力・次数・漸近解析・別不等式を探索しない。証明の根拠は本文のexact式と初等不等式であり、[verification.json](verification.json)の管理PASSではない。

<!-- FIXED_REFERENCES -->

[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[PC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md
[S02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/43d6533970e0eb4c430fe1cff2e798d681b707d4/docs/research/track_b_g6_w2a_general_order_normalizer_20261018/general_order_comparison_question.md
[S03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/43d6533970e0eb4c430fe1cff2e798d681b707d4/docs/research/track_b_g6_w2a_general_order_normalizer_20261018/baseline_normalizer_definition_matrix.md
