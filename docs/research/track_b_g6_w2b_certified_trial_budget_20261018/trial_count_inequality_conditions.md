# Full RAの十分trial数が対照以下となる正確な条件

[主資料](certified_budget_comparison.md)の同一contract、正のs/κ/λ/ell、既存finite-bit/測定前提を仮定する。o/u/bは**実際にbudgetが読む外向き上端**。新上界・sampler・数値探索は作らず、既存式を直接差し引く。

## 1. Ceil前：momentとrangeを同時に比較する

a∈{O,P,3}、b=B_a^+とし、

\[
D_a=\frac{2\kappa}{s^2}(ou-b^2)+\frac{4\lambda}{3s}(o-b),\qquad
T_F-T_a=\ell^+D_a.
\tag{C1}
\]

\[
c=\frac{2\lambda s}{3\kappa}=\frac{2s}{3(1+\rho)}>0
\]

で正の因子を除くと、**ceil前の必要十分条件**は

\[
\boxed{T_F\le T_a\iff o(u+c)\le b(b+c)
\iff u\le \frac{b^2}{o}-c\left(1-\frac b o\right).}
\tag{C2}
\]

strict版はすべて<、連続tieは=。ellは同じ正の値なので**連続比較の符号**から消えるが、整数gapには残る。p/x/m/K/η/ρ/s等を変えてこの条件を満たす探索は実施しない。

## 2. Moment/rangeのトレードオフ

| 実上端の関係 | ceil前T_F≤T_aの条件 | 含意 |
|---|---|---|
| b<o | b²−ou≥c(o−b) | Fullはrangeで不利。momentが単に少し小さいだけでは不十分 |
| b=o | u≤o | rangeが同じ。momentの差だけで連続値が決まる |
| b>o | ou−b²≤c(b−o) | Fullはrangeで有利。momentが不利でも補える場合がある |

b<oでは連続非劣にb²−ou>0が必要で、条件をsについて書けば`0<s≤[3(1+ρ)/2]*(b²−ou)/(o−b)`。moment差が0以下なら正のsで成立しない。b>oかつou−b²>0なら`s≥[3(1+ρ)/2]*(ou−b²)/(b−o)`、momentも有利なら全ての正sで連続非劣。これは**固定したsに対する同値な表示**であって、結果に応じた誤差幅緩和やprecision調整を提案するものではない。

単なるu<b_Pやu<b_3はC2と同値でない。Fullのmomentはo*uでありrangeはo。逆にu>b_aだけから整数N_F>N_aを結論してはいけない。例えば実上端としてu>bかつo≥bまで別途分かればT_F>T_aだが、整数化後はtieがあり得る。ここで新しい入力例を作成/評価していない。

## 3. Ordinary：実上端とroot経路までの静的対応

理想量B_F≤U≤B_oだけから、任意に取得されたupper endpointへ順序を移すのは不正。今回の固定[K] `root_interval` 45–52と[G]の**同じK**については、操作列を直接照合できる。

[K]の上端写像は`q_K(v)=2^(-K)*ceil(2^K sqrt(v))`（v≥0）。isqrt/floorとexact-square分岐のコードに対応し、vに対して単調である。これはroot routineの定義の直接書換えで、精度改変ではない。

- Full.Bはsum_roots(t_l²+t_(l+1)²,K)、Ordinary.Bは同じt_lと同順pair radicandのroot和。従って**b_O=o**がexactに一致する。
- [M]§7の既存envelopeでd_empty²≤t_0²+t_1²。[GC]/[U]§7の既存raw-reduced質量は0≤r_l≤1。固定SOURCEのuは、

\[
u=q_K(d_\emptyset^2)+\sum_{l=2,4,\ldots,m-1}q_K(t_l^2+t_{l+1}^2)r_l
\le\sum_{l=0,2,\ldots,m-1}q_K(t_l^2+t_{l+1}^2)=o.
\tag{O1}
\]

従って同じSOURCE/input/Kとexact rational加算・同じbudget条件なら、**T_F≤T_O、N_F≤N_O**。これは既存envelopeと固定丸め/加算への通常適用であり、新U、新normalizer上界、新しい一般比較定理ではない。U.hiをmin(U.hi,B.hi)へ差し替える処理もない。

さらに本域m≥5（m=3の整合域でも）、p>0、x>0ではr_2=1−χ<1かつq_K(t_2²+t_3²)>0。tailのexact rational係数を丸めず掛けるSOURCEのためO1のtailでstrict差が残り、u<o、T_F<T_Oまで読める。ただし**N_F<N_Oはceilのgap条件が別途必要**。整数tieを排除していない。

異なるK、別のroot/和評価、独立な任意外向き区間、b_Oとoの入力identityが未確認ならこの操作列の等式を使わない。実上端を照合するか、次の一般条件/区間条件へ戻す。Partial/Closed P3のhi順序は、理想normalizerの順序だけで埋めない。

## 4. 整数ceil：必要十分条件

`k_a=ceil(T_a)`とする。同じ実上端とrational ell/sに対して、

\[
\boxed{
N_F\le N_a\iff T_F\le k_a,
\qquad N_F<N_a\iff T_F\le k_a-1,
\qquad N_F=N_a\iff k_a-1<T_F\le k_a.}
\tag{I1}
\]

Fullの整数劣位は`T_F>k_a`。全て整数ceilの定義の直接操作であり、必要十分。T_F<T_aは非劣の十分条件だがstrict Nの十分条件ではない。T_F>T_aでも同じceil binならtieするので、D_a>0だけでもstrict N劣位は出ない。`T_a−T_F≥1`ならstrict N改善の十分条件だが、必要条件ではない。

o>0より`A_o=2 ell^+ κ o/s²>0`と書けば、`T_F=A_o(u+c)`。I1は上端uへの同値な閾値表示にもなる。

\[
N_F\le N_a\iff u\le k_a/A_o-c,
\quad N_F<N_a\iff u\le (k_a-1)/A_o-c,
\]
\[
N_F=N_a\iff (k_a-1)/A_o-c<u\le k_a/A_o-c.
\tag{I2}
\]

k_aも同じ認証式からのexact ceilであり未知の最適Nではない。τ/根の幅/分母をfloatに変換せず、正規化したq=A/B（B>0）ならceil(q)=(A+B−1)//BがSOURCEと一致する。T_aが整数の境界もI1に含まれる。s≤0は除外域で、有限Nとして比較しない。

## 5. 外側区間と未認証情報

通常のbudgetではo/u/bは「未知のideal量の区間」ではなく、**計算済みroot区間のhiという特定の有理数**である。そのhiを両方式ともそのまま使う。UやBのmidpoint/ideal値/loへ置き換えてtrialを減らさない。

まだbudgetの実上端が未確定だが、**それら実上端そのもの**について認証された外側区間`o∈[o_L,o_H]`、`u∈[u_L,u_H]`、`b∈[b_L,b_H]`がある場合、各lower endpointを非負に制約し、同じexact正κ/λ/s/ellの下で正の積/平方和の単調性からT_F/T_aの外向き区間を作れる。これは既存budgetへの通常の区間代入であり、新moment boundではない。相関を無視した区間は保守的でよく、独立確率と仮定しない。s等も区間なら正の下限・同じcontract/provenanceが追加で必要になる。

T_F∈[L_F,H_F]、T_a∈[L_a,H_a]、`n_F^- = ceil(L_F)`/`n_F^+ = ceil(H_F)`等と置くと、次は**認証の十分条件**。

| 分類 | 条件 |
|---|---|
| CERTIFIED_NONSTRICT_IMPROVEMENT | n_F^+≤n_a^-（同値な形はH_F≤ceil(L_a)） |
| CERTIFIED_STRICT_IMPROVEMENT | n_F^+<n_a^-（H_F≤ceil(L_a)−1） |
| CERTIFIED_TIE | 両N区間が同じsingleton整数k |
| CERTIFIED_FULL_WORSE | n_F^->n_a^+ |
| INSUFFICIENT_CERTIFIED_INFORMATION | 上の条件で向きが確定しない。相関/解析的同一性を使えば結論できる場合もあり、不成立とはしない |

正確なTが両方取得済みなら区間は一点となりI1が必要十分となる。今回どの入力についてもT/Nや新しい数値区間を計算していない。**mandatory STOP。**

<!-- FIXED_REFERENCES -->

[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[GC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
