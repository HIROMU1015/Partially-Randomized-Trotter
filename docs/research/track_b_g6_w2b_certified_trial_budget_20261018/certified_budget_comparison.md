# G7の認証budgetを同一条件で比較する

対象は同じ`M=P_m(−iσxR)`、`R=Σ_i p_iQ_i`。有限L≥1、正の明示有理p_i、Σp_i=1、有理0<x≤1、奇数m≥5、σ±1、Hermitian involution Q_i、同じcoherent X/Yとexact controlled-Q query前提。[M][GC]。geometry/basis/DF rank/L_Dは一般有理入力の本比較に非該当。m=3は既存整合の参照だけ、m=1とx=0 identityは4方式の比較域に入れない。finite P_mへのbiasと、exp(−iσxR)へのTaylor remainderは別であり、後者を今回評価しない。

## 1. 同一に固定する比較契約

| 項目 | 比較時の条件 | 既存SOURCE/実装との境界 |
|---|---|---|
| input/target | pの同順表、x、m、Q labelの意味、state/readout、同じfinite P_m | armごとにtargetやstateを変更しない。σは外側で保持し、adjoint/controlled相対位相も同じ |
| rational precision | 同じ0<η<1、ρ>0、positive int H/K_root、同じ丸め/外向きroot規約 | 固定defaultだけで全lawの充足を宣言しない。FullはF1 P0〜P5、canonicalはF1 §5の有限group・conditional law条件 |
| angle/provider | 同じangle誤差規約とstrict primitive error上界 ε_Rz≥0、exact controlled-Q前提 | F3のτとnative誤差を合わせて既存ε_Rzに収める証拠が必要。F3 PASSをnative誤差達成としない |
| axis error | 同じε_axis>0、共通bias β、同じremaining s=ε_axis−β>0 | 非正sは認証Nを返せない条件。方式ごとのprecision/bias緩和は禁止 |
| confidence | 既存α_axis=1/320、同じell^+=log640_upper(96)、familywise≤1/20の16 axes枠 | 元G7は2 target×4方式×X/Y=16 axes。条件式は各targetへの点ごとの主張で、全入力・新gridの同時保証ではない |
| call/count | Nは**各axisの十分全trial数**、同じaxes/target数、同じzero-fill | accepted数を分母にしない。硬い量子call上限Nと期待accepted数を区別 |

G7 `budget`のAPI引数はepsilon_axis/primitive_error。η/ρ/mはgeneratorから読むが、αと16-axis規約は返却定数、ellは固定関数である。任意confidence APIに変更しない。旧契約値ε_axis=1/200、ε_Rz=10^-6、η=ρ=10^-12、H=160/K_root=256は**既存G7の設定**として引用する。一般有理p/x/mにdefaultが十分であると評価していない。[F1][F3]

## 2. SOURCE正本の共通式

[G] `budget` 266–284と[GC]有限bit節を照合した。以下の値は、外向き上端を取得した**後**にはexact rationalである。

\[
\kappa=\frac{(1+\rho)^2}{(1-\eta)^{m+2}},\quad
\lambda=\frac{1+\rho}{(1-\eta)^{m+2}},\quad
\beta=2\,[3\rho+3(1+\rho)\,2\epsilon_{\rm Rz}],\quad
s=\epsilon_{\rm axis}-\beta>0.
\]

\[
T(M_2^+,W^+)=\ell^+\left(\frac{2M_2^+}{s^2}+\frac{4W^+}{3s}\right),\qquad
N=\lceil T\rceil.
\]

ell^+は`log640_upper()`の96項の有理和＋正のremainder。SOURCEの`9*log_from_z(1/3)+log_from_z(1/9)`は、文書の9 log2+log(5/4)=log640の上側評価に対応する。ここで関数を呼び出して定数を評価していない。Bernsteinの既存十分式であり、必要最小trial数・最適shot設計ではない。

## 3. 4方式の対応表

o=Full generatorの`B.hi`（**ordinary envelope**）、u=`U.hi`、b_O/b_P/b_3=各CanonicalGeneratorの`B.hi`とする。同じinput/K/sourceを保持したOrdinary行ではb_O=o（次資料§3で操作列を照合）。B_F=Fullの真のnormalizerはbudget入力に使わない。

| 方式 / G7 arm | 理想normalizerと実上端 | 認証moment M2^+ | 認証range W^+ | 認証十分全trial数 N_per_axis |
|---|---|---|---|---|
| Ordinary / ordinary | B_o=Σ_even l sqrt(t_l²+t_(l+1)²)、b_O=canonical B.hi | κ b_O² | λ b_O | ceil(ell^+[2κ b_O²/s²+4λ b_O/(3s)]) |
| Partial Return＋ordinary tail / partial_return_tail | B_P=B_P,3+T_(4,m)、b_P=**そのgroup根を直接丸めた和**のhi | κ b_P² | λ b_P | ceil(ell^+[2κ b_P²/s²+4λ b_P/(3s)]) |
| Closed P3＋ordinary tail / closed_P3_tail | B_3=B_3,3+T_(4,m)、b_3=そのgroup根和hi | κ b_3² | λ b_3 | ceil(ell^+[2κ b_3²/s²+4λ b_3/(3s)]) |
| Full RA / full_return | B_F=Σ_even reduced u d_u、**使用上界はu=U.hi**、oはordinary B.hi | **κ o u** | **λ o** | ceil(ell^+[2κ o u/s²+4λ o/(3s)]) |

全行でβ/s/ell/α/precision条件は§1–2で共通。Fullのmomentをκ B_F²やκ u²へ置き換えず、canonicalを実測varianceへ置き換えない。U.loをB_Fの下界としない。βは理想coefficient L1 errorと2 primitive rotationの既存共通保守予算であり、B.hiを係数bias3の代わりへ無断代入しない。既知ideal norm<3と、計算した上端がどれだけ保守的かは別である。

## 4. 上端の取得経路とSOURCE位置

`sum_roots` / `root_interval`は根ごとの外向き有理区間を加算する。p/x/m/Kが同じでも、group化・scale前後・root個数が異なると、理想normalizerの順序だけからhiの順序は出ない。各armの実`B.hi`を保持する。

| SOURCE位置（固定G） | 実処理 / 使用量 |
|---|---|
| 19–41 raw_reduced_masses/sum_roots/upper_normalizer | rawから最初からreducedの質量r_lをDP。u=q_K(d_empty²)+Σ_even l≥2 q_K(t_l²+t_(l+1)²) r_l。reduce後の長さ分布ではない |
| 103–143 CanonicalGenerator | χ、P3 root/distinct/first-label群、ordinary tailを作り、各group rootのhiを加えてb_aとする。rootのρ、group lawのη guardは別途必須 |
| 148–151 _ordinary | 同じt_l=x^l/l!のpairing。Partial/Closed P3は4以上のtailを保つ |
| 216–225 FullReturnGenerator | local kernel、order、labels、**ordinary B**、既存Uを保存。B_new取得はしない |
| 258–263 log640_upper | 同じ有理ell^+。任意αに対応する新log policyではない |
| 266–284 budget | m2/range/common_bias/remaining、positive margin guard、continuous、exact ceil、固定confidenceを返す |

例：Partialのdistinct groupはsqrt((1−χ)²(t_2²+t_3²))を**scale込みで**root_intervalへ渡す。これを`(1−χ)*root_interval(t_2²+t_3²).hi`へ置き換えない。[W2A]の理想式は定義対応、実上端はこのSOURCE経路が正本。

## 5. 比較可能性の正確な意味

4方式に`CERTIFIED_BUDGET_FORMULA`が存在し、同一前提下の条件式は`CONDITIONAL_COMPARISON`として扱える。式の評価に必要なo/u/b_aは既存の有限setup/local kernelとgroup算術だけで取得でき、未知B_F/Z/真の信号/全event表は不要である。今回その取得・評価は実施していない。

F3の非zero canonicalは未対応で、実IID/provider/全N測定・保存も今回未実施。よって「4方式のbudget式を比較できる」と「4方式を同じpreflight付き実装で完走した」は別。数学的meanは一般にM_tilde、元finite Mとの差は共通biasに戻す。物理回路のactual adjoint/位相/native errorの証拠は別である。

m=5のP5Closedは[別表](comparator_certification_matrix.md)に必ず残す。既存G9予算は異なるpolicyなので、G7行へ混ぜない。**mandatory STOP。**

<!-- FIXED_REFERENCES -->

[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[GC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[W2A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/43d6533970e0eb4c430fe1cff2e798d681b707d4/docs/research/track_b_g6_w2a_general_order_normalizer_20261018/baseline_normalizer_definition_matrix.md
[F1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/README.md
