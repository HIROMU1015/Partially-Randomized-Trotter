# 固定する分解・normalizer・取得条件

2026-10-18 JST。以下は既存式の転記・直接対応であり、新しい係数導出や比較定理ではない。`PROVED_IN_EXISTING_SOURCE`＝既存一般証明、`EXACT_FIXED_ORDER_RESULT`＝既存固定次数結果、`ROUTINE_DEFINITION_ALIGNMENT`＝直接の定義対応、`OPEN_COMPARISON_PROPOSITION`＝未解決比較、`REQUIRES_NONTRIVIAL_PROOF`＝追加の非自明証明を要する場合、`ACCESS_OR_CERTIFICATION_GAP`＝取得・認証不足、`NOT_ESTABLISHED`＝資料上未確定。未証明を偽としない。

## 共通の意味と6方式

`t_n=x^n/n!`、`b_l=√(t_l²+t_(l+1)²)`、`T_(k,m)=Σ_(even l=k,...,m−1)b_l`（空和0）、`χ=Σp_i²`、`μ_r=Σp_i^r`。係数は正の質量c、複素符号はeventに保持する。共通表記

`V_(d,w,i)(r)=(-iσ)^d exp[-iσ atan(r) Q_i] Q(w)`

は既存G7 eventの直接書換え。dは元の偶数次数、Q(w)の左から右の積順序を保持する。raw wordを隣接消去してもdの位相を勝手にreduced長へ変更しない。回転軸は単一Q_iであり、odd word全体ではない。理想meanはΣcV、normalizerはこの特定分解のΣcである。[G] 79–90,103–212、[M] §§1,6。

| 方式 | Exact finite target / valid order | 理想normalizer | 定義・SOURCEと既存保証 | 未確立の点 |
|---|---|---|---|---|
| Ordinary | P_m、奇数m≥5 | `B_o,m=Σ_(even l=0,...,m−1)b_l` | G 113–114,148–151、GCのordinary。各even raw wordとIID childの次数pairing。一般targetは既存 | Fullとの新しい資源比較は本対象外 |
| Partial Return＋ordinary tail | 既存partial P3＋次数4〜mのtail＝P_m | `B_P,m=B_P,3+T_(4,m)` | G 116–123,135、GC「強い三次対照」。下記partialの係数。`ROUTINE_DEFINITION_ALIGNMENT` | Full/Uとの一般次数順序は未確立 |
| Closed P3＋ordinary tail | 全P3集約＋同じtail＝P_m | `B_3,m=B_3,3+T_(4,m)` | G 124–135、GCの三次閉形式。`EXACT_FIXED_ORDER_RESULT`＋tailの直接対応 | Partialと同一lawではない。Full/Uへの一般次数比較は別 |
| Closed P5 | P5のみ。**m=5** | `B_5,5`（下式） | P5 11–123、PC「P5の独立再導出」。同じ理想ensembleをgroup化。`EXACT_FIXED_ORDER_RESULT` | digitalの丸め・proposal・weight・実装までFullと同じではない |
| Closed P5⊕ordinary tail | m≥7の**形式比較量**。`P_m=P5+Σ_(n=6,...,m)(-iσxR)^n/n!` | `B^form_(5⊕o,m):=B_5,5+T_(6,m)` | finite polynomial恒等式と既存paired tailの直接対応は`ROUTINE_DEFINITION_ALIGNMENT` | P5はself.m=5、G7に当該armなし。既存の統合sampler・そのdigital/取得契約は**NOT_ESTABLISHED**。完成済み既知方式とは呼ばない |
| Full RA | 一般奇数m≥5の同じP_m | `B_F,m=Σ_(even reduced u, |u|≤m−1)d_(u,m)` | M §§2–8、K 138–236、G 215–251。局所正値分解と生成は既存一般証明 | 全B_F,mの効率的取得・一般closed/partial順序は未確立 |

`P5⊕tail`行は数学的な有限和・正係数質量の形式benchmarkを固定するだけであり、branch混合確率、sampler、取得手順を新設計していない。元P5とtailの**tagを保持する直接和**を想定した係数和で、跨ぐreturnの再収集後の量ではない。形式量への不等式を解いても、既存の実装・予算・費用保証が完成したという結論にはならない。

## PartialとClosed P3：異なる係数とevent

`a_0=1−χt_2`、`s_P=t_1−χt_3`。既存partialは`R²=χI+D, R³=χR+RD`を使う。rootは係数`D_P p_i`、`D_P=√(a_0²+s_P²)`、角比`s_P/a_0`、childは全i。長さ2のdistinct parent `(j,k),j≠k` は係数`b_2 p_jp_kp_i`、childは全i、角比`x/3`、位相d=2。従って

`B_P,3=√[(1−χt_2)²+(t_1−χt_3)²]+(1−χ)b_2`。

Closed P3はroot child `a_i=p_i[t_1−(2χ−p_i²)t_3]`、`s_3=t_1−(2χ−μ_3)t_3`、`D_3=√(a_0²+s_3²)`、event係数`D_3 a_i/s_3`、角比`s_3/a_0`。parent `(j,k)` は`a=t_2p_jp_k`、`s=t_3p_jp_k(1−p_j)`、childはi≠jに比例p_i、角比`x(1−p_j)/3`。従って既存式は

`B_3,3=√(a_0²+s_3²)+Σ_j t_2p_j(1−p_j)√[1+(x(1−p_j)/3)²]`。

これらはGCの既存三次式とGのgroup/eventからの転記である。同じP3を表すことと、parent/child law・角度・normalizerが同じことを区別する。両方式とも高次は同じ`_ordinary(4)`を用いる。

## Closed P5：SOURCEにある係数とFullとの一致範囲

P5 19–23,50–74とPCの既存導出から、次を転記する（新しいTaylor係数の導出ではない）。

`a_(0,5)=1−χt_2+(2χ²−μ_4)t_4`、

`a_(i,5)=p_i[t_1−(2χ−p_i²)t_3+(5χ²−4χp_i²+2p_i⁴−2μ_4)t_5]`、`s_(0,5)=Σ_i a_(i,5)`。

`A_jk=t_2−t_4(3χ−p_j²−p_k²)`、

`S_jk=(1−p_j)t_3−t_5[(1−p_j)(4χ−p_j²−p_k²)−(μ_3−p_j³)]`。

長さ2 parentのa/sは`p_jp_k(A_jk,S_jk)`。そのchild iuの係数はi≠jについて`p_jp_kp_i[t_3−t_5(4χ−p_j²−p_k²−p_i²)]`。長さ4 reduced uのaは`t_4p(u)`、child iuはi≠first(u)について`t_5p(u)p_i`。各parentは既存の`d=√(a²+s²)`、child確率`a_iu/s`、角比`s/a`、phase`(-iσ)^|u|`で表す。

`r_(4,j)`を「長さ4 raw wordが最初からreducedでfirst=j」の質量とすると、P5のgroup平方根和は

`B_5,5=√(a_(0,5)²+s_(0,5)²)+Σ_(j≠k)p_jp_k√(A_jk²+S_jk²)+Σ_j r_(4,j)√[t_4²+t_5²(1−p_j)²]`。

PCは同じ理想ensembleと記載し、P5の`parent_coefficients`がFullのP5係数をgroup化する。この**固定次数の既存対応**により`B_5,5=B_F,5`。一般mの結果・完全独立software再現・digital bytes一致へ拡張しない。ゆえにm=5でFullの理想normalizerがClosed P5より小さいという独立優位はない。

## Full、Uと取得条件

Mの既存定義は、`l=|u|`、`a_(u,m)=Σ_(n=l,l+2,...≤m)(−1)^((n−l)/2)t_n P_n(u)`、`s_(u,m)=Σ_(i:iu reduced)a_(iu,m)`、`d_(u,m)=√(a²+s²)`。event係数は`d a_iu/s`、角比`s/a`。追加のQ間関係で語を並べ替えない。

`U_m=d_(empty,m)+Σ_(even l=2,...,m−1)b_l r_l`、`r_l=Pr[長さlのraw IID wordが最初からreduced]`。**reduce後の長さ分布ではない。** `[M] §7–8, [GC] U節, [U] §7, [G] 19–41`により`B_F,m≤U_m≤B_o,m`が既存保証。

| 量 | 必要な情報・取得 | 無料としないもの／認証の境界 |
|---|---|---|
| B_o,m | 明示p/x/m、次数ごとのb_lの有限和 | 根の外向き区間と算術。全語辞書は不要 |
| B_P,m / B_3,m | χ/μ_3、P3の有限群係数、既存ordinary tail | root/群係数の取得、p表read、算術・入力bit長。GのBは理想値の外向き区間 |
| B_5,5 | P5のroot・ordered pair・first-label群、r_(4,j)の既存completion DP | PCのO(L²)群算術は既存の限定countで、全bit/online/native費用ではない |
| B^form_(5⊕o,m) | 上のP5式と次数6以上のb_lの形式和 | 当該統合generator・実proposal・digital認証は未取得。式の存在を実装済みとしない |
| B_F,m | 全parent dの定義、又はそれと同値な認証済み取得法 | K/Gは局所queryを持つが、正確な全和を無料取得しない。全辞書・全価格表も仮定しない |
| U_m | root局所query、Green前計算、raw-reduced質量DP、b_lの有限和 | Gの`U.hi`は認証上側。`U.lo`はB_F,mの下界ではない。新しいH/K仕様はW1担当 |

全比較側へ同じ合法なGreen query、closed式、IID、streaming/cache、return収集、IS、CTS layeringの機会を認める。全lawを無料oracleとして配らず、取得費用は記録する。本表は特定分解のnormalizer比較であり、すべての合法なIS/CTSを含む最適性命題ではない。

<!-- FIXED_REFERENCES -->

[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[GC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
