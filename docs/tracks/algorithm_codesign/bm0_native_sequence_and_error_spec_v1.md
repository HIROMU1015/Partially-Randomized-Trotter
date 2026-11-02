# BM-0 native実行列・誤差分解仕様案 v1

2026-10-05 JST。**設計案／GPT review required。実装済みalgorithm・実行contractではない。**
返却計画書 §7–10の構成を展開する。新しい入力、行列評価、signal、circuitは生成していない。

## 1. 入力・primitive・作用順

`H=A+B+R+cI`。A/Bは互いに素なdeterministic generator集合。one-bodyは分離したgeneratorとして
一度だけ割り当てる。Rの物理split・sampling分布は固定し、re-factorization、PF係数探索を同時に入れない。
DF規約は`D_i=lambda_i F(G_i)^2`、`F(G)=sum_pq G_pq a_p^dagger a_q`。
実snapshotにあるone-body補正、係数の1/2、抽出identityはadapterで照合する。架空の係数変換を行わない。

primitive `E_i(t)=exp(-it D_i)`は、当該DF generatorのbasis変換、対角演算、復帰とcontrolled phaseを
含む論理blockを指す。`exp(-it A)`や`exp(-it B)`を無料のnative primitiveとしない。
各groupには結果前に固定した順序`A_1,...,A_a`、`B_1,...,B_b`を持たせる。

時系列list `L=[E_1,...,E_s]`のoperatorは`U(L)=E_s ... E_1`（右端から作用）と定義する。
`reverse`は同符号・同時間の逆順でありadjointではない。
`S_A(u)`の時系列はA forward half-sweepとreverse half-sweep。
同じ中央generatorの二halfはexact fusionしてよい。`S_A(-u)=S_A(u)^dagger`だが、
一般に`S_A(u)`自体はHermitianではない。ここでのsymmetric/self-adjoint formulaはこの時間反転性を指す。

一macro-step、`h=T/q`、整数`m>=1`の理想時系列を次で定義する。

```text
B forward half-sweep(h/2)
m copies of S_A(h/(2m))
exact R(h)
m copies of S_A(h/(2m))
B reverse half-sweep(h/2)
scalar relative phase c*h, exactly once
```

これはpalindromic列である。式ではouterから順に

\[
V_m(h)=E_{B_1}(h/2)\cdots E_{B_b}(h/2)
 [S_A(h/(2m))]^m e^{-ihR}
 [S_A(h/(2m))]^m E_{B_b}(h/2)\cdots E_{B_1}(h/2).
\]

全q macro-stepのoperatorとscalar phase`exp(-icT)`を返す。全signed timeを保持する。
control branchにのみかかるscalar relative phaseをglobal phaseとして削除しない。

## 2. m=1、flat、fusion

flat対照は現current partial S2のdeterministic forward half-sweep → R → reverse half-sweep。
Aの**full symmetric step**をRの両側へ置くnested m1とは一般に異なる。
特にR=0でもnestedは`[S_A(h/(2m))]^(2m)`を含み、m1が`S_A(h)`になるとは限らない。
group固定nested m1と、現native flat S2を別identity・別対照で保存する。

許すfusionは、zero-time identity削除と**隣接する同一generatorのexact指数**の時間和。
group内、反復境界、macro境界とも同じ規則を適用する。異なる非可換generatorを並べ替えない。
既存partial S2のmacro境界fusion・scalar phase集約はflat対照にも適用する。
一般relative Gaussian融合は別途wrapper検証を必要とし、今回は検証済みとしない。

非退化a>=1、b>=1、R blockが間にある場合、fusion後のA指数block数は
一A半区間で`m(2a-2)+1`、一macroでその2倍。Bは一macro`2b`、q境界で同じB1が隣接すれば
`q-1`回のfusion候補となる。これは論理blockの構造countでありgate costではない。
empty group、zero generator、commuting reorder、消えたRなどは別に正規化する必要がある。

## 3. ideal leading BCH分解

`X_i=-iD_i`、`X_A=sum_i X_Ai`、`X_R=-iR`とする。固定m、h→0の形式展開を使う。
symmetric BCHの三次係数を

\[
C(X,Y)=-[X,[X,Y]]/24+[Y,[Y,X]]/12
\]

と定義する（[SPRINT v1](https://arxiv.org/html/2606.30741v1) §III.1、Eq.(18)の既知式）。
A内部の三次係数は

\[
K_A=\sum_{i=1}^{a-1}C(X_{A_i},\sum_{j>i}X_{A_j}),
\qquad \log S_A(u)=uX_A+u^3K_A+O(u^5).
\]

B内部・group間・A/R cutを含むmに依存しない係数は

\[
K_{\rm floor}=C(X_A,X_R)+
 \sum_{i=1}^{b}C(X_{B_i},X_A+X_R+\sum_{j>i}X_{B_j}).
\]

従ってこの**tail-centered構成**では

\[
\log V_m(h)=h(X_A+X_B+X_R)
 +h^3\{K_{\rm floor}+K_A/(4m^2)\}+O(h^5).
\]

返却計画書 §7.1の説明用二層式はAのfull intervalをm分割するので`K_A/m^2`。
本native列は二つのhalf intervalなので内部係数は**`K_A/(4m^2)`**。
両構成の係数を混ぜない。a=1ならK_A=0で、細分化したexact A指数は融合可能。

flatの登録deterministic順序を`D_1,...,D_d`とすると、同じideal tailに対する係数は
`K_flat=sum_i C(X_Di, X_R+sum_{j>i}X_Dj)`。
flatとnestedのleading評価にも同じBCH係数・DF backendを使う。

同じgroup、同じqに対して`K_m-K_1=(1/m^2-1)K_A/4`となるのはleading係数の差である。
cutが相殺できるのはこの形式差だけ。finite-Tのoperator差、coherent-signal bias、有限RTE誤差が
同じように相殺するとは主張しない。`K_floor`は実誤差の下界ではない。
`T^3/q^2 (chi_floor+chi_A/(4m^2))`というnorm上界のleading modelには相殺・高次項の情報がない。
これは有限時間accuracy certificateでも新しい一般定理でもない。

## 4. finite-RTEを挿入したalgorithmの区別

各R(h)を、source-bound canonical finite-RTEの独立occurrenceで一度だけ置換する。
補正後平均を`M_R(h;r,K)`、正normalizationを`B_R(h;r,K)`、physical sample meanを`M_R/B_R`とする。
Rに由来する抽出identity phaseの置き場所をadapterで一度だけ固定する。
同じq/r/K、同じsplit・分布なら、mを変えてもtail occurrence数q、total signed tail time T、
`B_total=B_R^q`は変わらない。qを変える比較ではnormalization・bias・shotsも変わる。

全corrected meanはnative列の積で、physical sample meanはその`B_total`分の一。
coherent signalはこのfirst momentを使う。random unitary channelのdiamond-distance保証と交換しない。
`M_R`はunitaryでなく、ideal列の時間反転・odd-order BCH構造をそのままfinite列へ移さない。

finite化前の指数fusionとfinite化後のexact circuit simplificationを分離する。
一般に`P_K(A)P_K(B) != P_K(A+B)`。Kのラベルがpolynomial degreeと同じとは仮定せず、
現finite-RTE sourceの次数規約を結果前に固定する。二occurrenceを時間和の一polynomialへ変えない。

shot式は返却計画書 §10の共通十分条件を使う案：axis a∈{Re,Im}で
`s_a=epsilon/sqrt(2)-b_a-u_a>0`、
`N_a=ceil(2 B_total^2 log(2/alpha_a)/s_a^2)`、`G=sum_a N_a Cbar_a`。
corrected signalのbias b、数値guard u、alpha分配、cost単位の結果前固定が必要。
この式は最適shot数でもcompiled総costでもない。I2で得たbをI1設計に戻したらoracle-assistedと表示する。

## 5. 実行前に必要なsemantic obligations

| Obligation | 今回の状態 |
|---|---|
| 時系列listとoperator、m1/flat別identity、各fusionの同値性 | 数学仕様案。実装/testは未実施 |
| signed time、scalar/control phase、zero/empty group | 義務を記載。BF用semantic testsだけでBM wrapper検証済みとしない |
| DF block／one-body／tail eventの粒子数sector保存 | 完全な論理block単位で要確認。hardware分解の全gateにsector boundを仮定しない |
| ideal／corrected mean／physical meanの一致 | synthetic end-to-end testを新sourceで要実施、今回は0 |
| finite-T remainder、finite mean数値guard | 未証明／未固定。leading式だけでaccuracy合格を出さない |
| algorithm identity・candidate・結果のatomic保存 | 新namespace/sourceで要設計。旧BF markerは消さない |

[DF評価・既知対照](bm0_df_information_and_prior_art_v1.md)と[pilot案](bm1_small_model_pilot_proposal_v1.md)を
同時にreviewする。新science authorizationはない。
