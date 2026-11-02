# G4-A：B2 budget-policy分離とデジタル化の独立認証契約・証明

2026-10-09。認可元は[GPT G3再レビューv2](../../research/track_b_G3_scientific_review_20261009_rereview_v2.md)§13。
基点3b0fa70b47848e72ef9a9c9e13afc3164962f7cd。旧STOP/marker/resultを変更せず、新G4として一回検算する。
この証明は条件付き数学・保存値算術であり、外部再現・情報理論的下界・旧D0 witnessではない。

## 比較class

旧2-qubit distinct-basis controlled finite P3、p=(3/4,1/4)、σ=+1、既存x={1/8,1/4}。
7 prototypes×3 precisionの21 columns/x、旧phase/内部IID label/cost/strict error/workspaceを固定する。
理想B2はordinary/PTSC-K0/Aの凸混合と、各active prototypeの3 precisionへの非負配分。
同じevent辞書上の任意full-support proposalを許す。real/imagで同じtask policy・lawを使う。
new dictionary、Pauli cancellation、既知寄与除去、stratification、state-dependent variance、別confidence方式は含めない。

## 全混合への還元

representation aのactive group gごとにprecision fraction f[g,e]を持つ場合、pure assignment
πの係数を `theta[a] product_g f[g,π(g)]` とすれば、group/precision marginalが元の配分へ一致する。
Σπの係数はtheta[a]であり、group norm・label law・cost/errorを固定したevent係数cの凸分解となる。
従ってordinary9＋PTSC27＋A27の63 pure profilesが、今回の理想B2 coefficient classを生成する。
inactive groupは分解に現れずzero massを除算しない。

profile vのK_v=Σc_i h_i≥0、s_v=ε−Σc_i d_iとする。s_v>0のK_v/s_vの下界をr≥0とする。
正sではK_v≥r s_v、非正sでもK_v≥0≥r s_v。任意凸混合では同じ線形不等式K≥r sが成立する。
これはK/s緩和の完全性。finite-shot/range目的そのもののpure-profile最適性を意味しない。

## confidence費用下界

d_i=2δ_i、m2=Σc_i²/q_i、L=max c_i/q_i、s=ε−d^T c>0。
固定policy `n=ceil[ell(2m2/s²+(4/3)L/s)]` の出力はn≥2ell m2/s²。
任意q>0、Σq=1でCauchy–Schwarzにより `(Σq C)(Σc²/q)≥(Σc sqrt(C))²`。
従ってtwo axesの費用は `G_C=2nΣq C≥4ell(K/s)²≥4ell r²`。
T=0のeventを含んでも不等式は有効。ISのinfimumを達成する必要はなく、cost0を置換しない。
1Qは `n(2Σq C_1Q+5)=2nΣq(C_1Q+5/2)` とし、h_i=sqrt(C_1Q+5/2)を使う。
rangeと整数切上げを落とす向きは下側。物理的に必要な最小shotsの下界ではない。

ell=ln10560についてexp(37/4)<10560を、正項Taylor60項＋幾何tailで有理認証する。
下界出力は保守的な37 r_lo²。根は384-bit dyadic/isqrtで囲み、r_loは63個のratio lowerの最小。
別にln10560の上下をpositive atanh seriesで算出し、保存J1のBernstein式を再検査する。
旧G2/G3のroot/log/profile/certificate helperはimportしない。

## 対称なデジタルclassへの延長

同じevent集合上で、ある理想B2 cへ対応する非負tilde cと、e≥||tilde c−c||_1を持ち、
tilde s=ε−e−d^T tilde c>0を要求する。coefficient L1をobservable biasとして課金する。
全eventでh_i+r d_i≤rが成立すれば

`h^T tilde c−r tilde s = (h^T c−r s) + (h+rd)^T(tilde c−c)+re ≥0`。

理想側線形不等式とL1双対性、h+rdの非負性から従う。同じrがdigital classの任意proposalへ延長される。
これを各x・T/1Q priceで実データ全event上に認証する。
G3のnorm midpoint化は理想profileへ対応し、保存coefficient intervalsがideal normを囲むことを平方比較で確認する。
degree residualだけを許す旧K2/K3の任意点がこのL1対応を持つとは主張しない。
G3 J1にも同じe＋synthesis biasとq/weight/shot/readout規則を支払わせるため、digitalizationの優遇はない。

## 実装・検査・分岐

source：`scripts/tracks/algorithm_codesign/g4_independent_certificate.py`、stdlibのみ。
sourceeventを旧R1 raw JSONに照合し、word/rotation/phase/IID probability、prototype式、normalized方向intervalを再確認。
既存2 xの126 profilesをT/native1Q/readout付き1Qの378価格計算で再構成する。
G2既存目的区間とのoverlapは事後照合であり、その区間を新下界入力に使わない。
G3各xの保存J1 1Q選択lawを一つずつ、別実装でq、exact cancellation、ideal norm、bias、m2/L、shots、費用まで検査する。
新candidate・新sampling・LP・matrix・合成は0。13 synthetic/off-domain testsのみ。

cap wall600s/CPU480s/AS512MiB/output16MiB、一回/retry0、exclusive marker。
x=1/4の同一finite lawがT下界とreadout付き1Q下界を下回り、bridge条件が閉じればG4-B技術条件PASS。
x=1/8の非分離を新law/precision探索で救済しない。未成立・証明failureならBを実行せずGPTへ返す。
Bは別source/scope固定後だけ、一次文献のCTS finite operator specializationを使用する。
G4-Cは次検証の比較設計のみ。全工程後mandatory STOP、科学的採否はGPT。
