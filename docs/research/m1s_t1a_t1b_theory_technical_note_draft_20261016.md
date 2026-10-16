# Technical note草稿：共有±部分乱択のcoherent信号保証と成立域

2026-10-16 JST。**DRAFT／NOT ADOPTED**。正式な論文原稿・投稿claim・研究方向の採択ではない。

この草稿は[Checkpoint 2 GPT原本][GPT2]で限定受理された数学的成果を、[T1a][T1A]、[T1b][T1B]、[後続addendum草稿][ADD]から整理したもの。新規証明を構築していない。独立新規性・実分子での資源価値は未認証。

## 要旨案

Pauli/Hermitian involution-tailとexact deterministic halfを持つ共有±乱択lawの平均干渉信号を扱う。固定有限次元、固定整数r、小step、有限減衰予算で、全Hamiltonianを基準にしたsecular参照への全step一様複素誤差を与える既存証明を整理する。全Hの対象固有値が非縮退なら、pair-sum共鳴があっても参照振幅はJensen下界を持ち、参照位相は理想相対位相と一致する。単一対象blockでは先導減衰係数比が等式で定まり、共鳴では複数のrateが残り得る。位相補正とouter splitting、有限角、mode mixingを分離する。RPE統計・一般DF-native・native総費用の保証は対象外。

## 1. 対象と既知の構成

H=HD+HR、HR=lambda_R E P_j、P_j²=I、同位置label共有、時系列は同順sign reversal、位置間/step間IID。決定論halfはexact exponent。Directional control・乱択exponential・palindrome・共有乱数自体を新しい回路構成と主張しない。

厳密な平均写像は `Psi_delta(X)=D Phi_theta^r(DXD)D`、`Phi_theta(X)=E[A_j X A_j]`、`D=exp(iδHD/4)`、`theta=lambda_R δ/(2r)`。通常のchannel共役とは違うcoherent first momentであるが、first momentの研究一般が未踏だという意味ではない。[追加照合][LIT]には平均operator信号を明示的に扱う先行研究もある。

## 2. 成果の配置と固定出典

| 構成要素 | 固定出典・受理 | 草稿での主張範囲 |
|---|---|---|
| 完全乱択の平均化 | T1a §§3–6、GPT1 | Nθ²≤D_capの全N一様。局所高次を含む |
| 部分乱択への移送 | T1b §§3–5、GPT2 §§3–4 | βsδ²≤D_capの全s一様。β=lambda_R²/(4r) |
| Modeと左右weight | T1b §7、GPT2 §5 | (SP)・gap/δ条件下のRiesz射影とresidual |
| 共鳴を許す参照floor | GPT2 §6 | E_k非縮退、Jensen下界。今回の新しいCodex定理ではない |
| 係数比・等イベント | T1b §10、GPT2 §8 | 指定law、W>0、小step。費用認定ではない |
| 三次位相・outer/dressing | T1b §§6/8/9、GPT2 §5 | 解析係数と明示剰余を区別 |
| 共鳴反例 | T1b §12、GPT2 §7 | 定義したsigned-involution lawだけ |

## 3. 主評価の記載案

T1bではHS空間上のA=J_H/2、Q=beta D_R≤0、Q_0=厳密pair-sum pinching、u=sδ²とする。受理済みの主誤差は

\[
|z_s-z_s^{H,\rm sec}|\le\sqrt n
\left[Ku|\delta|+u q_n+2|\delta|\kappa_f\{1+u(q+q_0)\}\right].
\]

記号と全証明はT1b原本を参照する。固定modelのη=0では全s一様O(abs(δ))。global HSの√n、逆pair-sum gap、coupling norm、βへの依存を隠さない。系サイズ・gap・rを同時変更する保証や実用的な必要gate数ではない。

## 4. 共鳴時のvisibilityと単一指数の区別

全Hの非縮退固有stateについて、GPT2が受理した表示は

\[
z_s^{H,\rm sec}=e^{iT_{\rm rel}E_k}m(u),\quad
m(u)=\sum_\ell w_\ell e^{u\mu_\ell},\quad
w_\ell\ge0,\ \sum_\ell w_\ell=1,\ \mu_\ell\le0,
\]
\[
e^{-u\beta(W_k+v_k)}\le m(u)\le1.
\]

単純な対象pair-sum blockなら下界の単一指数が等式となる。共鳴反例はこの等式の普遍化を否定し、正の下界・理想参照位相を否定しない。真の有限δ信号は上の複素誤差を控除し、その誤差がfloorより小さい場合だけ位相を換算する。一般guiding state・縮退E_kには同じ系を主張しない。

## 5. 係数比と資源の境界

W=1−<h²>、v=Var_j<P_j>、h=HR/lambda_R。指定ordinaryと(SP)でrho=(W+v)/(2W)、W>0。同じ総イベント位置数Rでの独立±比較は各枝R/2を使い、native価格とは分ける。非縮退・共鳴ありではGPT2の参照visibility下界の比較として述べ、単一実rate比の等式としない。

rho単独の総資源STOPは不採用。Visibilityとshots、control単価、phase bias、精度配分、D/R費用が変わる。高rhoは当該減衰機構の優先度を下げる材料、低rhoは追加検討の材料になり得るが、自動の科学GO/研究終了規則ではない。

## 6. 論文化前に残ること

近接先行のmean operator/variance理論と、明示lawの長時間C-sec保証との厳密な還元可能性をさらに科学判断する必要がある。Yi–Crossonの有効Hamiltonianスペクトル解析は状態・位相分離の既知枠組みである。今回の追加文献照合は技術比較であり、独立priorityの認定ではない。

量子化学への接続では、DF-native eventとPauli-tailのlaw/normalizer/価格差、数値oracle state、全空間の非縮退性、有限stepと固定r極限の違いが残る。Gate 0草案はheadroomの準備であり、分子数値結果ではない。T1c、Q4/Q5、一般DF拡張は別契約・承認を要する。

本草稿の新規性・論文としての着地点・technical noteの正式採択はGPT/ユーザーに残す。**新科学計算0、新規証明0。草稿公開後mandatory STOP。**

[GPT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62554ee127a2bec1206367cf1754df2ff49fa000/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md
[T1A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md
[T1B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md
[ADD]: ../tracks/algorithm_codesign/m1s_g0t1_post_checkpoint2_theory_addendum_draft_20261016.md
[LIT]: ../tracks/algorithm_codesign/m1s_checkpoint2_prior_art_delta_20261016.md
