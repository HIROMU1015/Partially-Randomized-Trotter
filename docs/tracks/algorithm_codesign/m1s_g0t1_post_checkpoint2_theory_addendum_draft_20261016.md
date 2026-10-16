# M1-S Checkpoint 2後：理論addendum草稿

2026-10-16 JST。**DRAFT／NOT ADOPTED／NO SCIENTIFIC EXECUTION AUTHORIZED**。

本書は[Checkpoint 2 GPTレビュー原本][GPT2]の受理済み結果を整理するaddendum草稿である。**新しいCodex証明・独立科学レビュー・新規性認定ではない。** T1a/T1bの原本を変更せず、当時のpendingや留保を履歴として保持する。[technical note草稿][NOTE]、[Gate 0事前調査・契約草案][GATE]、[追加文献照合][LIT]を併せて参照する。

## 1. 今回の権限と出典

ユーザーはCheckpoint 2後の成果整理・read-only source調査・Gate 0事前登録案・追加文献照合・公開を承認した。指示原本は[保存先][USER]。Gate 0/E0、保存データからの新しい科学量、行列評価・対角化、sampling、compile、T1c・Q4/Q5・一般DF拡張は未承認。Technical noteの正式採択、Gate 0の条件/閾値、研究方針変更も今回行わない。

| 結論 | 固定資料 | 今回の扱い |
|---|---|---|
| T1aの固定有限系一様保証 | [T1a][T1A]、[GPT1][GPT1] | 限定受理済みとして再掲 |
| T1bの移送、mode、位相、outer、条件付きrho | [T1b][T1B]、GPT2 §§3–5/14 | 限定受理済みとして整理 |
| 共鳴を許すJensen floor | GPT2 §6（Claude付録A.8の独立確認） | 本addendumの中心。既存レビューの結果を記録 |
| 等イベント数比較 | GPT2 §8 | 先導係数・参照下界に限定 |
| rho単独の一般総資源STOP | GPT2 §9 | **不採用**。限定費用代数と分ける |

GPT2原本は52,863 bytes、SHA256 `89266825e483dbc334180fc4125dfd2e56f77ec6f35bf51f5bfb9dc3741f2774`、保存commit `62554ee127a2bec1206367cf1754df2ff49fa000`。全文を確認してbyte copyした。Claude原本 `レビューinner(2).md` は今回のlocal/attachments/利用可能ref・GitHub既定branch検索では未取得。GPT2に記されたSHA256は**引用情報のみで、今回の独立同一性確認ではない**。Claude全文を再生成せず、判断・補足の採用範囲は取得済みGPT2による。

## 2. 受理済みのT1a/T1b保証

共通scopeは固定有限次元、Pauli/Hermitian involution、共通線形角、同位置共有±、位置間・step間IID。T1bは全H=HD+HRの孤立対象固有state、exact deterministic half、固定整数r、δ=δ_rel→0、u=sδ²有界、βu≤D_cap、β=lambda_R²/(4r)を使う。固定δでr→∞、model/次元/gapの同時極限は受理域外。

T1aのΦ(X)=E[A_j X A_j]はCPTP共役ではない。HS散逸補正Dと厳密同一周波数pinchingを使い、局所高次累積を含め、全NでNθ²≤D_capに一様なO(abs(θ))のC-sec複素誤差を与える。定数18、near/far、入力√nは[T1a §§2–6][T1A]のまま。

T1bは

\[
\Psi_\delta(X)=D\Phi_\theta^r(DXD)D,\quad
D=e^{i\delta H_D/4},\quad\theta=\lambda_R\delta/(2r),
\]
\[
A=J_H/2,\quad Q=\beta D_R\preceq0,\quad
Q_0=\sum_\omega\Pi_\omega Q\Pi_\omega,
\quad\omega=(E_a+E_b)/2
\]

とし、全Hに基づくC-secを `z_s^{H,sec}=<k|exp(isδA)exp(uQ_0)(I)|k>` とする。受理済み主上界は

\[
|z_s-z_s^{H,\rm sec}|
\le E_{\rm sig}:=\sqrt n\left[Ku|\delta|+u q_n
+2|\delta|\kappa_f\{1+u(q+q_0)\}\right].
\]

KはT1b固有の局所三次剰余定数、q_n/κ_fは全Hのnear/far結合量。η=0で全s一様O(abs(δ))になるのは固定有限modelの主張。次元√n、逆pair-sum gap、βへの依存を削除しない。詳しい定義と証明は[T1b §§3–5][T1B]、受理はGPT2 §§3–4。今回再証明・数値検証していない。

## 3. Jensen型signal floor：既存レビューからの追加記録

**GPT2 §6が受理した前提は、対象E_kが全Hの非縮退固有値であること。pair-sum共鳴は許す。** E_kkをe_0、対象blockをQ_{E_k}=Π_{E_k}QΠ_{E_k}と書くと

\[
z_s^{H,\rm sec}=e^{iT_{\rm rel}E_k}m(u),\qquad
m(u)=(e_0,e^{uQ_{E_k}}e_0),
\]
\[
\boxed{e^{-u\gamma_{\rm sh}}\le m(u)\le1,
\qquad\gamma_{\rm sh}=\beta(W_k+v_k).}
\]

ここでW_k=1−<k|h²|k>、v_k=Var_j<k|P_j|k>、h=HR/lambda_R。0≤v≤W。従ってC-sec参照は正の振幅を持ち、その位相は理想相対位相と一致する（mod 2π）。

**GPT2の根拠の整理**：非縮退性からΠ_{E_k}I=e_0。負半定値・HS自己随伴Q_{E_k}のspectral measureでは、m(u)は非負weight、和1、指数の平均rateが−γ_shとなる減衰指数の和。GPTがこの表示へJensenを適用して上記下界を独立確認した。これは**既存レビューの説明の再掲**であり、今回Codexが新しい補題を追加した記録ではない。

真の信号にはE_sigを別に控除する。GPT2 §6の条件付き系は

\[
E_{\rm sig}<e^{-u\gamma_{\rm sh}} \Longrightarrow\
|z_s|\ge e^{-u\gamma_{\rm sh}}-E_{\rm sig}>0,
\quad|\operatorname{Arg}(z_s/z_s^{H,\rm sec})|
\le\arcsin(E_{\rm sig}e^{u\gamma_{\rm sh}}).
\]

対象energy縮退、一般guiding stateには同式を拡張しない。保存された数値stateの残差が小さいことだけで全Hの非縮退性を認定しない。

## 4. 単一指数の等式と共鳴反例の射程

(SP)：(E_a+E_b)/2=E_kのpairが(k,k)だけ、なら `m(u)=exp[−u beta(W+v)]` が等式で成立する。**(SP)はこの単一指数等式の十分条件で、Jensen下界には不要**。T1b原本の「共鳴ではfloor未保証」は当時の提出範囲として保持し、非縮退対象についての後続の補足を本書へ分離する。

受理済み三準位例では `m=(2/3)exp(−W D_damp/2)+(1/3)exp(−2W D_damp)`。否定するのは単一rate等式の一般化で、正のfloorやC-sec参照位相ではない。異符号の同じinvolutionを別eventにするlawを使うため、非冗長Pauli-string表やDF-native一般の反例としない。非冗長化はlawの変更であり、元の反例の減衰をそのまま移さない。

## 5. 同イベント数とrhoの意味

GPT2 §8の比較は同じH、分割、law、対象state、理想相対時間。Rを**概念的な乱択gate位置の総数/step**とし、independent ±ではRを偶数にして各枝R/2、sharedとordinaryではr=Rとする。

| 方式 | 位置数の対応 | C-sec先導係数 |
|---|---|---|
| Shared ± | r=R、同位置label共有 | lambda_R²(W+v)/(4R) |
| Independent ± | 各枝r=R/2、列は独立 | lambda_R²W/(2R) |
| Ordinary | 片枝r=R、角はsharedの2倍 | lambda_R²W/(2R) |

W>0、指定比較law、(SP)下で

\[
\rho_\infty^{(H)}=(W+v)/(2W)\in[1/2,1].
\]

非縮退・共鳴ありでは、GPT2はJensenによる**参照visibility下界の比較**として受理した。複数rateの信号を単一の実rate比の等式としない。W=0は0/0で適用外。

有限の各rを固定した小step漸近の比較と、δ固定でrを増やす運用は別。Rはnative gate価格・回路depth・shotsではない。Directional controlと片枝control、独立二枝のmultiplexingの価格を同じと仮定しない。既存PR-RTEのTaylor law/normalizerとの完全同値も未認証。

## 6. 位相・dressing・outerの受理範囲

GPT2 §5が受理したT1bの三次位相は、outer symmetric splitting、tail finite-angle、散逸Qによるmode mixingに分かれる。Exact HDでもouterはlocal O(δ³)、累積O(u abs(δ))として残る。解析級数の三次係数と、有限δの明示一様剰余を区別する。

状態dressingの `norm(D|k>−exp(iα<HD>)|k>)<=abs(alpha) sigma_k(HR)`、非直交mode射影のgap依存、rateの絶対差・分母付き相対差は別の量。固定αでrを増やす有限step系列の式は未証明。T1cの新規研究は今回開始しない。

## 7. 総資源STOP規則と草稿の境界

GPT2 §9は、同shot schedule・同精度配分・同D費用・同event単価・同候補集合という**限定費用モデル**の不等式を受理した。一方、visibilityでshotsが変わる、control単価・biasが違う一般のRPE総費用にrho単独の上限を移す規則は不採用。

rho≥0.85、rho≤0.7、abs(alpha)sigma≤0.1等を正式な継続/終了・実用適格の閾値にしない。これらは過去レビューの例示である。定数や目標削減率は[Gate 0草案][GATE]の未決欄に残す。今回、費用式の新規評価・最適化もしていない。

## 8. 限定claimと停止

論文化の草稿claimはGPT2 §11.3の範囲、すなわち明示lawの長時間C-sec保証、非縮退対象の共鳴floor、条件付き係数比、位相成分の分離まで。独立新規性、分子での有用性、DF-native、一般guiding state/RPE/総費用は未確立。追加文献照合も新規性の最終認定ではない。

**本addendumとtechnical noteは草稿。正式採択・新規定理の追加・新しい研究契約は行っていない。科学計算0、新規解析証明0、T1c0。資料公開後mandatory STOP。** 次はGate 0草案の対象・比較・評価指標・判断基準のユーザー/GPT採択判断で、科学計算にはその後さらに別実施承認を要する。

[GPT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62554ee127a2bec1206367cf1754df2ff49fa000/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md
[T1A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md
[T1B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md
[GPT1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4899608ea222777e64e706fda2b7e54b84c86247/docs/research/track_b_m1s_g0t1_checkpoint1_gpt_independent_scientific_review_20261016.md
[NOTE]: ../../research/m1s_t1a_t1b_theory_technical_note_draft_20261016.md
[GATE]: m1s_gate0_headroom_preflight_and_prereg_draft_20261016.md
[LIT]: m1s_checkpoint2_prior_art_delta_20261016.md
[USER]: ../../../artifacts/track_b_m1s_post_checkpoint2_preflight/2026-10-16/v1/user_instruction_original.txt
