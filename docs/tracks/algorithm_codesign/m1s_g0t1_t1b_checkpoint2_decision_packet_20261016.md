# M1-S G0-T1／T1b：Checkpoint 2 Decision Packet

2026-10-16 JST。**`T1B_ANALYTIC_TRANSFER_SUBMITTED_WITH_DOMAIN_CHECKPOINT2_PENDING_MANDATORY_STOP`**。

**部分乱択の固定r・小step・有限減衰系列で、全Hを基準とするC-secへの一様複素信号上界を著者証明として提出する。** 単一pair-sum対象blockなら、候補rho^(H)は明示したordinary参照との先導減衰係数比として成立する。一方、孤立H固有stateだけでは単一rateを保証できない。非零HRと非可換HDを持つ解析的共鳴例で、その一般化は不成立と示す。

**著者の数学的提出とGPTの科学的受理・新規性・研究継続判断は別。** Checkpoint 2レビューは未実施・開始未承認。数値計算・Gate 0・Q4/Q5・DF拡張へは進まない。

## 1. 今回の承認と入力

ユーザーの添付指示は「G0-T1のT1bに限定した新規理論研究の実施を明示的に承認します」と明記した。[承認記録][AUTH]と[指示原本][USER]を保存した。既存S0の契約を再採択せず、今回の新しい実行承認を別記録として追加している。

- [正式S0][S0]：`373b8e4b9831f4d00bdc00ffc741ac61e03f0841`。
- [T1a証明][T1A]と[Checkpoint 1資料][CP1]：`4e7b18dd2aa14c127bce9e248c63996c62a055cc`。
- [Checkpoint 1 GPT原本][GPT1]：byte保存commit `4899608ea222777e64e706fda2b7e54b84c86247`、43,022 bytes、SHA256 `f24bf90e3991cd4ed7f7513a75bc1380c03f30ccd90fc09f4d16efea3ba5a32d`。限定T1a受理、独立新規性未認定、限定T1b推奨を根拠とする。今回GPTレビューを再実施した扱いにしない。
- [T1bの定理と全証明][THEORY]、[claim一覧][CLAIMS]、[証拠索引][INDEX]が今回の出力。必須資料のGitHub取得と公開後のbytes/SHA照合を記録する。

研究対象は固定有限次元、Pauli/Hermitian involution、exact deterministic half、同位置共有±、位置間・step間IID、固定r≥1、δ_rel→0、sδ_rel²有界。全Hの孤立固有stateを用いる。一般DF event、内部HDのPF、固定δでr→∞、r/系サイズ同時増加を追加していない。

## 2. 主定理を読むための最小辞書

\[
H=H_D+\lambda h,\quad h=\mathbb E P_j,\quad
\theta=\lambda\delta/(2r),\quad D=e^{i\delta H_D/4},
\]
\[
\Psi_\delta(X)=D\Phi_\theta^r(DXD)D,\quad
A=J_H/2,\quad Q=\beta D_R,\quad
\beta=\lambda^2/(4r),\quad u=s\delta^2,\quad\beta u\le D_{\rm cap}.
\]

HS空間で `D_R=−E(J_j−L_R)^2/2<=0`、J_j(X)=P_jX+XP_j、L_R=E J_j。全Hのpair-sum周波数は `omega=(E_a+E_b)/2`、Q_0は厳密同周波数pinching。

参照は `z_s^{H,sec}=<k|exp(isδA)exp(uQ_0)(I)|k>`。同じ物理時間の理想信号は `exp(i sδ E_k)`。

## 3. 著者が証明した主張

### 3.1 全H C-secへの一様移送

証明§§3–5でΨと `exp(iδA+δ²Q)` の収縮性、二次係数一致、明示局所誤差K abs(δ)^3を導く。`m=norm(H_D)+lambda`、`a=norm(A)`、`q=norm(Q)`、固定解析上限δ_0で

\[
K=m^3/6+(a+\delta_0q)^3/6+a q+\delta_0q^2/2.
\]

T1aの定数18を転用していない。η≥0についてexact resonance以外をnear/farへ分け、q_nはnear blockのnorm、κ_fはfarのordered coupling-norm/周波数差の和とする。全sでβsδ²≤D_capなら

\[
\boxed{|z_s-z_s^{H,\rm sec}|\le\sqrt n
\left[Ku|\delta|+u q_n+2|\delta|\kappa_f\{1+u(q+q_0)\}\right].}
\]

η=0ではq_n=0で固定有限モデルのκ_fは有限。従って全sに一様なO(abs(δ))。次元/逆gap/βの依存は残り、model familyやr増大への一様性はない。Φの固有値やCPTP性から推測した証明ではない。

### 3.2 Modeと減衰係数の十分条件

追加条件(SP)：`(E_a+E_b)/2=E_k` を満たすpairは(k,k)だけ。この条件は主block定理には不要で、scalar系だけに用いる。単純な全Hilbert空間のground/最高stateは満たすが、sector内gapだけでは不十分。

証明§7は `(Psi_delta−I)/delta` とnormalなiAを比較し、pair-sum gap gに対する摂動比を明示する。小δでrank1の非直交Riesz射影P_deltaを得て、`norm(P_delta−P_0)<=4 abs(delta)T_*/g`、左右mode weightと全s一様residualを与える。単に固有値だけで実信号を同定していない。

\[
W_k=1-\langle k|h^2|k\rangle,
\quad v_k=\operatorname{Var}_j\langle k|P_j|k\rangle,
\quad0\le v_k\le W_k.
\]

(SP)なら

\[
\gamma_{\rm sh}=\beta(W_k+v_k),\qquad
z_s^{H,\rm sec}=e^{is\delta E_k}e^{-u\gamma_{\rm sh}}.
\]

有限δのmode log誤差は証明(18)で `abs(log lambda_delta−iδE_k+δ²gamma_sh)<=C_log abs(delta)^3`。rateの絶対差、phase累積、ratioの分母を別に扱う。一般の全H stateで `1−<h>²` をWの代わりに代入してはいけない。

### 3.3 Outer bias・有限角・dressing

exact HRを中央に置いた対称参照S_deltaと全H exponentialの差はlocal O(δ³)、反復後O(u abs(δ))。K_out=(m³+a³)/6を提示した。exact HDだからouter biasが零とは扱わない。主定理のKに含まれる誤差との二重計上を避ける。

三次logは `iδA+δ²Q+iδ³T`、Tはouter commutator係数とtailの有限角係数を含む。単純対象modeではさらにQによる逆gap加重mixingが三次位相に入る（証明(19)）。(SP)域での解析級数の同定であり、有限δ一様保証には別に導いた明示剰余を使う。

状態vectorのdressing O(abs(α)σ_k(HR))、HS mode射影O(abs(δ)T_*/g)、rate絶対誤差O(abs(δ))を区別した。rate相対誤差には正のrate下界、ratioにはordinary分母下界が必要。状態の近さだけで相対rateやrhoを保証していない。

## 4. rhoの結論と比較するlaw

| law（同じH・分割・相対時間sδ） | 規約 | (SP)下の先導rate |
|---|---|---|
| Gate共有± | 各枝r、角θ、同位置label共有 | beta(W+v) |
| 独立± | 各枝r、角θ、左右列独立 | beta W |
| Ordinary | 片枝r、角2θ、HD half各δ/2 | 2 beta W |

独立±ではmeanが `V_delta X V_delta`、V_delta=D(E A_j)^rDとfactorizeし、共有のΦとは異なる。ordinaryは `O_delta=exp(iδHD/2)(E exp(i2θP_j))^r exp(iδHD/2)`。この正確なmeanから係数を導出した。本書の独立±指数gateは、PRのTaylor RTEと同一law・同一費用だとは主張しない。

**旧G0の実際の候補式**

\[
\rho_\infty^{(H)}=\tfrac12+\frac{v_k}{2(1-\langle h^2\rangle_k)}
\]

は(SP)、W>0、上記ordinary参照の下で、**先導mode減衰係数比として`HOLDS`**。独立±に対する同rでの改善ではない。finite-step mode ratioには証明(24)の分母付き誤差があり、finite-sの信号半径比・イベント数比・RPE/費用比へは格上げしない。

| 適用域 | rhoの分類 |
|---|---|
| (SP),W>0,明記したordinary、固定r小δ | `HOLDS`（先導係数比に限る） |
| 有限δ、mode gap/分母下界あり | `HOLDS_WITH_CORRECTION`（証明(24)のrate ratio誤差） |
| W=0 | `NOT_APPLICABLE`（0/0） |
| 共鳴blockで複数rate、単一係数に還元不能 | `NOT_APPLICABLE`（当該single-rate claim） |
| 任意共鳴blockの代替ratioや一般分子の有用性 | `UNRESOLVED`／未検証 |

この条件付きHOLDSは**著者の数学的提出**で、Checkpoint 2の独立科学認証ではない。

## 5. 解析例・反例と保証の強さ

全て手計算。分子geometry/basis/DF rank/L_Dは該当せず、数値window/数値validationの結果ではない。

1. **非可換二準位。** P=Z/X等確率、lambda=1、HR=(Z+X)/2、HD=(Z−X)/2、H=Z。W=1/2,v=1/4なのでrho=3/4。誤ってWを1−<h>²=3/4へ置換すると2/3になる。実際の旧候補式への反例ではなく、単純代入への反例。有限D_capで正の参照floorがあり、主定理のO(abs(δ))と両立する非空δ域を明示した。
2. **非零tail・非可換決定論部を持つ三準位共鳴。** H=diag(−1,0,1)、target|0>。Pは|0>と(|−>+|+>)/sqrt(2)を交換し、残る直交vectorを固定するinvolution。Pと−Pを確率(1±ξ)/2でdraw、0<ξ<1。h=ξP、HR=lambda ξP、HD=H−HR、W=1−ξ²,v=0。対象stateは孤立しているがpair-sum共鳴があり、C-secは **(2/3)exp(−W D_damp/2)+(1/3)exp(−2W D_damp)**。単一exp(−W D_damp)とは異なる。許可されたHermitian involution classの反例で、Pauliだけのsubclassに同じ反例があると主張しない。

主定理は2のblock信号も正しく近似する。scalar rhoの不成立をblock保証の失敗、全M1-Sのno-goに読み替えない。共鳴例の単一係数は初期微分には一致するが、有限減衰系列の全区間のrateではない。

## 6. 次元・gap・alias・情報アクセスの留保

global HS入力√n、全spaceのcoupling norm/逆pair-sum gapを使う。state-specificな安価な量を得たわけではない。計算に必要なexact state・gap・weightはoracle診断量であり、実機に無料で供給されるとの仮定ではない。

δに伴いstep eigenphase gapも縮む。証明§13は非alias域でstep gap≥(2/pi)abs(δ)g、摂動≤δ²(q+K abs(δ))を比較した。固定gでは比は零へ進むが、finite-angle aliasと小gap familyを一律排除していない。Pauli-tailがsector外へ出る場合は全spaceが必要。保存量の名称だけで縮小しない。

参照floorがあるときだけ単一主値位相へ換算できる。共鳴blockでは一般にfloor未保証。shot freshness、統計集中、multi-round、RMS energy、state preparation、native価格は対象外。

## 7. 既知性とCheckpoint 2で判断する科学的論点

一般技法は既知のTaylor/Duhamel/有限次元平均化・Riesz摂動で、これ自体を新発明としない。T1aから単に記号を変えた箇所と、今回新たに導いた部分乱択のβ・full-H W/v・outer/位相・mode・ordinary/独立±・共鳴例を、証明§§15–16で区別した。

G0-pre/P0/P0.5とGPT1の文献監査を再利用し、Campbell/KGW/Daviesの一次ページの書誌/abstractのみ限定確認した。全PDFの再読・網羅的新規性調査・独立priority認定はしていない。PRの独立RTEとのlaw/normalization差、random PF集中と無条件first momentの差を保持する。

Checkpoint 2の論点は次の通り。

- 定理1の規格化、局所剰余、near/far、mode/log係数、ordinaryの時間対応とrho域を独立検算し、著者主張を科学的に受理できるか。
- 全Hの有限減衰coherent保証・位相補正・scalar化条件・共鳴の限界が、既知平均化の直接系以上に情報を持つか。
- 独立論文の中心claim候補を「固定有限次元の部分乱択coherent信号保証とscalar化の成立域」に限定して検討する価値があるか。それでもtechnical noteで十分か。
- 全space定数の保守性、(SP)/分母条件、実分子での情報アクセスを踏まえ、継続・範囲縮小・technical note・STOPのどれが妥当か。

これらは**判断の材料**であり、Codexによる採択・研究継続推奨・全M1-SのGOではない。

## 8. Gate 0で確認する価値があり得る事項と未解決

Checkpoint 2が必要と判断する場合の候補としてのみ、実分子の到達space・coupling-weight・pair-sum gap、W/vと分母、floor、global boundの非空性を挙げる。現時点で計算対象・閾値・L_D・delta窓・H4/H6実行を採択していない。小rhoだけをGate 0 GOにしない。

残る数学的問題は、共鳴blockを含む有用なstate-specific rate/phase保証、size/gap依存の軽減、Pauliのみの共鳴障害の自然な範囲、実装HD内部PF、一般DF event。後二つは初期scope外で別契約を要する。有限資源・RPEへの拡張は未実施で、理論の肯定結果を理由に自動開始しない。

## 9. 証拠状態・実施範囲・mandatory STOP

| 区分 | 状態 |
|---|---|
| 解析的研究 | 一様移送・条件付きmode/rho・位相/outer・反例を著者提出 |
| T1a数学的受理 | GPT1により限定受理済み。旧著者記録は保持 |
| T1b独立科学的受理・新規性 | 未実施・未認定 |
| 文書/参照/hash/公開bytes | 証拠JSONで技術確認。数学的受理の代替ではない |
| 新科学数値・行列対角化・simulation/sampling | 0件 |
| H4/H6の新科学量・回路/compile・forecast | 0件 |
| 既存SOURCE/result/classification/auth/marker | 保持 |
| Checkpoint 2 GPTレビュー | 開始未承認、未実施 |

今回のT1bの実施承認を再度求める必要はない。著者成果と未解決を揃え、正式S0§13および今回指示§14の **「Checkpoint 2のGPT重要科学レビューは、ユーザーがその時点で明示的に開始を承認した場合にのみ実施」** に従い、**ここでmandatory STOP**。次はCheckpoint 2重要科学レビューの開始判断である。Gate 0/E0、Q4/Q5、DF拡張の実行許可はない。

[THEORY]: m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md
[T1A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md
[CP1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_checkpoint1_decision_packet_20261016.md
[S0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/373b8e4b9831f4d00bdc00ffc741ac61e03f0841/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md
[GPT1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4899608ea222777e64e706fda2b7e54b84c86247/docs/research/track_b_m1s_g0t1_checkpoint1_gpt_independent_scientific_review_20261016.md
[AUTH]: ../../../artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/user_authorization_record.json
[USER]: ../../../artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/user_instruction_original.txt
[CLAIMS]: ../../../artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/analytic_claim_inventory.json
[INDEX]: ../../../artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/evidence_index.json
