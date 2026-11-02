# N3幅比較 v2：数学・対照・証拠の実装契約

2026-10-12レビュー系列。[完了済み科学レビュー](../../artifacts/n3_width_comparison_v2/2026-10-12/preparation/review_intake.md)第5–15節を具体化する。
実装・synthetic検証・未使用入力の固定は認可済み。**新しい6入力の科学実行は別の明示認可まで行わない。**
旧結果、S1、旧2-base gate、N1停止/N2保留を保持する。

## Hamiltonianと保証対象

実対称h、非負unordered density J、固定ν、一候補につき一mode dを空に射影。
H=Σh_ij a_i†a_j+Σ_{i<j}J_ij n_i n_j。残存Hamiltonian A_dはone-bodyとdensityの両方を含む。
trial上界a_d≤U_d、外部占有sectorの下界C_d≽c_d I、g=c−U>0を使うSchur保証の対象は0≤a_d−E0(Hν)≤δ。
真のground/gap、物理状態のRDM、時間発展、PF、shots、量子総費用を入力や評価に使わない。

Uはactive modeのlexicographic determinant組合せの先頭最大8件に対するexact期待値の最小。
全occupationを列挙した最良Uではない。各法で同じpoolを使う。
c=h_dd+σ_{ν−1}(h_A+diag J_{id})の認証下界。外部sectorの残存非負densityを落とす。
今回は外部occupation全列挙の強化を一次比較へ入れない。c≤Uの不受理は物理的縮約不能ではない。

## 7対照と同じ情報

| method | 有限実装 | 証拠・役割 |
|---|---|---|
| NORM | β²=||v||²、κ=0、s=−β² | 安価なscalar対照 |
| FIVE | 既存のGershgorin/spanによる5 shift。SPDをexact確認し逆二次形式を解く | OLD族の有効証拠。singular shiftは未使用 |
| OLD | 32-step concave λmin score探索、有限dyadic候補 | Kκ−μI≽0、s=νμ。solver失敗で全族不成立とはしない |
| OCC | 32-step concave最小ν固有値和score探索、有限dyadic候補 | Z≽0、Kκ−τI+Z≽0、s=ντ−trZ |
| SDP_EQUIVALENT | 同じ理想OCCの一般SDP、固定3δ tiers | 同保証の取得能力・費用の参照。理想保証への優位を主張しない |
| DENSITY_SOS | 固定real quadratic weighted SOSの5 Gram blocks＋粒子数等式 | 相互作用を保持する強い参照。有限実装の失敗は理想SOS不成立ではない |
| FULL_H_LOWER | full one-body σν(h)の認証下界L_H | 0≤a_d−E0≤U_d−L_H。cの取得を必要としない別経路 |

FIVEのshiftはr=min_i(h_ii−Σ_{j≠i}|h_ij|)、span=maxdiag−mindiagからλ=r+ρ span、ρ={0,1/20,1/10,3/20,1/5}。
元sourceの追加λmin refinementや新しいshift scanは行わない。v1の結果・FIVE証拠を上書きしない。
OLD/OCCのscoreはκ∈[0,g+ε_proj]。端点と最良normalized abscissaを40-bit dyadicへ丸めた上下のκを証拠候補にする。
全固定proposalを保存し、ε_projに通ったことだけでは早期終了しない。
SDP/SOSはδ={ε_proj,ε_proj/2,ε_proj/4}の3点、各1 solver call。結果後にtiers・precision・levelを増やさない。

## 同値SDPと証拠の向き

D=g+δ、Kκ=κh_A−vvᵀ。実装は

\[
\min_{0\le\kappa\le D,\tau,Z}\ q=\kappa U-\nu\tau+\operatorname{tr}Z,
\quad Z\succeq0,\quad K_\kappa-\tau I+Z\succeq0.
\]

これは科学レビューのFantope epigraph表示

\[
\rho_D=\max_{\gamma,t}\{v^T\gamma v-Dt:
0\preceq\gamma\preceq I,\operatorname{tr}\gamma=\nu,
t\ge0,t\ge\operatorname{tr}(h_A\gamma)-U\}
\]

と同じ理想値を持つ。κ/τ/Zの有理証拠はρ_D≤qを保証し、q≤δDなら受理。
PSD-LMIのdualからγ候補を得て有理化し、trace補正と有限のidentity混合を試す。
γ≽0、I−γ≽0、trγ=ν、tのepigraph条件をexact確認し、vᵀγv−Dt>δDなら当該δの**理想OCC条件族**の非可行を証明する。
数値objective／solver statusを厳密上下界としない。finite failureは未判定。反証は物理的な縮約不能、SOSや別法の不能を意味しない。
ν=1、ν=m、縮退・singular PSDも定義内で扱う。

## 固定density-aware weighted SOS

Fκ=κA−BB†。生成子は実係数の

\[
\{a_i\},\quad\{a_i^\dagger\},\quad\{a_j a_i:i<j\},\quad
\{a_i^\dagger a_j^\dagger:i<j\},\quad\{a_i^\dagger a_j:i,j\}.
\]

各blockのGram Q_b≽0と、(N−νI)に掛けるreal Hermitian quadratic／定数の自由係数を使い、

\[
F_\kappa-\ell I=\sum_b\sum_{ij}(Q_b)_{ij}O_{bi}^\dagger O_{bj}
+\sum_r\eta_r f_r(N-\nu I)
\]

の係数一致を数値SDPへ入れる。f_rはI、n_i、a_i†a_j+a_j†a_i。
CAR normal orderingの係数mapをmemo化し、CSC疎行列でCVXPYへ渡す。全hierarchy・実分子spin-adapted問題を実装したとは言わない。
この固定levelはOCC証拠を含む：R=Kκ−τI+Z、粒子Gram=R、正孔Gram=Z、Dの対角=κJ、number multiplier=τ、ℓ=ντ−trZ。
この包含と符号をsyntheticで全Fock恒等式と独立sector actionから確認する。

数値Gramを有理化・PSD修復すると係数一致が変わる。その残差Eを元係数から再構成し、各CAR monomialのnorm≤1からe=Σ|E_coefficient|を取り、s=ℓ−eを安全な下界とする。
独立checkerはproducerのCARをimportせず、固定ν-sectorのladder actionでFκ−ℓI−ΣGramを再構成し、Hermitian残差の最大絶対row sum≤eを検査する。
number multiplierは固定ν-sectorで消える。PSD修復・残差による下界の悪化を隠さない。
このsector checkerの規模はcomb(m,ν)。一般fillingで全pipelineが多項式とは主張しない。

## 幅とselector

証拠κ,sを固定して、許容幅のinfimumは

\[
\delta_w=\max\{0,\kappa-g,(\sqrt{g^2+4\max(\kappa U-s,0)}-g)/2\}.
\]

既存128 fractional bitsの外向きroot区間を再利用。独立checkerはpolynomialとboxで囲み・正のproof上端を検査する。
有限候補poolの最良上端を返し、理想OCC/SOS最適幅と呼ばない。反証なしで探索失敗から二分探索の下端を上げない。
同じ係数／ν／Uでcが強まれば、既存κ,τ,Zを維持してscalar幅を再検査するtransportを実装する。

同modeの採否・幅と、同じD_set内で一つでも削除できるH単位endpointを分ける。
safe上端が最小、同値なら固定mode順を採用。幅区間の重なりをstrict順位証明にしない。
mode候補の追加受理が2件でも、返す削減は最大1mode。可行候補なしは元H fallback、削減0。
cheap OR={NORM,FIVE,OLD,FULL_H_LOWER}を同じD_setで作り、OCC/同等SDP/SOSの各operational ORに共通で与える。
fallback／埋込みの利益をOCC固有gainへ数えない。ORは保存有効証拠のcoverage集計で、専用運用経路のcold timeを実測したとはしない。

## 誤差会計の訂正

将来|E_est−a_d|≤ε_solverを得た場合の元H区間は[E_est−ε_solver−δ_cert,E_est+ε_solver]。
E_estを報告する絶対誤差半径はε_solver+δ_cert、区間中点E_est−δ_cert/2ならε_solver+δ_cert/2。
未知の真の投影誤差を正確に引く処理ではない。今回のendpointは射影certificateで、後段solver/PF/統計契約は採択しない。
