# A1 Control VariatesとB1-OLの理論的比較

2026-11-07 JST。既存資料の静的調査と数式導出。実科学計算、保存数値結果の再解析、詳細科学レビュー、研究方針の採択は行わない。

## 1. 固定対象・出典・主張範囲

対象は保存linear H4、1.00 Å、STO-3G、legacy DF rank12、8 system qubits、全4電子sectorのN2保証ground、L_D=3、m0/q1/δ=0.02/r1/K2、coefficient_atol=0、X/Y、N1 actual finite-grid Qとselected full列である。理想logical gate・独立uniform53-bitモデルに限定する。PRNG独立性を証明したものではない。

以下は一つの測定軸についての導出である。Yはbit0→+1、bit1→−1の実測出力、すなわちY∈{−1,+1}。phaseは回路に含まれ、bitに二重適用しない。X/Yの平均は別々に推定し、量子費用は両軸を合計する。このtargetはattenuated複素meanであり、元Hamiltonianのenergy・18-round RPE・実機noiseへの保証ではない。

| 固定資料 | 本書で再利用する内容 |
|---|---|
| [N1 744e1a0・REPORT][n1report]、[LAW.json][law] | actual Q、exact質量・ν、5層、固定3-control。未知測定分散の取得を意味しない |
| [N3事前判定 a53c・§6][decision] | 層内定数controlの全分散恒等式、B1-OL分散、補正相殺。この既存整理を以下で展開 |
| [N3期待費用計画 a53c・§2,6–8][expected] | 固定履歴での期待費用、手法全体の期待費用、STOPと返却確率の区別 |
| [752-shot設計 5c27684・契約草案][pilot] | conditional train、unconditional A1評価、5層B1-OL、floor・整数quota・probe heuristic |
| [正式共通指示2026-10-25の同一bytes保存版][policy] | 許可範囲、理論・観測・仮説の区別、SOURCE保全、mandatory STOP |

本書は新たに母平均・母分散・価格を数値評価しない。定理・等式は下記の明示仮定からの解析的帰結であり、当該H4で改善量を観測した主張ではない。

## 2. 一般のCVに対する全分散分解

全正層hに対しp_h=Q(h)>0、Σ_h p_h=1、

$$
\mu_h=E[Y\mid h],\quad \sigma_h^2=\operatorname{Var}(Y\mid h),
\quad \mu=\sum_h p_h\mu_h,\quad \nu=E_Q[g].
$$

g∈R^dは二次momentを持つ既知平均のcontrol、λは評価前に固定した係数とする。定義

$$
Z=Y-\lambda^T(g-\nu),\quad
m_h=E[g\mid h],\quad
\Gamma_h=\operatorname{Cov}(g\mid h),\quad
b_h=\operatorname{Cov}(g,Y\mid h)
$$

から、

$$
E[Z\mid h]=\mu_h-\lambda^T(m_h-\nu),
$$

$$
\operatorname{Var}(Z\mid h)
=\sigma_h^2-2\lambda^Tb_h+\lambda^T\Gamma_h\lambda.
$$

全分散の法則を適用すると

$$
\boxed{\operatorname{Var}_Q Z
=\sum_h p_h\{\sigma_h^2-2\lambda^Tb_h+\lambda^T\Gamma_h\lambda\}
+\operatorname{Var}_{h\sim p}
 [\mu_h-\lambda^T(m_h-\nu)].} \tag{1}
$$

第一項が層内、第二項が層間の寄与である。Σp_h m_h=νよりE_Q Z=μ。独立なunconditional Q測定n件の平均の分散はVar_Q Z/n。この不偏性には最適係数を知る必要はない。

## 3. 現在の3-controlへの特殊化と重複

現在のgは各5層内で一定、すなわちg=g_h。したがってΓ_h=b_h=0であり、

$$
W:=\sum_h p_h\sigma_h^2,\quad
B:=\sum_h p_h(\mu_h-\mu)^2,
$$

$$
R(\lambda):=\sum_h p_h
 [\mu_h-\mu-\lambda^T(g_h-\nu)]^2\geq0
$$

と書けば、既存N3 §6の式は

$$
\boxed{A(\lambda):=\operatorname{Var}_Q Z=W+R(\lambda),\qquad
\operatorname{Var}_QY=W+B.} \tag{2}
$$

となる。A1は層間平均のうちcontrolで表現できる部分だけを除く。任意のλで層内分散Wは変わらず、悪い係数ではR(λ)>Bにもなる。「CVを適用した」だけでは削減を保証しない。

Σ=Cov_Q(g)、b=Cov_Q(g,Y)=Σ_h p_h(g_h−ν)(μ_h−μ)と置くと

$$
R(\lambda)=B-2\lambda^Tb+\lambda^T\Sigma\lambda.
$$

Σが可逆ならλ*=Σ^{-1}b、そうでなければ射影上の解Σ^+bを使う数学的記述が可能で、

$$
R_*:=\min_\lambda R(\lambda)=B-b^T\Sigma^+b,\qquad
A(\lambda)=W+R_*+(\lambda-\lambda_*)^T\Sigma(\lambda-\lambda_*). \tag{3}
$$

これはoracle係数の式であり、実装に真のbを与えることは許可していない。現在の契約はrank不足をminimum-norm解で救済せずSTOPする。

B1-OLはn_h≥1のfresh独立なQ(·|h)測定から

$$
\widehat\mu_{\rm OL}=\sum_h p_h\overline Y_h,\quad
E\widehat\mu_{\rm OL}=\mu,\quad
\boxed{\operatorname{Var}\widehat\mu_{\rm OL}
=\sum_h\frac{p_h^2\sigma_h^2}{n_h}.} \tag{4}
$$

を得る。層間平均の差は分散式に現れない。既知p_hを確定重みとして使い、ランダムな層出現数の揺らぎを排除しているためである。有限Qであっても、ここは独立・復元抽出の測定モデルなので有限母集団補正を掛けない。

同じλを全層に使い、層別平均へ現在のCVを加えても

$$
\sum_h p_h\{\overline Y_h-\lambda^T(g_h-\nu)\}
=\sum_h p_h\overline Y_h
-\lambda^T(\sum_h p_hg_h-\nu)
=\widehat\mu_{\rm OL}. \tag{5}
$$

したがって現在のcontrolによる補正は標本ごとに完全に相殺し、B1-OLに追加の削減を与えない。別々のλ_hとglobal νを無条件に組み合わせると相殺しないので、同じ推定量だとみなしてはいけない。

A1とB1-OLの利益は主に層間成分Bを対象として重複する。A1が取り除く成分とB1-OLの層間除去を独立の利益として掛け合わせたり足したりできない。

## 4. 5層に対する表現力：失われる一つの対比

[proposed_control_banks][banks]と[ConditionalSampler][sampler]の定義を静的に対応させる。a,bはN1固定label、L_iは適用順の位置。層番号は保存SOURCEの0–4を維持する。

| h | 条件 | (g1,g2,g3) |
|---:|---|---|
| 0 | n=0,L0=a | (1,0,0) |
| 1 | n=0,L0≠a | (0,0,0) |
| 2 | n=2,L0≠a | (0,0,0) |
| 3 | n=2,L0=a,L1≠b | (0,1,0) |
| 4 | n=2,L0=a,L1=b | (0,1,1) |

切片込みdesign [1,g_h^T]は5行4列でrank4。h=1と2の行が同一なので、任意の5個の条件付き平均を表現することはできない。affineな完全表現μ_h=a0+λ^Tg_hが可能である必要十分条件は

$$
\boxed{\mu_1=\mu_2.} \tag{6}
$$

証明：同じ行には同じ予測しか置けないので必要。逆に共通値を切片とし、他の3行を各slopeで順に合わせれば十分。μ_1=μ_2が当該H4で成立するかは未評価である。

4種類のpatternをkとし、

$$
P_0=p_1+p_2,\qquad
\bar\mu_0=\frac{p_1\mu_1+p_2\mu_2}{P_0}
$$

を同じzero-pattern群の質量と平均とする。この4群に関してdesignは飽和しているので、最適予測は各群平均となり、

$$
\lambda_*=
(\mu_0-\bar\mu_0,\ \mu_3-\bar\mu_0,\ \mu_4-\mu_3)^T,
$$

$$
\boxed{R_*=
p_1(\mu_1-\bar\mu_0)^2+p_2(\mu_2-\bar\mu_0)^2
=\frac{p_1p_2}{p_1+p_2}(\mu_1-\mu_2)^2.} \tag{7}
$$

最後の等号は加重平方和の展開による。これは保存値の再解析ではなく、全ての正p_hとμ_hに対する記号恒等式である。p_h>0と4patternの存在からΣは可逆になる。g3⊂g2という包含はrank不足そのものではない。

controlを増やさず学習精度だけ改善してもR_*は除去できない。gの任意の非線形関数も同一patternのh=1,2を区別できない。追加のorder情報は数学的には欠けた対比を識別できるが、それは層間表現の補完であり、新しい層内削減を生まない。実装・採択は行わない。

## 5. 有限学習：凍結した一回と学習を反復するpolicy

### 5.1 独立学習と独立評価

Tを学習データ、λ̂(T)を一回fitして凍結した係数とする。評価はTから独立なQ測定、nは固定と仮定する。

$$
E[\widehat\mu_A\mid T]=\mu,\quad
\operatorname{Var}(\widehat\mu_A\mid T)=A(\widehat\lambda(T))/n.
$$

全分散を再び使うと

$$
\boxed{\operatorname{Var}(\widehat\mu_A)
=\frac{W+R_*+E_T[\delta^T\Sigma\delta]}{n},\quad
\delta=\widehat\lambda-\lambda_*.} \tag{8}
$$

Var_T(E[μ̂_A|T])=0なので、係数の揺らぎを別のO(1)分散として二重加算しない。有限二次momentがあれば

$$
E[\delta^T\Sigma\delta]
=\operatorname{tr}(\Sigma\operatorname{Cov}_T\widehat\lambda)
+(E_T\widehat\lambda-\lambda_*)^T\Sigma(E_T\widehat\lambda-\lambda_*).
$$

第一項は係数の揺らぎ、第二項はQ最適係数からの偏りで、両方とも非負。係数の偏りと、fresh評価平均の不偏性を区別する。学習と評価を同じデータで行う場合、E[λ̂^T(ḡ−ν)]は一般に0ではなく、上式は適用できない。cross-fittingもfold間共分散を扱う別設計になる。

n=n(T)を独立評価前に決める場合の分散はE[A(λ̂(T))/n(T)]であり、E[A]/E[n]ではない。B1-OLも独立probe/cal履歴Tでquotaを凍結するなら、条件付き分散Σp_h²σ_h²/n_h(T)をTについて平均する。production応答を見て再配分する方法にはこの式を自動適用しない。

### 5.2 現在のconditional trainとunweighted OLS

[fit_and_freeze][cvsource]は切片付きunweighted OLSであり、5層の測定件数をQ質量で加重するWLSではない。各層r_h件を用意した理想実数算術では、zero-patternのfit値は

$$
\widehat m_0=\frac{r_1\overline Y^{tr}_1+r_2\overline Y^{tr}_2}{r_1+r_2},
$$

他3patternのfit値はそれぞれのtrain平均。r_hが等しい設計では、zero-patternの母fit値は(μ_1+μ_2)/2であり、一般にはQ最適のμ̄_0と異なる。中心化designにνを使っても、切片のあるOLSの行重みは変わらない。

r_1:r_2を固定して学習を増やすと、

$$
m_{tr}=\frac{r_1\mu_1+r_2\mu_2}{r_1+r_2}
$$

へ収束する。その極限係数のQ評価での追加損失は

$$
\boxed{R(\lambda_{tr})-R_*
=P_0(1-P_0)(m_{tr}-\bar\mu_0)^2.} \tag{9}
$$

証明：λ_tr−λ*=(μ̄_0−m_tr)(1,1,0)^Tであり、g1+g2はzero-pattern以外の指示関数なのでVar(g1+g2)=P_0(1−P_0)。したがって学習増加だけでQ最適係数に到達するとは限らない。ただしこの損失の実際の大きさは未評価で、WLSへの変更提案・採択を意味しない。

### 5.3 Range guardとSTOP

式(8)は係数を定義してfresh評価を実施するモデルの分散である。学習guard通過に条件付けて使うなら、Tについての平均もその条件付き分布に変える。失敗履歴に推定値を補ったり、式(8)だけからcampaign全体の返却確率を主張したりしない。

既存[freeze][sampler]はrank4および保守的box幅

$$
D_{\rm box}=2+\sum_{j=1}^3|\widehat\lambda_j|\leq4
$$

を要求し、失敗時にclip/refit/control削減で救済しない。rank成立は係数精度・母分散削減・range合格を保証しない。

実supportの幅は

$$
D_{\rm support}\leq
2+\max_h\widehat\lambda^Tg_h-\min_h\widehat\lambda^Tg_h
\leq D_{\rm box}.
$$

依存するindicatorをboxで独立に囲うため保守性がある。数学的なλ*がguard集合内にあるとも限らない。guardによる不返却はCV不偏性の反証ではないが、同じ返却サービスの費用比較には返却確率を含める必要がある。本書はguardを変更しない。

## 6. 同shot数・期待RZ予算におけるoracle比較

ここでは学習・probe・cal費用を一旦外し、係数と配分を固定する。連続配分で整数・最低shot制約を緩めた理想比較である。σ_h=√σ_h²とする。

### 6.1 同じ評価shot数N

比例配分n_h=Np_hが実現可能なら、

$$
V_{\rm prop}=W/N\leq A(\lambda)/N=V_A,
$$

等号の必要十分条件はR(λ)=0。Cauchy–Schwarzより

$$
\sum_h\frac{p_h^2\sigma_h^2}{n_h}\geq
\frac{(\sum_h p_h\sigma_h)^2}{N}.
$$

S=Σp_hσ_hと置けば、正分散層のNeyman配分n_h=Np_hσ_h/Sで

$$
\boxed{V_{\rm OL,opt,N}=S^2/N\leq W/N\leq V_A.} \tag{10}
$$

S²≤WはJensenによる。N>0で等号V_OL,opt,N=V_Aとなる必要十分条件はR(λ)=0かつ全正質量層のσ_hが同じこと。零分散層への0配分はoracle緩和の極限であり、未知平均を測定せず知れる実装を仮定しない。

### 6.2 同じ期待compiled RZ評価予算B_eval

固定profileのpath価格c(ω)に対しc_h=E[c(ω)|h]、c_Q=Σp_h c_hとする。価格は同じselected full no-prep回路のもので、X/Y別に定義する。以下c_h>0、B_eval>0。層内で価格とYが相関していても、独立fixed-n測定の期待費用には平均価格でよい。

A1はn_A=B_eval/c_QによりV_A=c_Q A(λ)/B_eval。B1-OLはΣn_h c_h=B_evalの下で

$$
K:=\sum_h p_h\sigma_h\sqrt{c_h},\qquad
n_h=\frac{B_{\rm eval}}{K}\frac{p_h\sigma_h}{\sqrt{c_h}},
$$

$$
\boxed{V_{\rm OL,opt,cost}=\frac{K^2}{B_{\rm eval}}
\leq\frac{Wc_Q}{B_{\rm eval}}
\leq\frac{A(\lambda)c_Q}{B_{\rm eval}}.} \tag{11}
$$

導出：K²≤(Σp_h²σ_h²/n_h)(Σn_hc_h)。後半はK²≤(Σp_hσ_h²)(Σp_hc_h)=Wc_Q。等号はR(λ)=0かつσ_hが√c_hに比例する場合に限る。すべて零分散の退化例も含める。

したがって**回路価格の層間差だけでは、現在のA1が費用最適oracle B1-OLを上回る理由にならない**。B1-OLも同じ価格差を使って配分を改善できるためである。c_h=0の層では無制限の無料shotという退化があり、追加制約なしに上記の内点解をそのまま使わない。

期待予算を使い切る前に実現価格を見て停止するhard-budget samplingとは異なる。後者には停止規則と推定量の別解析が必要。本書は新しい停止規則を導入しない。

## 7. 残る層内成分と将来のcontrol

pathをω、t(ω)=E[Y|ω]と記号的に定義すると、二値測定では

$$
\sigma_h^2
=E[1-t(\omega)^2\mid h]
+\operatorname{Var}(t(\omega)\mid h).
$$

現在の層内定数gは両項とも減らせない。層内で変化するpath control u(ω)なら第二項の一部を減らせる可能性があるが、pathから予測できないBorn測定ノイズの第一項は残る。tの実計算は本作業で行わない。

E[u|h]=ν_hが正確に既知なら、

$$
\widehat\mu_{\rm hybrid}
=\sum_h p_h\{\overline Y_h-\beta_h^T(\overline u_h-\nu_h)\},
$$

$$
V_{\rm hybrid}
=\sum_h\frac{p_h^2}{n_h}
 \{\sigma_h^2-2\beta_h^Tb_h^u+\beta_h^T\Gamma_h^u\beta_h\}. \tag{12}
$$

固定β_h・独立測定という条件下である。最適β_hのoracle削減は(b_h^u)^T(Γ_h^u)^+b_h^uで、正なら同quotaで厳密に削減できる。ν_hの推定誤差、有限学習、control測定費用は別途必要。uが層内一定ならb_h^u=Γ_h^u=0で追加削減は0となる。

## 8. 既存理論との関係と未確定事項

[Glynn–Szechtman (2002), §2, Theorem 1, §7/Theorem 6][glynn]は、多重CVのL²射影、推定係数の漸近理論、層indicator CVとpost-stratificationの関係を扱う一次文献である。本書ではその既知の枠組みとの対応を確認し、現在の5層・3-controlへの代数的特殊化を明示した。文献の漸近同値は128-shot pilotの有限標本同値や費用優位の定理ではない。B1-OLは直接層別取得であり、IID測定の事後層別と区別する。

式(1)–(12)はそれぞれ明記したモデル内の関係であり、μ_h、σ_h、c_h、R_*、係数誤差、guard通過率の実値は未確定。現行実装・pilotとの費用比較条件は[A1_ADVANTAGE_CONDITIONS.md](A1_ADVANTAGE_CONDITIONS.md)、研究仮説は[A1_FUTURE_RESEARCH_DIRECTIONS.md](A1_FUTURE_RESEARCH_DIRECTIONS.md)を参照する。

[n1report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/744e1a09661ebba6841fe5faf456f88c41c47780/docs/research/o2_a1_new_target_n1_20261106/REPORT.md
[law]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/744e1a09661ebba6841fe5faf456f88c41c47780/artifacts/o2_a1_new_target/2026-11-06/n1/LAW.json
[decision]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a53c1d6eedea0d85fb484425db9012aa89738129/docs/research/o2_a1_n3_decision_expected_rz_20261106/N3_RESULT_DECISION_FRAMEWORK.md
[expected]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a53c1d6eedea0d85fb484425db9012aa89738129/docs/research/o2_a1_n3_decision_expected_rz_20261106/EXPECTED_RZ_COST_GUARANTEE_PLAN.md
[pilot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5c27684d904aed9f4f9f2787752c917bb46d684c/docs/research/o2_a1_n3_scope_redesign_20261106/N3_PILOT_CONTRACT_DRAFT.md
[policy]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5c27684d904aed9f4f9f2787752c917bb46d684c/docs/archive/server1_main_collection_20261103/b05c68a693d31e3a15652faccfc4e6f0a05f8c34/research_project_instructions_unified_2026-10-25.md.original
[banks]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5c27684d904aed9f4f9f2787752c917bb46d684c/src/trottertracks/algorithm_codesign/a1_connections.py
[cvsource]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5c27684d904aed9f4f9f2787752c917bb46d684c/src/trottertracks/algorithm_codesign/a1_control_variates.py
[sampler]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5c27684d904aed9f4f9f2787752c917bb46d684c/src/trottertracks/algorithm_codesign/a1_n3_connection.py
[glynn]: https://web.stanford.edu/~glynn/papers/2002/GSzechtman02.pdf
