# A1がB1-OLより有利になる条件・限界

2026-11-07 JST。静的調査・理論整理のみ。数値的優劣・科学的GO/NO-GOを確定しない。

固定対象・出典・証明は[A1_VS_B1OL_THEORY.md](A1_VS_B1OL_THEORY.md)を参照する。以下は一軸で、同じactual Q、state、selected full回路、価格profile、targetを比較する。記号は

$$
A=W+R(\lambda),\quad S=\sum_h p_h\sigma_h,\quad
c_Q=\sum_h p_hc_h,\quad K=\sum_h p_h\sigma_h\sqrt{c_h}
$$

を引き継ぐ。係数・配分はfresh評価前に固定し、まず完了するfixed-n設計を考える。c_h>0、必要な分母>0を仮定する。

## 1. 先に区別する比較対象

| 比較 | 理論的な結論 | 限界 |
|---|---|---|
| 同じ評価shot数、比例層別が実現可能 | W/N≤A/N | 任意の整数quotaが比例配分と同じ性能とは限らない |
| 同じ評価shot数、連続Neyman最適B1-OL | S²/N≤W/N≤A/N | 最適σ_hを実装が知るとは仮定しない |
| 同じ期待RZ評価予算、連続費用最適B1-OL | K²/B_eval≤c_QW/B_eval≤c_QA/B_eval | 最低shot・整数・probe誤差・startupを含まない |
| 実際の固定quotaと係数 | 下記の必要十分不等式で比較 | 母分散・期待価格は今回未評価 |
| train/probe/cal込みの同精度・同返却サービス | 全期待費用・精度・返却確率の比較が必要 | 観測RZやpilot scoreだけでは確定しない |

**現在の3-controlに、理想的な最適B1-OLを上回る固有の層内分散削減能力はない。** 一方、上のoracle関係は、全ての有限実装・保証経路・追加費用に対する一様な費用優劣を述べない。

## 2. 同じ評価shot数でA1が勝つ必要十分条件

両方式に評価shot数Nを与え、B1-OLの正整数quotaがΣn_h=Nを満たすとする。

$$
\Pi_N:=N\sum_h\frac{p_h^2\sigma_h^2}{n_h}-S^2\geq0,\qquad
G_N:=A-S^2=(W-S^2)+R(\lambda)\geq0.
$$

Π_Nは連続最適配分からの分散penalty、G_NはA1とoracle B1-OLのgapである。直接の差は

$$
\boxed{V_A<V_{\rm OL}\quad\Longleftrightarrow\quad
\Pi_N>G_N.} \tag{A1}
$$

これは**同じ評価N・同じ凍結λ・同じ母分散モデル内での必要十分条件**。最低shot、整数丸め、不正確なproxy、固定shot目的と費用目的の不一致などはΠ_Nを増やす可能性があるが、「floorがある」だけでは十分条件にならない。

係数がλ*に近くR_*が小さければA1のgapは小さくなり、有限quotaに勝つ可能性は高まる。しかしλ*であってもG_N≥0であり、oracleへの勝利にはならない。R(λ)<Bは同shot数のB0に対するCV改善の必要十分条件であるが、有限B1-OLへの勝利の必要条件ではない。B1-OLの配分penaltyが大きければλ=0でも勝ち得るためである。

比例配分が正整数で実現し、実B1-OLが少なくともその分散を達成するならV_OL≤W/N≤V_A。この条件下でA1の厳密な分散優位は不可能である。

## 3. 同じ期待compiled RZ評価予算

B1-OLの凍結shot比率をa_h=n_h/Σn_j>0、Σa_h=1として

$$
F(a):=\left(\sum_h\frac{p_h^2\sigma_h^2}{a_h}\right)
       \left(\sum_h a_hc_h\right),\quad
\Pi_c:=F(a)-K^2\geq0,\quad
G_c:=c_QA-K^2\geq0.
$$

期待RZ予算に合わせてこの比率を連続的にscaleするモデルでは

$$
V_A=\frac{c_QA}{B_{\rm eval}},\qquad
V_{\rm OL}=\frac{K^2+\Pi_c}{B_{\rm eval}},
$$

$$
\boxed{V_A<V_{\rm OL}\quad\Longleftrightarrow\quad\Pi_c>G_c.} \tag{A2}
$$

これも**このモデル内の必要十分条件**である。層ごとの価格差はF(a)とKを両方変える。高価格層のfloorが大きな費用penaltyを生むことはあり得るが、費用最適配分にも同じ価格情報を与えればΠ_c=0に近づけられる。価格差そのものをA1に有利な十分条件としてはいけない。

同shot数の最適配分はa_h∝p_hσ_h、期待RZ制約での最適shot比率はa_h∝p_hσ_h/√c_hであり、目的関数が異なる。752-shot草案のscoreは後者をproxy化した方向だが、Σn_h=128を固定したheuristicであり、厳密なfixed-shot最適性も費用最適性も主張できない。

## 4. 全量子費用・整数制約・最低shot数

同じ「shot数」が学習等を含む全取得数N_totを意味する場合は、追加shotをt_A,t_OLとして、比較式は

$$
\frac{A}{N_{\rm tot}-t_A}<
\frac{S^2+\Pi_{N_{\rm OL}}}{N_{\rm tot}-t_{\rm OL}},
\quad N_{\rm OL}=N_{\rm tot}-t_{\rm OL}
$$

となる。両分母が正で所定quotaを実行できる場合に限る。同じ評価shot数と同じ全取得shot数を混同せず、層ごとに価格が異なるなら全shot数の同額化も期待RZの同額化とは区別する。

### 4.1 固定履歴の追加費用を含める

F_A,F_OLをtrain/probe/cal等の評価前量子RZ費用とする。共通setupも必要な範囲で含め、どの費用を実現値・期待値として固定したかを明記する。古典秒・メモリ・compile回数は別vectorであり、RZに足さない。

共通総期待予算B_totから、整数丸め等で使わない予算u_A,u_OL≥0を除き、評価に用いる量を

$$
B_A=B_{\rm tot}-F_A-u_A=n_Ac_Q,\quad
B_{\rm OL}=B_{\rm tot}-F_{\rm OL}-u_{\rm OL}=\sum_h n_hc_h
$$

と定義する。a_hはこの実quotaの比率とすれば、

$$
\boxed{\frac{c_QA}{B_A}<
       \frac{K^2+\Pi_c}{B_{\rm OL}}} \tag{A3}
$$

がA1の分散優位の必要十分条件である。n_Aもn_hも整数の実式と同じ内容で、未使用予算を実消費に見せない。

連続モデルu_A=u_OL=0なら、同じ目標分散v>0を満たす最小評価量について

$$
C_A(v)=F_A+\frac{c_QA}{v},\qquad
C_{\rm OL}(v)=F_{\rm OL}+\frac{K^2+\Pi_c}{v},
$$

$$
\boxed{C_A(v)<C_{\rm OL}(v)
\ \Longleftrightarrow\
v(F_{\rm OL}-F_A)+\Pi_c>G_c.} \tag{A4}
$$

**十分かつ必要なのは、この不等式全体であり、個々の理由の列挙ではない。** 比率aを固定してscale可能な、guardを通過する完了policyのvariance比較に限る。

特にoracle B1-OLではΠ_c=0なので、A1が勝つにはF_OL>F_Aが必要で、さらにv(F_OL−F_A)>G_cであれば十分でもある。同じstartupなら不可能。小さなvでは1/vの評価費が支配し、固定startupの利益は相対的に消える。ただし実際の保証policyではcal量もvに依存し得るので、この漸近説明を無条件に移さない。

### 4.2 実行可能なquota集合

整数実装の正しい比較対象は、許可されたfloor f_h、shot cap、費用制約等からなる集合Nを用いた

$$
\min_{(n_h)\in\mathcal N}\sum_h p_h^2\sigma_h^2/n_h
$$

である。oracle連続解はその下界で、SOURCEのheuristicは通常その上側にある一つの点。最低shotがあることによる損失と、その制約下でさらに不適切なquotaを選ぶ損失を区別する。

全正層にf_h≥1を要求する未知平均推定では、予算がΣf_hc_h未満ならその層別実装は実行不能となる。一方A1は少数のunconditional評価を定義できる場合がある。これは**実行可能性の差の可能性**であり、A1が所定精度や返却率を満たす十分条件ではない。零分散層を標本分散0だけで除外してはいけない。

### 4.3 752-shot設計から言える限定事項

[固定草案][pilot]は各軸でA1 train=5×16、B1-OL probe=5×8、両方式の評価=各128、B1-OLの評価floor=各層4、両軸合計752と定める。今回この草案を実データに適用していない。

同じ条件付きlawと価格で全train/probeを完了するモデルなら、一軸の期待追加量子費用は

$$
F_A^{pilot}=16\sum_h c_h,\qquad F_{\rm OL}^{pilot}=8\sum_h c_h.
$$

したがって、この設計ではA1の学習費が安いことを期待優位の理由にはできない。上式は完了する所定段階の期待値であり、実現価格やSTOP込み費用の恒等式ではない。

128評価shot同士の優劣と、これらの追加費用まで同額化した優劣は異なる。各層floorによる損失が大きく、A1のRが小さいなら有限実装に勝つ余地はあるが、今回それを数値認定していない。希少層ほど本来の最適quotaが小さい可能性はあるが、p_hだけでσ_hやc_hを決めることはできない。

## 5. 学習・calibration・rangeの扱い

### 有限係数

理論文書の式(8)–(9)が示すとおり、独立学習の係数誤差はoracleより非負の分散損失を加える。conditional trainでrank4を得ても、Q最適係数への近さ、Rの小ささ、guard通過率は分からない。均等trainとunweighted OLSの重みの差も残り得る。

「controlとYの相関が強い」はRを小さくする方向の説明だが、それだけで(A1)–(A4)を満たすとは言えない。B1-OLが同じ層間差を既に除去しているためである。

### 保証付きshot数

分散vの一致は、同じ(ε,α)精度保証の一致ではない。例えば真の単発分散Aと幅Dを使う標準Bernsteinの十分条件は

$$
n\geq\left(\frac{2A}{e^2}+\frac{2D}{3e}\right)\log(2/\alpha),
\qquad e>0.
$$

ここでは独立有界測定、既知の正しい母分散・幅という仮定を明示した式にすぎず、実Nを計算していない。SOURCEでは独立calでvariance上界を得る既存保証経路を用いる。A1のD_boxとB1-OLの層別幅、信頼水準の分割、Σp_he_h≤ε−b_engine、有限candidate配分・capが必要shotを変える。

現在のD_box≤4 guardは係数の大きさを制限する。生bit幅2より大きな保証幅や保守的calは、母分散改善の利益を相殺することがある。逆に層ごとのcalの追加費用やunionの保守性がB1-OLに不利となる可能性もある。いずれも**実装された十分条件の費用差**であり、推定器の最適必要shot数の比較ではない。

既知でexactなνならcentering biasは0。将来νを推定するなら、λ̂^T(ν̂−ν)による誤差とその取得費が増え、独立性・bias/varianceの再整理が必要。現在のexact νを無料の新しい推定νに置き換えない。

### 学習からやり直す期待値とSTOP

[N3期待費用計画§6–8][expected]を再利用する。Hをtrain/probe/calと配分freezeまでの履歴、F_m(H)をそれまでの全実現RZとすると、fresh評価が全quotaを完了する場合、

$$
E[C_A\mid H]=F_A(H)+n_A(H)c_Q,
$$

$$
E[C_{\rm OL}\mid H]=F_{\rm OL}(H)+\sum_h n_h(H)c_h.
$$

これをHについて平均して初めて学習からやり直す計画期待費用になる。nとpilot価格の経験的平均を独立とみなして掛け合わせない。c_hを分離できるのは未来のlawと平均価格がHから独立で固定だからである。

guard/cap/word制限/価格欠測等でSTOPする実policyについては、計画費用と1attempt実費期待値を区別する。完走者だけの平均費用、安い早期中断、同じseedの一回値は、同精度・同返却率の改善の代用ではない。UNKNOWNを0としない。A1 train失敗でB1-OLも未実行となるcampaign順序を、単独B1-OLの安さ・勝利と読まない。

## 6. 必要条件・十分条件・可能性の整理

| 命題 | 分類・適用条件 |
|---|---|
| 現在のA1が追加費用なしの連続最適B1-OLに厳密勝利する | 理論文書式(10)(11)の仮定下では不可能 |
| 同Nの有限quotaに勝つにはΠ_N>G_N | (A1)の仮定下で必要十分 |
| 同期待評価RZで勝つにはΠ_c>G_c | (A2)の仮定下で必要十分 |
| 同分散vの総期待費用で勝つにはv(F_OL−F_A)+Π_c>G_c | (A4)の連続・凍結・完了モデルで必要十分 |
| Oracle層別にもstartup込みで勝つ | F_OL>F_Aが必要、さらにv(F_OL−F_A)>G_cなら十分。同じモデルに限る |
| floor、整数化、高価格希少層、probeノイズ、保守的calがある | A1が有限実装に勝つ理由になり得る可能性。単独では十分でない |
| R_*が小さくλ̂が安定しguardを通る | gapを小さくする可能性。B1-OL超越の証明ではない |
| 層内で変わる新controlに非零の条件付き共分散がある | 理想固定係数の同quota削減には十分。学習・追加費用込み優位には不十分 |
| 1pilotでA1のscoreが小さい | 当該有限pilotの診断。母分散・期待費用・一般優位の十分条件ではない |

この一覧はどちらかを一般的勝者として採択するものではない。現在のA1の優位性を期待するとすれば、まずoracleとの構造的gapを上回る**有限実装上のpenaltyまたは追加費用差**を説明しなければならない。現時点でその不等式の実H4での成立は未確定である。

[pilot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5c27684d904aed9f4f9f2787752c917bb46d684c/docs/research/o2_a1_n3_scope_redesign_20261106/N3_PILOT_CONTRACT_DRAFT.md
[expected]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a53c1d6eedea0d85fb484425db9012aa89738129/docs/research/o2_a1_n3_decision_expected_rz_20261106/EXPECTED_RZ_COST_GUARANTEE_PLAN.md
