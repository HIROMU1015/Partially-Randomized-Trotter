# SCVの有限confidence接続

2026-11-07 JST。**静的SOURCE・既知理論の適用整理。新科学結果、保証付き実行、正式採択ではない。**
調査版・対象・保全境界は [REPORT](REPORT.md)、実装対応は [再利用表](SCV_REUSE_AND_VALIDATION_MAP.md)。
224-shotの結果・途中データは取得していない。

## 1. Targetと保証の単位

対象は保存linear H4、1.00 Å、STO-3G、legacy DF rank12、8 system qubits、全4電子number sector、
$L_D=3$、$m=0,q=1,\delta_{\rm time}=0.02,r=1,K=2$、coefficient_atol=0、X/Y。
[N1固定報告][n1]のactual finite-grid law $Q$とselected full logical列、
[N2固定報告][n2]の保存state、後続の[engine認証][engine]を区別する。
理論基準は元の**保存DF Hamiltonian**の厳密ground。N2 stateはその数値近似であり、
元の未近似分子Hamiltonianや実機noiseへの保証は含まない。
価格はQiskit1.3.0 / basis=(rz,sx,x,cx) / opt2 / seed17 / full no-prep。

一軸の$B=1-2\,{\tt bit}\in\{-1,+1\}$、$p_h=Q(h)>0$、$\mu_h=E[B\mid h]$、
$\mu=\sum_{h=0}^4p_h\mu_h$を使う。scalar/identity/event phaseは回路に既に含まれ、bitへ二重適用しない。
理想uniform53-bit wordモデルでの独立・復元抽出を仮定する。異なるPCG64 seed/domainの記録だけで
数学的独立性を証明したとはしない。

区別する主張は次の三つである。

| 主張 | 必要な根拠 | 今回の状態 |
|---|---|---|
| 同じ有限signalの不偏推定・分散式 | 条件付き中心化、fresh評価、固定quota | 既存理論とSOURCEを確認 |
| 所定$\epsilon,\alpha$の十分shot数 | 正しいrange、独立calの分散上界、全失敗予算、bias headroom | 接続設計可能。SCVの実値・実接続は未検証 |
| energy/RPEや期待総費用の優位 | 全round信号・decoder・価格・返却を含む追加条件 | 未確立 |

## 2. 不偏性と分散削減

既存[SCV設計 §1–5][design]をそのまま出発点とする。$T$を学習履歴、
$u_h$をpath metadata control、$\nu_h=E_Q[u_h\mid h]$とし、
$\widehat\lambda_h(T)$を評価前に凍結する。

$$
Z_{h,i}=B_{h,i}-\widehat\lambda_h^\top(u_{h,i}-\nu_h),\qquad
\widehat\mu_{\rm SCV}=\sum_h p_h\overline Z_h .
$$

評価が$T$から独立な$Q_h$に従い全quotaを完了するなら、
$E[Z_{h,i}\mid T,h]=\mu_h$、$E[\widehat\mu_{\rm SCV}\mid T]=\mu$。
係数推定が不正確でもこの平均保存は成立する。層別$\lambda_h$へglobal平均を組み合わせると、
一般にはこの不偏性を失う。同じ評価bitで係数・群・quotaを再選択する場合にもこの議論は使わない。

$$
\tau_h^2(\lambda)=\sigma_h^2-2\lambda^\top\gamma_h+\lambda^\top\Sigma_h\lambda,\quad
V_{\rm SCV}(T)=\sum_h p_h^2\tau_h^2(\widehat\lambda_h)/n_h .
$$

$\sigma_h^2=\operatorname{Var}(B\mid h)$、$\gamma_h=\operatorname{Cov}(u_h,B\mid h)$、
$\Sigma_h=\operatorname{Cov}(u_h\mid h)$。
oracle $\lambda_h^*=\Sigma_h^+\gamma_h$に対して

$$
\tau_{h,*}^2=\sigma_h^2-\gamma_h^\top\Sigma_h^+\gamma_h,\qquad
\tau_h^2(\widehat\lambda_h)=\tau_{h,*}^2+
(\widehat\lambda_h-\lambda_h^*)^\top\Sigma_h(\widehat\lambda_h-\lambda_h^*).
$$

これは既知のCV射影式の条件付きlawへの適用であり、非自明な新定理を採択するものではない。
一次文献は[Glynn–Szechtman §2][glynn]。pseudo-inverseは理論記号であり、rank失敗を救済する実装ではない。
有限trainの二次損失がoracle利益を上回れば、rawより分散が大きくなる。
履歴に依存するquotaでは平均する量は
$E_T[\sum_h p_h^2\tau_h^2(\widehat\lambda_h(T))/n_h(T)]$であり、
平均分散を平均shot数で割らない。

最初の候補はh=1だけscalar indicator、他h=0,2,3,4は取得前から$\lambda_h=0$。
同quotaでの追加削減はh=1の項だけに現れる。旧3-controlは各層内で定数であり、
条件付き中心化後は0となる。[A1対B1-OL理論][a1theory]の層間利益をSCVへ二重計上しない。

## 3. Bornノイズの残留

解析記号$m(\omega)=E[B\mid\omega]$について、

$$
\sigma_h^2=E[1-m(\omega)^2\mid h]+\operatorname{Var}(m(\omega)\mid h),
$$

$$
\tau_h^2(\lambda)=E[1-m(\omega)^2\mid h]+
\operatorname{Var}\{m(\omega)-\lambda^\top(u_h-\nu_h)\mid h\}.
$$

metadata CVが減らせるのはpath平均変動の説明可能成分であり、固定pathのBornノイズは残る。
h=0はpath固定のためmetadataによる追加利益がない。
回路価格との相関だけではbitとの相関を示さない。
$m$、groundベクトル、Born確率、$\gamma_h$、真のvarianceは推定器へ渡す無料oracleではない。
今回これらの実値を計算していない。

## 4. Residual rangeは母分散と別の入力

[FrozenCV.output_range][cv]は、実際のbinary64 slopeをFractionへ凍結し、indicatorのboxを囲う。
$u\in[0,1]$なら

$$
\ell=-1+\min\{\lambda\nu,-\lambda(1-\nu)\},\quad
r=1+\max\{\lambda\nu,-\lambda(1-\nu)\},\quad D_{\rm box}=r-\ell=2+|\lambda|.
$$

実supportの幅はこのbox幅以下。Bernsteinで必要な$|Z-EZ|$の上界にも$D_{\rm box}$を使える。
$D$はrangeの**全幅**であり、半幅や$|Z|$上界へ読み替えない。
[fit_cells][scalar]はrank2と$D_1\leq4$を検査する。他4層の幅は2。
224-shot準備版にはguard4が実装されているが、同commitの科学実行草案は未採択と記録されている。
本調査は別セッションの採択状況を確認・変更していない。

general vectorのboxは$D_h\leq2+\sum_j|\lambda_{hj}|(b_{hj}-a_{hj})$。
旧indicator APIへ幅2のsigned featureをそのまま渡すことはできない。
観測標本の最大・最小からsupport全体のrangeを縮めない。
rank/range失敗後のclip、refit、追加shot、別controlへのfallbackは固定契約で禁止されたままである。

## 5. 既存calibrateの実式

固定560c6092版の[a1_n3_connection.calibrate][n3]は、独立・固定件数$k_h\geq2$のcal値と
既知区間$[\ell_h,r_h]$、$\delta_h^{\rm cal}$を受け取る。
不偏sample variance $s_h^2$から

$$
U_h=\min\left\{\frac{D_h^2}{4},
\left(\sqrt{s_h^2}
+D_h\sqrt{\frac{2\log(1/\delta_h^{\rm cal})}{k_h-1}}\right)^2\right\}
$$

を上側に丸めて返す。保証は、凍結係数に条件付けて
$P(\tau_h^2>U_h\mid T)\leq\delta_h^{\rm cal}$。
根拠は[Maurer–Pontil **v1 Theorem 10 (3)**][mp]を区間正規化して適用したもの。
同論文の同一sampleで直接平均を囲うempirical Bernstein式と混同しない。
variance cap $D_h^2/4$もbounded-variableの決定論的上界である。

SOURCEではFractionのsample variance、100桁mpmath.iv、外向きdyadic端点を使用する。
sample variance=0でも有限$k_h$の$U_h$は正となる。
幅4のcalは幅2より補正項・capが大きく、少数calでは実分散の差が上界へ現れない可能性がある。

関数自体は数列・区間だけを受け取るので、stream独立性、正しいcell、frozen係数、固定cal件数は
呼出し側で検査する必要がある。[evaluation_inputs][cv]は旧単一CV用のstage/stream検査を持つが、
SCV全5層のplanを構築する機能ではない。

## 6. 既存sufficient_shotsの実式

[sufficient_shots][n3]は正のheadroom
$e=\epsilon-b_{\rm engine}-b_{\rm center}$に対し、二側失敗確率$\delta^{\rm prod}$で

$$
N_B=\max\left\{1,\left\lceil
\left(\frac{2U}{e^2}+\frac{2D}{3e}\right)\log\frac{2}{\delta^{\rm prod}}
\right\rceil_{\rm outward}\right\},
$$

$$
N_H=\max\left\{1,\left\lceil\frac{D^2}{2e^2}\log\frac{2}{\delta^{\rm prod}}
\right\rceil_{\rm outward}\right\},\qquad N=\min(N_B,N_H)
$$

を返す。これは最小必要数ではなく**十分数**。
Bernsteinの係数は、$|Z-EZ|\leq D$に対する二側tail
$2\exp[-ne^2/(2\tau^2+2De/3)]$に対応する。
原式は[Vershynin, First Edition Theorem 2.8.4][bernstein]で確認した。
Hoeffding経路はvarianceを使わない。cal後にどちらを選んでも、同じ有効上界とfresh評価を使う限り、
「二式を比較した」ことだけを理由とする追加の失敗確率倍付けは不要である。
calibrationを省略して既知rangeだけでHoeffdingを使う別の実行案も可能だが、
現SCV driverの自動fallbackとして採択したものではない。

層別には各層の統計誤差$e_{b,h}>0$を与え、

$$
\sum_h p_h e_{b,h}+b_{{\rm engine},b}+b_{{\rm center},b}
+b_{{\rm arithmetic},b}\leq\epsilon_b
$$

とする。globalなengine allowanceを各層で何度も差し引かない。
既存choose_allocationはglobal headroomを先に取り、各層のsufficient_shotsへengine_error=0、
centering_error=0で渡す。数値合成誤差がexact Fractionで0ならその理由を記録する。
finite-target外へのbiasを追加する場合は、別項として予算化し、既存の有限target保証へ紛れ込ませない。

## 7. 失敗確率・凍結順序・返却

将来の最小順序は次のとおり。今回実行しない。

1. LAW/catalog/controlと$\nu_h$・range規約を固定する。
2. fresh trainでh=1のX/Y係数を一回ずつfitし、rank/rangeと実係数を凍結する。
3. 固定件数のfresh calを全5層・X/Yで取得する。h=1はFrozenCV.apply(cal)、他層はraw。
4. 全cellの$U_{b,h}$を作り、許容誤差配分・整数quotaをcal/probeまでの履歴で選び凍結する。
5. cal/trainから独立なfresh productionを全quota取得し、層別質量で合成する。

calとproductionの独立性は、calからquotaを決めてよいことと両立する。
係数をcalで選び直してから、同じcalを「固定係数のcal」と扱うことはできない。
誤差配分gridからの選択は、先に得た全cell同時variance上界の上で行う。
productionを見てquotaを変えたり、calを有望な時点まで増やしたりしない。

方式familyを$\mathcal A$として、信号精度の同時claimには

$$
\sum_{a\in\mathcal A}\sum_{b=X,Y}\sum_{h=0}^4
(\delta^{\rm cal}_{a,b,h}+\delta^{\rm prod}_{a,b,h})\leq\alpha_{\rm signal}
$$

を明記する。一方式の精度保証と、二方式を同時に比較するfamilyは区別する。
既存confidence_cells(alpha,H)は「4方式×2軸×cal/prod」を前提に各$\alpha/(16H)$。
2方式×2軸×5層へ同じalphaを渡すだけなら予算を余らせる保守的配分にはなるが、
SCVの全round・返却失敗を含む新配分にはなっていない。明示familyへの薄い接続が必要。

精度保証は全guard通過履歴に対して同じ形で成立するため、OLSの「oracleに近い」ことを
保証する$\delta_{\rm train}$は精度証明の必須入力ではない。
係数品質・費用改善の確率保証を別に主張するなら、その学習失敗率も必要となる。
rank/range不合格は不返却の確率へ含め、calibration失敗予算で代用しない。

finite word cap、worker/価格失敗、途中STOPで結果を選別すると完走条件付きcoverageは自動ではない。
有効な仮想fixed-n列との対応の下で主張できる基本形は
$P(\text{returnかつ精度違反})\leq\alpha_{\rm signal}$。
所定の返却成功率には、別途
$P(\text{abortまたは精度違反})\leq\eta_{\rm abort}+\alpha_{\rm signal}$
の管理が必要。未知の$\eta_{\rm abort}$を0にしない。retryで失敗履歴を消さない。

## 8. 分散削減がshot削減になる条件

同じ$e,\delta$について、丸め前のBernstein項同士だけを比較すると

$$
U_{\rm OL}-U_{\rm SCV}>\frac{e}{3}(D_{\rm SCV}-D_{\rm OL})
$$

のときSCV側の値が小さい。これは既存shot式の代数的比較で、科学的有望性閾値ではない。
真の$\sigma^2-\tau^2$を左辺のcal上界差へ代入してはならない。
実際の判定には、両方式の$\min(N_B,N_H)$、外向きceil、層別誤差配分、整数floor、
cal費用・train費用まで用いる。同quotaでの分散低下だけではshot数削減を証明できない。
rawの幅は2、active residualのbox幅は$2+|\lambda|$なので、Hoeffding単独には
variance低下を利用する経路がなく、range拡大が不利になる。

## 9. Biasと未取得情報

| 項目 | 有効な扱い | 現在の不足 |
|---|---|---|
| control平均 | actual $Q_h$のexact Fraction。近似なら$\sum_h p_h|\lambda_h|\,|\widetilde\nu_h-\nu_h|$を加算 | 実H4のSCV平均・選択出力は今回未取得 |
| engine/state/bit-grid | 同じ全support・同じ回路の一様座標allowanceを一回加算 | N2単独では総engine誤差を閉じない。後続認証へのbindingが必要 |
| finite signalと理想signal | PF、有限RTE、CDF/law、selected列などの差を別予算化 | 固定finite targetの不偏性からenergy biasが0とはいえない |
| 係数学習 | frozen actual係数のresidualをcalする | 224-shot sample varianceだけでは$U_h$未取得 |
| 十分N | fresh calと既知range・headroom・明示family | SCV用cal/plan adapterは未実装、数値$\epsilon/\alpha/N$未採択 |
| 返却 | 全guard/STOP/未価格tailを保存 | 返却率・途中STOP補正の保証は未取得 |

独立calで残差を測るため、oracle係数誤差を別に推定してvarianceへ二重加算する必要はない。
ただしengine上のcalで得たvarianceはengine bit lawのものである。
signal平均は一様engine allowanceで理想基準へ移せても、ceil/guardを含む**費用policyの分布**
まで同じになるわけではない。[総費用文書](SCV_TOTAL_COST_COMPARISON.md)で別に扱う。

224-shot探索で候補・係数を選んだ場合、将来それを固定履歴として扱い、
独立calとfresh productionで将来の精度保証を構成することは検討できる。
探索に使ったsample varianceを遡及的にcalibrationと呼んではならない。
探索済み係数を使う単発deploymentと、trainから再学習するpolicyは別の費用対象である。

[n1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/744e1a09661ebba6841fe5faf456f88c41c47780/docs/research/o2_a1_new_target_n1_20261106/REPORT.md
[n2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5e9bd08068738cdf1070bdc43bee8194b113ae33/docs/research/o2_a1_new_target_n2_20261106/REPORT.md
[engine]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdf00cee3c87d804845497afc89ac6a71f3b3f59/docs/research/o2_a1_engine_qualification_20261106/REPORT.md
[design]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/84e021b6a8decde7fa3ed3fa44d7148cc5428061/docs/research/intrastratum_cv_design_20261107/STRATIFIED_CV_DESIGN.md
[a1theory]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8d8a290fa4d4cf24f968aea27953ad5422e418b3/docs/research/a1_b1ol_theory_20261107/A1_VS_B1OL_THEORY.md
[cv]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_control_variates.py
[scalar]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/scv_scalar.py
[n3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_n3_connection.py
[glynn]: https://web.stanford.edu/~glynn/papers/2002/GSzechtman02.pdf
[mp]: https://arxiv.org/pdf/0907.3740v1
[bernstein]: https://webapps.math.uci.edu/~rvershyn/papers/HDP-book/HDP-1.pdf
