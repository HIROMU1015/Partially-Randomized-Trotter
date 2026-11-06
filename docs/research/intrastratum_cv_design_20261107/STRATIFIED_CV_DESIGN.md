# 層別＋層内CVの設計

2026-11-07 JST。既知のCV・条件付き分散の枠組みを、固定N1 lawへ記号的に適用する設計。実装・新数値評価・詳細科学レビュー・正式採択なし。
対象・law・SOURCEは[候補調査](INTRASTRATUM_CONTROL_CANDIDATES.md)と同じ。

## 1. 推定器と不偏性

一軸のbit出力B∈{−1,+1}、Qの層質量p_h、μ_h=E[B|h]、μ=Σp_h μ_hとする。
Tは評価前の独立な学習/必要ならprobe履歴。u_h(ω)はpath metadataだけの有界vector、ν_h=E_Q[u_h|h]はexactな既知平均。
λ_h(T)、正整数n_h(T)を評価前に凍結する。異なる層のfresh復元抽出はTと独立とする。

$$
Z_{h,i}=B_{h,i}-\lambda_h^\top(u_h(\omega_{h,i})-\nu_h),\qquad
\widehat\mu_{\rm SCV}
=\sum_{h=0}^4 p_h\,\overline Z_h.
$$

$$
E[Z_{h,i}\mid T,h]=\mu_h,\quad
E[\widehat\mu_{\rm SCV}\mid T]=\mu.
$$

ゆえに固定quotaを完了するモデルでは、係数が有限trainからの不正確な値でも不偏。異なるhでλが違ってよいが、中心は必ずν_hを使う。
global νをλ_hごとに引くと一般にΣp_h λ_hᵀ(E[u|h]−ν)≠0であり、targetが変わる。

同じevalデータで係数・候補・quotaを選び直す方法にはこの条件付き証明を使えない。cross-fitting、path-dependent sampling、optional stoppingは別設計として扱い、今回の案に含めない。

## 2. B1-OLとの差とoracle係数

$$
\sigma_h^2=\operatorname{Var}(B\mid h),\quad
\Sigma_h=\operatorname{Cov}(u_h\mid h),\quad
\gamma_h=\operatorname{Cov}(u_h,B\mid h),
$$

$$
\tau_h^2(\lambda_h)
=\sigma_h^2-2\lambda_h^\top\gamma_h+\lambda_h^\top\Sigma_h\lambda_h.
$$

独立層の分散は

$$
V_{\rm SCV}(T)=\sum_h\frac{p_h^2}{n_h(T)}\tau_h^2(\lambda_h(T)),\quad
V_{\rm OL}(T)=\sum_h\frac{p_h^2}{n_h(T)}\sigma_h^2,
$$

$$
V_{\rm OL}(T)-V_{\rm SCV}(T)
=\sum_h\frac{p_h^2}{n_h(T)}
\{2\lambda_h^\top\gamma_h-\lambda_h^\top\Sigma_h\lambda_h\}.
$$

有限の任意のλでは差が負にもなる。Σ_hが正則ならλ_h*=Σ_h^{-1}γ_h。特異な場合の理論式はΣ_h^+γ_hで、

$$
\Delta_h=\gamma_h^\top\Sigma_h^+\gamma_h,\quad
\tau_{h,*}^2=\sigma_h^2-\Delta_h,\quad
\tau_h^2(\widehat\lambda_h)
=\tau_{h,*}^2+
(\widehat\lambda_h-\lambda_h^*)^\top\Sigma_h
(\widehat\lambda_h-\lambda_h^*).
$$

共分散行列の性質からγ_hはΣ_hのrange内にあり、この射影式が成り立つ。pseudo-inverseは理論記述であり、実装のrank失敗救済を採択したものではない。
scalar indicatorならΣ_h=ν_h(1−ν_h)。その値はlabel lawで求まるが、γ_hはfresh bitを使って学ぶ未知量。真の信号を与えない。

同じquotaでoracleの追加削減はΣp_h²Δ_h/n_h。全ての活動層でΔ_h=0なら追加利益なし。
Tを反復するpolicyの分散はE_T[Σp_h²τ_h²(λ̂_h(T))/n_h(T)]であり、平均分散を平均Nで割った値ではない。係数誤差と学習法のbiasは非負の二次損失として残る。

一般の多重CVの最適係数とL²射影は[Glynn–Szechtman (2002), §2](https://web.stanford.edu/~glynn/papers/2002/GSzechtman02.pdf)の既知枠組み。上式は各条件付きlawへの適用と展開で、128-shotでの漸近最適性を主張しない。

## 3. 現在の3-controlが効かない理由

現行gは各hでg_h一定。条件付き中心では g−E[g|h]=0 が各標本で成立し、どんなλ_hでも補正0。
同じglobal λとglobal νを全層へ加える表現でも、

$$
\sum_h p_h\lambda^\top(g_h-\nu)=0
$$

により合成後に相殺する。これが[理論8d8a290 §3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8d8a290fa4d4cf24f968aea27953ad5422e418b3/docs/research/a1_b1ol_theory_20261107/A1_VS_B1OL_THEORY.md)の結果。
現在のindicatorを非線形変換しても層内定数である。新候補はこの定数subspaceの外にある自由label情報を使う。

## 4. 減らせる成分とBornノイズ

m(ω)=E[B|ω]を解析用の記号とする。controlへ渡す数値ではない。

$$
\sigma_h^2
=E[1-m(\omega)^2\mid h]+\operatorname{Var}(m(\omega)\mid h),
$$

$$
\tau_h^2(\lambda_h)
=E[1-m(\omega)^2\mid h]
+\operatorname{Var}\{m(\omega)-\lambda_h^\top(u_h-\nu_h)\mid h\}.
$$

Cov(u,B|h)=Cov(u,m|h)。削減できるのはpathに依存する平均変動のうちuの線形spanで説明できる部分。
Δ_h≤Var(m|h)であり、pathを固定したBornノイズ E[1−m²|h]は残る。
h=0はpath固定なので、後者のpath分散が最初から0。どのmetadata CVでもこの層のBorn分散は減らせない。
同fragmentや同labelが価格を予測できるだけではγ_h≠0にならない。

## 5. 平均・数値・range

保存grid質量と位置制約からν_h、E[uuᵀ|h]を有限和/積和で求める。label/candidate identity、law hash、層条件を中心化表へbindし、別lawのνや理想ρを代入しない。
実平均の新規算出は今回していない。exactとは手法の表現可能性で、値が取得済みという意味ではない。

u_j∈[a_hj,b_hj]なら、

$$
D_h\leq 2+\sum_j|\lambda_{hj}|(b_{hj}-a_{hj}).
$$

indicatorでは D_h≤2+Σ|λ_hj|、signed-supportでは幅2を使う。
既存FrozenCV.output_rangeは[0,1] indicator用なので、一般の[-1,1]特徴をそのまま渡してはならない。
OLSのbinary64係数をexact rationalとして凍結し、actual係数でrange guardを検査する既存方針は再利用できる。

ν̃_hに誤差e_hjがある場合は

$$
b_{\rm center}\leq\sum_h p_h\sum_j|\lambda_{hj}|e_{hj}.
$$

独立に推定した平均でも、実現した中心化誤差を無料/0としない。今回の候補はexact finite和を第一案とする。
共通測定engineの一様座標誤差b_engineは、exactに中心化したmetadata補正の期待値0によって増幅されず、Σp_h=1でそのまま加算できる。これは固定law/独立train/有界評価の条件下の座標保証である。

## 6. 配分・学習・総費用

固定λと正の期待価格c_h=E[full RZ|h]に対し、同Nの連続oracle配分は n_h∝p_h τ_h。
同期待評価RZ予算Bでは、

$$
n_h\propto p_h\tau_h/\sqrt{c_h},\quad
K_{\rm SCV}=\left(\sum_h p_h\tau_h\sqrt{c_h}\right)^2,\quad
V_{\rm SCV,opt}=K_{\rm SCV}/B.
$$

B1-OLはτ_hをσ_hへ置換。oracle係数ではK_SCV≤K_OLだが、finite train・整数floor・guardで同じ利益が得られるとは限らない。
metadata control自体は回路を短縮しない。c_hは同じで、期待される利益は必要測定数を減らすこと。

評価前量子費F_SCV、F_OL、目標母分散v>0を置く連続完了モデルでは、

$$
C_{\rm SCV}(v)=F_{\rm SCV}+K_{\rm SCV}/v,\quad
C_{\rm OL}(v)=F_{\rm OL}+K_{\rm OL}/v,
$$

$$
C_{\rm SCV}<C_{\rm OL}
\Longleftrightarrow
K_{\rm OL}-K_{\rm SCV}>v(F_{\rm SCV}-F_{\rm OL}).
$$

これはvariance目標でのoracle会計で、ε/α保証の必要費用ではない。finite量の推定誤差・cal・cap/STOPの返却率まで含む保証は別途必要。
現実の費用は学習・probe・cal・評価全てのfull interrogationを課金し、classical setup/生成/fit/中心化/compile/cache/time/RSSを別vectorにする。

希少層は直接条件付き取得でき、1/p_h回の量子棄却を必要としない。しかしtrain一件はfull価格。全層で係数を学ぶ必要はなく、事前にλ_h=0と固定する層を設けてもtargetは変わらない。これは学習失敗後の救済とは区別する。

配分はresidual varianceに依存し、residualは係数精度に依存する。両方を少数データから同時に最適化すると不安定になるため、最初の案は一層・一特徴・固定quotaに限定する。将来最適化する場合もtrain→独立cal/probe→quota freeze→fresh productionの依存順を保つ。

## 7. 有限confidenceへの接続

独立calからV_h^upper、正しいD_hを得た場合、各層の精度e_h・失敗δ_hに対し、既存Bernstein/Hoeffding経路を再利用できる。

$$
n_h\geq
\left(\frac{2V_h^{upper}}{e_h^2}+\frac{2D_h}{3e_h}\right)\log(2/\delta_h)
$$

またはHoeffdingの十分数。Σp_h e_h+b_engine+b_center≤ε、全method/axis/層のcal・production失敗予算和≤αが必要。
sample variance=0をV_h=0にしない。既存calibrateと人工testにはこの点の確認がある。新しいε/α・cal数・総Nを今回は採択しない。

guard/rankでのSTOPがTだけに依存するなら、通過Tの下でもfresh評価の期待値はμ。word cap、実行失敗、価格欠測等で評価途中にSTOPするpolicyの返却条件付き不偏性・coverageは別問題である。失敗を捨てた完走者だけで保証せず、有限wordモデル/返却失敗を契約の成功確率・費用へ明示する。
実現RZを見て都合よくhard-budget停止する方式も今回のfixed-n証明の対象外。

## 8. SOURCE再利用と必要な将来の最小差分

| 既存資産 | 再利用できる部分 | そのまま使えない点 |
|---|---|---|
| a1_n3_connection.ConditionalSampler、PathChoice、validate_batch | actual Q、全5cell、適用順metadata | 新uの中心化表・provenanceをbindするadapter |
| a1_control_variates.ControlBank.joint_mean/pattern_mass | fixed-label積moment・包含排除 | arbitrary fragment群/continuous特徴・条件付きνのbankは未実装 |
| fit_and_freeze/FrozenCV/require_disjoint | intercept OLS、凍結rational、range/独立性の方針 | 元bank.meansはglobal。層別ν_hが必要。旧rank4/3-control固定を流用しない |
| production_mean / pilot_mean | cell-IDを保持する加重合成とquota検査 | modelは現在一つ。層別model_hとλ=0層への最小接続が必要 |
| Domain/Streams/manager・journal・reconcile | stage/axis/cell独立、bit先行、全費用・停止保全 | METHODS/serializerは旧方式固定。新identifierまたはpaired-eval schemaの拡張が必要 |
| SavedPrices/selected full provider | 同policy、Qiskit1.3.0/rz,sx,x,cx/opt2/seed17 | 真の期待価格は既知でない。cache hitも課金 |
| calibrate/sufficient_shots | bounded residual、outward十分数 | confidence_cellsは4方式前提。新campaignの配分をそのまま使えない |
| 既存人工test | conditional law、frozen fit、range、独立性、positive層、価格 | 新conditional-mean/cell-model/paired会計testは将来必要 |

特に旧cv.stratified_meanへ新controlの値を層IDとして渡してはいけない。元5層と新uの値が一致するとは限らず、h=1,2の同一旧patternも区別する必要がある。
新control・推定器・adapterは今回一切実装していない。test SOURCEを読んだが、新しいtest実行もしていない。
