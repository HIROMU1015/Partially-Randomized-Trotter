# 共通frame：二次moment・強いgrouping・全費用

2026-11-08。条件と記号は[演算子接続](COMMON_BASIS_OPERATOR_CONNECTION.md)。全て紙上の条件付き比較であり、実H4の新群・Born分布・variance・価格・break-even値は算出していない。

## 1. 同じtargetを扱う三方式

| 方式 | 操作と費用上の性質 | 保持すべき条件 |
|---|---|---|
| 通常selected PR | support限定G_l†→controlled D_l→G_l、D half blocks・wrapper・phase。basisは非制御 | diagonal_only・境界reuse/fusionを維持。全Gaussianを制御する弱い対照にしない |
| 共通frame label周辺化 | branchのA†/B、共通full V_f、system readout、d_f(z)依存ancilla軸とweight | selected→full演算子差、新joint測定/軸の誤差、群質量・全support |
| 既知direct/grouping | 上と同じframe/branch/平均observableを認める。A=B†ならsystem-onlyも検討 | 同じ元入力・情報・最適化・取得費を両側に認める |

同じq_l、共通frame、branch準備とreadoutを既知groupingへ与えると、**操作・outcome確率・推定量・費用が共通frame候補と一致する。** 抽象的なPOVM包含だけでなく、同じ具体的なbranch列と対角関数が再現される。独自の増分利益は残らない。

[Huggins et al.](https://arxiv.org/pdf/1907.13117) §IIのbasis rotation groupingは、共通軌道回転後のoccupation・occupation積を同じreadoutから得る既知原理である。ただし同論文のenergy target、postselection/noise性能を今回の複素full PRへ移植しない。[Mitarai–Fujii](https://arxiv.org/pdf/1901.00015) Results 1–3は適用条件を持つ間接→直接測定の既知原理。一般のA/Bで同論文又はHugginsを引用するだけでancilla-free化できるとは言わず、上の明示branch接続と必要な有料操作を戻す。

出版社のHuggins本文は取得時に認証redirectで読めなかったため、同題author preprintの本文を確認した。[2021年掲載先/DOI](https://www.nature.com/articles/s41534-020-00341-7)。網羅的新規性調査・重要科学レビューではない。

## 2. M2・variance・range

理想共通frameならp_f(z)はlabelに依存せず、m(z)=|d_f(z)|≤1。両軸で

\[
E[Y_a]=\mu_{f,a},\quad M_{2,f}=\sum_zp_f(z)|d_f(z)|^2,\quad
V_{f,a}=M_{2,f}-\mu_{f,a}^2 .
\]

通常label mixtureの±1 bitはM2=1。全labelで|e^{iχ_l}D_l(z)|=1なら

\[
1-M_{2,f}=\frac12\sum_{l,l'}q_lq_{l'}\sum_zp_f(z)
 |e^{i\chi_l}D_l(z)-e^{i\chi_{l'}}D_{l'}(z)|^2 .
\]

同じconditional mean・取得数でのvariance差はこのM2差に等しい。共通φ/χではr(z)=Σq_l σ_l P_{S_l}(z)として

\[
M_{2,f}=\cos^2\phi+\sin^2\phi\,E_{p_f}[r^2],\quad
\Delta_f=\sin^2\phi(1-E_{p_f}[r^2]).
\]

理想support多様性があっても、p_fが乗るzで関数が同じなら利益0。小角度ではsin²φが利益を抑える。未知p_f・ground振幅からΔやvarianceを無料で計算して群/配分を選ばない。今回実H4のその値は全てUNKNOWN。

出力は[-1,1]、より狭いrangeはdのuniform boundが別途あれば使える。新しい数値実装で|d|>1等が出た場合にclippingして保証したことにしない。A=B†なら強いsystem-only出力(Re d,Im d)が可能で、そのM2は各座標の二乗平均。branch型より同等以下になり得るが、保存H4でA=B†を確認したわけではない。このsystem-only条件なら同じzからRe/Im dの両方を計算でき、そのreadout共有も強いbaselineへ認める。一般A/Bのbranch型ではX/Y軸の別取得費が残る。

## 3. 全wrapperの交換関係

selected PRはsupportに必要な列だけを保持するので、個別basis変換が短い可能性がある。候補はfull basisが重くなる一方、label別中央rotationとlabel取得を対角関数の計算へ移し、一つのsystem readoutから群内関数を扱う。この交換をfull costで比べる。

branch準備は diag(A†,B)を実装する。例えば

\[
\operatorname{diag}(A^\dagger,B)
=(I\otimes A^\dagger)\operatorname{diag}(I,AB)
\]

なので、素朴にA/B全体を二重に制御するだけの価格を必然としない。既存diagonal_only・basis reuse・fusionを候補と対照の両方に適用する。ただしこの書換えで常に安くなるとも、compiled価格が加法的とも仮定しない。

固定f・軸aの候補費は、説明用に

\[
\bar C_{\mathrm{grp},f,a}
=E_{z}\,[C_{\mathrm{branch}\ A^\dagger/B,V_f,
system\ readout,axis(z),ancilla\ readout,feedforward}]
\]

と置く。全括弧を一つのwrapperとして扱い、元selected価格をbasis gate数だけで差し引かない。system readout、ancilla制御、条件付き軸、追加qubit/latency、境界最適化損失を落とさない。動的compiler/profile/hardware対応は未確認。

q_l・群質量取得、original/selected bindingと誤差証明、d(z)のO(|f|)対角符号和・trig/angle、記録/cache等の古典費を別に計上する。現M prototypeの2^n分岐列挙を大きな入力で効率的とみなさない。今回group集計・全z評価・新数値certificateは0。

量子のsetup/calibrationも含め、軸別 G_J=S_J+Σ_f N_{J,f,a} Cbar_{J,f,a} を同じ資源座標で比較する。T/CX/RZ/depthは別々、古典時間・取得費も別記。**状態準備費だけ**を既存scopeに従って除外し、fresh状態の消費数は隠さない。制御・basis・readout・校正は除外対象ではない。

## 4. 改善が成立し得る具体的条件

固定f・同じconditional取得数なら、selected label mixtureの V_mix=1−μ² とV_grp=M2−μ²を比較する。正のV_grpの場合、漸近費用×varianceで必要な条件は

\[
\frac{\bar C_{\rm grp}}{\bar C_{\rm mix}}
 <\frac{1-\mu^2}{M_2-\mu^2}.
\]

setup/校正・basis差biasで有効精度が変わるならそのまま十分条件ではない。V_grp=0を有限sample variance0から認定せず、比の発散を利益と主張しない。

さらに強いlabel別stratificationでは、同予算・連続配分の理想係数を

\[
K_{\rm sel}=\sum_{f,l}w_f q_l\sqrt{V_{{\rm sel},fl}\bar C_{{\rm sel},fl}},
\quad
K_{\rm grp}=\sum_f w_f\sqrt{V_{{\rm grp},f}\bar C_{{\rm grp},f}}
\]

と置き、固定量子予算Bに対する最小varianceはK_J²/B。取得・学習費を含めた適用可能性、独立性、全positive層の保持が必要。未知varianceと価格を知るoracle配分は実行アルゴリズムではない。M2が下がるだけでこの強いselected baselineに勝つとは言えない。

対照groupingが同じq/frame/readout/取得を使うなら C_grp=C_known、V_grp=V_known、setupも同じ。**同じ最適化を認めた既知groupingへの増分は0**。通常selected PRを上回る可能性は既知groupingの適用価値としては残るが、今回独自のアルゴリズム改善ではない。実H4の数値的な優劣・break-evenは未取得。

## 5. 有限confidenceと新誤差

独立な各群/軸の固定N_f、range幅D_f、variance上界v_fに対し、weighted estimatorの V_sum=Σ_f w_f²v_f/N_f、R_max=max_f(w_f D_f/N_f)。Bernstein型の条件

\[
2\exp[-\epsilon_{\rm stat}^2/(2V_{\rm sum}+
 2R_{\max}\epsilon_{\rm stat}/3)]\le\alpha_a
\]

へ接続するには、正当なv_fとrange enclosure、独立校正・凍結、全校正費が要る。Hoeffdingは Σ_f w_f²D_f²/N_f のみに依存し、D_f=2のままならM2低下だけで十分Nは減らない。X/Y、校正失敗等のα配分は別途固定し、今回数値を採択しない。

元F_selをtargetとするなら ε_stat≤ε_a−b_frame,a−b_new_engine,a−b_other,a が正であることが必要。frame差と新測定engineの誤差を重複・欠落なく会計し、旧engine上界を代入しない。新しいallowance/ε/α/seed/Nを今回発行しない。

## 6. 判定・次の最小事項

**総合判定C：保存finite targetへの必要な数値保証は未成立。** 理想接続が成立する条件下のbaseline比較はB相当で、既知groupingへ還元される。Aに相当する「強いgroupingにもない利益」は特定していない。一般的な測定法の無効性やO2終了ではない。

この独自改善を理由とする新instrument・H4群探索・価格benchmarkは推薦しない。既知groupingを実用対照として別途評価する方針なら、最初に必要なのは、元selected角の理想列と共通full角の理想列の**一様差証明の方法**、新readout誤差、同じ最適化を施したfull費用取得の有限仕様。追加quantum simulationでなく、これらの取得仕様を先に明示する。

実科学入力が必要になった場合、固定N1/context・有限support集合、対象関数/回数、保存target差とallowance、baseline/profile、STOPを明確にした別途許可が要る。今回その入力集合を保存catalogから選ばず、科学SOURCEや実行契約を発行しない。
