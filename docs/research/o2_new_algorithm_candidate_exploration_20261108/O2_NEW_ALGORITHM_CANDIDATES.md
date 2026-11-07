# O2新候補：label平均を測定へ移す方法と、局所差分の別測定

2026-11-08 JST／SERVER1。出発点は[O2統合成果][SYNTH]（`d28164b5d51e0ca5a0ade74ded8ff176b443dab8`）。**2案を設計候補として残し、最初の技術確認には案Mを推奨する。** これはrepositoryに未接続の操作の提案であり、学術的新規性・性能順位・科学的GOの認定ではない。新規実装、人工量子simulation、科学計算、旧rawの数値再解析は行っていない。

## 1. 保存するtargetと、今回使える情報

対象は一般に、既知のactual finite-grid law Qの下での複素平均

\[
 Z=\mathbb E_Q\langle\psi|U_\omega|\psi\rangle
\]

である。量子channelの平均や、絶対値の二乗へ変更しない。通常のHadamard bitはXで実部、Yで虚部、bit0を+1、bit1を−1へ写す。scalar、identity、event phaseもUへ含める。PF/truncation biasと、このtargetの測定誤差を分ける。

将来の接続先として参照するN1は、保存linear H4／1.00 Å／STO-3G／legacy DF12、L_D=3、m0、q1、δ=0.02、r1、K2、coefficient_atol=0.0。N2の全4電子sector groundとN1 actual Q・selected full policyを使う場合でも、今回の新測定へ旧engine誤差保証をそのまま移せない。N1 payloadを数値評価して適合性を調べる作業は今回実施していない。[N1・N2・旧engineへの固定入口][INDEX]

既知として使うのは保存された整数CDF質量、order/label、係数符号、対角Pauli support、basis/gate列、明示された角度・phaseである。未知のstate vector、E0/gap、Born確率、path信号、相関、母分散、compiled価格を無料のoracleとして使わない。状態|ψ⟩を通常の有料な実験入力として準備できることと、その振幅を古典側が知ることを区別する。

## 2. 案M：共通basis内のlabel平均を、system readoutとancilla測定へ移す

### 2.1 操作と適用条件

ある一つのevent slotについて、残りのpath情報ηとlabel群gを条件に、**実際のselected target**を

\[
 U_{\eta,l}=A_\eta V_g^\dagger R_l V_g B_\eta,\qquad
 Q_l=V_g^\dagger D_lV_g,\quad D_l^2=I
\]

と書ける場合に限定する。D_lは符号付きZ/ZZ（identityも可）の対角演算子、固有値d_l(z)は±1。R_lはD_lそのもの、または同じ角度の回転exp(−iφD_l)。条件付き確率q_l=Q(l|η,g)をactual整数CDFから得る。元の(η,g)周辺lawを維持し、群内のlだけを量子操作から除く。

**A、B、V、回転角φが群内で共通であることが必要。** original fragment/basis IDの一致だけでは足りない。supportごとのselected basis変更や境界簡約でA/Bまで変わる場合、同じ物理演算子への厳密な対応を示すか、共通gate列を持つ部分群へ分ける。近い行列・同じhash種別・同じchannelという理由だけでまとめない。該当する複数label群の存在は実H4ではUNKNOWN。

実験では、元のcontrolled R_lを実行する代わりに

\[
 |\Phi_\eta\rangle=\frac{|0\rangle A_\eta^\dagger|\psi\rangle
                  +|1\rangle B_\eta|\psi\rangle}{\sqrt2}
\]

を準備する。H後、ancilla=0でA†、ancilla=1でBを適用し、systemへVを適用してcomputational basisでzを測る。逆状態準備は要求しない。制御0側のA†、scalar phase、逆gate順序の実装費を省かない。二つの枝の干渉を残し、system測定結果を使ってancillaをreadoutする。

### 2.2 Product slot：同じ一回のsystem readoutで群内labelを平均

\[
 r_g(z)=\sum_{l\in g}q_l d_l(z),\qquad |r_g(z)|\le1.
\]

XまたはY ancilla測定の符号s_aに対し、出力をs_a r_g(z)とする。例えばXでは

\[
 \mathbb E[s_a r_g(z)\mid\eta,g]
 =\operatorname{Re}\langle\psi|A V^\dagger
       (\sum_lq_lD_l)V B|\psi\rangle.
\]

Yも同様に虚部を返す。label別±1測定の条件付き二次momentは1であり、新出力のそれはE[r_g(z)²|η,g]≤1。同じ平均を保つので、同じ(η,g)取得・同shot数では分散を増やさない。これは未知のpath平均を計算するCVではなく、**追加のsystem測定結果で、可換labelの測定情報をまとめる**操作である。

### 2.3 Rotation slot：平均したobservableの固有基底でancillaを読む

R_l=exp(−iφD_l)、c=cosφ、s=sinφとする。Z_η=⟨ψ|AB|ψ⟩、W_ηg=⟨ψ|AV†(Σq_lD_l)VB|ψ⟩と書くと、群の信号はcZ_η−isW_ηgである。system結果zの後に、次のancilla observableを測る。

\[
 O_X(z)=cX+s r_g(z)Y,\qquad O_Y(z)=-s r_g(z)X+cY.
\]

両者の固有値は±m_g(z)、m_g(z)=√(c²+s²r_g(z)²)≤1。既知のzとmetadataからequatorial測定軸を選び、±1 bitにm_g(z)を掛ける。m=0なら出力0とし、その分岐の測定軸は任意に固定できる。X/Yを同時取得したと主張せず、各座標は別の有料attemptで得る。

\[
 \mathbb E[Y_a\mid\eta,g]=\{\operatorname{Re},\operatorname{Im}\}
            (cZ_\eta-isW_{\eta g}),\quad
 \mathbb E[Y_a^2\mid\eta,g]=\mathbb E[m_g(z)^2\mid\eta,g]\le1.
\]

この等式は上の枝状態とobservableからの記号的導出であり、実験結果ではない。個々の回転labelに対応するancilla observableは一般に可換ではないので、全部の元bitに共通の古典的joint lawがあるとは主張しない。平均したHermitian observableを直接測ることで成り立つ。

sourceのRTEはcosφ I−i sinφ Pを使い、通常のR_P(θ)=exp(−iθP/2)とはθ=2φの関係にある。product列を先に、rotationを最後に適用し、event phase (−1)^(n/2)も保持する。既知の共通phase e^{iχ}は上記のX/Y observableをχだけ混合して扱う。label依存phaseを共通とみなさず、群分割か別の導出を要する。

### 2.4 改善の意味と、相殺する費用

群内でd_l(z)が割れ、|r_g(z)|<1の測定結果に正確率があり、rotationならs≠0であるとき、上の二次moment差は厳密に正になる。rotationの差はsin²φ E[1−r_g(z)²|η,g]なので、小角度では統計的利益が二次に小さくなる。一方、|r|=1のsupportしかなければ統計的利益はない。**実H4でどちらになるかは計算していない。** sup_z |r_g(z)|=1なら、rangeだけの十分shot数上界は改善しないことがある。厳密なsup評価は対角Z/ZZ多項式の最適化になり得るため、安価と仮定しない。有限confidenceへ進むには、独立calibrationまたは正当な解析的上界の費用も必要。

元のcontrolled product/rotationは除けるが、A†/Bの枝分け、V、systemの全bit readout、rotation版のfeed-forwardとancilla軸回転が加わる。前後のbasis相殺を失う可能性もある。新価格は**full interrogation**で比較し、状態準備のみ同じ規則で除外する。本案はsystemを破壊的に読むため、attemptごとにfreshな入力状態を使う比較を前提とし、状態の再利用を無料で保てるとはしない。mid-circuit readout、latency、追加logical bit・誤差、古典r(z)評価とcacheも別計上する。RZが減ってもhardware時間や総費用が減るとは限らない。

q_lは整数質量の正確な比、r(z)はその有理和として記述できる。一つの群のr評価は素朴にはO(|g|)の符号付き和であり、効率化しても群構築・context照合・cache費を記録する。安い(η,g)周辺生成が得られるかも適用条件であり、一般の全path集計を無料と仮定しない。群選択の整数samplingは既存のconditional取得を適合確認の上再利用する。

上の平均/M2の式はideal gateと理想Born測定での式である。実際のequatorial軸・mの数値生成、system/ancillaの測定確率とbit-grid、出力丸め、全support誤差には新しい保証が要る。actual Qを保存しただけでは測定側の数値biasまで消えない。旧exact Fraction kernelへの無断置換・既存b_engineの流用は不可。旧engineは新しいsystem測定と条件付き軸を認証していない。

### 2.5 既存法との差と選別

既知のdirect/sequential measurementへの置換はMitarai–Fujiiの先行研究に含まれる。[一次論文](https://arxiv.org/abs/1901.00015v2)。本案の提案部分は、PRのactual label lawを共通basisで平均し、そのobservableをfull-wrapper干渉の途中へ接続する設計である。一般の測定置換や可換groupingそのものを新発明と呼ばない。

SCVとの操作差はmetadata特徴量ではなく、従来捨てていたsystem readoutを使うこと。A2との違いは非unitary差分状態のLCU/herald/成功率補正を使わないこと。Closed/CTSとの違いは語の生成・return集約でなく、平均observableの測定に置き換えること。単独labelでの既知測定置換にも利益があり得るので、それを除いた**群平均固有の追加利益**も分けて比較する。

**分類：KNOWN_METHOD_APPLICATION + ACTIONABLE_CANDIDATE（次の限定技術確認用）。** target一致の条件を満たすか確認する価値があり、実科学実行可能・量子資源改善支持済みという意味ではない。

## 3. 案D：同じPauli生成子の局所回転差を、小係数付きの別測定へ分ける

### 3.1 操作と記号的な改善機構

fine U_fは元のtargetを維持する。同じlayout/生成子で一部角度だけが違う補助回路U_cを**既知の外部入力**として与える。coarseだけにtargetを変更しない。O2で新たなnative合成・PF/split探索を行う案でもない。

P²=Iなら、R_P(θ)=exp(−iθP/2)について

\[
 R_P(\theta)-R_P(\varphi)
 =-2i\sin((\theta-\varphi)/4)R_P((\theta+\varphi)/2)P.
\]

U_f=G_M^f⋯G_1^f、U_c=G_M^c⋯G_1^cとして

\[
 U_f-U_c=\sum_jG_M^f\cdots G_{j+1}^f
 (G_j^f-G_j^c)G_{j-1}^c\cdots G_1^c
 =\sum_j\gamma_j V_j.
\]

相違するslotが上の回転ならγ_j=−2i sin((θ_j−φ_j)/4)、V_jはprefix、midpoint回転、P、suffixの明示unitaryである。制御付きgateの差そのものをscaled unitaryとみなすのではなく、**underlying system U**を分解してから各V_jのcontrolled Hadamard測定を作る。XXPlusYYなど一般Gaussian gateへP²=Iの式を無条件に適用しない。

元law Qでpathを取得し、U_cの平均を有料のbase測定で、Σγ_j⟨V_j⟩を独立した補正測定で求めて加える。Jをp_j>0で取得し、γ_jのphaseを測定軸に入れ、出力|γ_J|s_a/p_Jとすれば、補正の平均は正確に一致する。理想的なp_j=|γ_j|/Γ、Γ=Σ|γ_j|なら

\[
 \mathbb E[Y_{\rm corr}^2\mid\omega]=\Gamma(\omega)^2.
\]

局所差分が小さくΓも小さい場合、補正測定のnoiseがその係数で小さくなる。A2の差分state LCUの成功率補正でnoiseが残る問題とは異なる。ただしbase測定の分散・費用は残り、補正だけの利益を全体へ移せない。

### 3.2 Finite-gridと全費用による制約

一般の三角係数に比例するp_jは53-bit一語では正確に表せない。実装時は**実際の**正確な取得確率p_jへ重みを合わせ、二次momentをΣ|γ_j|²/p_j、rangeをmax_j |γ_j|/p_jとして扱う。理想Γ²を実装保証へ流用しない。正のγをgridでゼロにしてはならず、全γ=0の処理も厳密に分類する。三角係数/phaseの誤差、samplingのword数、正規化と全path range評価の追加費もUNKNOWN。

全gate差が小さくてもslot数に応じてΓが累積する。midpoint回転は元回転より高価になり得る。補正V_jはfull prefix/suffixを持つため安い局所回路だけでは測定できない。coarseの有料base、差分sampling、両段階の学習/calibration、full-wrapper価格・失敗attempt・古典費を含める。shot数を数値設定せず、将来は独立なbase/補正の総分散V_c/n_c+V_corr/n_corrと総費用で、同じtargetの直接測定と比較する。

### 3.3 重複と技術的な位置付け

これは既知のtelescoping/multifidelity/quasiprobability分解の適用候補で、単独で新理論とはしない。TE-PAIは角度補間により浅い回路と測定負担を交換する既知の関連法だが、その主分解は**channel/superoperator**であり、ここで必要なcoherent amplitudeの不偏性を直接保証しない。[一次論文と式(8)–(10)](https://arxiv.org/html/2410.16850v2)。counts/orderと局所defectの構造を使う研究も既にある。[一次論文](https://arxiv.org/abs/2606.23544v1)。それらを強い比較対象に含め、新規性は未認定とする。

C-Sのsame-position residualと局所回転差の代数を共有する。本案を残す理由は、**各shotへcoarse＋residualをcoherentに埋め込む代わりに、baseと加法的な補正を別の有料shot群へ分ける**点にある。[C-Sの固定SOURCE][CS]を再開・変更せず、O3の角度/provider最適化をO2へ持ち込まない。既存C-Sと同じestimator/費用構造に還元されるだけなら本案を除外する。

**分類：KNOWN_METHOD_APPLICATION + PROMISING_BUT_INCOMPLETE。** 既知の安いcoarse候補と全pathの小さい補正負担がまだ得られておらず、案Mより先の実装は推奨しない。実数精度・finite sampling・phase・全費用を同時に満たすSOURCEは未接続である。

## 4. 再利用するSOURCEと、まだ存在しない差分

以下のpathは特記以外[SYNTHの固定tree][TREE]。静的に接口・意味論を確認したもので、module/runnerを実行したという意味ではない。

| 既存SOURCE | 再利用するもの | そのままでは足りないもの |
|---|---|---|
| `src/trotterlib/rte.py`：`_make_event`, `event_unitary` | product→rotation順、sign/phase、共通order角の定義 | actual Qは別の整数CDF由来。float event_probabilityを流用しない |
| `src/trotterlib/df_rte_circuit.py`：`DFRTEComponentCircuitSpec`, `DFRTEApplicationBasisChoice`, `DFRTEBasisPlan` | signed Z/ZZ、original/selected basisの区別、identity semantics | 案Mの群内でA/B/Vが一致する証明・eligibility条件 |
| `src/trotterlib/df_partial_s2.py`, `df_rte_qiskit.py` | system evolution、明示basis/diagonal列 | 案Mのcontrol0 A†/control1 Bの枝配置、system readout |
| `src/trotterlib/rpe_hadamard_interrogation.py` | diag(I,U)、X/Y/bit符号、full interrogation対象 | 一般equatorial軸、条件付きreadout、重み付きdecoder |
| `src/trottertracks/algorithm_codesign/a1_n3_connection.py`：`ConditionalSampler`, `MeasurementFacade`, `Streams`, `SavedPrices` | actual-Q条件付き取得、bit-only隔離、独立stage、費用記録 | 新しい(η,g) marginal、system outcomeをfacade内で重みへ写す接口。旧±1専用range/variance式の更新 |
| 同directory `a2_lcu_access.py` | fine/coarseの明示system gate、phaseを含める規約、全attempt会計の参照 | 案M/DともLCUそのものは使わず、旧herald decoder/証明を流用しない |
| 同directory `return_aggregation.py`, `g9_p5.py`, `parent_allocation.py` | finite-bit/正support、Closed/Parentの強いbaseline、既知law扱い | 全path価格score・normalizerの新oracleは提供されない |
| `src/trotterlib/df_rpe_hadamard_compiled_cost.py` | 同profileでのfull no-prep価格取得・既存cache規約 | mid-circuit/feed-forward価格と誤差、案Dのbase/correction全段階会計 |
| [C-S SOURCE][CS]：`same_position_residual.py`, `cs_v0.py` | 局所回転差、coherent residualとの重複確認に限定 | 案Dの別shot estimator/有限取得仕様。旧科学契約は再利用許可されていない |

## 5. 今回の限界

式は設計提案の記号的確認であり、人工テストPASS・実装資格・科学的優位性の証拠ではない。実入力の群存在、分散差、full価格、十分shot数、総費用はすべて未取得。旧A1/SCV/S2/A2の負材料を一般no-goへ拡張せず、既存成果・STOP・C1凍結を維持する。次の限定技術作業は[提案書](O2_NEXT_RESEARCH_PROPOSAL.md)に分けた。

[SYNTH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d28164b5d51e0ca5a0ade74ded8ff176b443dab8/docs/research/o2_research_results_integration_20261108/O2_RESEARCH_SUMMARY.md
[INDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d28164b5d51e0ca5a0ade74ded8ff176b443dab8/docs/research/o2_research_results_integration_20261108/O2_EVIDENCE_INDEX.md
[TREE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/d28164b5d51e0ca5a0ade74ded8ff176b443dab8
[CS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e69d1dfbff0f8ed924109c3c5f30087fc122f595
