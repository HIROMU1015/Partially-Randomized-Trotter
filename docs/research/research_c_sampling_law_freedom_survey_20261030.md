# 研究C：PR sampling lawの未利用自由度・既存手法との対応

2026-10-30着手、2026-10-31 JST完了。起点SOURCE `fb0c803f1cf2b5a06122b3799b8a5149c3f67f97`、独立branch `codex/research-c-sampling-law-freedom-survey-20261030`。

**技術調査の結論：`NO_DISTINCT_FREEDOM`。確認した変更案から、既知Euler／IS／Parent allocation／線形辞書分解／全path再重み付けと区別できる、実装可能な追加改善機構を特定できなかった。今回の系列では追加科学実験を準備しないことを提案する。** 任意の辞書・全sampling lawに改善が存在しないという定理ではない。Track BのD＋E採択、既存STOP、研究C全体の方針を変更しない。正式な研究終了の採否はユーザー／研究Cへ戻す。

新規科学計算・保存値再集計は0。人工の決定論的2×2演算と既存人工回帰だけを実施し、**新9＋既存4＝13 tests PASS**。新sampler、optimizer、科学runner、実行可能科学契約は作成していない。実H4の配列・QPY decode、PF/RTE、shots再算定、sampling、build/compile、fit、候補探索は行っていない。

## 1. 現在地と今回の確認範囲

対象文脈は保存直鎖H4・1.00 Å・STO-3G、Nα=Nβ=2／sector36、通常number PR-aware CDF8 γ=.1の同じJ/U、元順LD2、partial ordered symmetric S2、δ=h=.1、s1/r128/K0、m0..12。元g補正κ、核scalar一回、tail identity抽出、own-DF ground-state oracle、状態準備費用除外、既存P0条件を維持する。

次は[189-call結果](research_c_roundwise_det_pr_execution_20261030.md)の引用であり、新しい比・順位・費用集計ではない。

| 保存方式 | Ideal仮定の条件付き総RZ | 引用の意味 |
|---|---:|---|
| 同J/U強決定論S2 s1 | 101,825,929,455 | 全frameの対照 |
| PR CDF8 s1/r128/K0 | 234,117,788,135 | 有限32校正価格を使う条件付き点値 |
| Round別D/PR最小割当 | 101,825,929,455 | 全13roundがD。固定点モデル内の混合改善なし |

Evidence比較はPR全law合成・implemented-law誤差未資格で`UNQUALIFIED`。仮定付き点比較を実CA、母平均優位、PR一般の不可能性としない。副指標CX depthの点値ではPRが低く、主RZの結果を全指標の支配へ広げない。

研究Cの[割当設計](research_c_roundwise_det_pr_allocation_design_20261030.md)、[native構造調査](research_c_pr_native_structure_optimization_design_20261030.md)、[M2](research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md)、[M4-native128](research_c_pr_aware_df_M4_native128_execution_20261028.md)の記録と関連SOURCEを確認した。既存のtail exact-basis-key共有、head相対Gaussian融合、RTE符号修正、固定価格のscopeを再利用し、それらを新機構として数えない。

### Track Bの取得と既知結果

必須4資料は指定commitのGitHub本文とGit blobを取得・照合した。Branchにない原本は移動せず、読み取り用コピーと固定URLを[取得索引](../../artifacts/research_c_sampling_law_freedom/2026-10-30/technical/reference_index.json)へ保存した。補助3資料は同じ固定Git objectから取得した。

| 固定資料 | 確認した事項 |
|---|---|
| [Parent allocation終了記録][B-CLOSE]、commit4301c0f… | D＋Eは正式採択済み。限定成果を保持し、同じdevelopment fixtureの追加探索を区切る |
| [資源構造][B-RESOURCE]、同commit | R0/G/D/S4のtarget・情報access・T/CX/N・setup/onlineの相違。安いprimitiveやB低下だけでは全task利益にならない |
| [次期Decision Packet][B-NEXT]、同commit | 四群は未採択。full-path lawやon-demand生成という名前だけでは方法差にならない |
| [Parent既知性監査][B-PARENT]、commitbc649c… | Euler＋線形分割、固定辞書IS、標準KKTは既知。SPSへの包含と完全な実装同値を区別 |
| [相関監査][B-CORR]、同commit | marginal反例、全path IS、実±1 outcomeのmoment、PRHS/qSHIFTとのtask差 |
| [G6対応][B-G6]／[G9結果][B-G9]、同commit | return集約、local query、zero-fillとnormalizer取得、closed P5とgeneral full、matched CTSの差 |

Track BのP3＋tailは同じfinite P5を実現する対照で、次数を下げて精度を緩めた比較ではない。Closed P5とgeneral local fullは理想return ensembleを共有するが、normalizer access／zero-fill／有限bit生成と必要trial数が異なる。S4のcommon L1最小性とnoncommonの角度・価格変更、正価格ISとの交換は既済。少数qubit・finite P5・r1/2のT価格や改善率をH4 K0/r128へ移送しない。過去の認証・grant・one-shot条件はその履歴に残す。

## 2. 現行finite-RTE law：SOURCEに基づく仕様

以下は固定SOURCEの式を整理したもの。無限Taylorのexact evolutionと有限targetを区別する。一次資料は[元PR Appendix A.2][PR]と[Wan Appendix C][WAN]。紙面の回転符号はVの定義に依存するため、SOURCEの`cosφ I−i sinφ P`で統一する。

### 2.1 Hamiltonian、辞書、係数

元のtailからidentityを抽出し、

\[
H_R^{\rm original}=\zeta_R I+H_R^0,\qquad
H_R^0=\sum_\ell a_\ell\bar P_\ell
=\lambda_R R,\quad R=\sum_\ell p_\ell P_\ell,
\quad\lambda_R=\sum_\ell|a_\ell|,\quad
p_\ell=|a_\ell|/\lambda_R,\quad P_\ell=\operatorname{sign}(a_\ell)\bar P_\ell.
\]

各\(P_\ell\)はHermitian involution。一般対称Jのframeをそのまま一個のinvolutionとしてsampleしていない。\(\bar P_\ell\)は実際にはGaussian回転したZ/ZZで、basis key・support・frame来歴を持つ。非zero係数は微小でも保持する。κ・決定論prefix・核scalarと\(\zeta_R\)はsampling massに含めない。

一h-stepをr microstepへ分け、\(\tau=\lambda_R h/r\)。SOURCEのKは**最大even order**で、有限polynomialの次数はK＋1である。Even \(n\in\{0,2,\ldots,K\}\)について

\[
w_n=\frac{|\tau|^n}{n!}\sqrt{1+(\tau/(n+1))^2},\quad
B_K=\sum_n w_n,\quad
c_e=w_n p_\ell\prod_{j=1}^n p_{\ell_j},\quad p_e=c_e/B_K,
\]
\[
V_e=(-1)^{n/2}e^{-i\phi_nP_\ell}P_{\ell_n}\cdots P_{\ell_1},\qquad
\phi_n=\arctan(\tau/(n+1)).
\]

回路時間順はproduct labels \(\ell_1,\ldots,\ell_n\)、最後にrotation label。非可換wordをrotationの反対側へ交換しない。Coeff符号はsigned Pへ吸収し、unsigned回路では`unsigned_rotation_angle`と`product_sign_phase`が戻す。\((-1)^{n/2}\)もcontrolled相対位相として必要。

\[
T_{K+1}(\tau R)=\sum_{j=0}^{K+1}\frac{(-i\tau R)^j}{j!},\qquad
\mathbb E_{p_e}V_e=T_{K+1}/B_K.
\]

**K0**ではorder0だけ、\(T_1=I-i\tau R\)、\(B_0=\sqrt{1+\tau^2}\)、\(p_e=p_\ell\)、\(V_\ell=e^{-i\arctan(\tau)P_\ell}\)。これはangleτの通常qDRIFT ensembleと同じ平均ではない。Taylorのleading orderやHamiltonianが同じでも、`atan(τ)`をτへ置換できない。K0でP3/P5の高次数returnが隠れているわけでもない。

### 2.2 IID多step・full kernel・信号

各microstepが独立なら、積の順序を保ったまま

\[
\mathbb E[V_r\cdots V_1]=(T_{K+1}/B_K)^r.
\]

核scalarを除いたκ/prefixのhalf-sweepをD_f/D_rと書くと、補正kernelは

\[
F_h=e^{-i(c+\zeta_R)h}D_r(h/2)T_{K+1}^rD_f(h/2),\qquad
\mathbb E W_h=B_K^{-r}F_h.
\]

Referenceは核scalarをbase generatorの両halfへ含める表記、nativeは最後のancilla phaseへ一回載せる表記を使う。これらを重ねて二回加算しない。F_hはnormalizationを戻したfinite operatorで、unitaryとは限らない。

\(n_m=s2^m\)個の**fresh independent occurrence**をcoherent接続すると

\[
\mathbb E W_{m}=A_mF_h^{n_m},\quad A_m=B_K^{-r n_m}>0,\quad
z_m=\langle\psi|A_mF_h^{n_m}|\psi\rangle.
\]

固定s1/r128では\(n_m=2^m\)。同じsample kernelを繰り返す場合の\(\mathbb E[W_h^{n_m}]\)へ置換しない。\(A_m\)はnormalization attenuationであり、**一般に\(|z_m|\)そのものではない**。Finite PF/RTE bias、oracle、合成、実law誤差は別量である。

既存accuracyは同じraw complex-signal単位へ
\(d_m=A_m(\epsilon_{\rm PF,m}+\epsilon_{\rm RTE,m})+\epsilon_{\rm syn,m}+\epsilon_{\rm law,m}\)
を戻し、\(\rho_m\ge A_m-d_m\)、\(\beta_{\rm det}\le\arcsin(d_m/A_m)\)を使う。DF[Ha]をこの和へ足さない。未知syn/lawはnullのまま。

以上は理想実数のfinite lawの仕様である。実装はbinary64の係数・B・CDFとPCG64を使い、component確率もその和で再正規化する。式の厳密一致、理想IID、丸め後のjoint streamをこの静的調査で認証していない。旧C2-Bの別h/rに対する条件付きtable boundを、現r128や新qへ移送しない。

### 2.3 実装対応

| SOURCE・関数 | 本調査で確認した役割 |
|---|---|
| [rte.py](../../src/trotterlib/rte.py) `finite_rte_distribution/_make_event/event_unitary/finite_taylor_operator` | w/B/p/phase、順序、degree K＋1、符号・normalizationの分離 |
| 同 `iter_sample_rte_events` | PCG64でorderを選び、n＋1 labelsをcomponent pから選ぶ。量子shotsではない |
| 同 `finite_rte_corrected_operator/finite_rte_attenuation` | T^rとB^−rを別に扱う |
| [connection](../../src/trottertracks/c1_stage3_connection.py) `split_native_model/configure_rte` | 元J/U・identity抽出・signed mass・既存law接続 |
| [prototype](../../src/trottertracks/pr_dictionary_validation/prototype.py) `make_tail/ordered_event` | Fraction係数からpとλ、符号を生成。無料の任意q fieldはない |
| [Stage B](../../src/trottertracks/c_pr_aware_df_stage_b.py)／[reference](../../src/trottertracks/c1_stage3_reference.py) | 独立CAR atom、full tail、kernel順序、fresh occurrence、corrected/raw比較 |
| [修正済みnative bridge](../../src/trottertracks/c1_stage3_native_bridge.py)／[circuits](../../src/trottertracks/pr_dictionary_validation/circuits.py) | 格納φとunsigned回路角の区別。旧connectionの負atom angle検証を使う経路へ戻さない |
| [accuracy](../../src/trottertracks/c1_stage3_accuracy.py) `assess_round` | ±1測定、radius、phase、既存Hoeffding。Weighted outcomeへそのまま転用しない |
| connection `compose_serial_cost/serial_native_sequence` | 全occurrence fresh、fenced serial kernel価格＋shell。event加法proxyではない |

## 3. Dictionary固定で確率だけを変える場合

### 3.1 通常の変更と重要な例外

固定\(T=\sum_e c_eV_e\)、\(p_e=c_e/B\)に対して、qだけを変えれば一般には\(\sum q_eV_e\ne T/B\)。Normalized targetを戻すには\(w_e=p_e/q_e\)、補正targetなら\(w_e=c_e/q_e\)をoutcomeへ掛ける。位相だけでは\(|w_e|>1\)の倍率を実現できない。

一方、**辞書にaffineな線形従属性がある場合**は

\[
q\ge0,\quad\sum q_e=1,\quad\sum_e(q_e-p_e)V_e=0
\]

を満たす別qがあり得る。K0の共通角では\(\sum(q_\ell-p_\ell)P_\ell=0\)。この場合weightなしで同じnormalized meanを保持できる。固定辞書だからqが常に一意、と断定してはいけない。重複列の質量移動・相殺・より一般の線形係数選択は、既知のdictionary/係数最適化の自由度である。

SPSは元々**process matrix**上の分解であり、このoperator条件と無条件に同一ではない。\(C(V)=\operatorname{diag}(I,V)\)を用いると、controlled channelのoff-diagonal actionはVを読む。Off-diagonal projection、または\((\operatorname{Ad}_{C(V)}-\operatorname{Ad}_{C(-V)})/2\)というsigned liftを明示すれば同じ線形design問題として記述できる。これは本調査での対応導出と[既存Track B監査][B-PARENT]の再利用であり、SPSがH4用の安価なsamplerを提供したという主張ではない。[SPS II][SPS]

H4保存辞書のnullspace/rankや新qは今回計算していない。全Fockのoperator関係とsectorだけの関係、basis gauge一致と回路価格一致は区別する。関連する物理operator/情報取得なしに線形従属性の利益を既知としない。

### 3.2 現行SOURCEで必要な変更

現在の`RTEEvent`は`event_probability=event_coefficient/event_normalization`を検証し、`_make_event`も同じpからcと確率を生成する。Sampled meanのAPIは**unweighted**であり、現行accuracyも±1 outcome用である。RNGへ任意qを渡すだけではISにはならない。

ISを実装するなら、元c/Bと実qを分離し、元のV・φ・phase・順序を維持し、outcomeのlikelihood weight、全pathの積、実qでのmoment/range、finite-bit誤差とconfidenceを接続する必要がある。`RTEComponent.probability`を変えて現行constructorを通すだけでは、元cを保持した証拠にならない。辞書まで変えるなら新c/V/Bを別target来歴で固定する。今回はどの変更も実装・採択しない。

## 4. Normalization・測定統計・native価格の交換

### 4.1 同じmeanと分散を同一視できる条件

Unweighted Hadamardの各軸outcomeをY∈{−1,+1}とすると、固定状態・同じ理想readoutでは\(\mathbb EY=\mu\)、\(Y^2=1\)、\(\operatorname{Var}Y=1-\mu^2\)。従って**同じraw complex signalを持つなら、その条件下で各X/Yの一shot marginalと分散は同じ**。同じnormalized operator meanもこれを十分に含む。ただし別軸の同時測定分布は主張しない。

同じHamiltonianだけ、同じ補正Tだけ、異なるB、weighted outcome、異なる合成誤差／readout、shot間依存の場合にはこの結論を使えない。Ensemble内のpath期待値の分散と、実±1測定の分散も別である。

固定normalized meanをISで推定する場合、実outcomeはZ=wYで

\[
\mathbb EZ=\mu,\quad\mathbb EZ^2=\sum_e p_e^2/q_e,\quad
\operatorname{Var}Z=\sum_e p_e^2/q_e-\mu^2,\quad |Z|\le\max_e p_e/q_e.
\]

\(\sum q_e(w_e\mathbb E[Y|e])^2\)は実測second momentではない。量子測定ノイズを落として改善を推定しない。一般に上のsecond momentは1以上。補正targetを推定するならB²も戻す。

### 4.2 既知cost-aware IS

固定係数c、固定辞書、正の固定価格CではCauchy–Schwarzより

\[
\left(\sum_e c_e^2/q_e\right)\left(\sum_e q_eC_e\right)
\ge\left(\sum_e|c_e|\sqrt{C_e}\right)^2,\qquad
q_e^*\propto |c_e|/\sqrt{C_e}.
\]

これは既知のcost×second-moment proxyの特殊化である。[Resource-Optimal IS, II Theorem 1][IS]。全circuit parameter/pathをeとする場合も同じ原理に含まれ、「eventからwhole-pathへ替える」だけを新理論としない。ゼロ価格には表示式を適用できず、恣意的なfloorを無償導入しない。

このproxyは有限confidenceのceil付き十分N、最大weight、coherent bias、合成精度、online費用の総最適性を保証しない。角度が変わってCも変わる外側問題は、この固定価格定理だけで解けない。[IS III][IS]のbias不変は同じ実装pathを再重み付けする場合に限り、別の辞書・角度へ無条件に移さない。

### 4.3 Parent配分とL1最小性

K0のparentは\(a=1,b_i=|\tau|p_i\)（時間の符号は\(\sigma\)へ）。既存familyは

\[
z_i\ge0,\quad\sum_i z_i=a,\quad c_i=\sqrt{z_i^2+b_i^2},\quad
\phi_i=\operatorname{atan2}(b_i,z_i),\quad
\sum_i c_ie^{-i\sigma\phi_iP_i}=aI-i\sigma\sum_i b_iP_i.
\]

三角不等式により\(\sum_i c_i\ge\sqrt{a^2+(\sum b_i)^2}\)。Common \(z_i=ab_i/\sum b_i\)は等号を達成する。つまり**現K0は既にこのparent family内のL1最小点**。非commonなら角度／native価格を変えられるが、Bをさらに小さくする新自由度ではない。Eulerの再配分としてTrack Bに実装・検証済み。[B-PARENT]／[SCU Note 5][SCU]／[PTSC Eq.(75)][PTSC]

この下界のscopeはparent familyだけ。一般辞書に対する必要条件は\(B\ge\|T\|_2\)で、\(\sqrt{1+\tau^2}\)が全unitary辞書の最小値とは主張しない。Pauli collection・既知の演算子関係・別unitary分解でBが下がる数学的可能性は残る。その構成・control・情報取得・whole-kernel価格を閉じた新方法は今回得られていない。一般LCU枠組みへの包含だけを、全生成手順と費用の完全同値にも、独立新規性にも置き換えない。

### 4.4 多stepの交換と費用scope

同じ補正Tを保持してBだけ増やすと、独立N microstepのraw meanはB^−Nに縮み、補正outcomeのrangeはB^N、second momentはB^(2N)。理想固定absolute-errorのleading shotsはこの二乗負担を受ける。位相taskの実際の十分shotsは既存radius/PF-RTE/syn/law予算へ戻す必要があり、B²だけで再評価済みとはしない。ここでNはmicrostep数で、科学call数ではない。

必要な比較は、同精度の\(N_{\rm shots}\mathbb E[C_{\rm complete\ kernel}]\)とshell／coherent反復の整合である。単event価格和・switch数・小angle・B低下をP0 whole-kernel価格と同一視しない。Classical setup、normalizer/dictionary取得、finite-bit tables、likelihood評価、online生成は別座標で報告する。未測定費用を0にせず、CPU秒とRZを換算係数なしに加算しない。

## 5. 多step相関とcontrolled位相

IID path lawを\(P(\omega)\)、順序・head・phaseを含む実装を\(W_\omega\)とする。Unweighted qが元meanを保つ条件は

\[
\sum_\omega(q(\omega)-P(\omega))W_\omega=0.
\]

各stepのmarginal一致だけでは不十分。より強い十分条件として、各stepの条件付きoperator meanが全historyに対して同じ固定meanなら、tower propertyで積の平均を因子化できる。一般のhistory-dependent lawではこれを仮定しない。

例として\(V_i=cI-isQ_i, R=\sum p_iQ_i\)、pをstationaryとするpersistence kernel \(K_\eta(j|i)=(1-\eta)p_j+\eta\delta_{ij}\)を取ると

\[
\mathbb E_\eta[V_2V_1]-(cI-isR)^2=-\eta s^2(I-R^2).
\]

一般には非zero。これは既存[相関監査][B-CORR]の反例を再利用した2×2検証であり、新lawの提案ではない。

同じtargetを全path ISで保持するなら、単純likelihood補正が定義できるsupport上でq>0とし、\(W_{\rm wt}(\omega)=B_{\rm tot}P(\omega)/q(\omega)\)を実±1 outcomeへ掛ける。

\[
M_2=B_{\rm tot}^2\sum_\omega P(\omega)^2/q(\omega),\quad
L=B_{\rm tot}\max_\omega P(\omega)/q(\omega).
\]

これは既知IS。Full support条件はこのP/q法の条件であり、別の線形分解がsupportを減らす全可能性を否定するものではない。独立blockならweight積、history依存なら条件付き確率とnormalizerの積を保持する。Basis切替を減らしてもM2/rangeや生成費用が増え得る。異なるshot同士の依存まで入れるなら、既存IID Hoeffdingへは接続できず別のconfidence論証が必要になる。

[PRHS][PRHS]はLiouvillian/channelのsigned correlated block、[qSHIFT v3][QSHIFT]はobservableのhistory-dependent Taylor matchingである。相関を使う方向自体は既知。現在のcontrolled finite-RTE first momentへの移送にはtarget／bias／正のsampling確率／weight／相対位相の別証明が要る。過去のqSHIFT表示不整合は[B-CORR]の記録に帰属させ、今回それを解消したとは扱わない。どちらの外部性能値もH4へ移送しない。

Channelだけの一致を避ける最小例はVと−V。\(\operatorname{Ad}_V=\operatorname{Ad}_{-V}\)だが\(C(-V)=Z_{\rm control}C(V)\)で、X/Y信号は異なる。Complex/sign位相は\(p\ge0\)のunitary枝に保持するか、明示したclassical weight／測定frameへ戻す。有限bit後のphase・probability・weightを未確認のまま同じraw signalと呼ばない。

## 6. 自由度の分類・既存法との対応

下表は今回の全検討集合であり、候補の大量生成・ランキングではない。科学的な性能negativeと、未実装・未資格・同値性を分ける。

| 区分／変数 | 保存target | 既知対応・Track B状態 | B／moment・range | Native価格・取得条件 | 今回の判断 |
|---|---|---|---|---|---|
| 固定辞書、一般q＋p/q | 補正またはnormalized operator mean。Raw unweighted lawは一般に変わる | Cost-aware IS。既実装・監査済み | B固定、M2/range変化 | 同じVでもpath価格分布は変化。正価格／likelihood取得が必要 | 新機構なし |
| 固定辞書、affine nullspaceのq | 同じnormalized meanをunweighted保持 | 線形係数最適化／SPSへの明示lift。H4固有のnullspaceは未評価 | B・unweighted±1 marginal固定にできる | Basis列の価格は変わり得る。関係式と全scopeの証明が必要 | 既知自由度。H4利益は未評価 |
| Parent z／Euler angle | 同じfinite T | S4 familyに包含。D＋Eで同fixture探索終了 | Commonがfamily内B最小。非commonは増加し得る | Angle-dependent価格。既知固定価格KKTは実P0最適性でない | 再提案しない |
| 相殺・Pauli collection・unitary辞書変更 | 明示すれば同じfinite T | CTS／SCU／return aggregation／線形辞書分解。G/Dで検討済み | Bを下げる可能性、取得依存 | Pauli展開や別unitaryの取得・controlled実装は無料でない | 独立した安価な生成差なし |
| Joint path q／history | ordered-product mean、または別高次target | 全path IS／PRHS／qSHIFT。単純相関は既存反例あり | 積weight／history M2・range。Marginalだけでは保存不可 | Switch節約とlikelihood／normalizer費用を会計 | 新たな同target生成機構なし |
| Basis共有・control・compile | 同じV/pathとphase | Native既済共有／相対融合、D2境界簡約、研究C構造調査 | 理想B/law不変、合成誤差は別 | Whole-kernel価格を取得して判断。未観測節約を仮定しない | 別名の新lawにしない |
| Normalizer／dictionary query、setup/online | 同じ理想meanでも実trial lawは異なり得る | G6 local query、G8 cold/warm、G9 closed-P5/zero-fill既済 | Unknown Bの無料再正規化不可 | CPU/bit/メモリ・miss/selector・生成費用を取得 | 新しい非列挙手順・取得優位を特定できず |

### P3／P5の転用限界

G6では\(Q_i^2=I\)の形式returnを集約し、parent依存angleを取得する。Closed P5は少数群のBを使い、general local fullはordinary envelope B_ordからacceptance B_full/B_ordのzero-fillを行う。全trial分母でのM2はB_ord B_full、range B_ordで、accepted-onlyの再正規化を無償に行えない。[B-G6]／[B-G9]

Literal CTSはPauli情報をcollectする。SCUの有限／層別構成もあり、「CTSは必ず全wordを列挙する」とは言えない。一般Gaussian回転involutionはPauli word一個ではないので、CTSと同情報・同費用で扱えるかは別条件。Return、Euler、normalizer回避という部品自体は既知で、再命名では独立機構にならない。

現K0の一microstepはP1で、高次数returnの余地はない。r microstep全体をcollectする場合のtargetは\((I-i\tau R)^r\)。これは総時間のTaylor polynomial P3/P5でも\(T_{r}(r\tau R)\)でもない。G6の有限P_m係数やclosed-P5式をそのまま代入すればtargetが変わる。全pathを新係数で集約する余地を論理的に排除はしないが、既存の分解問題を超える具体的な生成／native削減手順は未取得である。

## 7. 一次資料の限定確認と主張範囲

以下の関連本文を今回取得した。元PRのHTML404はPDFで回収した。全論文・全branchの網羅監査、priority判定、詳細GPT科学レビューは実施していない。論文全文はrepositoryへ複製せずURLと関連箇所を残す。

| 一次資料・取得版 | 関連箇所と本報告での用途 |
|---|---|
| [Günther et al., PR v2][PR] | Appendix A.2 A18–A29。現finite lawの元となるeven pairing・positive coefficients・multi-step sampling |
| [Wan–Berta–Campbell v2][WAN] | Appendix C C4–C5／finite cutoff。既知common-angle construction |
| [Cugini–Atif–Subaşı IS v1][IS] | II Eq.(4),(6)–(10)、III。固定protocol/pathのcost×moment、bias不変のscope |
| [Peetz–Smart–Narang SCU v2][SCU] | II.1/II.3、IV.2、Note5 S13–S15。Operator辞書とEuler/CTS、Pauli access |
| [Koczor SPS v2][SPS] | II.1–II.3 Eq.(1)–(6)。Process辞書・L1最適化。Operatorとの対応は本報告のliftを明記 |
| [Zeng et al. PTSC v2][PTSC] | Eq.(75)付近のidentity＋involution Euler pairing。Trotter orderのKを本SOURCEのcutoff Kと混同しない |
| [Cugini PRHS v1][PRHS] | Main Liouvillian定義とsigned path構成、補足のordered word条件。Finite-RTE targetの直接保存とは異なる |
| [Lee–Choi qSHIFT v3][QSHIFT] | III/III.1のhistoryとobservable Taylor matching。古典係数生成費用を無償としない |

数学の対応はSOURCEと上記本文に基づく本調査の導出であり、各論文がH4/K0/r128のwhole-native資源を最適化したという引用ではない。一般枠組みに包含されることと、同じ情報・有限bit・手順・費用での完全同値性は別。未確認の完全同値性を独立新規性の証拠にも使わない。

## 8. 人工検証・保全・実施量

[新test](../../tests/management/test_research_c_sampling_law_freedom.py)は既存`rte`のnormalization／enumeration／event_unitary／finite Taylorをそのまま呼ぶ。人工のsigned X/Z、正負τ、K0/K2だけを使い、確率列は固定値、RNGを一切呼ばない。追加数式は独立referenceの小行列恒等式で、新アルゴリズム実装ではない。

| 人工確認 | 件数 | 結果 |
|---|---:|---|
| 既存lawのfinite mean・符号・順序・B、Aと信号絶対値の違い | 4 | PASS |
| q変更のmean破れ、p/q修復、実±1測定second moment | 1 | PASS |
| Affine依存辞書でのunweighted q変更と同じBernoulli marginal | 1 | PASS |
| 既知Parentのmean恒等式、common最小B、端点 | 1 | PASS |
| 同marginal相関の非保存、ordered path IS、M2/range | 1 | PASS |
| System-channel同値とcontrolled phaseの違い | 1 | PASS |
| 既存回帰：paired Taylor、identity、signed product、異種occurrence normalizer | 4 | PASS |
| **合計** | **13** | **PASS** |

[log](../../artifacts/research_c_sampling_law_freedom/2026-10-30/technical/tests.log)／[JUnit](../../artifacts/research_c_sampling_law_freedom/2026-10-30/technical/tests.xml)。既存回帰は`tests/test_rte.py`の上記4testだけを再利用し、分子テスト・科学runnerは起動していない。Artificial PASSはH4性能・実law・全law・科学GOの証拠ではない。

[技術照合記録](../../artifacts/research_c_sampling_law_freedom/2026-10-30/technical/verification.json)では既存SOURCE118件、前段までの原本・契約・SOURCE・grant・ledger・marker・STOP等2830 pathのhash一致を確認した。必須Track B4文書のGitHub readbackも固定Git blobと一致した。これは原本保全の確認であり、旧科学結果の数値再解析ではない。

Task-local venv/cache/tmpで既存依存をread-only参照した。依存install・共有venv編集・全体設定変更・他job操作はない。小さい決定論的代数検証に重いworkloadはなく、CPU/RAM/worker/wall人工capを追加していない。起点のproduction SOURCE、旧契約・raw・grant・ledger・marker・STOPは無変更。日付は20261030の依頼識別子を維持し、中断後20261031に同じ技術scopeを継続した。

## 9. Q1～Q9と次の判断

| 問い | 回答 |
|---|---|
| Q1 変更できる部分 | p/weight、affine係数、Euler z/angle、辞書、joint law、native実装、normalizer access。ただしtarget/phase/統計/取得費用を同時に定義する必要がある |
| Q2 既済部分 | Euler線形分割、fixed cost IS、Parent z/KKT、P3/P5 return、zero-fill、full-path IS、basis共有は既知・既済。新名称で重複候補にしない |
| Q3 同じfinite meanでBを下げる余地 | Common K0 parent内では既に最小。一般辞書・collectによる可能性は否定しないが、既存分解原理を超える安価な具体機構は特定できず |
| Q4 B増加と安いnative | 既済Parentの交換。多stepのraw半径・weight M2/range・bias・十分N・whole価格が必要。単価低下だけでGOにしない |
| Q5 相関の保存条件 | Joint operator和の一致、または適切な全path likelihood補正。各stepのmarginalだけでは不可。Shot間の依存も区別する |
| Q6 実装可能な独立機構 | 今回はなし。Controlled lift、finite-bit probability/weight、normalizer／dictionary取得の未資格を残す。一般枠組みへの包含を完全実装同値としない |
| Q7 K0/r128への適用 | IS／既済Parentは数学的には適用可能だが新原理ではない。P3/P5 returnは現P1やT1^128へ直接移植不可。新r/Kの採択なし |
| Q8 強決定論との競争 | 保存主RZの条件付き点モデルでは全Dが選択された。新lawのwhole価格／精度／取得費用は未評価で、勝利予測も不可能性証明もない |
| Q9 次の実験の価値 | 今回は独立機構の条件を満たさず、追加scienceはNO-GO提案。候補0・実行可能契約0。広いPR研究の終了判断へ拡張しない |

上表Q5の条件は式の\(\mathbb E[V_r\cdots V_1]\)に対するもので、各stepに関する条件を表す。共同meanの恒等式と単に同じ周辺確率を保つことを区別する。

**ユーザー／研究Cの次の作業**：この「既知Euler／IS／Parentの再配分による追加探索」を区切る提案の採否を判断する。D＋Eを解除する必要は今回認めず、その変更案も出さない。

**Codexの次の作業**：公開・remote SHA/tree・SOURCE・旧原本・成果物を照合後、mandatory STOP。現在の依頼内で未完了の科学計算はない。新しい具体的な生成機構が提示された場合だけ、保存target、既知法との差、weight/phase、実装可能な取得費用を紙上で先に判定する。今はその候補がないため、仮の科学call枠やH4実行契約は作成しない。追加H4再解析／sampling／native校正／fit／別分子／詳細科学レビューへ自動進行しない。

[B-CLOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[B-RESOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_resource_structure_and_bottlenecks_20261015.md
[B-NEXT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_next_algorithm_research_decision_packet_20261015.md
[B-PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/parent_allocation_prior_art_correspondence_20261014.md
[B-CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md
[B-G6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md
[B-G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[PR]: https://arxiv.org/pdf/2503.05647v2
[WAN]: https://arxiv.org/pdf/2110.12071v2
[IS]: https://arxiv.org/html/2603.13495v1
[SCU]: https://arxiv.org/html/2407.21095v2
[SPS]: https://arxiv.org/html/2402.15550v2
[PTSC]: https://arxiv.org/html/2212.04566v2#S4.E75
[PRHS]: https://arxiv.org/html/2608.29756v1
[QSHIFT]: https://arxiv.org/html/2604.26263v3
