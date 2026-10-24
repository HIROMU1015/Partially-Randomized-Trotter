# H4 Pauli PR全lawの実装残差上界転送：R2静的設計

2026-10-25 JST。基点 `21fa1d08cf53422bd6e44e832ce8cc453f6d3ebf`、専用branch `codex/non-df-h4-central-r2-design-20261025`。

**全lawへの転送不等式は構成できる。現在のopt1 compiled familyが保存予算を満たす保証は、既存証拠だけでは確定できない。**
有望なのは、C-target→central入力のoperand/相対位相誤差と、入力→compiledの誤差を分け、全域で成立するgate/rewrite上界を合成する方法。
まずuniformな証明を目指し、必要な場合だけ同じ証明をeffective lawで重み付けして改善する。
単なる追加trajectory診断や保存256本の最大/平均では、この不足を埋められない。

今回は静的SOURCE調査・数学的設計・人工検証だけ。新実H4数値評価、Arb状態作用、compile、sampling/RNG replay、state生成、N/G再解析、新科学grantは**すべて0**。
新しい全lawの数値ρは取得しておらず、`PR_law_rho=null`、正式certificate未受理、旧N/G=nullを保持する。
これは独立の科学レビュー・最終evidence freezeではない。数式の導出、SOURCEの対応、数値証拠、未解決の前提を区別する。

## 1. 固定対象・証拠

Linear H4／隣接1.00 Å／STO-3G、Pauli-JW184非identity項、PR retained prefix27／tail157、S2、T=.8、q8/r4/K4、δ=.1。
system8＋ancilla wire8、little endian、同一保存raw ray／exact norm νを使用する。DF、H2 S2/Yoshida4系列、PR母平均回路費用tailは対象外。
Compilerは既存D2のQiskit1.3.0、rz/sx/x/cx、optimization1、seed17、backend/coupling/layout/routingの明示指定なし。
古いgeneric全body controlへ戻したり、別compilerへ交換したりしない。

| 役割 | 固定SOURCE／保存資料 |
|---|---|
| 元law・Pauli append・wrapper/compiler | `0c329b30a81accb35c212be8defa66fee0dd0f12`、下記source参照 |
| Cのexact encoded law | `e80d7510880f8db993b777c627bb8e95b05a070d`、保存model/result |
| D2 central control | `74515d8eb4f073c40fc783681a3775dbca89e9b1` |
| 旧R2案 | [旧rho準備](non_df_h4_central_rho_preparation_20261024.md)、`h4_central_rho_plan.drafts`のR2 |
| 6局所Arb結果 | [最新結果](non_df_h4_central_arb_pf_pr0_sine_20261025.md)、[全6件の原表](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/six_local_results.json) |

基点commitはGitHubから取得してSHA一致を確認した。前段のpush失敗は当時の記録として保全する。
[今回のSOURCE索引](../artifacts/non_df_h4_central_r2_design/2026-10-25/source_evidence.json)、[lawの保存値引用](../artifacts/non_df_h4_central_r2_design/2026-10-25/law_QUOTED.json)。
保存6件はすべて局所数値上界が予算内だが、PR0/PR1以外・order4・一般compilerへの外挿はしない。

## 2. Effective event lawとC-target

### 2.1 確率の正本

元SOURCE `iter_sample_rte_events` は、orderを選択後、componentをk+1個抽選する。最初の抽選IDはrotation、残りがproducts。
`_make_event`は**実行順をproducts→rotation**へ格納する。保存`event_probability`は名目浮動小数値であり、全lawの重みの正本にはしない。

実装lawは、NumPy1.26.4のbinary64 CDFと53-bit uniform grid／right-searchから得たC保存値を使う。
Component質量をp_j、order質量をπ_kとすると、IID uniform-wordという既存仮定の下で

\[
 p(e)=\pi_k\,p_{j_R}\prod_{\ell=1}^{k}p_{j_\ell},\qquad
 p(i)=\prod_{t=1}^{qr}p(e_t),\quad k\in\{0,2,4\}.
\]

有限CDF端点cの累積個数は`ceil(2^53 c)`。Cの`grid_masses/verify_probability_mapping`を再利用できる。
今回CDF・確率・normalizationは再計算せず、保存された有理数を引用した。

| order | 保存effective質量 | products | 最後のrotation | order phase |
|---|---|---:|---:|---:|
| 0 | 2242933535092243/2251799813685248 | 0 | 1 | +1 |
| 2 | 4430213036831/1125899906842624 | 2 | 1 | −1 |
| 4 | 5852519343/2251799813685248 | 4 | 1 | +1 |

157 componentの全質量・label/signは引用JSONに保存した。Nominal重みは`|τ|^k/k! sqrt(1+(τ/(k+1))^2)`、名目確率はその正規化だが、R2でeffective質量へ上書きしない。
PCG64列の無条件独立性を証明したとはしない。新しいRNG認証を必須条件へ追加せず、既存IID仮定を明示する。

### 2.2 演算順序と位相

P_j=s_j σ_j（s_j=±1、σ_jはPauli word）とする。Cの個別eventは

\[
 E_e=(-1)^{k/2}\exp(-\mathrm i\widehat\phi_kP_{j_R})
                 P_{j_k}\cdots P_{j_1}.
\]

\(\widehat\phi_k\)は保存binary64角をexact dyadicとして解釈した値。保存角は順に
`.08880799816476508`、`.029672025283184623`、`.01780655959824602`。
理想atanを再計算して置換しない。Pauli同士の可換性を仮定せず、符号と順序を保つ。

各stepはretained27個forward→4event→retained27個reverse→native scalar phase。これを8回適用する。
C retained角は`exact(binary64 coefficient) × exact(binary64 δ) / 2`。
Native scalarは保存operandのexact sumを位相へ使う。Cの`phase(x)`はexp(+ix)。名目constantを新計算して入れ替えない。

P_jはHermitianかつP_j²=Iなので、**未観測order4を含む各E_eもtrajectoryもexact unitary**。
全lawのC平均は、M=Σ_j p_jP_j、R_k=Σ_j p_j exp(−iφhat_k P_j)として

\[
 \mathbb E[E_e]=\sum_k\pi_k(-1)^{k/2}R_kM^k.
\]

これは独立なID抽選と積の順序から得られ、異なるP_jの可換性は不要。Cの`finite_implemented_mean`に一致する。
さらに独立eventの平均をstepへ挿入するとCの`evolve`と対応する。
各経路がunitaryであることと、law平均行列自体がunitaryであることは別である。

### 2.3 Bとraw norm

B_savedは既存値 `1449450801590761/1125899906842624`（約1.28737）を**一度だけ**使用する。
Cにはsaved log_B上界包含true・Pauli event rangeの数値証拠があるが、旧`normalization_certificate_verified=false`と正式未受理は維持する。
R2でBを名目B_K^(qr)やB_upへ交換しない。B_upはshot range用、母平均費用の重みでもない。

ν=ψ†ψは保存raw vectorのexact norm。同一νで各信号を割り、ψ自体や出力stateを再正規化しない。
C保存`corrected_encoded_mean`はraw norm付きなので、lawのnormalized信号へ対応させる際はcomponent/ν。

## 3. 全lawへの転送定理と誤差所有

`z_i=ψ†U_C,i ψ/ν`、`m_C,i,a=Re(z_i)`または`Im(z_i)`、`m_out,i,a`を実compiled wrapperのancilla-Z期待値とする。
同じevent lawに対するbias差を

\[
 \rho_{PR,a}=\left|B_{saved}\,\mathbb E_p[m_{out,i,a}-m_{C,i,a}]\right|
 \le B_{saved}\,\mathbb E_p[|m_{out,i,a}-m_{C,i,a}|]
 \tag{1}
\]

と上界化する。有限和の三角不等式であり、256標本の推定量ではない。
これは回路の数学的測定実装残差で、実機noise・shot揺らぎ・新しい状態準備誤差は加えていない。

V_iをactual binary64 operandで定義するcentral入力bodyとし、SOURCE上でinputがdiag(I,V_i)のHadamard wrapperになることを全eventで確認する。
η_i=||V_i−U_C,i||、κ_i,a=||W_out,i,a−W_in,i,a||とすると

\[
 |m_{in,i,a}-m_{C,i,a}|\le\eta_i,\qquad
 |m_{out,i,a}-m_{in,i,a}|\le2\kappa_{i,a},
\]
\[
 |m_{out,i,a}-m_{C,i,a}|\le\min\{2,\eta_i+2\kappa_{i,a}\}.\tag{2}
\]

第一式はnormalized bra-ketとcontrol恒等式、第二式は||Z||=1・両wrapperがunitaryであることによる。
**body→Hadamard信号では係数1、一般wrapper→Z期待値では係数2**を区別する。
compiled側もdiag(I,Vout)の構造を全域証明できるなら係数1へ改善できるが、現在は仮定しない。
最終measurementを除いたunitary prefixと同じancilla/qubit配置が前提。layout等が変わるなら明示した対応が必要。

| 辺・不確かさ | R2での扱い |
|---|---|
| reference→C encoded law、既存bias/normalization | 旧Cの所有。R2へ再加算・再解析しない |
| C selected operator→binary64 central入力 | η。retained operandと集約relative phaseを一度ずつ評価 |
| central入力→compiled wrapper | κ。数値再合成・書換え・gate loweringを含む全域証明が必要 |
| bound自体の数値評価誤差 | 将来Arb端点へ包含。物理的なη/κとして別に二重加算しない |
| Cの既存`circuit_semantic_allowance` | 部分的なηと重なる。独立の追加誤差として丸ごと加算しない |
| PR費用母平均tail | 測定残差とは別。R2の成功で解決したことにしない |

Cの`total_error_upper`には既存semantic項も含まれる。今回その項を差し引いたり、保存ρ予算を広げたりしない。
既存C/L会計を変更せず、(1)の直接残差を固定予算に照合する設計に限定する。
将来正式会計へ統合する際のedge所有・重複は別の判断であり、ここで正式certificate gateを通さない。

## 4. Uniform／law-weighted／分解の比較

### 4.1 Uniform

全allowed iについてη_i≤D、κ_i,a≤K_aを証明すれば

\[
 \rho_{PR,a}\le B_{saved}\min\{2,D+2K_a\}.\tag{3}
\]

必要なのは有限集合からの観測最大ではなく**全域の証明**。確率を数値評価しなくても成立する利点がある。
単一Pauliのexp(−iαP)ではoperator差≤min(2,|α−β|)、unitary積の差≤primitive差の和を使える。
2qL個のretainedとqr個のrotation、最大Kqr個のproductsというSOURCE上限から、stateを使わず構成できる。
L=27、q=8、r=4、K=4は固定。これは命令構造の式であり、新しいcompiled費用上界ではない。

### 4.2 Law-weighted

SOURCEで定めた排他的stratum S_sに対し、全i∈S_sの信号差上界d_sと**正しいeffective質量**w_sがあれば

\[
 \rho_{PR,a}\le B_{saved}\sum_s w_sd_s.\tag{4}
\]

order/eventごとに加法的なoperator上界が証明できるなら、全trajectoryを列挙せず

\[
 \bar e=\sum_k\pi_k\left[k\sum_jp_je_{prod,j}+\sum_jp_je_{rot,k,j}+e_{order,k}\right]
\]

を使用できる。例えばη_i+2κ_i,a≤D_a+Σ_t e(e_t,a)が**全経路で**成立する場合、

\[
 \rho_{PR,a}\le B_{saved}\min\{2,D_a+qr\bar e_a\}.\tag{5}
\]

成分sign phaseを集約phase項で数えるならe_prod/e_orderへ再度加えない。
Compilerの1Q run融合はevent境界を跨ぐため、孤立eventのcompile誤差を足すだけでは(5)の仮定を満たさない。
Rewrite列全体へのattribution、または境界・run長を含む一様なK_aが必要。
Unseen componentの重みを0にしたり、256標本の出現率でw_sを代用したりしない。数学的重み付けに新しいconfidence allocationは不要。

### 4.3 分解は両方法と組合せる

Uniform/law-weightedは外側の集約法、C→input→compiledは内側の誤差所有の分け方であり、三者は排他的な選択肢ではない。
推奨は**分解＋uniformを第一段階**とし、全域証明が揃い、uniformの保守性が支配するときだけlaw-weightedへ進むこと。
全law意味論やcompiler上界が未証明なら、重み付けだけを精密化してもGOにはならない。

| 案 | 既存証拠で可能なこと | 主な不足 | 実用性／負荷 |
|---|---|---|---|
| 全lawの自明bound 2B | unitaryなCと測定rangeからの式 | compiler全域の正常出力という定義条件 | 数値科学計算不要だが今回の小さい予算には役立たない |
| (3) uniform | Pauli unitary恒等式・有限回数・same-angle部分・6局所の整合性 | phase DAG／全compilerのoperator誤差証明 | 第一候補。H4 state作用不要。rare orderの最悪値は保守的 |
| (4)(5) law-weighted | Cのeffective質量・非可換平均式 | stratumごとの全域bound、compiler境界の帰属 | 低次momentなら軽い。157^(k+1)のevent全列挙や全trajectory列挙は採らない |
| 保存256の平均・最大 | 有限cohortの観測だけ | 全域range/母平均への証明 | 全law保証の方法として不成立。追加標本でもこの不足は消えない |
| 新order4の局所Arb | 将来の不具合発見・stress診断には有用 | 別trajectory全域への転送 | 現時点では非推奨の既定工程。科学実行・QPY生成/compileが別承認で、全law保証を単独では与えない |

## 5. C→inputで小さくできる部分・位相の注意

D2は一つのnonidentity Pauli evolutionをbasis変換／parity CX／CRZ／逆変換へ展開する。
全orderに対し、productsはchronologicalなcontrolled X/Y/Z。負productとorder位相は元bodyのaggregate phaseをancilla P(φ)へ**一度だけ**移す。
このsymbolic control恒等式は全label/sign/orderに拡張可能。selected QPYのPASSをその証明の代用にはしない。

- RetainedのC角α_b=c_b δ/2とinput角β_b=fl(2 c_b (δ/2))/2の差は、保存binary64 operandのexact算術で有限に評価できる。全retainedで2q回ずつ。
- Event rotationのC角は保存atanのbinary64値。D2の`2*(±1)*angle`は有限・非overflow域で二進の倍数変換なので、この辺には新しいatan近似誤差を入れない。符号もcoefficientで保持する。
- Signed productはexact Pauliとして作用する。Cのsignとliteralの負product位相の対応を証明し、productの順序は交換しない。

Native scalar exact sumをσとし、N_-を負product数、N_2をorder2 event数とすると、Cのscalar/signをまとめた位相は
`q σ + π_exact (N_-+N_2)`。Literalは`math.pi`／`np.angle(±1)`のbinary64値、浮動小数加算、QuantumCircuitのphase setterを通る。
後者はPython側で`CircuitData.global_phase`へ渡される。**modulo処理を正確と仮定したgamma_nの貼付けだけでは不十分**。
位相の上界には、π定数差、加算丸め、modulo/remainderの実装とwinding数を含め、mod 2πの距離を正当化する。

SOURCEからN_-+N_2≤4qr。期待数を使うなら、s_-=Σ_{j:s_j<0}p_jとして
`E[N_-+N_2]=qr[(2π_2+4π_4)s_-+π_2]`。
これはcomponent負符号の**確率**の式であり、円周率πとは添字で区別する。今回はH4値を代入評価しない。
実数位相差をoperator差へ移す`|exp(ix)−exp(iy)|≤min(2,dist_2π(x−y))`を使える。
Bodyのglobal phaseはcontrol後にrelative phaseとなるため、`Operator.equiv`等で消してはならない。

既存Cの`semantic_bound`は解析的overcount候補として再利用を検討できるが、操作数・丸めモデル・phase setterの全域対応が必要。
単に保存`TARGET_MET`や大きいoperation countがあるだけで、R2のη証明として受理しない。

## 6. Compiler辺と未観測programの定義

目標familyは、全allowed event列iから**同じ元append→D2 central→各軸wrapper→固定transpile**へ写すalgorithm mapである。
すべてのiで正常終了し、同じqubit/measurement意味を持つnative prefixを返すことが前提。
保存cohortとindex限定controllerは有限run用で、全lawの科学実行を許可するものではない。

現在は未観測event列ごとのcompiled QPYも、全入力を覆うcompiler誤差証明も保存されていない。
「全てのQPYを新生成する」は必須ではなく、固定algorithmに対する一様な証明でもよい。
ただし新しいtemplate/compilerで代用するなら実装familyが変わり、旧価格・旧局所証拠へ無断で統合できない。

固定SOURCEで確認できた主な辺：

1. `default_compiler/transpile_and_measure_cost`はbasis・opt1・seed17を渡す。backend/coupling指定はない。
2. `level1.py`は既定translation/optimization stageを選択する。無coupling/無initial_layoutではlayout/routingを設定しない。
3. `Optimize1qGatesDecomposition.run`は`euler_one_qubit_decomposer.optimize_1q_gates_decomposition`というaccelerated数値処理へ渡す。
4. したがってPythonの置換候補・gate数低減条件だけでは、全run長・全角度・特異点近傍・丸め/threshold・global phaseを覆うoperator上界は得られない。

必要なのは一般compiler全体の巨大な認証基盤ではなく、**このrestricted Pauli入力familyが到達する変換だけ**の数学的保証である。
固定gate definitionsのCRZ/controlled Pauli/native lowering、RZのangle合成・簡約、1Q数値再合成、繰返しoptimization、phase/layout/measurementの保存を覆う。
実際のrun長上限やrewrite回数が必要なら、全native命令数の構造証明を使う。RZ費用の推定モデルや「compilerは減らすだけ」を意味論上界に使わない。
既存Arb bridgeのgate作用丸め上界は**与えられた一つのcompiled programを数値評価する誤差**であり、compilerが入力をどれだけ変えたかというκの一様証明ではない。

## 7. 予算を満たす実用性とorder4

最厳条件の保存exact予算：cosine `19792741170690161541/7631344236317346665332736`、sine `102330934580088554157/30525376945269386661330944`。
それぞれ約2.5936e-6、3.3523e-6。変更しない。
全域boundから得る十分条件は(3)または(5)の右辺≤各保存予算。上界が超えただけでは、実際のρ超過や科学的不適格は証明しない。

**Order4を未評価のまま自明rangeで切り捨てる案は、この予算では不十分。**
F4を少なくとも一回order4が起こる事象とすると、IID下で`w4=1−(1−π4)^(qr)≥π4`。
保存π4=`5852519343/2251799813685248`は`1/400000`より大きく、B_saved>1なので、
`2 B_saved w4 ≥ 2 B_saved π4 > 1/200000 = 5e-6`。
これは両保存予算より大きい。**この粗い十分上界のorder4項だけで既に不足する**という有理数の数学的比較であり、H4の新しいρやw4を数値計算した結果ではない。
実ρが大きいとの意味ではない。Order4の各pathにも、product/sign/phase/compilerの小さい構造上界を与える必要がある。

C→inputには有限operandと解析的Pauli恒等式があり、stateに依存しない低負荷の構成が見込める。
6局所値が非常に小さいことは実装整合性の観測として有用だが、全域のκを小さいと仮定する根拠にはならない。
**現在のopt1でのend-to-end実用性はUNDETERMINED**。大きいH4状態作用・標本拡大を先に行う判断は支持しない。

## 8. 既存コード再利用・人工検証

| 既存資産 | 再利用する範囲 | 保持する限界 |
|---|---|---|
| C `grid_masses/verify_probability_mapping` | 正しいeffective law、今回toy grid検証 | 新H4 CDF評価は未実施 |
| C `finite_implemented_mean` | 非可換M^kの平均式、2〜3qubitで全42toy eventとの比較 | H4 law mean再実行なし |
| PR0 selected kernel／C pauli/phase | exact retained operand・chronology・sign・norm・Arbのscalar演算 | 個別H4 targetを新評価しない |
| D2 `_tokens/_rotation/central_control_body` | restricted gate familyのSOURCE証明対象 | 今回新body/回路/compileを生成しない |
| 既存Arb bridge | 将来のgiven-program評価、gate/kernel誤差転送 | compilerの全域同値保証へ流用しない |
| 旧R2 draft | 同law/B/state・未観測order4・静的scopeの境界 | 旧draftを編集せず、新契約も実行無効 |

新規7＋既存selected-target検証4、**人工11件PASS**。2〜3qubitのtoyのみ。
非可換law factorization／order0,2,4 phase／order4の順序と負符号／law-weighted加法性とB一回／unknown層の非zero扱い／sine +Im／raw norm／有限gridを確認した。
既存4件ではretained exact積とbinary64丸め値の差、native scalar exact sum、共通norm・targetへのB非適用、order4を含む小系の256bit dense比較を再利用した。
このPASSはH4の全域実用性やcompiler保証の証拠ではない。主担当自身の検査で、別agentの独立レビューではない。

## 9. 最小の次段階と非実行契約

[次契約案](../artifacts/non_df_h4_central_r2_design/2026-10-25/next_contract_DRAFT_DISABLED.json)はexecution_authorized/dispatch_enabled=false、科学runなし。
ユーザーのチャット指示が今回の許可であり、承認証明・署名・承認専用ledgerを新設していない。

1. **最優先：restricted compilerとphase setterのSOURCE証明を閉じる。** H4科学call0。上の到達gate族・数値再合成・phase DAG・停止/回数のboundを明示する。閉じなければUNKNOWNを残して設計を止める。任意の「roundoff allowance」を導入しない。
2. **必要な場合のみ：scalar boundの限定包含評価。** 保存27 retained operandの丸め差batchと、全域phase-DAG上界batchの2族。64→96→128bit、未決時だけ、最大6scalar batch・縮約最大3。H4状態作用/compile/samplingは0。先にcompiler上界の証明がなければ全ρを確定できないため、この数値実行を既定の次動作にはしない。
3. Fixed compilerを一様に保証する道が閉じなければ、解析可能なtemplate lowering等への変更を**ユーザーの研究判断**へ戻す。旧価格や既存compiled結果をその新familyの証拠として流用しない。変更・追加compile・samplingは今回の案に含めない。

Scalar評価でも保存値から新しいH4上界を計算するため、**対象・SOURCE・precision・call・STOP条件を確認した別の明示実行承認が必要**。
現時点で不可欠な新しいH4状態作用は特定されていない。Order4の追加局所診断は、具体的なcompiler仮説の反証目的が定まった場合にだけ別設計する。
正式certificate受理・N/G・母平均費用tailは別scopeのまま。

## 10. 保全・Git・終了

旧SOURCE/raw/結果/grant/ledger/marker/STOPを変更せず、新設計資料・人工テストと必要な索引だけを変更する。
確認対象の科学SOURCE・引用ファイルのSHAと、基点との差分で保全を確認する。全依存hashや承認証明の新基盤は作らない。
最新版運用資料の§8.4/10を適用。手元ファイルは§10.2までで§10.3がないため版差を記録し、現在の直接指示を適用した。
CPU/RAM/workerの恣意的上限・共有設定/他process変更なし。静的/人工小系なので新たな重い並列科学jobは不要だった。
技術設計commit後にpush/remote SHAを照合し、公開状態を同じartifact directoryへ記録する。

次の担当：**ユーザー**がcompiler/phaseの限定証明継続または実装family再設計の方向を判断する。
**Codex**は指定された技術範囲の証明・人工検証を進め、scalar数値評価や新科学処理は別の明示承認後だけ実施する。
設計保存・commit/push確認後**mandatory STOP**。追加科学計算・正式受理へ自動移行しない。

## SOURCE参照

- [event法則と演算順序](../src/trotterlib/rte.py)：`_make_event/iter_sample_rte_events/event_unitary`。
- [元Pauli入力](../src/trotterlib/pauli_signal_adapter.py)：`append_pauli_event/trajectory`。
- [保存law exporter](../src/trotterlib/h4_legacy_export.py)：`build_export`。
- [C演算・law平均・semantic項](../src/trotterlib/h4_ball_evidence.py)：`grid_masses/finite_implemented_mean/semantic_bound/_tier`。
- [central合成](../src/trotterlib/h4_d2_pauli_control.py)、[wrapper/compiler設定](../src/trotterlib/fixed_signal_evaluation.py)、[transpile呼出](../src/trotterlib/rte_compiled_cost.py)。
- [selected C-target](../src/trotterlib/h4_arb_pr0_preparation.py)、[given-program Arb](../src/trotterlib/h4_arb_bridge.py)。
- [今回の人工テスト](../tests/test_h4_r2_static_design.py)、[artifact入口](../artifacts/non_df_h4_central_r2_design/2026-10-25/README.md)。
