# 研究B：Outer SplittingのRTE精度・同精度費用の限定検証設計

2026-10-25 JST。**静的調査・未採択の科学実行設計。新しい数値結果・性能順位・実行許可はない。**

提案の中心は、Yoshida4について既存 `(r_RTE,K)=(2,2)` と **(4,2)／(2,4)** を比べ、A2／A3対照でD近似の影響を分けること。その後、同じ単一時間のphysical coherent signal誤差・confidenceを満たす候補について、必要shot数とfull controlled wrapperの期待費用を合わせる。今回行うのはその設計までである。

## 1. 固定起点と引き継ぐ事実

起点は結果公開commit `b1d9f46ee583d86666deb433a4fc4c20dad20225`。独立branch `codex/research-b-outer-rte-cost-design-20261025` で本書だけを追加する。完走科学SOURCE **`b85dd328423de5b1159f11939835f04db90f3187`**、[実行報告][RESULT]、[実行前契約][CONTRACT]、[前段の設計][PREP]、[B2-D結果][B2D]、[静的費用調査][STATIC]を確認した。結果報告と固定PF費用SOURCEはGitHubからも取得した。

以下は既存報告の引用。今回CSV／raw JSONを開いて新しい集計・差分解析・ランキング・fitはしていない。

対象はsynthetic 2-mode、one-body＋3 DF因子、N=1／mode0、L_D=2 weight-ranked prefix、T=±0.18、基本q=2／δ=±0.09、refined-S2のみf=2、D内部A2 Yoshida4／A3 published Y8m10b、n_D=1、r_RTE=2／K=2、complex128、correctness atol=1e-12、phase診断無効。分子geometry／AO basis／分子DF rank探索は非該当。全入力・数学・normalization gate PASS、16行・906 dispatch・490 keyを完了したローカル科学結果で、CI／独立外部再現ではない。[実行報告][RESULT]

| Outer | A2のcorrected physical signal誤差（T=+0.18） | R occurrence／execution | 既存報告からの制約 |
|---|---:|---:|---|
| S2 | 2.26738e-7 | 2 | Outer改善の対照 |
| Refined-S2、f=2 | 5.66296e-8 | 4 | 分割を細かくする対照 |
| Yoshida4 | 6.75687e-10 | 6 | A3では4.52693e-10。D近似とfinite RTEの負担が現れる |
| SABA2 | 2.90506e-9 | 4 | RTEだけの高精度化でouter誤差まで消えるとは期待しない |

Yoshida4のfinite-RTE operator差は約4.89611e-10。A3＋Yoshida4のideal-tail signal誤差は約7.0036e-11、finite-RTE signal差は約4.5818e-10。A2ではD近似差も無視しにくかった。Yoshida4はnormalization積・raw signal誤差も大きい。**Corrected meanの改善をraw精度・shot数・総資源優位と同一視しない。** [同報告 §3–5][RESULT]

## 2. RTEの定義と、r／Kで変わるもの

### 2.1 記号を実装に合わせる

`r_RTE` は**時間分割数**でありTaylor次数ではない。`K=finite_taylor_order` は非負の偶数cutoffで、paired event次数 `n=0,2,…,K` を保持する。通常のTaylor多項式は **K+1次まで**。この区別は `rte.py` の `finite_taylor_operator`／`make_rte_config`／関連testsに基づく。[RTE SOURCE][RTE]

既存のidentity分離後のtailを `R=λ_R H̄_R`、`||H̄_R||≤1` とし、stage jのsigned durationをt_j、分割数をr_j、cutoffをK_jとする。λ_Rはretained involution係数のL1で、DF factorのweight ranking proxyとは別である。

\[
x_j=\lambda_R t_j/r_j,\quad
P_{K+1}(x)=\sum_{\ell=0}^{K+1}(-ix\bar H_R)^\ell/\ell!,\quad
A_j=P_{K_j+1}(x_j)^{r_j}.
\]

既存lawのpaired weightとnormalizationは

\[
w_n(x)=\frac{|x|^n}{n!}\sqrt{1+\frac{x^2}{(n+1)^2}},\quad
B_K(x)=\sum_{n=0,2,\ldots,K}w_n(x),\quad
N_j=B_{K_j}(x_j)^{r_j},\quad N=\prod_jN_j.
\]

Raw meanはA_j/N_j。独立occurrenceをD列の間にorderedに挿入し、`M_corr=N M_raw` とする。各物理occurrenceを積に残し、同じspecの計算共有によって重複因子を落とさない。`compose_finite_rte_occurrences` はtailだけの積であり、D interleavingは既存outer graph側で扱う。[RTE SOURCE][RTE]

Signed timeはrotation角・meanに保持する。N_jは|x_j|で決まるが、負時間を正時間に置換してよいわけではない。有限多項式は一般にnonunitaryで、A_j(−t)をA_j(t)のinverseとはみなさない。Freshな負時間sampleとactual inverseは別である。Scalar／event relative phase、control-zero branchを保つ。

### 2.2 調整方針

| 変更 | 精度・normalization | 実行費・注意 |
|---|---|---|
| r:2→4、K=2固定 | 各short-stepの|x|が小さくなる。有限Taylor切捨てとnormalization上包絡を下げる方向 | 各occurrenceのevent数が増える。Basis共有後のfull費用はevent数比例とは限らない |
| K:2→4、r=2固定 | 通常Taylor保持が3次→5次。固定xでは正のweightを追加するのでB_K、N_jは非減少 | 高次数eventのsupport／確率・長さ・期待費用が変わる。精度向上とshot range増加を別に評価 |
| q増加、同じT | Outer幅と各R durationが小さくなり、D内部のtau・finite-RTE biasも変わる | D stage・R occurrenceが増える。同じrを各stageに使うとtotal eventも増える |
| f増加（S2対照） | 同じTでS2の分割を細かくする | qfの実step数を数える。S2でqだけを増やした同一列を別候補として二重登録しない |
| D内部A2→A3 | D由来の差を切り分け、outer／RTEの残差を見やすくする | 長いD列を支払う。RTEの精度・Nそのものを改善する変更ではない |

`step_taylor_truncation_residual_bound` は最初の省略次数K+2以降のscalar exponential tailを使い、occurrenceは `(1+ρ_j)^{r_j}−1`、不均一列は `∏_j(1+ρ_j)^{r_j}−1` で合成する既存接口がある。Dがunitaryで同じordered productであること等を保って適用する。小さい|x|での局所的な先頭スケールは `O((λ_R|t_j|)^{K_j+2}/r_j^{K_j+1})` だが、full signalの実誤差・符号・単調性を保証しない。[SOURCE][RTE]

既知のinvolutionモデルでは `N≤exp(Σ_j λ_R² t_j²/r_j)` というnormalization上包絡がある。これはnative費用や実varianceそのものではない。有限Kの実Nは実装のB_Kから求め、上包絡と区別する。Infinite RTEのunbiased性をfinite Kへ無条件に移さない。[原論文 Appendix A.2–3、(A23)–(A36)][PAPER]

明示Kを渡す場合、`truncation_tolerance=1.0` は実装metadataでありphysical誤差目標ではない。`choose_finite_taylor_order` の一short-step用toleranceへ、全signal予算をそのまま渡さない。次回もKを明示し、stage別boundと実biasを併記する。今回、bound・N・event確率・新しいmeanの数値評価は行っていない。

### 2.3 方式別の問い

- **Yoshida4：** 長い正負stageでtail負担が現れたため、r増加とK増加を一つずつ比較する価値がある。A3はtail改善の余地を見る対照、A2は短いDの実用候補。A2のD誤差が残ればRTEだけを改善してもfull biasが下がり切らない可能性がある。
- **SABA2：** 既存報告ではfinite-RTE差が比較的小さく、まずouter qを一段増やす候補を置く。RTE条件を一律に大きくする全探索はしない。弱摂動性のnorm／commutator診断は保持するが、一般四次や自動適用閾値は認定しない。
- **Refined-S2：** 細かい分割で高次法と同じbias目標へ届くかが費用比較の要点。RTE高次数化だけでouter誤差を解決しようとせず、分割数を一つ追加する対照を提案する。到達不能なら候補集合内で未達と報告し、無制限探索しない。

## 3. 同精度比較の対象と予算

### 3.1 単一時間のphysical signalを共通targetにする

`z_phys(T)=⟨ψ|exp(−iHT)|ψ⟩`、`z_c(T)=⟨ψ|M_corr(T)|ψ⟩` と定める。同じH、split、ψ、sector、T、scalar、X/Y規約で、最終推定量ẑについて `Pr(|ẑ−z_phys|>ε)≤α` を目指す設計とする。これは単一時間の複素絶対誤差であり、energy誤差、RPE位相誤差、全round confidenceとは別である。

| 誤差層 | 定義と資格 |
|---|---|
| Operator直接誤差 | `||M_corr−U_phys||₂`。正規化ψのsignal biasを上から抑えるが、特定状態の直接biasと同値ではない |
| Algorithmic signal bias b | `|z_c−z_phys|`。Primaryのdeterministic bias。Outer、D近似、finite-RTE差は既存5対象の中間参照で診断 |
| 数値的不確かさ u_num | complex128参照・独立列照合の数値的識別限界。correctness atolをそのまま信頼区間／certificateにしない |
| 実装誤差 γ | 実際のfinite-bit law、回転合成、controlled phase、normalization近似等によるcorrected meanのずれ。未取得を0としない |
| 統計誤差 s | Fresh IID測定のX/Y推定誤差。Biasとは別枠でconfidenceを設定する |

`b+u_num+γ+s≤ε` の予算を契約時に確定する。Outer／D／RTE差の三角不等式は保守的な補助評価に使えるが、norm値を差し引いて成分を推定したり、誤差の相殺を独立の保証へ変えたりしない。

**誤差目標候補はε=1e-8、1e-9の2件。** 前者は既存報告でSABA2とYoshida4が同じ精度領域に入るかを問う値、後者はYoshida4のD／tail負担とA3対照の価値を識別するための値である。既存値の引用を設計動機とするだけで、正式採択・全候補の適格認定はしない。値は化学精度や特定装置への実用性の主張ではない。

初回の公平な報告案はalgorithmic biasと数値的不確かさを合わせた枠 `(b+u_num)≤ε/2`、implementation枠ε/10、統計枠2ε/5を全方式共通にする。Confidenceは例えば各返却complex signalでα=0.05を候補とし、X/Yへα/2ずつ。T両符号・候補選択を含む同時主張をするなら、その単位を明示してfailure budgetを追加配分する。**いずれも未採択で、shot数の数値計算は今回しない。**

固定配分は比較を読みやすくする基準。その次に、事前登録集合内で方式ごとのr／K／step／D精度を選び、`s=ε−(b+u_num)−γ` の残余予算を使う評価も可能とする。ただし両方式に同じ予算規則・実装誤差の資格を適用する。観測後に一方式だけ余裕を与えたり、不利な候補だけ追加探索したりしない。

実装γが未資格なら、まずanalytic biasと構造metadataだけのpilot結果とし、end-to-end同精度保証は未確立と表示する。Raw meanの合成／law誤差をγ_rawで評価する場合、corrected側にはN倍の影響がある。Normalizationの近似誤差も別に含め、旧compilerの受理tolをそのまま新signal予算へ流用しない。理想連続回転／理想samplerを仮定した参考モデルを出す場合はその仮定を明記し、実装資格と混ぜない。小さい目標による大きな統計負担が判明しても、phase／energy targetへ黙って変更しない。

### 3.2 Shotとconfidenceの方法（今回数値計算なし）

固定candidateでは全stageのduration／r／Kが決まり、Nは各shotで同じ既知の正数。X/Yそれぞれのraw測定結果を `O_a∈{−1,+1}` とすれば、corrected estimatorは `N·mean(O_a)`。

\[
\operatorname{Var}(N\bar O_a)=\frac{N^2\operatorname{Var}(O_a)}{m_a},\qquad
\Pr(|N\bar O_a-N\mu_a|>u_a)
\le 2\exp[-m_a u_a^2/(2N^2)].
\]

従って `m_a≥ceil[2N² log(2/α_a)/u_a²]` とし、`u_X²+u_Y²≤s²`、`α_X+α_Y≤α` にすれば、複素統計誤差をunion boundで扱える。これは独立・有界測定の既知Hoeffding関係を今回のcorrected estimatorへ適用する設計で、新しいPR誤差保証ではない。[Hoeffding (1963)][HOEFFDING]、[PR論文のnormalization／測定関係][PAPER]

Worst-caseでは `Var(O_a)≤1`。実varianceはraw mean等にも依存し、Nの大小だけでは決まらない。RTE trajectoryと量子測定を合わせたfresh IID条件を使う。固定trajectoryを多shot再利用すると条件付き平均とensemble平均を分ける必要があり、同じ式を無条件流用しない。新外側乱数でN自体が可変になる場合も別のweighted range／moment解析が必要で、本案には追加しない。

初回は `u_X=u_Y=s/√2` を共通とし、その後は同じ制約下で `m_X E[C_X]+m_Y E[C_Y]` を最小にする軸配分を検討できる。Worst-case boundとvarianceを用いる推定、必要shot数と実shot simulationは区別する。Confidenceを満たすには後者を必ず実行するわけではない。

`finite_rte_signal_validation._hoeffding_shots` は半径・phase budget経由の既存式であり、今回の複素絶対誤差をそのまま引数へ入れる関数ではない。`rpe_hadamard_failure_validation` のbinomial／Clopper–Pearson／fresh seed処理は将来の検証資産だが、RPE round構造や旧shot数を今回へ移送しない。新しいshot計算・simulationは現在0件。

## 4. 費用の内訳と同精度の目的関数

| 層 | 記録する項目 | 別扱いにするもの |
|---|---|---|
| D evolution | 内部family／n_D、実現kernel列、D run、block指数、合法融合、basis往復、signed角度 | Literal指数数はnative価格でない。異tau近似を時間和の一つの近似にしない |
| RTE tail | R occurrence、各t_j／r_j／K_j、event次数・確率、basis共有、phase、N_j、Taylor bias | 同じtailでも異durationのlaw・期待費用を共通値にしない |
| Processor／control | 今回A2／A3はprocessorなし。Controlled diagonal、scalar／event relative phase、control-zero、ancilla処理 | A4を追加しない。System-only同値性をcontrolled相対位相の検証に代用しない |
| Wrapper | 同じψの準備、Hadamard X/Y、measurement、native gate、depth、必要ならreset | 準備・測定を候補によって無料にしない。装置時間が未取得ならgate価格と分ける |
| 推定 | Nで補正したrange、bias予算、m_X／m_Y、confidence、可変trajectory費用 | Classical compile標本数とquantum shot数は別 |

将来の費用計測の第一指標案は、固定compilerでの **shot加重expected RZ count** とし、CX、RZ depth、CX depth、total depth、circuit sizeを併記する。これは論理回転ベースのproxyで、Clifford+T総費用の認定ではない。RZにはClifford角等もあり、非Clifford回転の分類は角度・簡約規約を明示して別記する。T-countを求めるなら離散gate合成法・精度・位相保存を別途固定する。[Qiskit 1.3の離散合成接口][QISKIT]

候補c、精度εについて期待総費用の形式は

\[
G_c(\epsilon,\alpha)=\sum_{a=X,Y}m_{c,a}
\,\mathbb E_{\omega\sim p_c}[C_{c,a}(\omega)],\qquad
G_{\mathcal M}^{\min}=\min_{c\in\mathcal C_{\mathcal M}:\,qualified}G_c.
\]

Cは同じ入力準備＋full controlled evolution＋measurementを含むscopeとする。準備費を除く参考表も出せるが、別scopeと表示し、最終比較ではshot数の異なる候補に共通準備費が掛かることを残す。深さは一回路の量であり、m×平均深さは逐次実行時のwork proxyとしてのみ記録する。並列wall timeや最大回路深さと混同しない。

固定shotsならcostと測定outcomeがtrajectory内で相関していても期待費用はm E[C]でよいが、outcomeに応じる停止／adaptive shotsは別の契約になる。Compile時間・古典前処理費は量子資源表とは別に記録する。MCでE[C]を測る場合はその標本誤差・confidenceをquantum estimationのαと別に予算化し、区間が重なる費用順位は保留する。

最小費用を主張できるのは**事前登録した有限候補集合内**だけ。候補が全て未達なら `NO_QUALIFIED_CANDIDATE_IN_REGISTERED_SET` とし、方法一般のNO-GOや無限費用とは呼ばない。費用未計測の候補を0円として選ばず、native価格未取得なら構造的tradeoffの報告にとどめる。

## 5. 調査した既存SOURCEと再利用経路

PRの固定参照は起点b1d9f46e版。リンク先の科学SOURCE・他研究結果を変更・移植したことにはしない。

| SOURCE／tests | 再利用できるもの | 今回への不足・制約 |
|---|---|---|
| [b2d_evaluator/outer.py、outer_backend.py、evaluation.py、dense_backend.py][OUTER] | 4 outer／A2-A3列、ordered5対象、D／outer／normalization gate、共通ψ signal、expm cache、weak診断 | Graphの共通Settingsはr／K一組。次のcase別r／Kを同時に扱う最小拡張が必要 |
| 同 `outer_execution.py`／`input_package.py`／`launcher.py`／scope | InputRecipe、SOURCE／input binding、fresh output、失敗保存、実績／STOP | Registryは旧16行固定。新候補集合とstage-specへ固定し直す必要あり。旧906／490を流用不可 |
| [rte.py][RTE]、`tests/test_rte.py` | 既存config／law／moments、signed phase、heterogeneous N、truncation bound・composition | 科学importを伴うので今回は静的読取りのみ。Testsも今回実行していない |
| [df_partial_s2.py][DF] の `_append_block`／basis／phase、diagonal primitives | 各D blockのbasis inverse→diagonal→basis、signed evolution_time、diagonal-only control | request／builderはpartial-S2用、任意outerとstandalone D内部PFのrendererには最小glueが必要 |
| [df_rte_circuit.py][RTECIRCUIT]／[df_rte_qiskit.py][DFRTE] | DFRTEEventPreparation、stageごとのoccurrence request、event順、basis共有、product符号とrelative ancilla phase | 新outer内での全trajectory・controlled列は未検証。負時間meanをinverseにしない |
| [rte_compiled_cost.py][COMPILE] | `transpile_and_measure_cost`、CompilerSettings、numeric-angle cache、CompiledMetricAccumulator、exact／MC event価格、occurrence非加法性の計測 | Event単価の加算だけではfull-wrapper境界最適化を反映しない。別DF／分子の旧価格は移送不可 |
| [df_partial_s2_repeated_cost.py][REPCOST]、[df_rpe_hadamard_compiled_cost.py][HADCOST] | Full repeated trajectory価格、X/Y同一trajectory、期待値重み、MC統計、workload planning | 一つの共通時間幅／RTE configと旧partial-S2前提。Heterogeneous outerへ無変更では使えない |
| [rpe_hadamard_interrogation.py][WRAPPER] | H–controlled evolution–H、YのSdg→H、bit→±1、phase保持 | 強いDFPartialS2RepeatedCircuitResult型、短q=1,2,4、状態準備を除くscope。任意outer／refined長stepへの意味論を明示して最小一般化する必要 |
| [pr_dictionary_validation/circuits.py、gates.py][GATES] | Actual adjoint、occupation準備付きwrapperの構造、compiled operator／relative-phase照合、既存Qiskit1.3のphase保持lowering | Dictionary Dは別モデルのoccupation演算、列も旧sampling law。`build_evolution` を今回の非可換DF-Dへ直接流用不可 |
| `tests/test_df_rte_qiskit_builder.py`、`test_df_partial_s2_cost.py`、`test_df_partial_s2_repeated_cost.py`、`test_rpe_hadamard_interrogation.py` | Signed event／controlled diag(I,U)、basis共有、全trajectory重み、cache、X/Y符号、phaseの検査箇所を確認 | 回路生成・operator評価を含むtests。名前に人工／regressionとあっても今回実行しない |
| `tests/management/test_pr_dictionary_compile_phase.py` | 既存位相欠落への回帰・official definition展開の再利用候補 | NumPy／Qiskit／Operatorを実行するtestで、本作業の許可外。旧許容差を新精度予算へ無条件流用しない |
| [PF processed_cost.py][PFCOST]／[pf_decomposition.py][PFDECOMP]、固定20d64c2 | S2列・同term融合、kernelと端点費の分離という考え方 | H-chain Pauli/RZ-layer表をnative DF価格にしない。今回A4なし、processor費は0と明示する |

既存RZ/CX compilerとphase保持処理を一つの固定profileへ揃える。旧プロトコルのprofile・価格・成績が別synthetic入力でそのまま有効とはしない。全Fock空間のcontrolled-zero branchとrelative phaseを検査し、N=1 sectorでの信号一致だけをwrapper正当性の代用にしない。

全event／trajectory列挙はKやrで急増し得る。個別event空間はcomponent数Lに対し `Σ_{n even≤K}L^{n+1}`、full trajectoryは各stageのevent選択の積となる。将来のbuild／compile契約は実metadataで展開規模を確認し、小さければexact weighting、重ければ既存MC／層別計測を再利用する。Full-wrapperをcompileする少数の実trajectoryを必要とし、古いevent加算proxyだけで確定費用としない。具体標本数・compiler budgetは採択時に固定し、今回は数値計画を実行しない。

## 6. 次の最小科学検証候補（未採択・未承認）

第一段階は**analytic bias・normalization・構造的費用の限定診断**とする。Shot計算・native合成を同時に無条件開始しない。共通条件はStage 1と同じsynthetic入力、N=1／mode0、L_D=2、T=±0.18、n_D=1、phase診断無効、complex128／correctness atol1e-12、既存係数・identity policy・independent analytic moments。新PF・sampling law・別fixtureは提案しない。

| 組 | Outer／D内部 | q／f | r_RTE／K | 条件数（Tを除く） | 問い |
|---|---|---|---|---:|---|
| B | S2、refined-S2、Yoshida4、SABA2／A2 | q=2、refinedのみf=2 | 2／2 | 4 | 同じ新実行での共通対照。旧STOP済みrunをresumeしない |
| D | Yoshida4／A3 | q=2 | 2／2 | 1 | D誤差を小さくしたtail対照 |
| R | Yoshida4／A2、A3 | q=2 | **4／2、2／4** | 4 | rだけ／Kだけの増加、D精度との相互作用 |
| S | SABA2／A2 | **q=4** | 2／2 | 1 | RTEよりouter分割を改善する一段の候補 |
| F | Refined-S2／A2 | q=2、**f=8**（実16 step） | 2／2 | 1 | 高次候補と同じ小さいbias目標へ近づけるかを問う強いS2対照 |

**計11条件×2 T＝22行の案**。これはcase数の設計であり、新しいgraphや数値を生成した件数ではない。f=8は同精度対照のための粗い追加挑戦点を一つに絞った提案で、目標到達の予測・採択ではない。f=4／f=3や追加qを同時に並べない。S2をq=4にする列はBのrefined f=2と同じなので二重登録しない。

A3はYoshida4の3つのRTE条件だけに限定し、SABA2／refinedの全Cartesian集合へ広げない。RTEの(4,4)、stage別最適r／K、n_D=2、A4、別時刻は初回集合から外す。全signed R stageに、そのcaseのr／Kを明示適用する。Yoshida4の中央負durationとN_jの重複を残す。

この集合は等しい探索機会の全網羅ではなく、既存報告で見えた誤差要因を切り分ける小集合。各方式には同じ目標・予算規則を課し、登録した当該方式の集合内で選択できる。集合の大小・探索範囲を併記し、方法一般の最適性を主張しない。現集合でstrict目標を満たすSABA2／S2候補がなくても、無理な外挿・自動追加はしない。

### 最小契約の評価とSTOP

Primaryはcorrected physical complex signal bias、secondaryはoperator誤差、既存5対象の中間差、N_j／N、raw mean、実stage時間和、R／D構造metadata。ε候補の適格性は登録した予算規則で記述し、roundoffと同程度ならINCONCLUSIVE。単一|T|の結果を収束次数確認としない。
入力identity・非可換性、D内部独立数学列、未融合exact-D/R数学参照、signed RTE config、raw／corrected normalization、全correctness先行を維持する。Weak診断は閾値なし、correctness atolは科学的改善閾値やfinite-bit certificateでない。
Fresh namespaceを新たに定め、SOURCE・入力・全条件・環境version・実績・部分失敗・STOPを保存する。Retry／resumeなしを基本案とし、復旧例外はその実行許可で明示する。以前の技術復旧許可を将来の別gridへ流用しない。完了・FAIL・INCONCLUSIVE時はmandatory STOPし、native／shot段階へ自動進行しない。

## 7. 新scopeと実装準備の見通し

今回はSOURCE・launcherを変更しない。採択後の最小実装は、既存caseにr／Kを対応付ける設定と、登録集合を差し替えた新versionのregistryで足りる見込み。`build_outer_graph`／Input package／numeric backend／Evidenceを優先して再利用する。大きな汎用費用frameworkを作らず、旧16行registry・結果・STOPを保全する。

具体的なdispatch・exponential・RTE config数は、**採択gridと新SOURCEを固定してから**metadata展開し確定する。旧906 dispatch／490 key／80 expectationを転用しない。構成手順は次の通り。

1. 22行案の採否と各caseのT／outer／q／f／inner／n_D／r／Kを固定する。
2. Exact physical keyは入力identity＋Tで共有。D block keyは入力・block・実signed時間、独立参照にも必要な時間を含める。
3. RTE moment keyはtail・signed duration・r／K・tolerance・lawで識別。独立occurrenceのmetadataと計算共有を分け、N_j積を保持する。
4. 入力10高位境界とGaussian／sector処理の内包範囲、独立D／outer／rescaling、5対象signal、今回追加するTaylor bound評価の境界を登録する。新しいbound関数を追加した分も記録し、未計数を架空値で補わない。
5. Node全体・共有key・callsと保存schema／sizesをmockで照合し、fixed SOURCE・fresh outputを決める。Input matrixやactual r／Kの評価は実行許可前にしない。

Shot／native段階ではさらにfull controlled列のrenderer、stageごとの既存RTE request、wrapper、phase正しさ・compiled正しさ、trajectory law、event／build／compile標本数、費用SE、confidence配分を登録する。そのscopeはanalytic graphのcall数では表せない。Shared environmentの変更なしで空き資源を確認し、重ければcache・独立task並列化を使い、恣意的なCPU／RAM／worker上限を置かない。

## 8. 未確認事項、今回の実施内容、次の担当

**未確認：** 新r／K／q／fでのbias・N・目標到達、D内部をA3へ変える費用利益、full outerのcontrolled／compiled相対位相、実native価格・費用分布、必要shot数、finite-bit／合成誤差予算、同精度総費用順位。旧Stage 1で未評価だったものを今回の式やSOURCE読取りだけで検証済みにしない。

今回実施したのは固定報告・SOURCE・必要な一次資料の静的確認と本設計の文書化。実Hamiltonian／operator／signal／新RTE config／bound／N／shot数の数値評価、旧CSV／rawの数値再解析、回路build／compile／shot simulationは**0件**。コード変更がないため科学test・回路回帰を起動せず、文書リンク・整合だけを確認する。人工PASS・科学SOURCEの新固定は主張しない。旧SOURCE・raw・grant／ledger／marker・STOP、共通運用指示は変更しない。

**研究Bのユーザー：** ①22行案の集合、特にf=8／SABA2 q=4とA3対照の範囲、②ε=1e-8／1e-9とbias・統計・実装の配分、③confidenceの対象単位、④初回はanalytic診断のみとするかを採択する。
**Codex：** その具体的指示後、既存SOURCEへ必要な設定接続・mock確認・新scope固定をまとめる。ユーザーが固定条件での科学実行を明示的に許可した場合だけ計算する。次にnative計測／shot評価へ進むには、compiler・費用scope・実装誤差・標本／confidenceを具体化し、その範囲の別途許可を得る。
承認原文の保存・grant発行・packet等は不要。本書だけをcommit・pushし、remote SHA・本文を照合後 **mandatory STOP**。研究統括への通常報告やStage 2／全RPEへの自動移行はしない。

## 参照

[RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/docs/research/research_b_outer_splitting_stage1_execution_20261025.md
[CONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d29eea6617ab44c0fd4d41ae220359953ead4824/docs/research/research_b_outer_splitting_stage1_execution_ready_20261025.md
[PREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e46050bd85e2df86c980e501f727ff56d10823e/docs/research/research_b_outer_splitting_stage1_preparation_20261025.md
[B2D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/docs/research/b2d_stage1_execution_20261025.md
[STATIC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f5d60122f158b3320fadab86498963950f00734/docs/research/research_b_post_b2d_stage1_static_cost_outer_splitting_20261025.md
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/src/trotterlib/rte.py
[OUTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b1d9f46ee583d86666deb433a4fc4c20dad20225/src/trottertracks/b2d_evaluator
[DF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/src/trotterlib/df_partial_s2.py
[RTECIRCUIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/src/trotterlib/df_rte_circuit.py
[DFRTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/src/trotterlib/df_rte_qiskit.py
[COMPILE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/src/trotterlib/rte_compiled_cost.py
[REPCOST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/src/trotterlib/df_partial_s2_repeated_cost.py
[HADCOST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/src/trotterlib/df_rpe_hadamard_compiled_cost.py
[WRAPPER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/src/trotterlib/rpe_hadamard_interrogation.py
[GATES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b1d9f46ee583d86666deb433a4fc4c20dad20225/src/trottertracks/pr_dictionary_validation
[PFCOST]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/processed_cost.py
[PFDECOMP]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/pf_decomposition.py
[PAPER]: https://arxiv.org/pdf/2503.05647v2
[HOEFFDING]: https://doi.org/10.1080/01621459.1963.10500830
[QISKIT]: https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/qiskit.transpiler.passes.SolovayKitaev
