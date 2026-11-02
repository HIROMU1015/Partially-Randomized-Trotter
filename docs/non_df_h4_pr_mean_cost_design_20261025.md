# H4 Pauli central／DF-native PR：母平均回路費用の追加検証設計

2026-10-25。**有望なのは、Pauliの決定論的な全trajectory費用上界と、DFの統計的な母平均下界を組み合わせる方法である。** DF最大費用や全trajectoryのcompileを初手に要求する必要はない。既存SOURCEにはこの非対称な比較を構成する具体的な経路がある。ただし、H4へのscalar代入と信頼水準の選択は今回は実施しておらず、母平均の順位を新たに確定していない。

基点branch `codex/non-df-h4-data-catalog-20261025`、commit `7a83dc186bddcc35e528e38032814423ef1cba88`。GitHubから同refを取得・照合した。専用branch `codex/non-df-h4-pr-mean-cost-design-20261025`。今回は静的調査・数学的設計・人工count検証のみ。H4 scalar評価、保存値の新統計解析、N/G再解析、回路build/compile、sampling/RNG replay、状態作用はすべて0。R2-Cの数値誤差証明は継続していない。

## 1. 対象・保存証拠・未解決事項

固定linear H4／隣接1.00 Å／STO-3G、system8＋ancilla8、S2／T=.8／q8／r4／K4。Pauli-JW184非identity項・PR prefix27、DF-native tol-only1e-8/rank10・PR prefix1＋one-body。Qiskit1.3.0、basis `rz/sx/x/cx`、opt1、seed17、backend/coupling/initial_layout指定なしを対象とする。PFやPauli genericは今回の母平均価格比較へ混ぜない。

費用変数を `C_R,a(ω)`＝表現Rのtrajectory ωの**full Hadamard wrapper compiled RZ数**と定義し、`μ_R,a=E_pR[C_R,a]`を目標とする。state準備を除く既存の指標で、CX/depth等の換算はしない。pRは各表現固有のeffective event law。Bは信号の尺度とshot会計に入り、回路価格の期待値にもう一度乗じるものではない。

入口は[統合カタログ](../artifacts/non_df_h4_data_catalog/2026-10-25/README.md)、[catalog.json](../artifacts/non_df_h4_data_catalog/2026-10-25/catalog.json)、[comparison_data.json](../artifacts/non_df_h4_data_catalog/2026-10-25/comparison_data.json)、[gaps.csv](../artifacts/non_df_h4_data_catalog/2026-10-25/gaps.csv)。指定資料を読み、次の原保存fieldとも対応させた。[今回の引用JSON](../artifacts/non_df_h4_pr_mean_cost_design/2026-10-25/evidence_quotes.json)は元path・JSON pointer・snapshot commit・SHA・GitHub URL付きの転記だけである。

| 保存量 | Pauli central PR | DF-native PR | 元証拠 |
|---|---:|---:|---|
| 価格標本数 | 256 | 256 | central `statistics.json:/PR/n`、旧 `cost_summary.json:/df_native_PR/n` |
| 両軸の標本平均RZ | 1140.8046875 | 23678.44140625 | 同上 `/axes/{axis}/mean` |
| 保存SE | 1.059925039945424 | 46.718033123081106 | 同上 `/axes/{axis}/standard_error` |
| 保存nominal95半幅 | 2.0873215844123516 | 92.00231643184831 | 同上 `/axes/{axis}/nominal95_halfwidth` |
| 保存標本min／max、両軸 | 1097／1192 | 21754／25962 | central `/PR/axes/{axis}`、旧 `/df_native_PR/observed_min,observed_max` |
| 観測order0／2／4 event | 8157／35／0 | 8167／25／0 | `rare_cost_sensitivity.json:/{case}/order_event_counts` |
| any-order4 trajectoryのeffective確率 | 8.316595170215553e-5 | 3.481583274276441e-5 | 同 `/implemented_probability/any_order4` |
| DF未観測componentを含むno4 trajectory確率 | 0（未観測component集合が空） | 0.1001190406163259 | 同 `/implemented_probability/no4_with_unseen` |
| 全wrapper母平均upper | 未保証 | 未保証 | 同 `/valid_whole_wrapper_upper=null`、旧cost summaryもnull |

central価格原本：[A_AXES statistics](../artifacts/non_df_h4_a_axes/2026-10-24/run01/statistics.json)。DF価格原本：[旧cost_summary](../artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/cost_summary.json)。希少event原本：[rare_cost_sensitivity](../artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/rare_cost_sensitivity.json)。Pauliのlawはgeneric/centralで同じだが、価格はcentral原本だけを使う。旧rare JSONのPauli価格欄はgenericなので転用しない。

両表現とも保存256本ではcosine/sine価格が対応し、保存共分散を利用できる。一方、未観測trajectoryでも両軸価格が同一になるとは証明されていない。表現間の同じindexはpaired標本ではない。SE／nominal CI／観測最大は母平均の厳密上界ではない。

Pauliでは全componentの観測と全trajectoryの被覆は別である。DFの未観測component集合Uは保存資料で164成分、effective質量は `3687017268433/1125899906842624`。集合は**同じ標本から事後選択**され、effective mass=0の成分もあり得る。「Uを一度も観測しなかった独立試験」とは扱えない。no4-with-Uの保存質量は、その確定した集合をlawへ代入した記述値である。

未観測order4の価格、未観測DF componentを含むwrapper価格、および通常eventを含む残りの母平均価格は未確定。保存 `stratum_population_mean` は全層nullである。過去の `rank_flip_sensitivity` はremainder平均＝保存標本平均とした仮想条件付き感度で、母平均の保証条件へ昇格させない。

## 2. 正しいeffective law

`rte.py`の有限paired Taylor lawはk∈{0,2,4}。各eventはk個のsigned Pauli／DF involution productの後にrotationを一つ作用させ、order phaseは `(-1)^(k/2)`。PFのS2次数とは別で、H2/Yoshida4系列とも独立である。

保存binary64確率をsequential cumsum・terminal divisionでCDFへ写し、53-bit uniform wordと比較した区間長がeffective確率になる。対応する既存コードは `h4_ball_evidence.py:cdf_binary64/grid_masses/verify_probability_mapping`、保存質量の引用元はrare JSON、保存済みの希少事象式は `h4_conditional_cng.py:rare_law`。いずれも今回は呼び出していない。

| k | Pauli effective mass（exact） | DF effective mass（exact） |
|---|---|---|
| 0 | 2242933535092243/2251799813685248 | 8984229315059167/9007199254740992 |
| 2 | 4430213036831/1125899906842624 | 717504366589/281474976710656 |
| 4 | 5852519343/2251799813685248 | 9799950977/9007199254740992 |

nominal order確率はPauli `[.9960625813453219,.003934819613987418,2.5990406907583435e-6]`、DF `[.9974498244090985,.002549087577779853,1.0880131215414273e-6]`。これをexact effective massの代用にしない。component確率も同じCDF規約であり、成分vectorは原本参照とする。

M=qr、order確率p̃_k、component確率π̃_jとすると、既存のIID uniform-word product-lawモデルのもとで

`p(ω)=∏_e [p̃_{k_e} ∏_{l=1}^{k_e+1} π̃_{j_e,l}]`。

allowedはeffective確率が正のsupportで定義する。`P(any4)=1−(1−p̃_4)^M`。独立に固定されたUの質量uについては

`P(no4 with U)=(p̃_0+p̃_2)^M−[p̃_0(1−u)+p̃_2(1−u)^3]^M`。

式自体は固定集合に成立するが、保存Uの事後選択に対する信頼区間を与えるものではない。通常部分の母平均を保存平均と置くこともできない。

既存SOURCEは固定件数の保存・price/seed/event joinを確認する（`h4_pilot_science.py:run_case/integrate`、`fixed_signal_evaluation.py`）。その技術的整合性はPCG64出力間の確率的独立性の証明ではない。保存資料の `PCG64_independence_proved=false` を維持する。以下の統計的boundsはこの**明示されたIIDモデルに条件付く**。

## 3. Pauli側：全trajectoryに成立するcount upperの設計

### 3.1 既存の整数count補題を再利用する範囲

[R2-C報告§4.1–4.3](non_df_h4_r2c_compiler_phase_proof_20261025.md)と[保存Qiskit SOURCE](../artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/upstream_sources.json)のうち、**gate数の探索・採否**だけを再利用する。operator norm、libm、Euler数値誤差、phase丸めの証明は続けない。

- 固定Pauli central入力はH/S/Sdg/CX/CRZ/CY/CZ/P＋measurement。Target・layout・routingなし。UnitaryGateや高位の任意unitaryを含めない。
- `basis_search.rs`はnative gate cost=1、rule内cost和の探索。正常な最短cost探索・非overflow・固定libraryを前提として、選択したnative分解の個数は明示的な候補経路の個数以下。同cost ruleでも個数の上界は保たれる。
- `euler_one_qubit_decomposer.rs:compute_error_from_target_one_qubit_sequence`はTargetなしの場合 `(gate数,gate数)` を返す。native runの再合成はnative basis内、採用は総gate数の減少。inverse cancellationは削除のみ。したがって `C_RZ,out ≤ native総数_out ≤ native総数_initial`。
- **RZ数自体の単調減少は仮定しない。** 角度の丸めによりRZの出現が変わっても、正常に完了する固定count経路なら総native数によるupperを使える。

これは正常終了する固定compiler familyの条件付き構造補題。SOURCE/wheel対応、標準libraryの候補経路・入力gate閉包、例外・非有限処理を含む全allowed入力の定義を最終boundへ接続する静的確認が残る。正常完了しない入力に架空の価格0を与えない。構造countの不足を、全compiler数値誤差証明が必要という条件に拡大もしない。

### 3.2 Z/ZZ保持部を使う記号的な改善

`h4_d2_pauli_control.py:_rotation/central_control_body`を直接確認した。単一Pauli label jのsupportをs_j、X/Y/Z数をx_j,y_j,z_jとする。既存libraryでH≤3、S/Sdg≤1、CRZ≤4、CY≤3、CZ≤7、P≤1 native gateという候補経路がある。basis/parityを非制御とするcentral実装の一回のrotation／productについて

`v_j = 6x_j+8y_j+2(s_j−1)+4 ≤10n+2`、
`w_j = x_j+3y_j+7z_j ≤7n`。

vはXのH対、YのSdg/H/H/S、往復parity CX、CRZの和。wはlocal controlled Pauli productの和。演算順序・負符号・order phaseは保持し、符号等を集約したbody phaseはancilla P一つとして別途数える。relative phaseを無視して価格を下げない。

Pauli prefix27がZ/ZZだけであることは[既存構造報告](non_df_h4_cost_reversal_structure_20261024.md)・[統合報告§4](non_df_h4_comparison_synthesis_20261025.md)の保存証拠で確認済み。この保持labelでは `v_l≤6`。Lを保持項数として

`A_a = 2q Σ_{l≤L}v_l + 7 + 1_{a=sine}`。

7はaggregate Pとwrapper両Hの候補個数。sine Sdgをさらに一つ含める。全trajectoryで

`C_P,a(ω) ≤ A_a + Σ_events [v_rotation + Σ_products w_j]`

なので、一様な候補upperは

**`U_P,a^uniform = 12qL + qr[(10n+2)+7nK] +7+1_{a=sine}`。**

これは既存R2-Cの粗い個数式の保持部だけをZ/ZZ構造で絞った**記号式**。固定H4 operandを代入して値を出していない。保存1192をUとする案ではなく、order4や未観測の非可換productも含む。

さらに、`ν=Σ_k k p̃_k`、`v̄=Σ_j π̃_j v_j`、`w̄=Σ_j π̃_j w_j`なら

**`μ_P,a ≤ U_P,a^law = A_a+qr(v̄+νw̄)`。**

単純版は `12qL+qr[(10n+2)+7nν]+7+1_sine`。component/orderの独立性と同じlocal cost surrogateを全supportへ適用できることが条件。非可換性は順序を保つ個数upperを妨げない。有限個の成分・orderのscalar和であり、全trajectory列挙は不要。compiler相殺を無視する分はupperの保守性で、同じ相殺を別途減算しない。

## 4. DF側：全域upperを求めずに母平均lowerを作る

### 4.1 最小限の統計的lower

DF価格は非負。n_s個のIID価格標本について、CDF Fと経験CDF Fhatを使い、

`e_a = sqrt(log(2/δ_a)/(2n_s))`、
`P(sup_t |Fhat_a(t)−F_a(t)| > e_a) ≤ δ_a`

という二側DKW–Massart bandを使う。[Wei–Dudleyの研究論文§1](https://arxiv.org/pdf/1107.5356)の一標本不等式を参照し、[Reeveの証明・Corollary 2](https://arxiv.org/html/2403.16651v1)とも確認した。二標本KSの定数や条件を流用していない。

この一つのband事象上で、任意のt≥0についてleft limitを使うと、

`μ_D,a ≥ t P(C_D,a≥t) ≥ t [1−Fhat_a(t−)−e_a]_+`。

保存標本minimum `c_min,a` をtに取れば `Fhat(t−)=0` なので、

**`L_D,a^min = c_min,a [1−e_a]_+ ≤ μ_D,a`。**

DF保存min=21754は上記引用資料に存在する。今回はn_s・min・δを代入してlowerを計算していない。δも採択していない。この方法はDFの高価格tailに上限を要求しない。未観測の高費用はlowerを弱めず、未観測の低費用が持ち得る質量をCDF bandが確率的に扱う。全lawを被覆する決定論的下界と呼ばない。

bandは全thresholdに同時成立するため、**同じ標本でminimum／thresholdを選んでも**有効である。DF component集合Uを独立な層とみなす必要がない。ただし、任意の事後選択したcost functionや実装familyにまで同時成立するわけではない。固定DF実装、全256本のcost非依存保存・correct joins、選択／rejectionなしという前提を点検する。新しいpostselectionがあれば、このまま受理しない。

必要なときだけ、同じbandから `L_D,a(τ)=∫_0^τ [1−Fhat_a(t)−e_a]_+ dt` へ進められる。threshold選択への追加罰則は不要だが、保存価格の新解析が必要になるので今回実行しない。初手は保存minだけで十分かを調べる。

両軸は `δ_cos+δ_sin≤δ_cost` のunion boundで同時化でき、軸の独立性は不要。δ_costは価格標本の信頼度であり、量子shotのfailure budgetとは独立に指定する。既存のshot α=.025を無断で価格解析に流用しない。固定された全精度行に同じ軸boundsを使うだけなら、そのたびに新しい独立試験と扱う必要もない。

### 4.2 順位判定への非対称な接続

各軸で **`U_P,a < L_D,a`** を確認できれば、SOURCEのcount前提とIIDモデルのもとで母平均価格を順序付けられる。Pauliの価格標本平均を厳密upperへ変換せずに済み、DF側の高費用tail上界も不要になる。

十分な差があるかの**H4数値判定は未実施**。ただし、Z/ZZ保持部と有限KによるPauli構造upper、すでに保存されたDF minimumという具体的な入口があり、全域compiler数値保証や多数の追加compileに先行して検討する価値がある。これはSOURCE・数式からの実用性判断で、数値的な順位保証ではない。

統計モデルを採用しない完全決定論的比較では、DF側の有用な正のcompiled RZ下界が現在不足する。入力の保持gate数や隔離回路の価格は、全体compileの相殺を経た費用の下界ではない。自明な0だけでは優劣は判定できず、この方向の保証は未完成である。

## 5. 三方式の比較とDF structural counterの位置付け

| 方式 | 成立に必要な証拠 | 鋭さ・負荷 | 今回の判断 |
|---|---|---|---|
| 全trajectory uniform upper | 固定入力family・native lowering候補・総gate数非増大・正常終了 | 最大K/全supportを毎回払うため保守的。Pauli Z/ZZで保持部を絞れる。scalarのみ | Pauli upperの第一候補。DF upper単独から順位は出ない |
| Law-weighted upper | 上記＋effective質量とpointwise surrogate | Kを平均orderへ置換。Pauliは有限成分和。DFはbasis切替構造が必要 | uniformで分離しない場合の次候補 |
| 両側有限標本＋tail | 全域rangeまたはtail envelope、IID、適切な同時不確かさ | range罰則が大きいと256本では保守的。nominal t95だけでは不足 | 両表現の厳密平均区間が必要な場合に限定 |
| **非対称：Pauli upper＋DF DKW lower** | Pauli count前提、固定DF価格関数、IID/CDF band | DF range不要、保存minで開始可能、追加compile0の設計 | **優先**。数値分離は未確認 |

`df_rte_qiskit.py:estimate_df_rte_structural_size_upper_bound`は、basis往復instruction数、product support数、rotation一つ、最大order、event数、aggregate phaseから**untranspiled instruction upper**を返すprebuild/OOM guardである。compiled RZ upperではない。`estimate_df_rte_untranspiled_size_upper_bound`も同様であり、その値をそのまま価格上界にしない。

DF builderは `(basis_id,basis_hash)` が等しい隣接applicationでbasis reuse、identityでflush、異なるbasisへ切替、occurrence末尾でcloseする。元familyはfull Gaussian basis、step間はraw concatenation。各eventで常にbasisを二回払うという**worst-case upper**は可能でも、実費用の等式ではない。保持部＋event部の実測価格も非加法的である。

DF law-weighted設計では、basis primitiveのnative上界を先に証明し、active basis∈{none,各登録basis}を状態とする有限状態遷移へ接続できる。product→rotationの順、order0/2/4、identity flush、occurrence resetを保持する。固定basis数と成分数に対する有限和／行列伝播で済み、全trajectory列挙を要しない。位相gateは適用occurrenceのaggregate上界へ数え、重複加算しない。

ただし `native_df_gate_model.py` の `explicit_rz_sx_x_cx_v1` は**宣言されたlowering model**であり、現行Qiskit compilerの保証ではない。Gaussian/XXPlusYY・controlled RZZ等の到達分解と最終opt1 countへの接続が不足する。DFの両側boundsを要求するとこの費用・保守性が増す。非対称な比較が分離すれば、このDF拡張は不要である。

一般のrange付き推定では、`0≤C≤U` を全supportで証明してからHoeffdingの片側半幅 `U sqrt(log(1/δ)/(2n_s))` を使う。empirical Bernsteinなら片側半幅の例は

`sqrt(2 s² log(2/δ)/n_s) + 7 U log(2/δ)/(3(n_s−1))`。

s²はunbiased sample variance。下側はU−Cへ適用し、各tailにδを割り当てる。[Maurer–Pontil (2009), Theorem 4](https://www.learningtheory.org/colt2009/papers/012.pdf)。varianceが小さくてもrangeに比例する項は残り、未観測tailを標本varianceだけで除去できない。今回は新variance・信頼区間・最適sample数を計算していない。

固定order層の `μ=Σ_h p_h μ_h` も設計可能だが、未観測層にupperを置くだけでなく通常層の不確かさが残る。事後Uの層を使うには集合選択込みの一様評価、独立holdout等が別途必要で、単純な固定層t区間は不可。order4一例のcompileはdiagnosticであり、どの方法でも全母平均upperの代用にはならない。

## 6. 条件付きGへの影響：今回は判定式だけ

[保存central C/N/G](non_df_h4_central_conditional_cng_20261024.md)・[旧conditional結果](non_df_h4_conditional_cng_results_20261024.md)のNを固定したとき、母平均価格に対する十分条件は

**`Σ_a N_P,a(ε,ρ_P) U_P,a < Σ_a N_D,a(ε,ρ_D) L_D,a`。**

全独立ρ guard boxで保証したい場合は、保存N端点を使える範囲で

`Σ_a N_P,a^max U_P,a < Σ_a N_D,a^min L_D,a`

が十分条件となる。Pauli/DFのlaw、bias、B、headroom、Nを同一にしない。価格μの順序だけでG順位が自動的に決まるとはしない。今回この不等式へ数値を代入したり、Nを再計算したり、保存順位を更新したりしていない。

両軸を直接weighted価格Z_i=N_cos C_i,cos+N_sin C_i,sinにする場合は保存pairと共分散を保持する。軸のSEを独立として足さない。上の同時軸boundsは共分散推定を必要とせず、独立性も仮定しない。表現間はunpairedのまま。

価格confidenceとρ guardは独立の未解決事項。価格boundsが完成しても、実ρ所属、PR全lawの測定残差、DF C/L接続、正式certificate、旧actual N/G nullは解決しない。既存6局所Arbは選択trajectoryの数値精度証拠で、母平均費用tailの証拠ではない。

## 7. 必要最小の次作業とGO／STOP

[非実行計画JSON](../artifacts/non_df_h4_pr_mean_cost_design/2026-10-25/next_validation_plan_DRAFT_DISABLED.json)を保存した。`execution_enabled=false`、新grantなし、実行を起動するentrypointなし。

| 優先 | 担当・追加証拠 | 最小計算範囲・情報量 | GO／STOP |
|---|---|---|---|
| 1 | ユーザー：価格confidenceとIID条件の採用可否、scalar評価範囲を明示承認。Codex：count前提の残りをSOURCEで閉じ、Pauli uniform upperとDF minimum lowerを比較 | 両軸の有限scalar batch。H4 build/compile/sampling/state作用0、raw再集計0、N/G0。母平均価格の分離可否が分かる | 前提・confidence固定後のみGO。分離または前提不成立でSTOP |
| 2 | Codex：必要時だけPauli law-weighted上界、さらに必要ならDF DKW積分lower | 成分・orderのscalar和。積分版だけ保存価格の新解析を別指定。全trajectory列挙0 | 1で未分離かつ有望なとき別承認。分離でSTOP、DF拡張へ自動移行しない |
| 別問 | Codex：依頼があれば保存Nを重みとするG不等式を比較 | 新shot計算0、指定ρ断面／端点の比較だけ | 新しいG比較の承認が必要。実ρ所属を推定しない |
| 保留 | DF finite-state upper／新fixed sampling batch | 主問いがDF mean upperや逆順位も要するとき、または上記で不確かさが支配的なときだけ設計 | 実際の不足から必要性を判断。今はcall数・新科学条件を採択しない |

第一候補は**追加のH4回路取得なしでも答えを得られる可能性がある**。それを確認するscalar評価こそ次の最小科学作業で、今回は未実行。全列挙、256本再compile、大量order4抽選、全compiler数値保証は既定工程にしない。完全決定論的な両表現の保証を求めるなら、DF正のlower等が別途必要で、現在は実用的と判定できない。

## 8. 再利用・人工検証・保全

既存の `h4_d2_pauli_control.py`、C finite-grid law、`h4_conditional_cng.py`の保存定義、DF structural counters/basis reuse、保存Qiskit整数count補題、既存catalogと価格要約を再利用した。`saved_rte_cost_diagnostics.py`も確認したがH2固有の入力・解析なので実行／混用しない。新しい科学pipelineや承認証明の仕組みは作成していない。

[人工checks](../artifacts/non_df_h4_pr_mean_cost_design/2026-10-25/artificial_checks.py)は標準libraryのみ。人工2成分lawのorder0/2/4全組合せで期待surrogate式、既存DF counterのAST抽出によるtoy descriptor upper、未観測の安価／高価値を含む人工CDF bandからのlower、軸独立性を要しないweighted boundを検査する。回路object、compile、状態vector、H4 operand、乱数を使わない。CDF検査は与えたband上での決定論的補題検査で、IID仮定や確率的coverageの実証ではない。

適用指示は `/home/AbeHiromu/projects/AGENTS.md`、worktree AGENTS、最新版 `research_project_instructions_unified_2026-10-25.md` §8.4・§10・§10.3。主agent自身で確認し、別agentレビューとは呼ばない。共有環境や他jobへ変更なし。

[verification.json](../artifacts/non_df_h4_pr_mean_cost_design/2026-10-25/verification.json)に人工結果・引用field同値・差分範囲を記録する。旧SOURCE/raw/結果/grant/ledger/marker/STOPは変更せず、旧母平均upper/null、全lawρ/null、元native UNKNOWN、D1初回FAIL、正式受理falseを維持する。この設計で科学結果statusを昇格させない。

[publication_status.json](../artifacts/non_df_h4_pr_mean_cost_design/2026-10-25/publication_status.json)に公開状態を記録。専用branchのcommit・GitHub公開・remote SHA照合後、**mandatory STOP**。scalar評価、新compile/sampling、正式certificate受理へ自動移行しない。

公開した主成果物commit：**`4cc80251e3b248adad5f2e6b756ccaaba570b27f`**。GitHub接続のGit data APIで公開し、local準備tree `0b075dd35fcc0b2bd8c7a159291ce85a75b57584` と一致。同commitをgit fetchし、local HEAD／origin ref／GitHub refの同SHA・clean worktreeを確認した。この公開記録だけを追記し、最終HEADもremote照合してチャットへ記載する。**設計完了、mandatory STOP。次のscalar評価は未承認・未実行のまま。**
