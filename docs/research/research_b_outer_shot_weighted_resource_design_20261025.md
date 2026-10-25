# 研究B：単一複素signalのshot加重native資源評価設計

2026-10-25 JST。**DESIGN_ONLY／SCIENCE_NOT_EXECUTED**。作業起点 `cae3cd784dbdfcf2fe96902345379aca4b9606c7`。今回行ったのはSOURCE・保存schemaの静的確認、以下の接続設計、人工scalarテストのみ。新しいshot数、shot加重G、順位、逆転境界、rare確率、signal／normalizationの数値評価・再解析は実施していない。計算用SOURCE・launcher・科学outputは作っていない。

## 1. 現在の証拠と固定対象

| 証拠 | 固定科学SOURCE／結果commit | 到達点と保全する限界 |
|---|---|---|
| [22行analytic](research_b_outer_rte_22row_execution_20261025.md) | `26e8734aff3969dc8077cb058826d12394be28e1`／`33715848bfb83d49ec4e54d6eef481012a7441e4` | corrected physical complex signal直接誤差・normalizationの共通入力。u_numはnull |
| [A0](research_b_outer_a0_execution_20261025.md) | `1e3921ba16e5d531e9a3010cffa9ff717609a93a`／`31662b49157b15f2d1c03eef1c2faa07e2f905e7` | 90/128桁のEMPIRICAL_CONSISTENCY。包含・certified u_numではない |
| [B0](research_b_outer_b0_recovery_20261025.md) | `b190fe28b98d2f94e1509e19cd3d36a63b2bb99e`／`0db9b3ed189eb6cd9d66e070993dbd0e72d29104` | full-wrapper correctness・決定論的代表費用。γ certificateではない |
| [B1](research_b_outer_b1_expected_native_cost_20261025.md) | `11e2532a62d54a65f8fad5489be6b135f826011b`／`bef63edbd8cc564444b44e4e8cf23baa90f05fe5` | 固定3,072 trajectory・6,144 wrapper完了、INCONCLUSIVE_RARE_EVENT_COVERAGE |
| [rare感度](research_b_outer_rare_event_minimal_sensitivity_20261025.md) | `a8cf99156af692f0facaa1885ab9ed52b28c6f4c`／`cae3cd784dbdfcf2fe96902345379aca4b9606c7` | B4/B3・D1/R4のCONDITIONAL_SENSITIVITY_ONLY。rare費用は未知 |

上記は既存報告の状態の引用であり、今回の数値再検証ではない。全旧SOURCE・raw・grant・ledger・marker・STOP・個別契約は変更しない。

対象はsynthetic 2-mode、one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix、n_D=1、T=±0.18。Geometry／AO basis／分子DF rank探索は非該当。Quantum outcomeのnormalization積は粒子数Nと区別して **𝒩** とする。Native費用はB1と同じQiskit1.3.0、`rz/sx/x/cx`、opt1、seed17、approximation1、`qubits_initially_zero=False`、topologyなし、`official_definitions_angle_preserving_native_v5`の保存値だけを使う。

| ID | Outer | D内部 | q/f | r_RTE/K | 役割 |
|---|---|---|---|---|---|
| B3 | Yoshida4 | A2 | q2 | 2/2 | B4との主対照 |
| D1 | Yoshida4 | A3 | q2 | 2/2 | A3基準 |
| R2 | Yoshida4 | A3 | q2 | 4/2 | r変更対照 |
| R4 | Yoshida4 | A3 | q2 | 2/4 | K変更対照 |
| B4 | SABA2 | A2 | q2 | 2/2 | ε=1e-8の主比較 |
| F1 | Refined-S2 | A2 | q2,f8 | 2/2 | refined対照、条件付き費用表へ含める |

精度はε=1e-8／1e-9、目標は `physical_complex_coherent_signal_absolute_error`。比較するのは単一時刻・単一複素signalで、RPE round／phase／エネルギー推定・Stage 2・全RPE資源評価へ拡張しない。

## 2. 実際に確認した既存SOURCEと再利用判断

以下は本起点にあるファイルを静的に読んだ。関数の存在だけから研究Bへそのまま適合すると判断していない。

| SOURCE・関数 | 適合する部分 | そのまま接続しない部分／必要差分 |
|---|---|---|
| [`rpe_resource_accounting.py::_hoeffding_shots`](../../src/trotterlib/rpe_resource_accounting.py) | bounded-outcome scalar式 `ceil(2/e_coord² * log(2/alpha_axis))` | floatのlog・積・ceilで大きな整数の切上げを保証できない。式の参照・toy照合に再利用し、本番整数は下記interval helperへ |
| 同 `evaluate_rpe_round_candidate` | fresh IID、軸別α、classical MCとquantum shotsの区別 | partial-S2、RPE phase、radius下界、arcsin、`rho*sin(beta)/sqrt2`、q_m→round、cost-providerの状態準備除外などが固有。今回は呼ばない |
| [`rte.py`](../../src/trotterlib/rte.py)：`finite_rte_operator_moments`、`compose_finite_rte_occurrences`、`finite_rte_combined_attenuation` | corrected／raw／𝒩の数学的意味、異なるsigned occurrenceの積 | matrixを生成する関数は起動しない。保存config・scalar定義だけを利用 |
| [`outer.py::normalization_product`](../../src/trottertracks/b2d_evaluator/outer.py)、[`outer_backend.py`](../../src/trottertracks/b2d_evaluator/outer_backend.py) | 実occurrence列、rescale確認、複素signalの直接差 | 既存float積を保証付き上界とは扱わない。backendを新たに実行しない |
| [`outer_rte_execution.py::budget_metadata/bias_qualification`](../../src/trottertracks/b2d_evaluator/outer_rte_execution.py) | 今回の精度・予算・α・null資格、corrected physical signalの選択 | 既存の参照statusと新しい条件付きshot数を分ける。atolや観測biasだけでPASSを発行しない |
| [`intervals.py::scalar_context/rational_interval/endpoints`](../../src/trottertracks/pr_dictionary_validation/intervals.py)、[`gates.py::shot_count_from_enclosure`](../../src/trottertracks/pr_dictionary_validation/gates.py) | mpmath1.3.0のoutward scalar interval、exact Fraction端点、ceil判定 | helperだけを再利用。`score_enclosure`のliteral-price・固定η・4arm policyは使わない。private `_mpi_`依存を含むため対応版・演算範囲を実行前確認 |
| [`f2_statistics.py::qualified_shots`](../../src/trottertracks/pr_dictionary_validation/f2_statistics.py)、[`f2_execution.py::execute_f2`](../../src/trottertracks/pr_dictionary_validation/f2_execution.py) | 等座標配分時の `4*B²/eta_complex²*ln(4/alpha_complex)` とshot×軸費用の和 | F2有限target・4arm・finite-bit allowance・外部資格・旧grant契約は本研究へ移さない。既存guardを外す変更も不要 |
| [`b1_native.py::statistics`](../../src/trottertracks/b2d_evaluator/b1_native.py)、`rte_compiled_cost.CompiledMetricAccumulator(weighted=False)` | 保存された6指標のmean・不偏分散・SE・n | 新MC／compileを呼ばない。Accumulatorは軸間共分散を保持しないため、paired raw IDを読む薄い集計が必要 |
| [`rare_sensitivity.py::dec/finite_law/boundary`](../../src/trottertracks/b2d_evaluator/rare_sensitivity.py) | scalar有限law、未知費用を残すaffine式、SE矩形、旧確率8条件の保存値 | `IDS`・r=2・2比較・RZ/CXに固定。R2/F1へ`probability`／`analyze`をそのまま呼ばず、必要なscalar部分だけ新adapterへ。旧モジュールは変更しない |

旧PRリソース会計は本起点の別履歴 `33c7231fc7c8f3940106bd8fe473e3db775fbcb0` で実ファイルを読んだ。

- `scripts/d0_frozen_shot_rule_v1.py::axis_headroom/u_aware_shots`：UNKNOWN／EMPIRICAL／CERTIFIEDを分け、巨大整数域はLOG_DOMAIN_ONLYとする考え方を参照。旧の軸別biasから統計headroomを引くpolicy、normalization-logのbitwise条件、binary64安全域制限を今回へ機械的に移さない。今回は固定s予算を使う。
- `src/trotterlib/pr_cost_integration.py::saved_shots/paired_statistics`：保存analytic shotsとpaired費用の接続を確認。旧n8/n32 cohort、positive-RZ制約、binary64 SEをそのままn256／6metricへ流用しない。
- `scripts/d0_saved_value_reanalysis_v1.py::weighted_metric`：`m_X*C_X + m_Y*C_Y`をtrajectory内で先に作り、paired totalsの分散を使う方式を再利用する。旧 `shots < 2**52` 条件は一般的なshot上限ではなく、今回のexact integer経路へ持ち込まない。
- `src/trotterlib/h8_six_axis_reaccount.py::shots_from_accounting/replace_row`：既保存shotだけを読み会計を置換する経路であり、研究Bの新shot数を供給する実装ではない。旧値・旧選択・旧承認は流用しない。

## 3. Estimator・normalization・符号

`rte.py`の有限paired-Taylor scalar weightをB_j、各occurrenceのshort-step数をr_jとすると、`𝒩_j=B_j^r_j`、`𝒩=∏occurrence 𝒩_j`。Momentのraw mean operatorはcorrected finite operatorを𝒩_jで割ったもので、独立occurrenceとunitary D列を合成しても全体raw meanがcorrected mean/𝒩になる。`outer.py`はrawを全体𝒩でrescaleし、correctedと照合する構造を持つ。

Signed Yoshida stageはsigned τ_jを保ち、B_jはabs(τ_j)のeven paired weightから得る。負時間の有限polynomialを正時間の逆行列に置換しない。共通Bのq乗、R timeの符号付き総和からのB計算、cache key数への置換ではnormalizationを求めない。繰り返される実occurrenceの因子は全て残す。F1のq/fとR2のr変更も保存planの実数に従う。

[`rpe_hadamard_interrogation.py::build_prepared_hadamard_wrapper/append_hadamard_readout`](../../src/trotterlib/rpe_hadamard_interrogation.py)はmode0準備、ancilla H、ordinary `diag(I,U)`、XはH、YはSdg→H、ancilla測定を使う。Bit 0→+1、1→−1に写すとX期待値はRe、Y期待値は**+Im**である。B0の独立`wrapper_reference`も同じ規約。今回はその回路をbuildして検証したのではなく、既検証SOURCEの定義と実績を確認した。

各quantum shotで全R occurrence／short-stepをfresh IID抽出し、対応axisのoutcome O_a∈{−1,+1}を得るモデルとする。各軸で独立な標本列を用い、B1の256本やB0代表・測定結果をquantum shot集合へ再利用しない。

`X_hat=𝒩*mean(O_X)`、`Y_hat=𝒩*mean(O_Y)`。その平均はcorrected finite modelのsignalであり、physical signalとの差はalgorithmic bias側へ残る。Compiled実装／有限bit sampling lawと理想modelとの差も自動消失せず、γ等の未解決資格として扱う。

## 4. 誤差・confidence配分とscalar shot式

`|z_hat-z_physical| <= b+u_num+gamma+s` を設計上の三角不等式とする。固定配分は次のとおり。

| 項目 | 条件 | 現在 |
|---|---|---|
| Algorithmic＋numerical | b+u_num≤ε/2 | 観測bあり、u_num=null |
| Implementation／synthesis | gamma≤ε/10 | gamma=null |
| Statistical | s≤2ε/5 | 数値評価未実施 |
| Confidence | α_X=α_Y=0.025、合計α=0.05 | 単一(candidate,T,ε)の2軸union bound |

統計予算は最大値 `s=2ε/5` を使う提案とし、観測biasが小さいから余りをshot予算へ再配分する操作は行わない。等座標配分 `eta_X=eta_Y=s/sqrt(2)` なら、両軸の絶対差≤eta_aから複素絶対差≤sとなる。

Raw O_aのrange幅は2なので、Hoeffdingの両側boundは `2*exp(-m_a*eta_a²/(2*𝒩²))`。したがって十分条件は

`m_a >= ceil(2*𝒩²/eta_a² * ln(2/alpha_a))`

であり、`_hoeffding_shots(eta_a/𝒩,alpha_a)`と同じ数学式である。η配分を代入してsqrtを消すと

`m_a >= ceil(4*𝒩²/s² * ln(2/alpha_a))`。

最小必要shotの厳密最適値ではなく、保守的な十分shot数である。RPEのρ、phase budget、arcsin、位相分枝条件をここへ追加しない。X/Yの確率的独立性がなくてもunion boundは成立するが、各axis内は固定law・独立なtrajectory／測定のbounded-outcomeモデルを要する。

αは各(candidate,T,ε)に対するもの。6候補・両T・2精度を同時に覆うfamilywise budgetやwinnerのconfidenceは未配分。Cost MCのSE／参考normal区間にαを割り当てたことにもならない。

## 5. 大きな整数とnormalizationの数値条件

将来は次のscalar経路を推奨する。今回この経路へ実候補の値を渡していない。

1. ε・予算係数・αをexact Fraction、保存binary64のλ／τ／𝒩を`as_integer_ratio`等で正確に取り込む。短い表示桁や`exp(log(saved_N))`へ置換しない。
2. 22行の`rte_stages`とB1 provenanceのsigned duration、τ、r、K、順序・重複を照合する。明示したtargetはその保存finite model。A0が用いたexact Fraction時間とbinary64 config時間の差も入力意味として記録し、黙ってmodelを変更しない。
3. `scalar_context/rational_interval`のoutward演算で、各B_jのsqrtと積から`[N_lo,N_hi]`を作る。保存𝒩の値は参考値／estimator scaleとして別に保持し、float正常値やA0一致だけで𝒩の上界としない。scalar helperの対応版はmpmath1.3.0、既存90桁設定を初期案とする。
4. 例えば `U=max(N_hi,abs(N_used))` をexact rational上界として固定し、statistical expressionには𝒩の代わりにUを使う。`N_used`は明示的に記録したestimator scaleで、旧scaleを変更した場合も記録する。ここでUを使うことは安全側のshot定義であり、point 𝒩に対する最小ceilを主張しない。
5. `4*U²/s²*ln(2/alpha_a)` をoutward intervalで囲み、`endpoints`と`shot_count_from_enclosure`へ渡す。lower/upperのceilが一致した場合だけそのPython任意精度intを返す。straddle／nonfiniteは`SHOT_COUNT_UNRESOLVED`で停止・保存し、float丸めで強制確定しない。
6. Shot数、m*C整数積、Fractionの分子・分母をfloatへ落とさず、出力の巨大整数はdecimal文字列にする。必要ならlog値は表示診断だけに使う。`2**52`等をquantum shot数の科学的上限にはしない。

Normalization enclosureは固定scalar modelの演算保証に限られる。実PRNGのbit-level law、finite modelとcompiled meanの差、物理入力・signal計算全体のu_numを証明するものではない。

またN_usedと数学的normalizationが異なると、統計rangeを広げるだけでは平均のrescaling biasは消えない。安全な座標boundから、scale差に起因する複素誤差を `u_scale <= sqrt(2)*max(|N_used-N_lo|,|N_used-N_hi|)` と囲む設計とする。これは数値予算内の一成分であり、未取得のsignal側u_numを0で補う許可ではない。`u_num_total`の構成を明記し、全体が未取得ならdeterministic資格はnullを維持する。

γはcorrected physical signalと同じ尺度で必要。Raw compiled operator差の最大値をそのままγへ代入せず、normalizationによる尺度、全trajectory範囲、finite-bit law・実装の差を含む資格は未解決として残す。

## 6. 保存入力の結合設計

今回確認したのは以下の**schemaとSOURCEの意味**であり、値の新しい結合集計・budget判定は実施していない。

| 内容 | 固定入力とfield | 方針 |
|---|---|---|
| 共通bias | `33715848…:artifacts/research_b_outer_rte22/20261025/analytic-once/result.json`、`rows[].outputs.signals.corrected_physical_signal_error` | 6候補×両Tの主観測bias。`bias_qualification.observed_bias`／CSVの同名列とも照合。Operator誤差・raw誤差・corrected対outer誤差へ置換しない |
| 共通𝒩 | 同`outputs.normalization.normalization_product`、`occurrence_normalizations`、`outputs.rte_stages[].config/event_distribution` | 全occurrence、positive normalization、signed configの同一性を照合。半径から𝒩を逆算しない |
| A0補足 | `31662b49…:artifacts/research_b_outer_a0/20261025/independent-once/result.json`の`comparisons[].bias`、128桁rowsの`outputs.normalization/corrected_signal/physical_signal` | B1集合と重なるB3/F1/R4・両Tの6補足記録だけを表示診断へ接続。S1は対象外。全6候補共通の22行主入力を維持し、有利なbiasへ選択的に差し替えない |
| A0一覧 | 同日`COMPARISON.csv`の`corrected_signal_error`、`normalization`、`u_num/certified` | 高精度一致の既存分類を付記する。無い候補はNOT_AVAILABLEで、u_num=0ではない。新たなabs差・高精度signal評価は行わない |
| Native S0単価 | `bef63edb…:artifacts/research_b_outer_b1/20261025/STATISTICS.csv`／`iid256-once/result.json.statistics.groups` | candidate,T,axis,metricでjoin。6metricのmean/variance/SE/n=256を維持 |
| Paired原本 | 同`SAMPLES.csv`、必要なidentityは`trajectories.jsonl`／binding／inputs／graph | sample ID・physical_keyをX/Yで一致させ、n256と同一fixture/law/compilerを確認。Quantum shot用samplerとして再起動しない |
| Rare既存8条件 | `cae3cd78…:artifacts/research_b_outer_rare_sensitivity/20261025/saved-once/result.json.probabilities` | B4/B3/D1/R4×両Tのp_rareとunknownを参照し、旧境界は会計の意味の照合に用いる |
| R2/F1のrare | B1の保存occurrence probability／r/K | 現感度解析はこの4 candidate-timeを含まない。将来の許可範囲でscalar p_S0積・p_rareを4条件だけ追加評価する設計。今回未計算、0補完しない |

結合keyはcandidate名だけでなく、Tのcanonical Fraction、outer、D内部、q/f、n_D、r/K、H／sector／state／DF分割／array identity、scalar/identity分離、normalization convention、full-wrapper compiler profileを含む。異なるSOURCEの保存結果を使うため、commit同一だけを要求するのではなく、共通科学identityと差分の意味を照合する。

A0と22行でbudget境界の意味が変わるような不一致があれば、良い方を採用せず`INCONCLUSIVE_INPUT_NUMERICS`を明記する。観測atol、B0/B1有限標本のPASS、A0安定性をu_num／γへ変換しない。

## 7. 資格を分ける出力

将来の各(candidate,T,ε)行では次を別fieldにする。

| Field案 | 意味と許される状態 |
|---|---|
| `observed_bias_budget_status` | 保存bとε/2だけの参考比較：WITHIN_POINT_BUDGET／OUTSIDE_POINT_BUDGET。保証付きPASS/FAILではない |
| `deterministic_qualified` | b+u_num≤ε/2。u_num欠如ならnull／UNRESOLVED_NUMERICAL_UNCERTAINTY |
| `implementation_qualified` | corrected尺度のγ≤ε/10。γ欠如ならnull |
| `statistical_shots` | モデル・range・IID前提、normalization enclosure、整数ceilの状態とm_X/m_Y。条件付きのscalar十分shot数と明記 |
| `end_to_end_qualified` | 上記条件全ての成立が必要。現在u_num／γがnullなのでNOT_CERTIFIED。shot数が計算できても昇格しない |

観測biasが予算外の行にも、条件付きstatistical workの参考値は出せる設計にするが、同精度適格候補には含めない。Null資格だけを理由に、許可された条件付き会計全体を中断する必要はない。入力・math・ceilのFAILとは分ける。

## 8. S0参考G・未知rare費用・費用MCの不確かさ

各metric kについて、状態準備・control・evolution・X/Y終端・measurement込みの保存単価を使う。

`G_S0_hat,k = m_X*mu0_hat_X,k + m_Y*mu0_hat_Y,k`。

これはS0条件付き参考値。無条件費用の経験S0部分と未知rare項は別に

`G_hat,k(u) = (1-p)*G_S0_hat,k + p*m_X*u_X,k + p*m_Y*u_Y,k`

とする。真のGにはさらに `(1-p)*Σ_a m_a*(mu0_a,k-mu0_hat_a,k)` が残る。unknown u_X,k/u_Y,k、unknown variance、unconditional population Gはnullのまま保持する。Rare費用の母平均や全support上界をB0代表／B1最大値から補わない。RZが主指標、CX・RZ depth・CX depth・total depth・sizeが補助。Shot加重depthは逐次work proxyであり、並列実機wall timeではない。Measurement単体は各wrapperの規約確認に使い、今回の6metric最適化対象へ追加しない。

### Paired MCのSE

同じclassical trajectory iのX/Y費用から `W_i,k=m_X*C_i,X,k+m_Y*C_i,Y,k` を作る。将来、n=256のpaired原本から

`SE(G_S0_hat,k)^2 = [m_X²*s_X,k² + m_Y²*s_Y,k² + 2*m_X*m_Y*s_XY,k]/256`

を評価する。s_XYはtrajectory間の不偏sample covarianceで、`Cov(mean_X,mean_Y)=s_XY/256`。保存S0 mean/variance/SEはそのまま保持し、追加でX/Y joint momentだけを求める。Quantum X/Yが独立という仮定を、pairedなclassical費用MCへ移さない。費用SEは期待費用推定の不確かさであり、実際のquantum job総費用の分散でも測定αでもない。

Joint原本が結合できないときは、共分散を0にせず、`SE_bound <= m_X*SE_X+m_Y*SE_Y` というCauchy型の標本分散感度上限を示すか、SE_UNRESOLVEDにする。これは母分散の保証上限ではない。保存MCの256を512にせず、normal95を厳密coverageと呼ばない。

S0経験寄与のSEは `(1-p)*SE(G_S0_hat)`。未知rare寄与の不確かさはこれに含まれず、全GのSEを発行しない。別candidate間の共分散は未評価なので、比較では各候補の±kSE矩形の端点を使い、独立性を仮定した差のSEやp値を作らない。将来のSE感度もk=1,2に限定し、標本追加・bootstrapは行わない。

### Shot加重後の比較式

比較の正側P／負側Qを定め、`Delta_G=G_P-G_Q` とする。`M_i=m_iX+m_iY`、`u_i^m=(m_iX*u_iX+m_iY*u_iY)/M_i` はshot加重rare混合平均の**定義**で、u_X=u_Yの仮定ではない。

`A=(1-p_P)*G_S0_hat,P-(1-p_Q)*G_S0_hat,Q`、

`Delta_G_hat=A+p_P*M_P*u_P^m-p_Q*M_Q*u_Q^m`、

`u_Q^m* = A/(p_Q*M_Q) + [p_P*M_P/(p_Q*M_Q)]*u_P^m`。

Delta_G_hat>0ならQ低費用。未知費用はX/Y別に原式も保持する。旧の単価比較のslope `p_P/p_Q`や切片を、新しいshot比が異なる比較へ移植しない。未知費用非負だけで母平均順位を保証することもできない。今回これらの係数・切片・順位の数値評価は一切していない。

## 9. 最小実装差分

将来は既存b2d_evaluator配下の**小さな保存値adapter一つと薄い入口一つ**で足りる見込み。名前案は `shot_weighted.py`／`run_b2d_outer_shot_weighted.py`。今回は未作成であり、使用可能なlauncherではない。

必要な関数は次の境界に限る。

1. `read_saved_cases`：上記identity・schemaを照合し、12 candidate-timeとA0補足を読む。新しいHamiltonian／moments／signal evaluatorは呼ばない。
2. `normalization_enclosure`／`shots_for_complex_signal`：既存interval primitivesを利用したheterogeneous scalar積とsqrtを消したshot式。現 `_hoeffding_shots` や旧F2全driverのguardを書き換えない。
3. `paired_cost_moments`／`weighted_cost`：n256、整数・Fractionによるjoint statisticsと巨大shot積。再利用不能な旧cohort guardやfloat上限だけを持ち込まない。
4. `resource_expression`／`qualifications`：unknown rareの係数・SE感度・null資格を保持。Model scalar条件と物理的保証を混ぜない。
5. 既存SOURCE/fresh output、`registered_scope/FollowupLedger`、`Evidence`の部分結果・失敗・STOPを利用する。承認原文・ID・certificate・packet・grant等は作らない。

Input schemaの読み取り、symmetric αからのmetadata cache、反復Bjのscalar cacheを利用する。小規模な保存値処理が主で、compile pool／simulation frameworkを新設しない。実装後は実資源と計算量を確認し、必要なら候補ごとに並列化する。CPU/RAM/worker/wall-timeに恣意的な上限を置かず、他研究・共有環境へ干渉しない。

## 10. 次の科学実行scope案

**これは未実行・未承認の将来契約案**。実装・人工検証後に新しい計算SOURCEを固定し、以下の対象・入力・schema・STOPとともにユーザーの別途明示的実行許可を受ける。今回の設計commitは計算SOURCEではない。

固定gridは6候補×2T×2ε＝24候補条件、2axisで48shot要求、6metricで144資源行。候補を観測費用だけで除外しない。全24行に観測bias／未取得資格を表示し、F1はrefined対照として含める。

主比較はB4対B3、D1対R4、D1対R2の3組。ε=1e-8ではB4/B3とF1参考表を優先し、ε=1e-9ではD1/R2/R4を優先する。Priorityはgridや予算の変更ではない。RZ/CXの3組×2T×2ε＝12比較条件、2metricで24 affine比較を予定する。F1を含む全6候補のS0参考表を表示するが、自動winner／無条件最適性は認定しない。

| 予定境界 | 固定件数 | 入力→出力 |
|---|---:|---|
| 保存binding／case取得 | 12 | 22行・B1・補足A0・rare → 同一性が確認された12case。A0補足は対象3候補×2Tの保存6記録を添付 |
| 𝒩/range scalar enclosure | 12 | 各caseの全R occurrence → N_used、N enclosure、U、scale差診断。実R occurrence計88／short-step反復計200の因子を保持。内部sqrtや加算はこの境界に含む |
| 追加rare確率 | 4 | R2/F1×両Tの保存p0 → p_rare。既存4候補×両Tの8確率は読取り再利用 |
| Paired joint moments | 72 | 12case×6metric → 既存mean/varianceと新joint covariance。256本のpaired IDで各caseを一回走査、ε間でcache共有 |
| 観測biasのbudget比較 | 24 | 保存corrected physical bias・ε → 点予算の参考flag。新signal／新abs誤差評価なし |
| Scalar shot評価 | 48 | 24条件×X/Y → モデルの整数mとenclosure status。α/ηの対称性を使ったcacheは可、logical要求は48のまま |
| 条件付き資源行 | 144 | 24条件×6metric → 軸寄与、G_S0、SE、rare係数を残すG式 |
| 比較・逆転感度 | 24 | 3組×2T×2ε×RZ/CX → Delta_G式・shot加重rare境界・±1/2SE感度 |
| 資格・最終status組立て | 24 | 各条件の5資格field → 参考状態／nullを区別。End-to-end PASSの自動補完なし |

これらは設計上の高位処理数で、内部interval演算数・nested科学callを混ぜた実行実績ではない。G_S0の中央／±1／±2SE表示なら144×5＝720表示点、比較の同様の矩形端点は24×5＝120表示点を上限とする。固定β/αの探索、別対照、別時間・精度、標本追加は含まない。未観測rare費用を動かす追加数値gridは作らず、affine係数・境界として出力する。

### 予定出力schema・保存

新しいfresh namespace案：`artifacts/research_b_outer_shot_weighted/20261025/saved-once/`。今回ディレクトリは生成していない。使用済みoutputは上書き・resumeしない。

- `provenance.json`：新計算SOURCE、固定入力commit/path/blob、科学identity、選択したmodel／N_used、budget／confidence、interval版・演算条件、資源・call scope。
- `result.json`：12 normalization／rare記録、24 case-epsilon行、各48 axisのshot status、72 joint moments、144資源記録、24比較式、全未取得資格・限界。
- `SHOTS.csv`：48行。`candidate,T,epsilon,axis,s,alpha_axis,N_used,N_model_low,N_model_high,U,m_decimal_string,shot_status,deterministic_status,implementation_status,end_to_end_status`。
- `RESOURCES.csv`：144行。`candidate,T,epsilon,metric,n_S0,G_S0_fraction,SE_S0,p_rare,known_contribution,coefficient_u_X,coefficient_u_Y,unconditional_G=null`。
- `COMPARISONS.csv`：24中央式。SE端点を展開する場合は120行まで。同じconfidenceで母平均順位を断定するfieldは作らない。
- 既存Evidenceの`events.jsonl,partial_results.json,status.json,STOP.json`、必要時のみ`failure.json`。保存済み巨大rawを重複copyせず、固定入力参照を保持する。

`model_shots`は将来評価済みの場合のみ整数文字列となる。保証付き同精度shotやGのfieldは、u_num／γ／sampling条件が未解決ならnullのまま。原本・途中結果を逐次保存し、候補計算が全部終わる前に最適候補を発表しない。

### Gate・STOP

SOURCE／入力identity／範囲／符号順／n256／metric定義の不一致、非有限・不正probability、対応しないinterval API、整数ceil未解決は原因を保存してSTOPする。通常の丸め差は元精度とscalar enclosureに照らして記述し、科学的結論に無関係なbitwise環境差だけを理由に再計算しない。

u_num／γ欠如やunseen rareによるINCONCLUSIVEは予定された科学的限界として全行へ記録し、0補完や有利な候補への変更をしない。全scope完了・科学的不一致・解決不能の技術停止のいずれでもmandatory STOP。初回案では自動retry/resume・fresh outputでの追加attemptも含めず、技術復旧の採否が必要なら旧runを保全してユーザーへ戻す。

新sampling、native build/compile、H/operator/signal、quantum shot実行、bootstrap、A1包含・γ新規評価、全RPE・Clifford+T・physical qubit評価は全0。計算するのはanalytic shot数と保存費用からの条件付き会計であり、その数だけquantum shotを実行する許可ではない。

## 11. 今回の技術検証と未完了

既存 [`test_pr_dictionary_f2_scalar_intervals.py`](../../tests/management/test_pr_dictionary_f2_scalar_intervals.py) の純人工5件がPASS。Exact rational変換、outward arithmetic、独立log oracle、ceil straddle／nonfinite拒否、人工shot式を確認した。

新しい [`test_b2d_shot_design.py`](../../tests/management/test_b2d_shot_design.py) の純人工5件もPASS。ASTで切り出した既存scalar Hoeffdingと複素座標配分の係数、巨大exact integer ceil、既存budget metadataとnull、paired covariance、shot比によるrare境界slopeの変化を、実候補と無関係なtoy値だけで確認した。Trotterlib package／NumPy／SciPy／Qiskit／PySCFをimportせず、研究データは開いていない。

再現用の人工テストcommand：

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B tests/management/test_pr_dictionary_f2_scalar_intervals.py -q
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B tests/management/test_b2d_shot_design.py -q
```

人工10件PASSは研究Bのactual normalization enclosure、実shot数、G、数値順位、scientific資格の検証ではない。今回は実候補の式代入をしていない。

実行前に残る技術作業は、上記小adapterの実装、保存値の厳密なjoin、normalizationのtarget／scaleとupper-bound演算条件の固定、joint statistics、null資格と失敗保存のmock確認、新計算SOURCEのfreezeである。統計range／IID／実装差の主張範囲もprovenanceへ明記する。保証付きu_num／γとfamilywise配分は未取得の科学的資格であり、将来の条件付き会計が完了しても自動で取得済みにはならない。

今回の成果は設計・scope・人工検証。結果公開後は**mandatory STOP**。次の担当は研究Bのユーザー：この固定scopeの実装・数値実行を別途明示的に許可するか判断する。Codexはその指示後にだけ準備・実行へ進み、今回の技術設計許可から新shot数／G評価を開始しない。承認証明資料やgrant等を要求せず、通常の研究B進行の研究統括への報告も行わない。
