# PR＋RPE：推定器変更の適合条件

2026-11-06 JST。[文献](VARIANCE_REDUCTION_LITERATURE.md)に示した既知式・保証をPRのSOURCEへ対応させる。以下は既存恒等式の再掲と適用条件であり、新しい非自明な定理の証明・推定器実装ではない。

## 1. targetと分散の単位

元の物理Hの厳密基底状態 `H|E₀⟩=E₀|E₀⟩` を候補・baselineで共用し、状態準備費を除く。H_D/近似Hのground、真のE₀/gap、完全state vector、未知path信号を無料の入力にしない。厳密groundでもPFの固有状態・基準半径1・branch成功は自動的に成立しない。[既存費用地図][COST]

一round/一axisでωを**全interrogationの**古典path、Y_bを実Hadamard outcome、wをnormalization/importance weightを含む補正とする。既存平均は `μ_b=E[w(ω)Y_b]`。全pathのD halves、scalar、ordered control、actual adjoint、XはH/readout、YはSdg→H/readoutという規約を保持する。system channelだけが一致しても、このcoherent平均は保存されない。

通常の非加重IID `Y∈{−1,+1}` では `Var(Y)=1−(E Y)²`。同じ平均を別のpath生成法で実現するだけでは1-shotの周辺分散を減らせない。これは今回の中心制約である。

既存[相関監査§4.4][CORR]が用いる全分散の分解は、f_b(ω)=E[Y_b|ω]として

\[
\operatorname{Var}(wY_b)=\mathbb E[w^2(1-f_b^2)]+\operatorname{Var}(wf_b).
\]

第一項は量子測定内、第二項はpath間の変動である。保存path期待値の分散やその二乗は、実測値のvariance/M2の代用にならない。wYのM2はY²=1からE[w²]となる。既存Parent/G7/S2のこの会計を取り消さない。

## 2. A1：同じshotに既知古典controlを付加する

L-CVの既知形を、既に正しいPR axis測定の後処理として使う案である。実law qに対して既知のcontrol平均ν=E_q[g(ω)]と、別pilotで学習後に凍結した係数λを用い、

\[
X_{b,\lambda}=wY_b-\lambda^\top(g-\nu)
\]

を平均する。**出力が±1に限定されなくなる**ことが§1の周辺分散制約から外れる理由である。量子回路lawやfinite targetを変えることを必須としない。[CV4Quantum §2.1–2.5][CV]

| 確認点 | この案で用いる情報・未解決点 |
|---|---|
| controlの期待値 | orderと各labelの既知lawから計算できるfactorized関数だけを使う。未知の〈E₀|Uω|E₀〉をνにしない。実有限bit lawとνの対応が必要 |
| 同shot取得 | pathのorder/selected labels/phase等は既存generatorが持つ古典情報。gはそのmetadataから評価し、追加の量子observableを同時測定するとは主張しない |
| covariance | 量子noiseも含む実outcome wYとgのcovarianceをpilotから取得する。state vectorでf_bを計算して置換しない。最適λを既知入力としない |
| 残るnoise | path-only controlではE[w²(1−f_b²)]は残る。改善余地はpath間成分のうちcontrolで説明できる部分。未知のgolden controlを使用しない |
| bias | 係数を同じ評価データから選んだ単純CV平均にはbiasが生じ得る。L-CVのleave-one-outは既知だが項間依存があり、IID confidenceを自動流用しない。最小案は独立pilot後にλを固定 |
| range | 補正後outcomeの実rangeを取得する。raw±1や旧wだけのrangeを使わない。大きなλや悪条件共分散行列はrange項と古典費用を増やす |
| 追加費用 | 学習/校正のfull quantum shots、control moment/共分散取得、回帰、metadata記録・memory・online評価。候補にだけ無料pilotを与えない。baselineに不要な学習を強制することもしない |

例えば既知factorized controlは、既存RTEのorderを条件にしたlabel選択のindicatorやその積を入力にできる。個別factorと確率が既知ならL-CVのEq.(34)の取得方式を使える。ただし実用上は有限の小さいcontrol集合が必要で、全path indicatorを作らない。**order-onlyは層別baselineへ吸収し、その後のlocal-label情報に追加の相関が残るかを問う。** 有利な相関の存在は未確認である。

SOURCEの `rte.py` はbinary64/PCG64によるcategorical生成であり、理想pの式だけでfinite-bitの実qを認証したことにはならない。将来の実装では既存dyadic生成の再利用、または適切な丸めallowanceにより、実qとνを対応させる必要がある。centering誤差δνがあればλによる増幅をbiasへ戻す。旧sampler・SOURCEを黙って置換しない。

zero-fillを使う場合もgとνを全planned trial lawに対して定義し、zero trialを分母から除かない。X_cvはquantum-zeroでも古典補正項が非zeroになり得るため、zero時に全estimatorを機械的に0にしない。事前判定前のprefix実費と、全trialの生成費は残る。

## 3. finite-confidenceと全費用

既存G7/ParentのBernsteinは主に既知M2上界とrangeを使う十分Nである。CVの実varianceが低下しても旧M2/旧rangeをそのまま渡すだけでは利益を認証できない。次の既知手段を区別する。[O2-A][O2A]

| 統計方式 | 必要な条件と会計 |
|---|---|
| 固定n empirical Bernstein | λを独立pilotで固定し、fresh IID評価標本と既知の補正後rangeにL-EB Theorem 4を尺度変換・両側化して使う。固定nのCI幅から最小Nを逆算したことにはしない |
| 予算Nを先に決めるvariance校正 | λ固定後の独立校正でL-EB Theorem 10等からvariance上界を取得し、その不確かさ/失敗予算を含め既知Bernsteinへ渡す。学習・校正・productionの3標本区分は統計的独立性のためで、各段で再承認する管理flowではない |
| adaptive停止 | L-CS Theorem 4等のtime-uniform保証、bounded/predictable条件と固定targetを満たす必要がある。固定n式をpeekingに使わず、最小案には採用しない |

比較対象のraw canonical、stratified/allocated estimator、既知ISにも、実際に利用可能な同じvariance-sensitive保証と価格情報を認める。候補だけがempirical Bernsteinを使う差をCVの利益にしない。baselineが真varianceや最適層配分を無料に知るという逆の不公平も避ける。

固定trial数なら各資源aの旧目的は `Σ_(m,b) N_(m,b) E[C_full,no-prep^(a)]`。pilot/校正や異種回路がある本案は、その費用も足す。より一般には

\[
\mathbb E\!\left[\sum_{s\in\mathrm{train,cal,prod}}\sum_{k=1}^{\tau_s}
C_{s,k}^{\mathrm{full,no\text{-}prep},(a)}\right]
\]

で会計し、データ依存Nなら学習・校正データで条件付けたproduction費用から戻す。可変停止で独立性なしにE[N]E[C]としない。T/CX/RZ/1Q/depth、古典秒、memory、追加ancillaは別欄。測定時間をgateへ無根拠換算しない。

## 4. MLMC/correlationを採る前の条件

古典pathだけを共有した独立fine/coarse測定では、条件付き測定varianceの和が残る。MLMCの小さい `Var(fine mean−coarse mean)` をそのまま `Var(Y_f−Y_c)` に置換できない。未知の両測定確率を使う共通uniform-threshold outcome couplingを、実装可能なjoint measurementとして扱わない。[相関監査][CORR]、[L-ML §2.3.2][ML]

L-MLのscaled augmented-stateは測定モデルを変えるので、この単純障害の外にある。ただしPRへは、coherent full targetのlevel対応、state差のmoment条件、非unitary mapのdilation、S補正、成功率/再試行、observable測定と実native価格が必要。bounded momentだけでfinite-confidence rangeが得られたとはしない。L-QMLはcoherent oracleを使う別の計算モデルであり、元H groundを利用できる仮定から自由なstate反射・逆state準備・coherent samplerを追加しない。

inter-shot相関を新設するなら、各時点の条件付きmeanと全共分散を示し、IID保証を外す必要がある。PRHS/qSHIFTのinter-step高次化は変更mean/bias/signed weightの問題も含むので、C/O3の既存課題として扱う。相関という名称だけでshot削減を主張しない。

## 5. 全RPEへ接続するための条件（B）

A1は同じaxis meanを保つ限り、元のPF/finite-RTE/native biasを消さない。raw/corrected signalのscaleを統一し、各roundの正の半径下界・phase予算から座標残余sを決める。normalizationによる半径低下とweight/range増加を二重計上しない。

全X/Y・roundに、学習/校正/最終推定/比較に必要な失敗配分を戻す。成功したaxisだけを選別しない。既存18round branch復元とenergy alias条件、`E_hat=−phi_hat/δ`の符号、元Hへの系統誤差を維持する。旧binary outcome用の二項tailを、非binary CV outcomeへ流用しない。既存decoderの決定論的部分とaxis confidenceの統計部分を分けて再利用する。[V2][V2]

V1/V2は保存DF modelへの条件付き証拠で、未近似の分子H/厳密groundへの保証を補わない。元H/state対応が未確定なら、将来のpilotもmodel限定と明記し、元HのPR＋RPE優位を認定しない。短roundの一軸利益から全round利益へ外挿せず、各roundのpilot負担も含める。

## 6. 静的に確認したSOURCEと最小差分

本表の静読snapshotは固定 `b6305ee2e33436ed96d8aefdee15588676242000`（[tree][BASE]）。これは収録版の識別で、旧実行SOURCEを置き換えない。旧science SOURCE・結果対応は[前回R1の表][R1REUSE]へ戻る。

| 既存file／関数・test | 再利用できる部分 | 追加が必要な部分・制限 |
|---|---|---|
| src/trotterlib/rte.py：finite_rte_distribution、_make_event、iter_sample_rte_events、event_unitary | order/label law、selected_component_ids、phase、application順、normalization。factorized control入力の候補 | 現samplerを実行していない。実qに対するcenteringの取得・finite-bit allowanceとoutcome bindingが未接続 |
| src/trotterlib/rpe_hadamard_interrogation.py、df_rpe_hadamard_compiled_cost.py | 同じcontrolled evolutionにX/Y readout、measurement、状態準備除外の既存builder/価格scope | A1で新量子gadgetは不要。古典metadataと各実測bitを結ぶ小さい後処理adapterが必要。価格較正で同pathをX/Yへ共有しても、実測noiseの共有を意味しない |
| rpe_resource_accounting.py：_hoeffding_shots、evaluate_rpe_round_candidate | round/axis予算、cost provider、alphaの接口 | bounded±1とreference_radius=1の既存仮定をCVへ流用しない。補正後range/varianceと正当な半径を入力できる限定差分が必要 |
| Parent parent_allocation.py：reweight_events/event_statistics/error_accounting、parent_experiment/design.py：ledger | 実q/w、M2/range、bias、資源別価格。旧SOURCE4ccc1832… | 現ledgerのE[w²]はCVの共分散込みvarianceではない。未知covarianceを0として使わず、新しい後処理統計を別SOURCEで接続 |
| G6–G10 generator/kernel/native、whole-path会計 | 有限bit/support/phase/zero分母、strong Closed/CTS baselineの既存定義 | RA再実装・同ensemble比較の復活は不要。C1凍結を解除しない |
| s2_multi_event.py：pair_table/normalizer_enclosure/evaluate_law、既存runner/test | 同finite target・実q・weight・保存native価格・Bernstein会計の分離 | 条件付きsignal covarianceを保存する処理ではない。S2旧pair表を再解析してCV利益を作らず、旧r/degree/regularizer探索もしない |
| docs/research/pr_rpe_v2_branch_success_20261103/run_saved_v2.py：reconstruct_saved_means、evaluate_round | 既存phase/branchの対応。元reconstruct_rpe_phaseの再利用先 | 二項tail部分はCVに不適合。新outcomeのconfidenceをphase許容領域へ渡すだけを差分にする |
| tests/test_rpe_hadamard_interrogation.py：test_wrapper_expectation_matches_wrapped_controlled_signal、test_known_negative_sine_sign_matches_exp_minus_i_energy_time | 既存phase/axis規約を守る人工対照 | SOURCEを静読しただけ。今回のPASSではない |
| tests/management/test_research_c_sampling_law_freedom.py、tests/tracks/algorithm_codesign/test_g6_return_aggregation.py、test_s2_multi_event.py | marginal/weight/zero/有限lawの既存反例と検査位置 | 科学testも人工testも今回は起動しない。CV centering/独立fit/rangeの少数人工検査が後日必要 |

主要SOURCEへの固定直リンクは、[RTE law/event][SRC-RTE]、[Hadamard builder][SRC-WRAP]、[full価格境界][SRC-PRICE]、[RPE会計][SRC-RPE]、[Parent統計][SRC-PA]、[S2接続][SRC-S2]、[V2 adapter][SRC-V2]。既存test・runnerの所在は同じ固定treeの表記pathを参照する。

実装を選ぶ場合でも既存builder/decoderを全面再実装しない。必要なのはcontrol値・既知momentの取得、独立fit/校正の統計後処理、正しいconfidenceと全費用の限定adapterである。新しい回路・sampler・optimizer・管理pipelineは今回作成していない。

[BASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b6305ee2e33436ed96d8aefdee15588676242000
[CV]: https://arxiv.org/html/2502.08735v1
[ML]: https://arxiv.org/html/2604.26865v2
[CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md
[COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/END_TO_END_COST_ANALYSIS_MAP.md
[O2A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/docs/research/o2_sampling_return_technical_survey_20261106/O2_A_SAMPLING_STATISTICS.md
[R1REUSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/docs/research/o2_r1_server_candidate_screening_20261106/O2_R1_BASELINES_AND_REUSE.md
[V2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[SRC-RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/src/trotterlib/rte.py
[SRC-WRAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/src/trotterlib/rpe_hadamard_interrogation.py
[SRC-PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/src/trotterlib/df_rpe_hadamard_compiled_cost.py
[SRC-RPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/src/trotterlib/rpe_resource_accounting.py
[SRC-PA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/src/trottertracks/algorithm_codesign/parent_allocation.py
[SRC-S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/src/trottertracks/algorithm_codesign/s2_multi_event.py
[SRC-V2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/docs/research/pr_rpe_v2_branch_success_20261103/run_saved_v2.py
