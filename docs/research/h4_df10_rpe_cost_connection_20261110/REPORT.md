# H4 DF10：RPE費用会計・q依存価格の接続確認

2026-11-10 JST。**会計の人工・回帰14件PASS、D1保存N/Cの3ケースを準備した。実H4のN×Cは未実行。** 既存branch `codex/resource-eval-rpe-shot-error-connection-20261110`を継続し、D1科学SOURCE・結果と旧STOPは変更していない。

- 現在の会計は、明示した入力条件とcoverageの下で軸別N×Cを正しく扱える。既存の全round関数を再利用し、m=3だけの小計を全RPE費用へ誤表示しない最小adapterを追加した。実H4費用の検証結果ではない。
- DF10の直接価格は候補別に疎で、D1のT=.8ではPF決定論・PR探索/確認の単価が揃う。今回の固定資料には、DF10のRPE長時間域へ資格済みの既存価格モデルは見つからない。外国条件の係数は移植しない。
- **次に必要なのは保存T=.8のRZ小計3ケースの限定数値確認**で、新signal/fit/compileは不要。全13roundの比較には他roundのNと価格資格が必要だが、全qの新compileを現時点の必須作業にはしない。[最小数値計画](MINIMAL_NUMERICAL_PLAN.md)。

## 1. 条件・SOURCEを固定した入力準備

[完了D1][D1]の条件を保持する。Linear H4/1Å/STO-3G、tol-only 1e−8の保存DF10、Nα=Nβ=2、target保存DF-H同sector ground、保存数値state条件付き、P=0。εE=.00159360010199040 Ha、αtotal=.05/方式、βtotal/stat=.4/.2、h=.1、m0…12。D1のNは再計算しない。PF B0/prefix7は切捨てを含みFull PFとは異なる。PRはprefix1/R_outer4/K2、両者S2/δ=.1/qPF8/T=.8/m3。

| 準備ケース | candidate ID | 保存N_X / N_Y | 保存C_RZ,X = C_RZ,Y | 古典価格標本n |
|---|---|---:|---:|---:|
| PF決定論 | `49709717a6c3d405c60d` | 705 / 705 | 44411 | 1 |
| PR探索 | `14bd81a375400c9b0305` | 977 / 977 | 16857.11111111111 | 9 |
| PR独立確認 | 同上 | 977 / 977 | 16947.78787878788 | 33 |

[PREPARED_INPUTS.json](PREPARED_INPUTS.json)は固定D1 resultと準備SAVED_POINTSからのcopy/joinで、全資源metricの保存単価も保持した。探索/確認は独立ケースでありpoolしていない。各ケースのsubset_resources/full_rpe_resourcesはnull、resource_products_evaluated=false。入力SOURCE/file hashと元D1科学SOURCE311255aaを保持する。実H4のaccount_subset、既存aggregate、N solverをsentinelで遮断して準備を実行した。

保存Nはideal条件付きの十分数でactual Nではない。PR価格はPCG64/NumPy float64 CDFによる有限cohortであり、ideal finite-RTE lawとの距離・希少eventの期待費用・compiled実装誤差は未資格。law_contextにこの差を明記し、同一宣言contextだけをjoinするが、物理的law同一性の証明とはしない。有限標本平均を**明示的な条件付き価格proxy**として扱う場合の会計準備であり、真の母平均費用やactual成功保証にはならない。

## 2. 既存会計の確認と最小拡張

| 既存実装 | 確認した機能・制約 |
|---|---|
| h4_df10_rpe_connection.aggregate_supplied_axes | 軸別N、candidate/binding/compiler/full-wrapper、direct/model、全round coverage、幅のmax。expected_roundsは0始まり連続、m3単独では使えない |
| c1_stage3_accuracy.aggregate_totals | N×各軸単価の既存算術。X/Yは共通scalar shots、同scopeとcoverageを要求。accuracyフラグだけで本研究の全RPE成功が証明されるわけではない |
| Track D B6b aggregate_registered / B-C | 固定B4の共通Nと同CDF8/同P0のStrong/Reference価格を接続。共有校正/round間共分散も別componentで保持。DF10/PF-PRへ固定Nや係数は移植不可 |
| 旧DF12 RPE | rpe_resource_accounting / research_direction_ablation / decision_cost / full_opt2系にshot×単価・総和がある。V3/R1の保存72軸表は単軸とBOTH_AXESの列を区別。signal/N/providerを含む旧runnerは起動しない |

新規部分は[h4_df10_rpe_cost_connection.py](../../../src/trottertracks/h4_df10_rpe_cost_connection.py)。保存case準備・入力照合・部分roundの積和だけを追加し、全roundが揃う場合は既存aggregate_supplied_axesへ委譲する。旧adapterやD1計算コードを変更していない。部分会計は実mを保持し、未取得roundのN/Cを0で埋めない。full_rpe_resourcesは全入力roundが揃った場合だけ設定され、それでもenergy成功の認定は行わない。

測定符号は元SOURCEの`ax2b_native_df_v5.build_native_hadamard_wrapper`と`h46_long_time_native.trial`を静的確認した。control-0がreferenceで、H→controlled evolution→(sineのみSdg)→H→測定、P(0)−P(1)がcosine=Re(zraw)、sine=Im(zraw)。X↔cosine、Y↔sineを明示し、符号/軸を取り違えた入力は拒否する。ここで新しい回路をbuildしていない。

| 今回検査した論点 | 実装・人工検証 |
|---|---|
| Nと同candidate C、PF/PRで異なるN | candidate全tuple/H/state/snapshot/control/compiler/scope照合、独立ケース、軸別Nで積和 |
| 量子Nと古典n、normalization B | n/Bを積和の乗数へ使わない。BはD1のraw測定Nへ既に反映、N再計算なし |
| RZ/CX/depth/sizeの単位 | gate_count / circuit_layers / instruction_countを明示。depth加重和は累積circuit-layer費用で、実機wall timeではない |
| logical qubits | 保存幅のmaxのみ。今回の元価格では幅欠測なのでnullを維持、Nやround数で加算しない |
| direct/model/missing | direct分類とmodel SOURCE/q変数/coverageを分離、欠測や未知Nは算術結果なし |
| 部分/全round・cohort | m3小計専用statusと不足round一覧、全費用はnull。軸間のcohort混在や二重登録を拒否 |
| 不確かさ・科学資格 | total covariance/CIはnull、actual/全RPE成功はfalse。未知相関や外挿biasを0へ置換しない |

[人工・回帰記録](ARTIFICIAL_TEST_RESULTS.txt)は14件PASS。旧fixtureと既存2会計器を再利用し、部分m3・不等NX/NY・異なる方式N・n/B非乗算・全round委譲・旧scalar会計との一致・幅・軸/単位・外国binding・欠測・cohort・model q・α・準備の非数値性を検証した。初回α超過fixtureは和が上限ちょうどだったため、超過値へ修正して通過した。人工結果を実H4資源の数値検証、独立科学レビュー、immutable CIと呼ばない。旧15/7人工testの保存記録は再利用し、科学D1は再実行しない。

## 3. q・TとDF10直接価格coverage

qRPE=2^m、T_m=.1 qRPE。qPFは1回のT発展のPF分割数、δ=T/qPF。D1でqPF=qRPE=8になるのは採用δ=hの結果で、任意候補で成立する恒等式ではない。T固定でqPFを変える掃引ではδが変わる。RPE時刻増加の価格を、その掃引の単価比例だけで外挿しない。

[COVERAGE.json](COVERAGE.json)は主研究155529e2のbase/extension signalと6価格表に対する**候補メタデータの照合だけ**。D1の2構成系列に限定した48行、各candidate ID・T/δ/qPF・cohort・n・直接価格SOURCE・alias・model欠測・N所在を持つ。新しい平均/残差/比/fitは作っていない。

| 保存T | PF B0/S2/p7の直接価格qPF | PR S2/p1/R4/K2の直接価格qPF | .1×2^mとの関係 |
|---:|---|---|---|
| .8 | 1,2,4,8 | 8（探索n9/確認n33） | m3。RPEの保存Nがあるのは指定q8だけ |
| 1.2 | 4,6,8,16 | 16 | schedule外。mへ丸めて使わない |
| 1.6 | 2,8,16,32 | 32 | m4。δ=.1のPF q16は直接、PR q16は欠測 |
| 2.0 | 4,8,10,16,32 | なし | schedule外 |

これら以外のprefix/formula/R/Kにも主研究の直接価格があるが、同一構成と見なして置換しない。PF q1/2/4/8の再利用rowがexploration表にもあるのは同じ価格のaliasで、独立標本を増やさない。PRのreused_confirmation表はこのcarrierでは空。元価格compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。保存profileはreference-only、別compiler/domainへの資格を追加しない。

この限定corpusには、同DF10/RPE長時間条件の適格Ĉ、直接CとそのĈの対応残差、長q予測CIはない。別資料の有限時間格子用モデルも、同DF10・同候補/時間域/law/compilerへの適用証拠がなければ接続しない。別prefixの直接点も同一構成の証拠とはしない。全私有資産にモデル/価格が存在しないと断定していない。

## 4. 価格モデルの再利用範囲

[価格調査][PRICE]と[固定索引][PRICEINDEX]の保存検証を引用し、同じ残差・bootstrapを再解析していない。

| 方法・既存domain | 入力/資源/検証 | DF10への再利用と不足 |
|---|---|---|
| D Affine F/A/S | CDF8/LD2/S2/s1/r32/K0/h=.1、Strong/Reference full P0、各絶対価格とpaired差。Fはq2/4固定、Aはq2…32の88path fit、Sは[1,q,√q] | whole-path校正・独立q holdout・モデル差分の方法/コード。係数、208−224q、改善率を移植不可 |
| D basis/lane-run | 同CDF8のevent feature→RZ/size/depth。88pathの探索CV、保存残差/共有bootstrap係数共分散 | DF10側の対応featureと未来feature lawが必要。E[max]はmax(E)で代用不可。新特徴抽出/fitは今回0 |
| 構造的count/max-plus | 確定native列のcount/depth/幅とwire frontier、固定q4/q8の有限P0 bridge | SOURCE・interface・単位の方法。DF10 native列や一般P0後価格/全law期待価格の資格は未取得。安易なq比例を与えない |
| legacy DF12 V3/R1 | δ=.02/LD3または12/K2・opt2、q1/2 anchorとq8等holdout、共有anchorと感度 | direct/model表・固定N感度・相関を保つ会計方法。DF10 opt1へ係数/共分散を移植不可。R1を再実行しない |
| Track C/newA | compiler A modular/weighted P3価格、独立holdout・shared-cell共分散／native DAG・max-plus・別Reference/readout | 校正/holdout設計、scope分離の方法。weighted range・別law・別compilerのN/C/改善率は本研究へ数値移植不可 |

D q64は後続限定完了でraw8/post32/pairedχ8が保存済み。q64を未完了扱いしない一方、q≥128の直接価格と外挿biasは未資格。q8/16/32は凍結Fの検証で、A/Sの探索CVとは独立性が異なる。paired CX差の整数残差0だけでは、両方式の絶対期待価格を確定しない。q64の方式/軸/資源共分散、C7-4のq内36量/係数共分散、DF12の共有anchor結果を再利用できるが、DF10共分散の代わりにはならない。

直接/予測のずれを将来見る際は、表示丸め（保存精度と表示桁）、有限cohort変動（paired X/Y・共有校正）、構造モデルbias、ideal/finite-bit law差を分ける。旧DF12のq16/32 holdout残差とR1感度は既存結果であり、新規作業として要求しない。DF10の新残差/ΔG/感度は未計算。

## 5. 全13roundへつなぐための最小構成

| round / 時間 | N | 同構成の直接C | 適格Ĉ・追加事項 |
|---|---|---|---|
| m0…2 / .1,.2,.4 | 未評価 | このcorpusに同時刻の価格なし | initial round signalが必要。Ĉ未資格 |
| m3 / .8 | 指定PF705/PR977、条件付き保存済み | 3ケース準備済み | モデル不要。次は限定RZ小計だけ |
| m4 / 1.6 | 未評価 | δ=.1のPF q16あり、PR q16なし | PR q32/δ=.05には別候補の直接価格あり。候補とNを別途採択すればcompile不要の可能性 |
| m5…12 / 3.2…409.6 | 未評価 | このcorpusの時間域外 | 同H/stateのsignal・候補構成・価格モデル/直接価格の資格が不足 |

NとC/Ĉを別に保持して、まず固定Nに対する価格問題を切り分ける。将来のΔG_m=Σ_a N_m,a(C_m,a−Ĉ_m,a)は同candidate/単位の直接/モデルが揃い、別途数値許可がある場合だけ。全13roundでは未知のN/C・covariance・外挿biasを埋めない。DF10のround寄与やPF/PR総費用順位は今回判定していない。D/DF12で長q寄与が大きかった既存結果は検証優先度の参考で、DF10の寄与率ではない。

追加作業の優先度は次の通り。

1. **今すぐ新signal/fit/compileを要しない**：保存m3のN×C 3ケースを、別途明示許可後に一回検証。これは最小の会計接続確認。
2. **全RPEのNに本当に不足するもの**：他roundの同条件signal/reference/誤差資格。保存T1.6はまず再利用可能だが、ユーザーが今回の次案より先にその評価を許可したわけではない。未取得の初期/長時間signalは将来scopeを限定する。
3. **価格で不足するもの**：採択候補が決まってから既存直接価格を優先。T1.6の別q32のように保存価格を持つ候補もある。欠測の同candidateだけについて既存資産確認、同DF10のmodel検証またはcompileを必要に応じて提案する。新fit/全長q compileを既定にしない。
4. **保証・順位の判断**：ground/reference/u/実装/law誤差と全roundの価格不確かさを別に補う。順位がmodel不確かさに左右される場合は未確定を維持。複雑なshot配分/PR再最適化はその後に必要性を判断する。

## 6. 実施範囲・成果物・STOP

- [module](../../../src/trottertracks/h4_df10_rpe_cost_connection.py)、[入力準備CLI](../../../scripts/prepare_h4_df10_rpe_cost_connection.py)、[人工test](../../../tests/check_h4_df10_rpe_cost_connection.py)、[14件記録](ARTIFICIAL_TEST_RESULTS.txt)。CLIにN×C実行modeはない。
- [PREPARED_INPUTS](PREPARED_INPUTS.json)、[COVERAGE](COVERAGE.json)、[最小数値計画](MINIMAL_NUMERICAL_PLAN.md)。実N×Cやモデル予測/残差の成果物ではない。
- D1保存carrier `ffe8a71da98bec330ce1af109a9606f14e1fc598`、D1科学SOURCE `311255aaac39dc1c3321108c296bde7dc93dd543`、準備e91021ae2。元主研究carrier155529e2、signal科学SOURCEe417ffc9f。元SOURCE/provenanceを今回の公開commitへ付け替えない。
- 既存会計・モデルの本文確認、公開保存値の選択/照合、人工14件、入力表とschedule metadataの準備のみ。**実H4 N×C/新N/新残差・感度/fit・bootstrap/signal/sampling/回路build・compile/RPE simulation/新最適化/H6計算/全13round集計/詳細科学レビュー/Main Claim変更は0。**

H6担当worktree・進行中作業へ操作/制約を加えず、branch統合/削除/履歴変更も行わない。新しいapproval証明機構、grant、ledgerは追加しない。SOURCE/input/raw/結果/marker/KEEP/STOPを保全した。GitHub commit/push後、remote SHA/tree・公開成果物を照合して**mandatory STOP**。次担当はユーザーによる最小数値計画の実行判断、Codexはその明示許可後だけ実H4小計へ進む。

[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ffe8a71da98bec330ce1af109a9606f14e1fc598/docs/research/h4_df10_rpe_d1_20261110/REPORT.md
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caa592a807c533c36e06a94d2fd4e62af4a38734/docs/research/rpe_price_reuse_survey_20261110/REPORT.md
[PRICEINDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caa592a807c533c36e06a94d2fd4e62af4a38734/docs/research/rpe_price_reuse_survey_20261110/SOURCE_INDEX.md
