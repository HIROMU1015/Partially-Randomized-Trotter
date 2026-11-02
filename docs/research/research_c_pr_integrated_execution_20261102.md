# 研究C：PR-aware DF統合科学実行・複数更新から条件付き資源まで

2026-11-02 JST。保存直鎖H4・隣接1.00Å・STO-3G、number CDF rank8・元frame順LD=2、Nα=Nβ=2/sector36。固定SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4` と[統合契約](research_c_pr_integrated_contract_20261102.json)を無変更でfresh run01へ一度実行した。**Phase A PASS、数値・物理資格付きU更新5回、選定点のPF/RTE13 roundと新native32を評価した。543/543 COMPLETE、FAIL 0、retry/resume0。公開照合後mandatory STOP。**

[全結果](../../artifacts/research_c_pr_integrated/2026-11-02/run01/result.json)／[集計](../../artifacts/research_c_pr_integrated/2026-11-02/execution_summary.json)／[SOURCE](../../artifacts/research_c_pr_integrated/2026-11-02/technical/SOURCE.json)／[call/STOP](../../artifacts/research_c_pr_integrated/2026-11-02/run01/STOP.json)。旧19-callのresumeではなく、その受理点を初期Uとする独立実験である。過去の未使用callを転用していない。

## 1. 固定条件・Phase A（Q1）

γ=.1、ε=1e-8Ha、元f*=2.1839632053132894e-9Ha²/M*=2.2093719370722074Ha、signed symmetric J/real orthogonal U、元g補正κ・nuclear scalar・spin/frame順を維持した。各新Uは元のcold JからNewton-CG＋2 refinementを行い、前点Jのwarm startを使わない。

契約SHA256 `b5fdbe049e611864be84680d6c513fde94dc7019ccf1be1e71c19b75d1c20c3a`、SOURCE manifest SHA256 `7eda4aa319b9b861cfa2b5e8c25e3381b4670c275bdad2bdb073335df550aff5`。207 SOURCE/input35をpre/postflightで照合した。[準備報告](research_c_pr_integrated_preparation_20261102.md)の312 unique人工/回帰PASSは実行前の技術証拠であり、今回の科学結果や外部再現と同一視しない。

指定の旧`accepted_U_update_1.npz`はetaのみのeventである。同runのterminalとindependent stateのJ/U・eta、元Uからのgeometry、旧result/19-call STOPをexactに照合した。旧受理候補のsector norm=8.40106811677e-09Ha、guard=3.22512401671e-09Ha、和=1.16261921335e-08Haで、CA/8=.00019875HaをPASSした。Tensor Frobenius=9.77649661106e-09、general-K/raw CAR max-entry=3.33066907388e-15。これは既存engineering guardによる数値pilotで、厳密DF enclosureではない。

## 2. 複数更新・物理精度・選定（Q2〜Q7）

各受理Uを新しい指数写像chartの中心とし、独立physical tangentからd=-gを更新した。q={1/8,1/32,1/128}、α=F/|gᵀd|×qの有限探索。J一次・rank/nullspace/tail injectivity・latest-pair stability・same-point検査・値尺度Armijo・独立終端を通した候補だけを最適化上受理し、その後に物理予算を独立判定した。

| 点 | F | exact λ [Ha] | J∞ | physical U₂ | sector norm [Ha] | guard [Ha] |
| --- | --- | --- | --- | --- | --- | --- |
| B_initial | 0.0107924679369 | 0.725819387316 | 1.22270773034e-07 | 2.66969428805 | 8.40106811677e-09 | 3.22512401671e-09 |
| update001 | 0.00953383347155 | 0.682184823959 | 1.5417404204e-07 | 2.30271965057 | 7.86534542841e-09 | 3.22480732463e-09 |
| update002 | 0.00844268368307 | 0.641960871139 | 1.15878226115e-07 | 2.00166488456 | 6.62756891759e-09 | 3.22517087971e-09 |
| update003 | 0.00741926353396 | 0.601795161619 | 1.09201308738e-07 | 1.77341648428 | 6.0459695561e-09 | 3.22588652701e-09 |
| update004 | 0.00680011132685 | 0.576137880637 | 7.78424233177e-08 | 1.23314844206 | 6.48447814756e-09 | 3.22846468851e-09 |
| update005 | 0.00661750979453 | 0.568349854686 | 9.5932671593e-08 | 0.658935814277 | 5.23337351802e-09 | 3.22739270291e-09 |

最適化上の受理5回と物理適格受理5回は一致した。A＋B初期＋5受理点の計7物理評価はすべて予算内。新U評価10点、trial9件、inner30回（うちrefinement20回）、独立終端5件。全10点が登録reduced数値資格を満たした。

Q3：Fは0.0107924679369→0.00661750979453、exact λは0.725819387316→0.568349854686。Jは全受理点MET、U₂は2.66969428805→0.658935814277だがNOT_METである。J残差が単調に改善したという主張ではない。

Q4：停止理由は`FINITE_STEP_GRID_NO_ACCEPT`。5更新後の次方向で登録3qがすべて数値適格だったが、Fが上昇し値尺度Armijoを満たさなかった。数値未資格・correctness FAIL・DF予算FAILや64更新上限ではなく、登録有限step探索の不成立である。さらに小さいstepに降下点が存在しないことを示さない。

| update attempt | q | α | Ftrial | 実測F減少 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.125 | 0.000189281302374 | 0.00953383347155 | 0.00125863446537 | ASSESS_BEFORE_ACCEPTANCE |
| 1 | 0.125 | 0.000224747795546 | 0.00844268368307 | 0.00109114978848 | ASSESS_BEFORE_ACCEPTANCE |
| 2 | 0.125 | 0.00026339516238 | 0.00741926353396 | 0.00102342014911 | ASSESS_BEFORE_ACCEPTANCE |
| 3 | 0.125 | 0.000294882731016 | 0.00680011132685 | 0.00061915220711 | ASSESS_BEFORE_ACCEPTANCE |
| 4 | 0.125 | 0.000558978776285 | 0.006930489997 | -0.000130378670153 | REJECT_ARMIJO_OR_VALUE_SCALE |
| 4 | 0.03125 | 0.000139744694071 | 0.00661750979453 | 0.00018260153232 | ASSESS_BEFORE_ACCEPTANCE |
| 5 | 0.125 | 0.00190510264557 | 0.00728608506573 | -0.000668575271202 | REJECT_ARMIJO_OR_VALUE_SCALE |
| 5 | 0.03125 | 0.000476275661394 | 0.006747901738 | -0.000130391943473 | REJECT_ARMIJO_OR_VALUE_SCALE |
| 5 | 0.0078125 | 0.000119068915348 | 0.00662722819999 | -9.71840545499e-06 | REJECT_ARMIJO_OR_VALUE_SCALE |

Q5：事前規則（物理・数値適格履歴のexact λ最小、tieはF/早いindex）で`update005`を選定した。[selected.npz](../../artifacts/research_c_pr_integrated/2026-11-02/run01/selected.npz)のJ/Uは物理評価candidate SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`に結び付く。最終点を無条件に採用したのではなく、全履歴に同じ規則を適用した。選定点のtensor Frobenius=3.29090997756e-09、sector norm+guard=8.46076622093e-09Ha。

Q6：登録した旧PR-aware Stage Aのexact λ=1.44242988558Haに対し、新λ=0.568349854686Ha、new/old=0.394022517399（約60.60%低下）。定義・LD・rankをそろえた有限endpointの比較で、最適化済み母集団の最良性は示さない。

Q7：J/U同時一次収束は未達、選定候補はunconvergedである。DF予算を保ちながら複数更新でF・exact λ・U残差を下げられたことが今回の科学的改善。Latest-pair一致や独立勾配比較は真のreduced-gradient誤差上界・厳密降下保証ではなく、それらはnullのまま。

Raw Newton-CG30件のstatus内訳は`{'0': 28, '2': 2}`、登録J基準の内訳は`{'MET': 30}`。Raw precision lossを正常optimizer収束へ読み替えない。nit合計4075、nfev/njev各8259、VG8349、Hessian4077。Raw x/jac・pre-step規約・全traceを保全している。

## 3. 新候補固有のPF/RTE・shots（Q8）

旧候補の値を移さず、選定J/UからHamiltonian・signed tail law・ground/alias・partial ordered S2を再生成した。δ=h=.1、s=1/r128/K0、m0..12、own-DF ground oracle、状態準備費用除外。Sector9因子general-K/raw CARとtail/law、full256-sector対応、ground/aliasはPASS。

| m | PF operator error | 補正RTE operator error | attenuation | 登録PF/RTE状態 |
| --- | --- | --- | --- | --- |
| 0 | 6.31574602303e-05 | 5.06218777879e-06 | 0.999987382051 | POTENTIALLY_FEASIBLE |
| 1 | 0.000125932745527 | 1.01147746585e-05 | 0.999974764262 | POTENTIALLY_FEASIBLE |
| 2 | 0.000248852021026 | 2.01531196337e-05 | 0.99994952916 | POTENTIALLY_FEASIBLE |
| 3 | 0.000474946273723 | 3.97093875252e-05 | 0.999899060868 | POTENTIALLY_FEASIBLE |
| 4 | 0.00080413030634 | 7.50808952578e-05 | 0.999798131925 | POTENTIALLY_FEASIBLE |
| 5 | 0.0010832944683 | 0.000125980311981 | 0.9995963046 | POTENTIALLY_FEASIBLE |
| 6 | 0.00155762814569 | 0.000221344033793 | 0.99919277217 | POTENTIALLY_FEASIBLE |
| 7 | 0.003020752509 | 0.00040295373523 | 0.998386195957 | POTENTIALLY_FEASIBLE |
| 8 | 0.00593965763839 | 0.000754410453768 | 0.996774996277 | POTENTIALLY_FEASIBLE |
| 9 | 0.0118366582096 | 0.00150331972361 | 0.993560393204 | POTENTIALLY_FEASIBLE |
| 10 | 0.0236581789964 | 0.00300550103488 | 0.987162254944 | POTENTIALLY_FEASIBLE |
| 11 | 0.0472918511388 | 0.00601420229869 | 0.974489317585 | POTENTIALLY_FEASIBLE |
| 12 | 0.0945468045945 | 0.0120636015991 | 0.949629430088 | POTENTIALLY_FEASIBLE |

[旧M4公開報告](research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md)の既報値を引用すると、m12はPF .0716149979/RTE .0609453085/A .716844438であった。今回のm12はPF .0945468045945/RTE .0120636015991/A .949629430088。**補正RTEとattenuationは改善したが、PF誤差は増えた**。λの減少から全誤差の単調改善を仮定しない。旧shotsはideal16056/assumed16120という既報値で、新shotsは以下の候補固有評価。旧PF/RTE/shotsの再計算はしていない。

13/13 roundが`POTENTIALLY_FEASIBLE`。これはCA達成ではない。CA=.00159Ha・DF reserve1/8・βtotal=π/6・βstat=π/12・failure total=.01/axis=.01/26の既存RPE/Hoeffding十分式を再利用した。Ha単位のDF誤差とoperator/signal誤差を根拠なく単純加算せず、既存signal/energy接続で別々に扱う。

| lane | 同精度十分条件（宣言モデル内） | 総shots | RZ条件付き総費用 |
| --- | --- | --- | --- |
| evidence | False | null | null |
| ideal_exact_native | True | 14048 | 207919150188 |
| assumed_raw_signal_budget | True | 14104 | 208893540202 |

`evidence`は実装lawと全law synthesisが未資格なのでshots/総費用をnullとした。`ideal_exact_native`は理想合成・ideal finite-RTE IIDの反実条件。`assumed_raw_signal_budget`は既存.001A synthesis＋.001A lawを仮定した反実条件で、実際の未知誤差を0または既知boundに認定していない。Counterfactualのsame_accuracyは実CA達成・保証shotsを意味しない。

## 4. 新native32・条件付き費用（Q8〜Q9）

新候補の独立32 PCG64 trajectory（各128 events、新cohort/seed、非適応・旧標本poolなし）を生成し、固定P0（Qiskit1.3.0/opt1/seed17/rz,sx,x,cx等）でcompileした。Full256 CAR、full512 raw/compiled reference、actual adjoint、complex X/Yとunitarityを比較した。既存StrongBuilderのrelative head・exact basis-key tail共有、共通head/X/Y shellを使い、未実装の追加融合を仮定しない。

32 build/compile/各種full検査・metrics、128 QPYを保存。有限回路operator検査は全sampling lawの合成誤差上界ではない。各coherent occurrenceは独立trajectoryという法則を維持し、1標本回路の同一反復で量子費用を過小評価していない。32は古典校正標本数で、量子shots数ではない。

| metric | 旧PR平均 | 新平均 | 新mean SE | 新/旧価格 | 今回の観測範囲 |
| --- | --- | --- | --- | --- | --- |
| rz_count | 14076.4375 | 19825.625 | 214.461468031 | 1.40842631525 | 17668..22066 |
| rz_depth | 4342.28125 | 5632.1875 | 57.5863383619 | 1.29705727836 | 5093..6314 |
| cx_count | 5579.4375 | 7399.3125 | 66.6277338224 | 1.3261753537 | 6754..8114 |
| cx_depth | 3424.875 | 4152.59375 | 29.8969621452 | 1.2124803825 | 3878..4502 |
| total_depth | 9574.53125 | 12393.1875 | 124.337392495 | 1.29439104395 | 11227..13870 |
| circuit_size | 28014.4375 | 39603.9375 | 433.579953973 | 1.41369740156 | 35320..44214 |

Raw32の全成分比較max=5.72875080707e-14、compiled32=4.28789615561e-12、actual adjoint32=4.29228552179e-12（元component allowance1e-10）。最大complex X/Y差はraw 1.66278832933e-14／compiled 1.20642488545e-13（元allowance2e-10）。32/32すべてPASS。Frobenius/max-entry観測を全law operator norm boundへ昇格させない。


総費用は新候補のshotsと新32標本の平均価格を使う条件付き点推定。校正SEは共通平均に対する総重みWで伝播しており、13 roundを独立校正として扱わない。標本min/maxを全law support・厳密上界にせず、母平均優位・actual CA・保証付き総資源優位は未資格。

| lane | shots | kernel重みW | 総RZ点推定 | 総RZ descriptive SE | 旧PR比 | 保存強D比 |
| --- | --- | --- | --- | --- | --- | --- |
| ideal_exact_native | 14048 | 10487392 | 207919150188 | 2249141484.14 | 0.888096337507 | 2.0419077076 |
| assumed_raw_signal_budget | 14104 | 10536540 | 208893540202 | 2259681836.37 | 0.888320652609 | 2.05147688138 |

Ideal laneは旧PR総RZ 234117788135→207919150188（約11.19%減）。Exact λは旧Stage Aより約60.60%減だが、kernel平均RZは14076.4375→19825.625（約40.84%増）。旧報告のshots16056/W16631888に対し、新shots14048/W10487392が価格増を一部相殺した。すべてのPF誤差・kernel価格が改善したわけではない。Assumed laneも旧PR比0.888320652609という条件付き点値である。

保存強Dの主RZ benchmarkは101825929455で、新PRはidealで2.04190770760倍、assumedで2.05147688138倍。主RZでこれを上回っていない。一方idealのCX depthは保存強D比0.751387249051で、全副指標で支配されたという主張でもない。比率は有限cohort価格と宣言した誤差仮定の点比較であり、母平均優位・有意差・厳密同CA優位の判定ではない。


Q9：PR内部の候補固有比較と、強い決定論PFを上回る主張を区別する。保存強Dは全frame S2/s1・旧PR J/Uの固定benchmarkで、新J/Uでの決定論回路を今回生成していない。CA/time/failure/serial metrics/準備除外の共通部分はあるが、Dの数値合成資格とPRの全law誤差仮定は同じ証拠水準ではない。RC2も別の有限最適化endpointで、pairedかつ収束済みoptimizerの性能比較ではない。既存RC2/XDFを今回再最適化した、あるいは未資格結果を0補完して順位付けしたとは主張しない。

## 5. Calls・資源・保全

| logical unit種別 | 実績 |
| --- | --- |
| saved_load | 1 |
| initial_identity | 1 |
| physical_context | 1 |
| phase_A_assessment | 1 |
| objective_setup | 1 |
| optimization_driver | 1 |
| inner_J_solve | 30 |
| precision_state | 20 |
| reduced_admission | 10 |
| physical_assessment | 6 |
| independent_terminal_assessment | 5 |
| selection | 1 |
| reference_comparison | 1 |
| performance_driver | 1 |
| export | 1 |
| C.model_prepare | 1 |
| C.sector_factor_gate | 9 |
| C.law_prepare | 1 |
| C.sector_law_gate | 1 |
| C.ground_reference | 2 |
| C.kernel_reference | 1 |
| C.exact_evolution | 13 |
| C.round_reference | 13 |
| C.round_classification | 13 |
| C.feasibility_summary | 1 |
| C.shell_metrics_load | 1 |
| C.signal_qualification | 3 |
| C.round_accuracy | 39 |
| C.energy_assessment | 3 |
| C.shots_summary | 3 |
| C.car_context | 1 |
| C.raw_factor | 9 |
| C.full_sector_gate | 1 |
| C.reference_context | 1 |
| C.full_law_gate | 1 |
| C.gaussian_build | 9 |
| C.relative_gaussian | 4 |
| C.shell_load | 6 |
| C.shell_assessment | 3 |
| C.trajectory_sample | 32 |
| C.kernel_build | 32 |
| C.trajectory_reference | 32 |
| C.raw_operator_gate | 32 |
| C.compile | 32 |
| C.compiled_operator_gate | 32 |
| C.adjoint_pair_gate | 32 |
| C.XY_gate | 32 |
| C.kernel_metrics | 32 |
| C.calibration_summary | 1 |
| C.cost_calibration | 1 |
| C.serial_cost | 26 |
| C.total_aggregation | 3 |
| C.calibration_uncertainty | 3 |
| C.resource_comparison | 1 |


親81＋C462＝543/1759 units、COMPLETE543・FAIL0、retry/resume0。新ffsim fit・外部outer BFGS・RC2継続は0だが、inner Newton-CGの実反復は上記のとおり実施した。旧19/49/60/480-call等のSTOPと残枠は維持した。

[資源](../../artifacts/research_c_pr_integrated/2026-11-02/run01/resources.json)／[C資源](../../artifacts/research_c_pr_integrated/2026-11-02/run01/C/resources.json)。実行前affinity32・MemAvailable約54.7GB・load1.10/pressure0、Cは既存resource_planで32 workers/BLAS1。逐次U/line search/refinementは順序を保ち、独立13round/32nativeは並列化、Qiskitの既存排他条件を維持した。全体wall=1360.08937579秒、親CPU=20.675034497秒、C wall=1339.62005796秒/CPU=15875.3047881秒、親peak RSS=452988KiB/child peak=3415152KiB。親/子wall・CPUの包含に注意し、単純和や未測定speedupを主張しない。

[Postflight](../../artifacts/research_c_pr_integrated/2026-11-02/execution_postflight.json)でSOURCE207/input35、旧SOURCE19組/raw2579件、旧契約・grant/ledger/marker/STOP、準備技術証拠の不変性を照合した。旧137科学recordsは保全し今回1件を追記、当日noteはappend-only。科学STOP後は計算済みJSON/JSONLの報告集計・hash照合・公開だけで、追加NPZ数値decode/queryは0。ローカル証拠でありimmutable CI/外部再現ではない。

## 6. 次の研究判断（Q10）

今回の限定H4結果は、物理予算を保ってλを下げるPR-aware reduced更新と、その候補固有の性能会計への接続を支持する。一方、U一次未達・有限step探索不成立・全law合成/law/support未資格を残す。ここから系サイズ一般化や無条件の資源優位は導けない。

今回の主指標では、λ60.60%減が総RZでは約11.2%減にとどまり、保存強D主RZの約2.04倍である。したがって、H6への拡張を直ちに優先する根拠は十分ではなく、まずH4で候補固有のbasis/価格変化と強baselineの公平性を確認する選択が自然である。これは今回の限定証拠に基づく研究選択肢で、方式全体のNO-GOや正式終了判断ではない。

次の選択肢は**ユーザー／研究C**が判断する。H4での方法検証を優先するなら、**Codex**が選定点の局所方向・inner精度と有限step尺度を切り分ける契約、および同じ新J/Uを使う強決定論baselineの最小比較を準備する。これは提案であり今回のSTOP後に実行しない。系サイズ拡張は、この未収束とbaseline比較の限界を受け入れた独立一般化実験として設計する選択肢であり、H6への自動科学実行は行わない。新しいGPT詳細科学レビューも開始していない。

Branch `codex/research-c-pr-integrated-20261102`。SOURCE/契約は固定commit、結果/raw/QPY/call/STOP・本報告を別commitで公開しremote SHA/tree・公開本文・SOURCE/input/結果/STOPを照合する。**統合範囲を完了しmandatory STOP。**
