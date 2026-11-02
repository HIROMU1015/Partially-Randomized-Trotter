# 研究C：PR-aware DF Stage C統合準備・総資源評価設計

2026-10-27 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。

**保存B1/B2のscalar証拠を、既存RPE/Hoeffding・共通CA・serial native費用へ接続する専用経路を完成した。人工23＋回帰50＝73 tests PASS。新規実分子科学callは0。** 実H4のshots・総費用・新しい誤差を計算していない。C1は未承認164-callの保存値再解析案、C2はC1後の選択を必要とする非実行草案である。

## 1. 目的・SOURCE・固定成果

中心課題は、PR-aware DFで得たtail mass／attenuation／補正RTE／native単価の改善が、同じ最終エネルギー精度を満たすinterrogation総資源へ伝わるかである。状態準備込みの無条件総資源や一般的優位は今回の目標に含めない。

| 段階 | 公開commit／科学SOURCE／実装 |
|---|---|
| Stage A | 公開`9ae5942d02572498b6ee1b5f309ca504ed58c6ef`、科学`e1085c95f5c01b68566e17d543ace5832cb94061`、実装`5a185964090f2cb8d19d445d41cf520c4d8ed2c7` |
| B1 | 公開`1331ac5df02272cc6edfd46d7de8d69f70ce882b`、科学`7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9`、実装`366cb090a2c05c6d7ffeafdba2443bfc64b8a035` |
| B2 | 公開`5645bd70e1a6115df9ac8b6f64be3c0da5499666`、run02科学`a1cf0046967c206f414dbad568a75884ee076ebf`、修正実装`3e1e8d1446de04b9ebec5c0bcf455e7e7a47d17a` |
| 新Stage C実装 | `e9f704e330e8d281f3f136524997f8e156efe8ff` |

[Stage A](research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md)では同一初期値・通常number CDF8のBaseline／Proposedで物理DF予算を保持し、exact tail massを34.7246%削減した。[B1](research_c_pr_aware_df_stage_b1_execution_20261027.md)は141/141完了、両13/13 round潜在適格。[B2](research_c_pr_aware_df_stage_b2_execution_20261027.md)はrun01のQiskit技術STOP429を保全し、fresh run02で652/652完了、64 full512 correctness PASS、固定32標本/armでRZ標本平均が約8.32%小さかった。これらは保存結果の引用で、新しい計算や科学レビューではない。

B2 run01とrun02は同じsampling SOURCEを用いた技術再実行であり、**比較対象はrun02の32標本/armだけ**。114独立標本に合算しない。保存SOURCE identityとsampling digestを分けて保持し、旧STOPを変更・resumeしない。

- Baseline final J/U SHA256：`f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02`
- Proposed final J/U SHA256：`f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`
- 元input SHA256：`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`
- B2 sampling digest：`cdf79ed569127f5cb6c7ee017b76bce6422166a656bb3ae866de6e86917c45bf`
- [新SOURCE60件](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/technical/SOURCE.json) SHA256：`bc2620634a6560e0492337547db0574451dd6576749bcf4e502e4d1ab99088f0`
- [未承認C1契約](research_c_pr_aware_df_stage_c1_contract_draft_20261027.json) SHA256：`d2256190c1b0e5c5daf4a51e90e13d727e5836f7a0218321356f2eab85fd08ee`

保存H4・隣接1.00 Å・STO-3G、Nα=Nβ=2／sector36、通常CDF8両arm、元frame順LD2、元g補正κ・核scalar一度、partial symmetric S2、δ=.1／s4／h1/40／r16／K0、m0..12を維持する。CA=.00159 Ha、DF reserveCA/8、βtotalπ/6・βstatπ/12、候補failure.01・axis.01/26、own-DF ground oracle・状態準備費用除外。共通Gaussian／relative fusion／P0を引き継ぎ、新fit・γ/rank/frame探索はない。

## 2. 再利用とStage C接続

新規は`c_pr_aware_df_stage_c.py`、`run_research_c_pr_aware_df_stage_c.py`、同名management test。旧SOURCEは変更しない。

| 既存SOURCE／処理 | 今回の再利用と適合範囲 |
|---|---|
| `c1_stage3_accuracy.AccuracyTarget`／`assess_round` | 同じCA・phase room・failure、保存PF/RTE/A、raw complex-signal誤差から既存shots式へ接続 |
| 同`assess_energy` | 全13round・DF予算・own-oracle／alias条件、Ha誤差を一度だけ加算 |
| `rpe_resource_accounting._hoeffding_shots` | 既存ceil式をそのまま呼ぶ。新推定器を実装しない |
| `bounded_cost_intervals` | run02の32 kernel costs/armから点推定。support不明なら区間null。C2向け人工analytic support受入れも検証 |
| `c1_stage3_connection.compose_serial_cost` | n=s2^mのkernel反復＋head/X/Y shell、serial scheduled depth、校正標本数とshotsの分離 |
| `aggregate_totals` | 同精度条件を満たす場合だけ52 arm/round/axis費用をshots加重。未適格候補を混入させない |
| `c_pr_aware_df_stage_b*` | arm identity、保存PR政策、B1/B2 call分類・native profileの照合だけ。旧science runnerは起動しない |
| `c1_stage3_qualification`／`round_synthesis`／`repeat_bound` | 全law資格要件、supportと標本極値の区別、将来のuniform誤差伝播設計の出所 |
| `Calls`／`resource_plan` | 科学call／依存STOP／実資源に基づくscalar task並列だけ再利用 |

既存`evaluate_evidence_packet`は単一qualificationでround／energy／costを評価する部品であり、assumed scenario・未知証拠・別lawの校正を分離するschemaは持たない。このためその内部の`assess_round`／`assess_energy`／`aggregate_totals`を直接再利用する最小adapterにした。旧`compare_totals`はCDF/XDF固有のverdictを出すため呼ばず、Baseline/Proposedの点推定差・比を返す小さな比較処理だけを追加した。旧CDF/XDFの数値や判定labelは流用しない。

```text
Stage Aのexport bytes/hash ── armのidentityと来歴のみ
B1 result／ground／STOP ────── 保存DF・13round PF/RTE/A/signal・alias
B2 run02 SOURCE／STOP ──────── 共通native/P0・64sampleのcorrectness
        └ 32costs/arm＋shell ── mean-only calibration → 52 serial cost records
                                         │  同じcohortを全roundで再利用
                  ┌ evidence lane ────────┤ error未資格ならshots/total null
                  ├ explicit scenario1 ──┤ 既存accuracy → energy → total
                  └ explicit scenario2 ──┘ assumptionはqualificationへ昇格しない
```

## 3. Shots・energy・費用の意味

既存の式を変更しない。同じraw complex-signal normへそろえ、各roundで

`d_m = A_m*(εPF,m + εRTE,m) + εsynth,m + εlaw,m`

`ρ_m = A_m − d_m`、`βdet,m ≤ asin(d_m/A_m)`。正のρと`βdet+βstat≤βtotal`が必要で、coordinate toleranceは`ρ sin(βstat)/sqrt(2)`。既存Hoeffding十分shotsは`ceil(2 log(2/αaxis)/coordinate²)`で、X/Yに同じNを適用する。普遍的最小shotsや情報理論的下界ではない。極小Aによるunderflow／overflowでは`UNRESOLVED_NUMERICAL_SHOT_RANGE`・shots nullとし、clipしない。

DF norm＋guardのHa誤差はdへ加算しない。全roundがPASSした後、既存energy判定で`DF_bound_Ha + βtotal/(δ 2^M)`へ一度だけ加える。保存のDF／PF／RTEはbinary64 numerical pilotなので、仮に他の項の厳密boundが得られても、このSOURCEから自動的に厳密CA保証は得られない。Actual energyの測定・shots抽選は行わない。

各round・axisの費用は`C_ma = n_m*mean(kernel) + head + readout_axis`、`n_m=4*2^m`。Kernelは決定論prefixとrandom tailを含むcomplete controlled h-step。ShellはancillaのHadamard head/X/Yであり、決定論DF headと混同しない。Headと対応readoutはinterrogationごとに一度だけ加算する。登録serial fence・segment別compile政策のRZ/CX/depth/sizeを扱い、長回路全体の再compile最適depthとは呼ばない。

総費用は既存`aggregate_totals`で`Σ_m Σ_axis N_ma*C_ma`。Classical calibrationの32や64をshots／coherent反復の乗数にしない。実際の量子実行は**各shotの各coherent occurrenceでfresh independent trajectory**を仮定し、同じsample kernelをn回反復する政策ではない。定常な同じarm law／kernel/P0/profileに条件付け、線形な期待費用をsample meanで近似する探索的会計である。今回は実長回路を生成しない。

同じ32標本の校正誤差は全roundで共有される。Kernel平均のSEに掛かる重みは`W=Σ_m,axis N_ma*n_m`で、総費用の記述的SEは`W*SE_kernel`。独立13cohortのroot-sum-squareやsqrt(13)縮小は使わない。Shellは保存決定論metricsであり、この校正のSEには加えない。費用SEはquantum samplingの統計誤差・エネルギー誤差ではなく、保証CIにもならない。

## 4. Evidence経路と仮定付きscenario

| 経路 | 使用する誤差 | 出力資格 |
|---|---|---|
| `evidence` | 全law synthesis／implemented-lawは現時点null | shots／total未資格。B2 sampled残差で埋めない |
| `ideal_exact_native` | counterfactual exact synthesis=0、ideal reference law自身のerror=0 | 明示仮定下のshots／energy／有限bit費用proxy。実装誤差0の認定ではない |
| `assumed_raw_signal_budget` | 各roundでsynthesis=.001*A、law=.001*Aという未検証の仮定 | 固定感度scenarioのshots／energy／費用proxy。evidence-qualifiedと別schema |

2つのscenarioは**次のC1実行に採択するか判断する提案条件**であり、今回計算していない。.001は実測・certificate・新しいprecision allocationではなく、Aで正規化したround-level感度入力である。既存CA／phase／failure／科学allowanceは変更しない。Per-gate tolerance1e−10から導出した値でも、per-kernel残差でもない。

Ideal scenarioでB2 finite-bit価格をideal-law価格のproxyにする際には、**費用law間の転送も仮定**として明示する。Law間の平均費用一致を実測／証明したとは扱わない。実装law scenarioでも定常性とfresh occurrenceを仮定する。未知誤差を黙って0にしたり、scenarioの`CONDITIONAL_CA_PASS`をactual CA達成へ昇格させない。

各laneのround/energy/resources/qualificationを独立に保存する。トップレベルのactual CA、保証shots、保証総資源優位はnull。Synthesis/lawが不足すれば`UNRESOLVED_SIGNAL_ERROR`、数値が扱えなければ`UNRESOLVED_NUMERICAL_SHOT_RANGE`、十分条件未達はそのlaneのみINFEASIBLEとし、他の登録laneを評価する。Strict保証付き比較へ必要な入力を追加する拡張口は人工データで試験したが、今回C1の実saved loaderは全law error evidenceをNoneに固定し、外部から仮のboundを注入しない。

## 5. 全law合成誤差：取得できるものと不足

B2のfull512は**各sample内の全operator**を検証するが、sampling lawの全wordを検証していない。Sample最大残差は未観測trajectoryの上界ではない。Gaussian emitterのsingle-particle transfer check、general-K/CAR照合、analytic vacuum phase、MPFR160のdeclared phase、actual inverseは正しさを支えるが、rounding enclosureや全compiler pathの誤差証明ではない。

全trajectory列挙が不可欠とはしない。K0の中央eventは有限atom集合のsigned rotationであり、headも固定なので、**有限のprimitive／basis／relative transitionのuniform誤差を構成して、既存telescopingへ接続する**方法が候補となる。単位的factorについて`||ΠU−ΠV||≤Σ||U_j−V_j||`の構造を使い、shellのstate/action誤差からcomplex signalへ変換する。13roundの異なるtrajectoryでもuniform boundなら同じ伝播が使える。

ただしB2はwhole-kernel P0最適化を行う。Eventを個別compileした誤差表だけでは、境界を越えるP0変形後の全kernelを囲めない。さらにCAR／expm／Operator／normの数値参照誤差を囲む必要がある。既存`round_synthesis`や`repeat_bound`は**入力boundのcoverageを作る関数ではなく、既に有効なboundを伝播する関数**である。これらへsample最大値を渡して全law資格を作らない。

最小のC2候補は、固定native emitterとP0変形の構造監査からuniform上界の構成可否を先に判定すること。新しい実H4の回転／local reference誤差評価が必要なら、限定SOURCE・scope・callsを別に固定してから許可を受ける。未確認の新compilerやevent-wise価格へ切り替える案は比較条件変更なので、自動採用しない。

## 6. Finite-bit lawの技術調査

固定`src/trotterlib/rte.py:iter_sample_rte_events`は、既存component probabilitiesをbinary64で再正規化して`Generator(PCG64).choice`へ渡し、Taylor orderとatom indicesをsampleする。K0でも1 microstepに1 atomを選び、1kernelは16 events。B2のseed namespaceはarm/final-export/SOURCE/cohort/sampleを分けるが、seed分離自体は厳密なIID乱数や理想分布との差の証明ではない。

差は少なくとも、exact declared coefficientsからの確率計算・正規化の丸め、finite random-grid／choice実装、PRNGをIID streamとみなすモデル、量子shots／coherent occurrenceへの独立性に分かれる。B1 ideal finite-RTEのlaw error0は、そのreferenceの定義に対する0であり、実装law error0ではない。

もし明示したIID random-bitモデル下で1 eventのTV上界ηを得られれば、product lawのcouplingでn_m*r eventsへ伝播でき、ideal unitary-trajectory complex signalのlaw差は保守的に`min(2,2*n_m*r*η)`で抑える候補がある。Gate synthesisとshell errorは別管理する。この構造だけではηの有用な数値は得られない。PCG64の固定有限cohortからηを推定・認定しない。

既存保存atom係数／probability tableと固定NumPy choiceの実装意味を対応させる評価は、全trajectory列挙や追加samplingなしでできる候補である。ただし、確率値の再解析は新科学評価なので今回実施しない。必要ならC2の小さなtable解析へ固定する。実装のchoice内部sourceが今回local配布に無いことも確認したため、累積和境界・random-gridの正確なenclosureは未調査／未資格として残した。

## 7. 費用support・同時failure・保証強化

B2の32標本/armから得られるmean／variance／SE／observed rangeだけでは母平均区間は資格付けできない。`bounded_cost_intervals`には有効な全law support provenanceと明示的なalphaが必要であり、今回はsupport=Noneの点推定経路を使う。Adapterは`sample_min_max`などの観測極値をsupportとして拒否する。

有限event alphabetと固定P0 pass scheduleから、全wordの最終RZ/CX/depth/sizeを抑える構造的rangeを作る案を優先調査する。Whole-kernelのcompiler最適化後の価格を扱う必要があり、event個別compileの価格和をそのままrangeとしない。有効なrangeが取れなければintervalはnullのまま、母平均順位を保証しない。

C2で保証cost intervalを作る場合の提案は、**2 arm×6 metricsにcost alpha_total=.01を配分して各.01/12**。同じkernel cohortを13roundへ使ってもcost confidence eventは12個で、26や13倍へ増減しない。RPE candidate failure .01は候補ごとの値のまま、両候補のjoint energy事象はunionで最大.02、そこへcost failureを別に組み合わせる。これを勝手に「総失敗率.01の同時優位」と呼ばない。このcost allocationはC1では使用せず、C2採択候補だけである。

有効なkernel平均intervalは同じ線形重みで全roundへ伝播するので、独立roundによる幅縮小を仮定しない。Idealとimplementedの費用転送にはlaw差とcost rangeが両方必要で、未資格ならideal平均はnull。追加sampleはこの資格不足を治さない。有効intervalが得られてなお幅が研究判断を妨げる場合だけ、新しい固定cohortを別許可後に検討する。

## 8. 未資格項目と最小手段の優先順位

| 優先／項目 | 最小手段 | 新実分子評価・規模 | C1／強い結論との関係 |
|---|---|---|---|
| 1：全law synthesis | emitter/phase/P0のuniform構造と参照誤差を調査、既存telescopingへ接続 | 有限primitive/basis/local operator表が必要になる可能性。全word列挙を既定にしない | 現実のshotsを資格付けする必須入力。C1では明示仮定のみ |
| 1：implemented-law差 | exact係数→binary64 probabilities→choice mappingを固定モデルで評価 | 保存2 armのfinite table再解析、追加trajectory不要 | 実装lawのshotsに必要。Ideal scenarioだけなら仮定として分離 |
| 2：費用support／mean interval | all-word P0 range＋同時alpha、同じ32cohortを再利用 | 静的pass/bound設計から開始。必要なら少数local IR確認 | C1点推定には不要、保証付き母平均比較には必要 |
| 3：DF/PF/referenceの厳密囲い | 主張に必要な場合だけ独立enclosure | 既存数値結果とは別の科学条件と計算scopeを要する | Numerical pilotと数学的CAを区別するために必要 |
| 3：一般化・独立cohort・別分子・新規性 | 固定H4でのtradeoffを判断してから選ぶ | 今回は設計／実行対象外 | 初回pilotを妨げる必須条件にしない |

[機械可読qualification inventory](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/technical/qualification_inventory.json)／[C2非実行草案](research_c_pr_aware_df_stage_c2_design_draft_20261027.json)。C2はC1で有望な条件付きtradeoffが得られた場合に、ユーザーが最小branchを選ぶ。Executable entry・科学call上限はnullで、無制限計算の許可ではない。C1結果を見る前に全law列挙・大量compile・別分子を正式固定しない。

## 9. C1：最初の未承認科学実行案

**保存JSONだけの164-call再解析**を推奨する。新fit、operator/PF/RTE/DF norm再計算、NPZ/QPY decode、sampling、build/compile、量子shots抽選は0。保存PF/RTE/Aとnative costから、新しい必要shots・条件付き総費用を計算する処理自体が新科学実行であり、別途明示許可が必要である。

| Logical units | 最大数 |
|---|---:|
| 保存bundle load・identity gate・target/horizon validation | 3 |
| 2 arm attachment＋2 cost calibration | 4 |
| 2 arm×13round×X/Yのserial費用 | 52 |
| 2 arm×3laneのsignal資格 | 6 |
| 2 arm×3lane×13roundのaccuracy（shots算定を内部に含む） | 78 |
| 6 energy＋6 total aggregation＋6 shared-calibration uncertainty | 18 |
| 3 lane comparison | 3 |
| **合計** | **164** |

`LIMITS`と人工完走が一致した。Load／identityは384 sample JSON等を一括扱うgrouped unit、round_accuracyは既存shots反転を含む1unitで二重計上しない。費用は誤差scenarioに依存しないため52件だけ構成し、3laneで共有する。78round判定はX/Y共通Nを出し、各arm/laneの資源出力に26 interrogationセルが対応する。STOP・SOURCEのserializeは科学unitを追加しない。

Fresh outputは`artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01`、retry/resume0。SOURCE/input/arm/seed/raw-compiled identityの不一致、保存correctness矛盾は依存STOPし部分証拠を保全。未知資格はnull、十分条件未達はlane内INFEASIBLEとして記録し、他の固定laneは継続する。閾値・scenario・標本数の事後変更はしない。

保存物は元SOURCE/契約、identity、kernel校正、52 serial costs、78round accuracy、6 arm/lane results、3比較、共有SE、call/resource/STOP。実際のC1結果の証拠経路は未資格のままになる可能性を残し、仮定付き数値を保証付き結論へ昇格させない。

## 10. 人工・回帰検証と保全

| 試験 | 結果 |
|---|---:|
| Stage C専用：13round×X/Y、n=s2^m、shell一度、DF Ha分離、null伝播、減衰とshots・単価tradeoff、sample倍率拒否、共有SE、support拒否／人工analytic拡張、数値range、SOURCE/compiled/arm/旧run重複STOP、人工E2E／fresh／164-call | **23 PASS** |
| 既存accuracy/connection、H4 CA resources、B2共有Qiskit mutex回帰 | **50 PASS** |
| 合計 | **73 PASS** |

[新test log](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/technical/stage_c_tests.log)／[回帰log](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/technical/regression_tests.log)／[構造化tests](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/technical/tests.json)。入力は人工scalar／mock provenance／人工回路のみ。実H4のsaved arraysや新shots/totalは使っていない。ローカル人工PASSであり、実分子性能・CI／外部再現・科学GOではない。

```bash
"$EVAL_PY" -B -m pytest -q --tb=short -p no:cacheprovider tests/management/test_research_c_pr_aware_df_stage_c.py
"$EVAL_PY" -B -m pytest -q --tb=short -p no:cacheprovider tests/management/test_research_c_c1_stage3.py tests/management/test_research_c_c1_h4_ca_resources.py tests/management/test_research_c_pr_aware_df_stage_b2_compat.py
```

[Passive preflight](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/technical/passive_preflight.json)ではSOURCE60／saved evidence408、schema・SOURCE/contract/arm・saved gate verdictがPASS。`assess_round`／`assess_energy`／`bounded_cost_intervals`／`aggregate_totals`／`compose_serial_cost`を禁止する検査下でも通過し、新科学算定0を確認した。旧A/B1/B2の1361ファイルと旧数値SOURCEは不変。元inputはprovenance用hashのみ参照し、新decodeをしない。[技術STOP](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/technical/STOP.json)を保存し、fresh C1-run01は存在しない。

既存evaluation envをread-only使用し、task-local cache/temp・bytecode/cacheprovider無効で試験した。実資源はaffinity32CPU、MemAvailable約52.6GiB、pressure avg10/60/300=0。将来C1は独立6 scalar taskを既存resource_planで並列化し、同じ52 serial-cost recordsを共有する。重い行列・compileがなく、CPU/RAM/worker/wallの人工上限を追加しない。Nativeの将来再使用ではB2の共有build/compile/adjoint mutexを維持するが、C1はその操作を呼ばない。共有venv/cache、他job、OS設定、旧grant/ledger/marker/STOPは変更しない。

## 11. Q1〜Q5・次の担当

**Q1：技術的接続は完成した。** 保存B1/B2から既存shotsとserial総費用へ到達し、資格不足／仮定／数値不適格を分ける。現在の証拠だけで保証付きの実装shotsが得られる、という意味ではない。

**Q2：最小のblockerは全law synthesisとimplemented-lawのraw signal bound。** 前者はemitter/P0のuniform構造、後者は有限probability tableとchoice/IIDモデルの対応を先に調べる。費用supportは点推定には不要で、保証付き母平均比較に必要。追加sampleだけでこれらは解消しない。

**Q3：条件付きと保証付きは分離できる。** C1ではevidence laneのnullを維持し、2つの提案scenarioだけが仮定付き数値を出す。数学的保証はB1/DF/referenceの資格も要し、独立のC2証拠なしには出さない。

**Q4：初回案は164-callの保存scalar再解析。** 同じCAと登録13roundの既知誤差・A、B2のkernel/shell価格を使い、明示scenarioの必要shots／総費用と共有校正SEを算定する。新fit／sampling／回路・operator計算は0。今回はまだ実行していない。

**Q5：科学的総資源削減の主張には、未観測trajectoryを含む合成誤差、実装law、平均費用の資格、joint failure、最終CAの水準を揃える追加検証が必要。** まずC1の条件付きtradeoffを得て、強い主張に効く最小のC2だけを選ぶ。Kernel単価低下や仮定付きCA PASSだけで無条件の総資源優位を認定しない。

次は**研究C／ユーザー**がC1の2scenario・164-call案を採択して科学実行を許可するか判断する。許可後に限り**Codex**が固定SOURCE／契約／保存input／fresh outputを再照合して一度実行し、証拠公開後STOPする。C2、追加fit／sampling、GPT詳細科学レビュー、論文化の最終判断へ自動進行しない。今回もcommit/push・remote SOURCE／成果物照合後、**mandatory STOP**する。
