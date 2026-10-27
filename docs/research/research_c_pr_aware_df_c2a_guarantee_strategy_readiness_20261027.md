# 研究C：C2-A全law保証の実現可能性・代替保証方式

2026-10-27 JST。起点`1f2b6338815af34bfa58d6292d98e7b5f66f898e`。

**方針AはA-3（利用可能な現行環境の証拠では実用的な成立が困難）と判定する。数学的な不可能性の主張ではない。** Installed Rust/libmの一様保証を得るには、追加の専門的な数値実装証明が必要であり、同じ有限probe／SOURCEコメント調査を繰り返す方針は推奨しない。代替として**B1：保存された個々のraw／compiled回路の事後validated検証**を推奨する。これは研究方針変更の提案であり、ユーザーによる採択・科学実行承認はまだ受けていない。

新規37＋既存回帰193＝230 tests PASS。旧順序依存test 1件は明示除外したまま。新規実H4科学call0、保存scientific arrays/QPYの数値decode0、新build/compile・sampling・fit・C1再解析0。

[新B1契約案](research_c_pr_aware_df_c2a_guarantee_contract_draft_20261027.json)は**19 logical units、UNAPPROVED**。実行方法・SOURCE・入力・precision・判定・fresh outputを具体化した新しい有限scopeである。旧all-law契約の`executable=false`／call上限nullは変更しない。

## 1. 対象・既存資産・今回の差分

将来対象は保存直鎖H4・1.00 Å・STO-3G、通常number CDF rank8のBaseline/Proposed、Nα=Nβ=2/sector36、system8+ancilla1/full512、元frame順LD2、partial symmetric S2、δ=.1/s4/h1/40/r16/K0、m0..12、own-DF oracle／状態準備除外。元g補正κ・scalar・native/Gaussian/P0政策を維持する。旧C1の約83.1%は明示仮定付き総RZ点推定のままで、今回再集計・資格変更をしていない。

| 再利用資産 | 適合範囲・今回の用途 |
|---|---|
| `c_pr_aware_df_zsxx_enclosure` | exact dyadic・Arb192・gate意味論・outward norm・RLock・runtime条件 |
| `c_pr_aware_df_zsxx_uniform` | N≤67731、run長依存式、det4coset、branch superset、translation部分規則、whole composerの条件を再利用 |
| C2-A graph／`monotone_budget`／`uniform_preview` | 固定P0と採用回数条件、未知値のnullを維持。整数構造を再導出するだけの追加実験なし |
| C2 audit／Bridge、`repeat_bound`／`round_synthesis` | raw/reference/lawの分離とunitary telescopeを確認。Observed Bridge残差をenclosureへ昇格させない |
| `c1_h6_compiler_precision` | MPFR160 marker・bind/cast経路を確認。Policy変更なし |
| B2 QPY・SOURCE・STOP・`SharedQiskitCalls` | 将来の有限入力をhash固定。人工build/compile/adjointは既存mutexを使う |
| `c1_stage3_integrated.Calls` | 新scope19-callの記録と最初のFAILでの依存STOPに再利用。承認証明や新grantは作らない |

新規は[固定回路verifier](../../src/trottertracks/c_pr_aware_df_fixed_certificate.py)、[保証scope helper](../../src/trottertracks/c_pr_aware_df_guarantee_strategy.py)、[保存入力専用executor](../../src/trottertracks/c_pr_aware_df_guarantee_execution.py)、[runner](../../scripts/run_research_c_pr_aware_df_guarantee_pilot.py)、[人工test](../../tests/management/test_research_c_pr_aware_df_guarantee_strategy.py)。既存Arb evaluatorは1qubit captured pairに限定され、full native回路の全基底事後検証がなかったため最小限追加した。既存compilerや数値ライブラリを再実装していない。

## 2. RZ complex-exp・wheel/libmの追加調査

[SOURCE調査](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/numerical_source_findings.json)／[read-only binary照合](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/binary_resolution.json)。

固定Qiskit 1.3.0のRZ行列は、binary64 thetaから`0.5*theta`を作り、num-complex 0.4.6の`exp(±i theta/2)`を使う。有限thetaのhalvingにoverflowはなく、normal域での正確な2進縮小とsubnormal域のroundingを分ける必要がある。複素expはexp(real)、cos/sinへ進み、巨大finite angleではrange reductionも保証対象になる。Signed zeroの特殊処理もfinite値を単に非zero扱いして省略しない。

今回は実`StandardGate.RZGate._to_matrix`を人工thetaで呼び、**Python RZ.to_matrixの代用ではないRustの出力**をArbの理想RZへ直接比較した。0/−0、±最小subnormal、.3、π、π+ulp、1e20、−1e100、最大finite binary64を固定入力として囲った。これはcaptured-point証拠であり、一様aはnull。[人工出力](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/artificial_certificates.json)。

### 新たに確認した不足

1. glibc 2.35 [s_sin.c](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/ieee754/dbl-64/s_sin.c)は約0.55 ULPとrange reductionの記述を持つ。しかしそのコメントから、全active binary/inputを覆う検証済み定理とbuild対応までは得られない。
2. [e_atan2.c](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/ieee754/dbl-64/e_atan2.c#L56)の最大ULP記述は**random samplingに基づく**と明記される。これを全D_Nの保証定数にすることはできない。
3. x86 [IFUNC selector](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/x86_64/fpu/multiarch/ifunc-avx-fma4.h)はCPU featuresによりSSE2/AVX/FMA/FMA4実装を選ぶ。Generic C SOURCEとinstalled selected routineを同一視しない。
4. `_accelerate`をread-onlyでdlopenし、dlsym/dladdrでsin/cos/sincos/atan2/exp/fmodのlibrary path・library hash・base相対offsetを記録した。sin/cosはnearest export名が得られない内部offsetへ解決された。この観測は実際の全Rust callsiteの実行traceやwheelのFMA/contraction証明ではない。
5. [generic e_fmod.c](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/ieee754/dbl-64/e_fmod.c)はexact remainderを意図する整数的な算法を持つ。ここだけの限定証明は考えられるが、installed x86 wrapper、Euclidean補正、全phase経路や他のmath関数へ自動拡張できない。

glibc資料が示す精度目標／test ULP表、有限高精度比較、Arb精度増加は、一様aを埋める根拠にはならない。公開SOURCEとwheelの完全一致も未認証である。

| aの候補取得方法 | 判定 |
|---|---|
| 固定libmの精度規約 | 必要な全入力保証を確認できず、未採用 |
| SOURCE演算ごとの解析 | 実装・係数・range reduction・selected IFUNC・FMAの全証明が必要。単なる短いadapterでは解決しない |
| Validated argument reduction／trig | 独立referenceにはArbを使えるが、未知Rust出力を囲うにはRust自体の誤差モデルが別に必要 |
| binary64格子 | 有限でも全角度格子の列挙は巨大。SOURCEの強い制約／計算機証明なしで有限probeへ置換しない |
| 到達angle制限 | 既存SOURCE domainへ縮小済み。初期RZ角の全域とrewrite再入力、Euler poles/cutsが残る |

## 3. Euler／ZSXX・translation／phaseと方針A判定

SOURCE由来のRZ/SX/X、det4coset、条件付きnear-unitary D_Nは維持する。D_Nから未知M→実Rust出力のoperator imageを得るには、determinant／abs／arg／atan2、θ≈0/π/2/π、RZ省略、SX/X、mod endpoint、sequence phaseを同時に覆う必要がある。θ等のbranch supersetを得ても、それだけでは出力operatorの包絡にならない。

Arb ideal template imageやheld-V boxをRust imageに改名する方法は採用しない。`uniform_model_status`はfew-ULP、sample最大、captured最大、ideal boxを一様モデルとして拒否する。未知a／ζ／translation／phaseはnullで、既存`uniform_whole_bound`へ0を渡さない。

Initial translationでは旧CRZ halving／非制御RZZ identity／RZ inverseを再利用する一方、XXPlusYY/inverse、generic controlled RZZ、U/U3 parameter arithmetic、binary64 PIとの差、definition phase、全global phase処理が残る。MPFR160 markerは**外側のaccumulatorの種類**を変えるが、definition内部で先に作られたfloat phaseや、最後のcast/remainderを消さない。

確認できた経路は、definitionでのFloat phase作成・正規化、DAGのphase加算、markerを持つParameterExpressionならFloat moduloを回避、最後のbind/cast後のFloat modulo、実inverse時のphase否定・再正規化である。採用rewriteのphase加算回数は≤Nだが、**definition/substitution内部も含む全正規化回数は、この証拠だけでは固定できない**。既存pass callbackはpass間snapshotであり、全setter呼出しのtraceではない。実H4をdecode/compileしてその回数を測ってはいない。

Whole-P0の条件付き構造式

`epsilon_translation + 6Np/(1−Np) + N*zeta(D_N) + epsilon_phase`

は正しい入力証明とNp<1、exact ideal unitary、同一phase-sensitive operator2のもとでのみ使える。N≤67731では、既存m12信号係数65536との積は4,438,818,816になる。仮にζの一様値を得ても、この保守的な回数評価により増幅され得る。a=0という**仮想exact-nodeモデル**だけでもrun項の上界は約2.166e−9となるが、これは実H4のa・誤差・phase-roomではない。C1数値の再解析やCA判定はしていない。

[Aの成立可能性記録](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/A_feasibility.json)は**A-3**とした。現証拠に有用なall-input theoremがなく、残りは「同じ数値調査をもう一回」では解決しない専門的なmath-library／compiler proofである。Stock P0の全law保証が数学的に不可能、またはPR-aware DF仮説が科学的FAIL、と結論したわけではない。

Aを再開する条件は、(i)正当化されたinstalled backendの全域誤差モデルが具体的に入手できる、または(ii)研究Cが新しいcompiler政策／保証目標を明示採択する、である。今回はどちらも採択していない。

## 4. B1：固定回路を事後検証する実装

### 検証対象と根拠

[B1 method](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/B1_method.json)。対象は保存QPY内のgate種・qubit順・binary64 parameter・declared phaseが定義する**exact ideal gate circuit**。実際のP0が返した回路全体を独立評価するため、未知入力に対するRust math-libraryの一様精度を仮定しない。Compilerの内部phase誤りも、保存raw／compiledの距離として残る。

対応gateはid/X/Z/SX/SXdg/RZ/RY/P/H/S/Sdg/CX/RZZ/閉CRZ/閉CRZZ/XXPlusYY。既存Arb gate evaluatorを使い、追加2〜3qubit gateは既知の独立matrix式で評価する。Qiskit Operator、gate.definition、transpileを数学的referenceにしない。未知gateは自動展開せずSTOPする。Open control、非有限parameter、未知gate class、非登録measurementは拒否する。

全full-Fock計算基底column e_jについて、gateを局所blockとして作用させる。各gate後にball vectorを中心cとvector2半径へ戻す。Exact local unitary Bに対し

`||v−c||2≤E`、`||B c−c_new||2≤delta`なら`||B v−c_new||2≤E+delta`。

deltaはvalidated gate演算とその後のrecenter残差をoutwardに囲った値で、gate評価・丸め誤差を含む。これにより長い非可換runの逐次interval wrappingを抑え、Rustの実丸め順序を改変することもない。ここで評価するのは**出力されたexact gate意味論**であり、Rust内部の数値foldの再現ではない。

各columnのraw／compiled差のvector2上界u_jを得て、全columnについて

`||U_raw−U_comp||2 ≤ ||U_raw−U_comp||F ≤ sqrt(sum_j u_j²)`。

これは有限probeからの外挿ではなく、full basis全列の包絡である。最大column下界はoperator2下界にもなる。局所Gaussianをsector射影してcontrolled回路の代用にはしない。

Actual adjointは、保存actual inverse QPYと、対応する保存recordの数学的adjointを比較する。Inverseを新生成せず、phase否定、gate順逆、XXPlusYY/angle符号、SXdg等を独立に扱う。

### Shellとreferenceの限界

Head/X/Y shellは全gateがlast ancillaだけに作用することを確認し、`I_system⊗U_ancilla`のnorm恒等式で2×2へ縮小する。X/Yには既存SOURCE通りterminal measurementがあるため、**唯一の最後のancilla測定→classical0**だけを検査して除き、unitary prefixをHまたはH*Sdgへ比較する。任意measurementやsystem作用を落とす処理ではない。

このB1で保証するのはP0／actual-adjoint／shellの**固定gate列の意味論**。Rawが物理Hamiltonian/CARの正しいevolutionか、そのreferenceの丸め誤差、raw Gaussian vacuum phase等は別問題でnull。全law、CA、shots、総資源は未資格のまま。

### 費用と運用

d=2^n、local arity k、gate数Gに対して、全column評価は概ねO(G d² 2^k)。列を独立processへ分け、各processのgate cacheとArb contextを使い、既存FLINTの小matrix×batch vector積を利用する。全d×d行列を同時保持しない。

実際に実行する全coherent occurrenceをこの方式で資格付けするなら、未検証trajectoryを実行対象へ混ぜない運用が必要である。登録PRの独立trajectoryを、資格済み1 sampleの繰返しへ変更すればlawが変わる。B2の有限calibrationを全実行の代わりに使うこともできない。多数の量子shot/occurrenceに対するclassical検証負担は大きくなり得る。

今回の人工3qubit whole-P0ではoperator upper約4.657e−15、実actual adjoint約6.928e−16を得た。Exact rationalが規範値であり、これを実H4の誤差定数へ転用しない。3qubitのtimingとprocess-column pathは人工証拠として保存した。実H4のwall時間や成功は未測定。

## 5. B2／B3との比較と推奨

[機械可読比較表](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/strategy_comparison.json)に主張・仮定・scope・reuse・実装・科学規模・C1接続・未資格・負担を記載した。

| 方針 | 得られる主張 | 主な仮定／追加負担 | C1との接続・限界 |
|---|---|---|---|
| A：stock P0全law | 全登録wordの一様synthesis上界 | Installed math実装とbranch/translation/phaseの全証明。現証拠では困難 | 成立してもraw/reference/law/cost supportは別。旧仮定の一部を置換できるだけ |
| **B1：固定回路事後検証** | 検証したraw／compiled・inverse・shellのoperator上界 | 全基底Arb検証。今回人工実装済み | 未検証trajectoryや母集団へ拡張不可。有限実行scopeへ主張を限定 |
| B2：確率モデル下 | 誤差のtail／期待値の高信頼度評価 | IID、事前固定holdout、rare-event/failure配分、各sampleの有効なcertificate | 合致するlawで平均信号等へ伝播可能。PCG64 IID／CA／shotsは自動解決しない |
| B3：新compiler政策 | 新policyの保証しやすい変換だけのbound | 検証済みidentity変換、別math backend等。新設計・再資格付け | 旧B2費用／約83%を転用できない。RZ/CX/depthを公平に再測定する必要 |

**推奨はB1の最小proof-of-method。** 固定P0を変えずに実行可能な検証を得られ、結果のscopeが明瞭である。これを採択するかは研究C／ユーザーが判断する。Aへの同じ一般調査の反復は中断し、具体的な新しい証明資産が得られた場合だけ再開することを提案する。

B2は[B2設計資料](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/B2_probability_design.json)に限定した。確率変数はtrajectory Wごとの`I(delta(W)>epsilon またはcertificate未解決)`。事前固定のIID holdout n件でfailure0なら、信頼度1−alphaで`q≤1−alpha^(1/n)`。全M occurrenceのbad-event確率をbeta以内にするunion設計なら`M q≤beta`が別に必要で、zero failure設計のnは`log(alpha)/log(1−beta/M)`以上となる。小さいrare-event確率には大きいnが必要になり得る。

同じlawで`0≤delta≤2`を使えば、期待誤差は`E delta≤epsilon+(2−epsilon)q`。これは平均signalの系統誤差とcalibrationのstatistical failureを別に伝播する設計で、全word保証ではない。PCG64／seed生成だけでjoint IIDは証明されない。追加samplingは今回0で、旧探索cohortもholdoutへ再分類していない。旧C1のshotsやroomを使ったn算定は行わない。

B3の例として、exact identityとして検証できる変換だけのpolicy、または[CORE-MATH](https://core-math.gitlabpages.inria.fr/)等の明示的なcorrectly-rounded backendを用いた新policyが考えられる。ただし、そのライブラリの存在は現installed glibcの保証ではなく、backendだけ替えても全compiler branchが証明されたことにはならない。今回policy変更・導入・build/compileはしない。

## 6. 最小科学案：保存1 kernel＋共通shell、19-call

[新しい未承認契約](research_c_pr_aware_df_c2a_guarantee_contract_draft_20261027.json)。**実行可能な技術設計だが未承認**であり、今回実行しない。旧A契約を実行可能へ書き換えたものでもない。

- 入力：B2 run02の**baseline_00** raw／compiled／raw_actual_adjoint／compiled_actual_adjointと、head/X/Yのraw／compiled、計10 QPY。SOURCE・STOP・元入力の来歴もhash固定。
- 選択：最小indexを結果を見る前に固定した。誤差・費用によるsample選別でもarm優劣比較でもない。
- 評価：kernel raw対compiled 1件、actual adjoint 2件、shell raw対compiled 3件、raw shell対exact H/H*Sdg 3件。
- Call：**QPY decode10＋固定pair certificate9＝最大19 logical units**。内部basis columnsは各比較の定義に含む。新candidateやsampling単位を隠して追加するものではない。
- Kernel：full512の3比較、1536 column比較／3072 state-action。Shell：ancilla-only確認後の2×2を6比較、12 column比較。
- Precision：Arb192、FLINT3.3.1／python-flint0.8.0。Allowanceはoperator2で`1/100000000`。新しい固定回路engineering pilot閾値で、CA誤差配分ではない。
- 判定：upper≤allowanceで固定pair PASS。lower>allowanceでcorrectness FAIL。区間が閾値を跨ぐ／gate path未登録ならUNRESOLVEDとして停止。結果を見てprecision・threshold・policyを変更しない。
- 実行：fresh `artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/B1-run01`、retry/resume0。最初のcorrectness FAILで依存後続を止め、開始済みcolumn taskを回収して部分証拠を保存する。
- 禁止：新build/compile、fit、Hamiltonian/CAR/reference評価、追加sample、C1 shots／総資源、他arm/candidateへの拡張。最大19完了でもmandatory STOP。

通れば「保存されたこのkernelのP0誤差・inverse・shellをvalidatedに囲えた」と言える。Raw physical correctness、全law、PR費用削減、CA達成の証拠にはしない。これが次のA1/A2/A3一括実験より小さく、保証scopeを明示できる実行案である。

## 7. テスト・保全・SOURCE

[集計](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/tests.json)／[新37 PASS](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/strategy_tests.log)／[旧193 PASS](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/regression_tests.log)。

新規では実Rustの人工RZ、未知ULP拒否、独立local gate、3qubit P0・相対phase故障・実adjoint、全basis、長いrecentered非可換word、process-local parallel columns、測定shellの厳密なprefix抽出、人工QPY round-trip、19-call正常E2E／11-call FAIL停止／UNRESOLVED停止、SOURCE/input不一致・passive処理、B2式のIID/holdout不足、未検証execution scopeを確認した。

新規初回[log](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/strategy_initial.log)の3 FAILは人工testで存在しないQuantumCircuit.xx_plus_yyメソッドを呼んだ2件と、open-controlの拒否理由の順序1件。既存XXPlusYYGate.appendと早いopen-control拒否へ修正した。旧科学SOURCEやテストの変更、科学retryではない。

旧並列STOPのREPLAY／dependent Stopped例外順序testは引き続き1件deselect。既存の両順序を強制する代替testを回帰へ含め、primary journalとSTOPを確認した。旧非決定性そのものは未修正で、旧FAIL log／SOURCE／STOPを維持した。

[Passive preflight](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/passive_preflight.json)ではSOURCE128、保存input/provenance44（QPY10はhashのみ）、外部Qiskit Python SOURCE22、package11を照合した。QPY/NPZ decodeと新数値entryは遮断。Fresh科学outputは未作成。[保全](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/preservation.json)は旧1906件中1904件byte不変、2 READMEの入口追記だけ。

Own worktree/venv/cache、既存evaluation packagesはread-only。CPU affinity/quota／空きRAM／pressureを[記録](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/environment.json)し、Arb process-local contextと既存Qiskit mutexを使用。2 suiteを並列実行し、column pathも人工検証した。科学runnerのworker数は実quota・affinity・RAM・column task数で決め、CPU/RAM/wallの人工上限を設けない。実H4時間の実測やspeedupは主張しない。

- Branch：`codex/research-c-pr-aware-df-guarantee-strategy-preparation-20261027`
- [SOURCE128](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/SOURCE.json) SHA256：`e8180d8e0a5479df4bfda527fbe7610a7a3d812a6b7ff7c3fc077427d0d68e80`
- [技術STOP](../../artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/technical/STOP.json)：新規実H4科学call0。

## 8. Q1〜Q5・次の担当

**Q1：現証拠では有効な一様aを取得できない。** 実Rust RZの固定点は囲えたが、installed selected math implementationの全域誤差証明が不足する。ソースのsampling由来ULPを採用しない。

**Q2：全到達ZSXX imageは未資格。** Pole/cutを含むD_N、arg/atan2/mod/simplification/phaseの全モデルが残る。Captured/ideal imageとは明確に分けた。

**Q3：whole-P0の構造接続は既存式まで。** Initial/phase/raw-referenceを含むall-lawの完成は困難で、stock P0の追加専門証明が必要。一方、固定出力全体の事後検証は内部数値モデルを仮定せず実装できた。

**Q4：同じ一般調査によるA継続は推奨しない。B1を最小の次案に推奨する。** B2はその後のIID・holdout・rare-event設計、B3は別compiler政策の研究判断である。どれも今回採択していない。

**Q5：新しいB1限定19-callの実行可能な未承認契約を固定した。** 保存1 kernel＋actual adjoints＋共通3 shell。全lawやraw物理referenceの代替保証ではない。今回の科学callは0。

次は**研究C／ユーザー**がB1の有限scopeを採択して19-call科学実行を許可するか判断する。**Codex**はその別途明示許可後にだけ固定runnerを一度実行し、結果・call・STOPを公開する。今回のcommit/push・remote/SOURCE/成果物照合後は**mandatory STOP**。Aの追加科学計算、B1実行、sampling、C1再解析、compiler政策変更、別分子、GPT詳細科学レビューへ自動進行しない。
