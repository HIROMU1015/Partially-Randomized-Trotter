# Stage C：全round Reference構成と回路価格の準備

2026-11-06 JST。[共通報告・次の検証順序](REPORT.md)／[Stage B](STAGE_B.md)。**Stage Aの一shot対応をfull ordered pathへ適用する入口はある。一方、1 h-step gadgetの単純なq回反復、全roundの強い対照、長q価格・母平均の資格は未検証である。** 今回は既存SOURCEを読んだだけで、回路生成・QPY decode・compile・新実装を行っていない。

## C1. 再利用できるStage A資産

[Stage A結果][A]はlinear H4／1.00 Å／STO-3G／Nα=Nβ=2、number-CDF8／元順LD2／S2/s1/r32/K0／h=δ=.1、同endpointのsample00..31、complete event16後cut、**1 h-stepのfull wrapper**に限る。SOURCEは`2dd012dc77dcc877d5918c7d212819eb79b97557`、保存結果は`3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0`。

| 保存・SOURCE | 今回確認した再利用範囲 |
|---|---|
| [SOURCE.json][ASOURCE]／[STOP][ASTOP] | 元C SOURCE143・保存入力16・追加law・保存cohort96 binding、新adapterと再利用部品。128 compile、retry/resume0、mandatory STOPは維持 |
| [paired CSV][CSV]／[有限cohort集計][SUMMARY] | 同じpath・同じP0の2方式×X/Y。全32本でCX/depth減、RZ増、qubits9→17。母平均・全round費用には未資格 |
| [archive index][ARCHIVES] | 32 archiveにraw/compiled wrapper256 QPYとA/B/actual A†96 QPY、reference／observable等を保存。今回は展開・decode・再hashをしていない |
| [paired adapter][PAIRED] | `prepare_common`／`prepare_path`／`path_phase`／`run_prechecks`。保存32本・K0・h=.1・cut16への入口であり、任意qの実装ではない |
| [H4 adapter][BRIDGE] | `stored_events`／`control_one`／`directional_head`／`reference_gadget`。同じgate・global phase・actual inverse、2 CSWAP cascadeを再利用可能 |
| [correctness harness][CHECKS] | saved CAR／ordinary raw、direct両branch、AB、actual A†、真空ray／χ、complex coherence、X/Y induced observable、compile前後の検査 |
| [runner][RUNNER] | 旧`compile_one`／`verify_one`を再利用するP0 pipeline、全precompile PASS後compile、finite-cohort出力。旧一回のrunを別qで再起動しない |

既存P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1／initial-zero false／topology-free。位相保持の`compile_checked`→MPFR160 declared-phase→unrollを使う。complex128、component／unitarity1e−10、signal／数値Frobenius2e−10、rtol0、phase fitなしはStage Aの履歴である。長roundへそのまま十分と認定せず、新しい科学実行前に必要精度との対応を確認する。MPFR160を全数値評価・T合成の精度とは呼ばない。

## C2. 全ordered pathと一回のgadget

既存13-roundを候補にする場合、各shotのordered trajectoryは

\[
V_{\omega,m}=U_{\omega_{q-1}}\cdots U_{\omega_1}U_{\omega_0},\qquad q=2^m
\]

である。各Uはhead・独立なr32 events・逆順head・scalar/identityを含む1 h-step。**保存した一つのUをq乗するlawとは異なる。** [既存repeated request][REPEATED]も一occurrence/stepとstep-major順序を持つ。ただしlegacyのpreparation型をCのnumber-CDF providerへ自動的に置き換えない。SOURCEが同一でない型・位相規約の接続確認が必要である。

whole-path案は、この同じVをV=A Bへ分割し、[既存一shot恒等式][APPLICABILITY]の一回のgadgetを使う候補である。

| 条件 | 全roundで保つ内容 |
|---|---|
| ordered path | 全q occurrenceの同じevent列・係数符号・product→rotation順・決定論headを保つ。AとB用に独立halfを新生成しない |
| reference | full-Fock真空rを同じregisterで表す。各complete number-conserving Gaussian／Z／ZZ factorが真空rayを保つ構造は候補。full Vのray条件とχが必要で、実装の全law保証は未取得 |
| 分割 | whole ordered circuitの前後からA/Bを取り、AB=Vを維持。complete factor/eventの境界を使う案。Stage Aのevent16 cutを全qのbalanced cutと同一視しない。新しい分割規則は未選定・実行前に固定が必要 |
| actual adjoint | Aの全gateを逆順に実inverseし、scalar・event・Gaussianの位相も共役反転。順序を保ったまま角度だけ変えない |
| χ | path全体のvacuum phase。各h-stepのscalar/identityを一回ずつ会計するか、同じ総時間へ集約する。round全体へh-step一回分だけを入れず、集約後さらに各stepへ重ねない |
| normalization | 同じB_Kとq/rのattenuation、p_m、weight／trial規約を維持。χやunitaryへnormalizationを乗算しない |
| wrapper | freshな元H厳密ground⊗reference⊗ancilla、H→core→YのみSdg→H→終端測定。途中測定・reset・post-selectionで置換しない |

この条件の下で求めるものは、全Vに対するdirectとReferenceの**終端X/Y測定分布**の対応である。A/Bの可換性や元H groundが個々のrandom pathの固有状態であることは一回の既存内積恒等式の前提ではない。単一gadgetのcoherence対応から、gadgetを繰り返したchannelやsystem出力の一致を推論しない。

## C3. 短いgadgetをq回並べる案との区別

| 構成 | 信号保存の状態 | 費用の扱い |
|---|---|---|
| h-stepごとのReference coreを同じancilla・systemへq回接続し最後だけ測定 | **Stage Aでは未検証。** 各coreはdirect `diag(I,U)`そのものの置換ではなく、共通back-actionを持ち得る。前のgadget後のsystemに次の非可換pathを作用させても、元のVのcoherenceになるとは限らない | 各gadgetの2 CSWAP cascade・P(χ)と境界費を持つ。信号対応前にq倍価格を有効なRPE価格として採用しない |
| 各h-stepのfull wrapperをq回実行・測定／fresh reset | 各stepの一shot実験であり、元の一shot coherent q-step RPE signalではない | q回の読出しを全round interrogationの代用品にしない |
| 全ordered Vを一つのgadgetで測定 | 既存恒等式を適用する候補。全pathのray／phase／actual adjoint、compiled observableの確認が必要 | 2 cascadeはwhole interrogationに一組。A/B全長、cut・lane間分配・routing等は新価格であり、Stage A価格のq倍ではない |

複数gadgetの共通固有基底等の特別な条件は[既存適用資料][APPLICABILITY]で別扱いになっている。独立random halvesや元H ground仮定だけでその条件を満たすとは扱わない。もし短gadgetの反復案を継続するなら、共通back-actionを含む全列の分枝coherence／X/Yを直接照合する別の検証が必要。今回はその成立・不成立を新たに証明・数値判定していない。

## C4. 強いsymmetric対照と公平な費用境界

[既存baseline資料][BASELINE]とStage Aは、Gaussian basisを非制御とし、前半headをuncontrolled／後半逆順headをdirectional、中央RTEをordinary controlledにする。Stage Aのh-step direct coreは両branchが`diag(I,V)`へ一致する検査を持つ。全qでもこのbranch対応を維持し、対称化が単にhalf-time signalを作る別構成へ変わらないことを確認する。

同じordered request、law、scalar/identity、actual inverse、compiler profile、wrapper、精度、同じ適用可能な最適化機会を与える。対照をGaussian全制御へ戻したり、basis共有やhead relative fusionを外したりしない。whole-path Referenceを比較するなら、対照にも同じwhole-path費用境界を与え、合法なstep境界のhalf-sweep融合・phase集約を検討対象に残す。

[既存repeated builder][REPEATED]の`_build_boundary_optimized`は隣接する同じblock0のhalf-stepをfull timeへ融合し、constant／identity位相を集約する。CのStrongBuilderはrun grouping・diagonal fsum・解析位相付きrelative Gaussianを持つ。これらがすべて同一native whole-round pipelineで同時に実現済みという証拠ではない。必要なのは最小接続とcorrectness確認であり、異なる実装の節約値を足し引きした架空のbaselineではない。

Stage Aはhead relative fusion=true、cross-head/tail relative fusion=false、tailの連続同basis共有を共通条件にした。cut両端basis一致14/32ではReferenceの分割で共有が失われ、観測価格に含まれている。全qではstep境界とglobal cutの共有損失を別に追跡する。損失を未測定のまま相殺・控除せず、別cutを価格結果で事後選択しない。

### 旧serial accountingへ入れる際の制限

[c1_stage3_connection.py][CONNECTION]の`compose_serial_cost`は、制御kernelの`native_kernel_repeat_with_fences`と別head/readout shellを要求する。countはnative連結、depthは**scheduled serial sum**であり、whole-circuitの再compile depthではない。`serial_native_sequence`はmeasure/resetを持たないn+1-qubit kernel factoryをoccurrenceごとに呼ぶ。Referenceの17-qubit coreをそのまま渡せるinterfaceではなく、同じ意味のkernelでもない。

- Stage A full wrapperをkernel欄へ入れ、旧shellを加えるとH/readout等を重複計上し得る。
- qには既にcomplete r32-event kernelの回数を数える。rや価格標本32をさらに乗算しない。
- 旧`W μ_kernel+S`、共有校正SEの組み立ては、同じserial制御kernel政策に限る。whole-path Referenceのq依存や母平均が同じaffine形だと決めない。
- fence政策を比較する場合は両方式に同じ費用境界・depth意味論を明示し、まずReference側の信号保存を確認する。whole-path最適化価格とfenced serial価格を同じ種類の実測として混ぜない。

## C5. 資源ごとの必要データ

| 指標／負担 | 追加検証・記録すべき内容 |
|---|---|
| RZ count／RZ depth | χ補正、A†、cut・step境界の融合損失を含む実価格。RZ countからT countを自動換算しない。T合成を目的にする場合はrotation精度と全roundの誤差配分を別途指定 |
| CX count／CX depth | 既に強いdirectional対照からのpaired差。CSWAP分解とroutingのCXを含む。Stage Aで小差のpathもあり、長qでの符号を外挿しない |
| Total depth | whole wrapperの依存関係、二つのlaneの並列機会、共通ancillaの競合。2 cascade各8 CSWAPを無償fan-outで同時実行とはしない。scheduled sum／logical depth／実時間／T-depthを区別 |
| Qubits | Stage Aはdirect9／Reference17、追加8。whole-pathも同じ二register recipeならlogical幅は2n+1の候補だが、compiler workspaceやrouting、cat ancilla採用は別条件。qやNを掛けた値を必要peak qubitsとは呼ばない |
| CSWAP／P(χ) | whole-path一gadgetとq個gadgetの回数を区別し、全wrapper内で数える。低level分解・位相保持は同P0で照合するまで新価格を付けない |
| Routing／register維持 | Stage Aのtopology-free価格には実機接続制約が無い。対象backend/layoutを選ぶ場合だけ同じ条件で比較する。reference reset／維持・readoutの評価範囲を明示し、不明費用を0へ置換しない |

[旧accuracyのMETRICS][ACCURACY]はcount／depth／sizeの6項目でqubitsを含まない。peak幅は別に保持する必要があり、countのように全shotsで足す処理へ押し込まない。資源指標の重み・主指標は今回選んでいない。

## C6. 価格母平均・long-qの不足

Stage Aは保存32 pathの有限cohort結果で、全p_mの母平均を保証しない。新しいpaired母平均推定には、対象law／selection／sample単位と同じrequest上の方式間差を明示し、必要なcoverageと不確かさを扱う必要がある。X/Yを独立2標本に数えず、同じ校正を複数roundで使う場合の共分散を保持する。観測min/maxを真のsupportとみなさない。

保存pathを決まった順に連結したq>1 fixtureは、順序・位相・構造correctnessを調べる再利用候補になる。ただしその有限fixture群をfresh IIDのwhole-round価格標本とは呼べない。32本の巡回や同pathのq乗で母平均データを補わない。long-q母平均に新samplingが必要なら、対象と数・判定精度を別途承認scopeへ含める。

最小案は、先に少数の保存由来q>1 fixtureでwhole-path意味論を確認し、PASSした構成だけ短qの同P0 full-wrapper価格と独立holdoutへ進むこと。既存cost provider／境界補正の検証方法を再利用しても、legacy DF12 opt2の係数・q>32外挿やordinary kernelの係数をReferenceへ流用しない。必要な長q domainを覆えないなら未確認と報告し、最大qの直接compileや大量samplingを自動開始しない。

**Stage Cの現在地：** 再利用SOURCE、whole-path構成の適用条件、公平な比較の費用境界まで。q>1 Reference、全law expected価格、routing／合成精度を含む総価格は未取得。次の最小検証とSTOPは[共通報告](REPORT.md)を参照。

[A]: ../track_d_h4_paired32_reference_20261106/RESULT.md
[ASOURCE]: ../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/SOURCE.json
[ASTOP]: ../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/STOP.json
[CSV]: ../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/paired_resources.csv
[SUMMARY]: ../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/finite_cohort_summary.json
[ARCHIVES]: ../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/payload_archives.json
[PAIRED]: ../../../src/trottertracks/track_d_h4_paired_reference.py
[BRIDGE]: ../../../src/trottertracks/track_d_h4_reference.py
[CHECKS]: ../../../tests/track_d_h4_reference_checks.py
[RUNNER]: ../../../scripts/run_track_d_h4_paired_reference.py
[REPEATED]: ../../../src/trotterlib/df_partial_s2_repeated.py
[APPLICABILITY]: ../track_d_reference_control_feasibility_20261106/REFERENCE_APPLICABILITY.md
[BASELINE]: ../track_d_reference_control_feasibility_20261106/CONTROL_BASELINE_AUDIT.md
[CONNECTION]: ../../../src/trottertracks/c1_stage3_connection.py
[ACCURACY]: ../../../src/trottertracks/c1_stage3_accuracy.py
