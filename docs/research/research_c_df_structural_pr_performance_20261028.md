# 研究C：tail-P4 CDFのPR性能・native32・条件付き総資源

2026-10-28 JST。**449/449 logical units COMPLETE、correctness FAIL0、retry/resume0。** 新candidateの全13roundが登録feasibilityを満たし、事前登録GOに従って固定32 trajectoryのnative校正まで完了した。

**主仮定の総RZは275,422,425,481.5で、現行PR-aware CDFの234,117,788,135に対し1.176427倍（約17.64%増）。** Tail係数位置を216→144へ減らした登録P4案は、今回の保存二候補・固定政策・有限校正の条件付き点比較では資源を削減しなかった。母平均や厳密CAについての判定ではない。

## 固定対象とSOURCE

保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2/sector36、system8/full256、last ancilla/full512。通常number CDF8、元frame順0..7、LD2、head一般対称J/tail対角＋P4 edge、全U可変で得た保存γ=.1最終J/Uを使用した。元g補正κ、nuclear scalar一度、tail identity抽出、interleaved2i+σを維持した。

Partial ordered symmetric S2、δ=h=.1、s1/r128/K0、m0..12、n_m=2^m。CA=.00159Ha、DFreserveCA/8=.00019875Ha、βtotalπ/6、βstatπ/12、failure.01/axis.01/26、own-DF ground oracle/状態準備除外を共通化した。

- 起点：`0ad1b8ef06e979707b75d272333b525327af75b4`。独立branch `codex/research-c-tail-P4-PR-performance-20261028`。
- 科学SOURCE：`4ac0e5f96a5ad5009f6273d0fc6ff5657ff80aff`。開始後無変更。
- [固定段階付き契約](research_c_df_structural_pr_contract_20261028.json) SHA256：`789df4e7373e80a598560fc9a8e009fe6810ecf30cee67a2721ba6db478d7de6`。
- [SOURCE93](../../artifacts/research_c_df_structural_pr/2026-10-28/technical/SOURCE.json) SHA256：`324fdf9c09fada29b37916ab69569b0389f8beaa575d608bb10900a63c2fa505`。
- 新J/U export SHA256：`ec0dfe3d53e8209b343b294a103ab89405ac757d3e26164ae591344d4423b985`。
- 保存h/g/scalar packet SHA256：`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。元P-H4来歴：`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。
- [前段1 fit/21-call](research_c_df_structural_support_execution_20261028.md)、[今回の再利用・人工検証・449-call準備](research_c_df_structural_pr_preparation_20261028.md)、[execution preflight](../../artifacts/research_c_df_structural_pr/2026-10-28/execution_preflight.json)。

今回はチャットによる一括科学実行許可の範囲でStage A→条件付きStage Bを実施した。承認証明・承認原文保存・新grantなし。旧契約/paired role gate/runnerは変更・再起動していない。

## 第三候補・表現・law・ground

新`tail_p4` role・`CDF_tail_p4` NativeModel・候補固有atom IDで、既存general-K/NativeModel/splitを接続した。旧`proposed` roleや旧216 atom catalogを転用していない。Supportはhead `[10,10]`、tail各7上三角、nonidentity係数位置はhead各36/tail各24、tail計144。元rank8/frame/group順は維持し、thresholdやpruningを行っていない。

保存DF誤差7.4515615285e-9Ha＋guard3.2638787060e-9HaのPASSとSOURCE来歴を再利用した。DF norm・新fit・optimizerの再実行は0。新base＋8 sector raw-CAR因子とgeneral-Kを個別照合し、native段階で全full256因子/sector対応、full signed tail＋identityも照合した。全gateが登録atol1e-10/rtol0内でPASS。Full CAR/lawの最大entry差は1.77636e-15。

新λ_R=**1.516165932377233 Ha**（保存診断と一致）。旧Proposedは1.4424298855756243Ha。新144 atomの符号・probability・basis・order・identityからr128/K0 configを生成し、旧λやattenuationを移送していない。Identity係数とcatalog/components/configは[新tail原本](../../artifacts/research_c_df_structural_pr/2026-10-28/run01/reference/tail.json)へ保存した。

新own-DF ground energyは−2.1663874490149095Ha、eigenpair residual1.61997e-15、normalization差3.33067e-16、alias margin1.61799387799。Sector36の新eigsolve1回で、旧Proposed stateは転用しなかった。同一元物理Hamiltonianの保存physical ground/alias証拠はinput identity確認の上で再利用し、再eigsolveしていない。[新ground/alias](../../artifacts/research_c_df_structural_pr/2026-10-28/run01/reference/ground_gate.json)。

## 全13round：PF/RTE・attenuation（Q1/Q2/Q3）

各DF自身のHamiltonianに対する`||PF(h)^n−exp(−itH_DF)||2`と`||corrected_RTE(h)^n−PF(h)^n||2`を比較した。t=.1×2^m/n=2^m。旧値は保存M4、今回値は新J/Uから既存reference engineで評価したもの。物理元Hに対するDF誤差[Ha]と、以下の無次元operator/signal誤差を直接加算していない。

| m | 新PF | 旧PF | 新補正RTE | 旧補正RTE | 新A | 旧A |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.253001e-05 | 6.216159e-05 | 3.942886e-05 | 4.671886e-05 | 0.9999102 | 0.9999187 |
| 1 | 0.0001247218 | 0.0001239869 | 7.879006e-05 | 9.335757e-05 | 0.9998204 | 0.9998375 |
| 2 | 0.0002467649 | 0.0002453104 | 0.0001570359 | 0.000186069 | 0.9996409 | 0.999675 |
| 3 | 0.0004729301 | 0.0004701613 | 0.0003097956 | 0.0003670526 | 0.9992819 | 0.99935 |
| 4 | 0.0008051953 | 0.000800737 | 0.0005883138 | 0.0006967926 | 0.9985643 | 0.9987005 |
| 5 | 0.0009842907 | 0.0009624765 | 0.0009967836 | 0.001176041 | 0.9971307 | 0.9974026 |
| 6 | 0.001575405 | 0.001295195 | 0.001404323 | 0.001612117 | 0.9942696 | 0.994812 |
| 7 | 0.003060138 | 0.002451266 | 0.001980368 | 0.002122382 | 0.988572 | 0.9896509 |
| 8 | 0.006026652 | 0.004504607 | 0.003753421 | 0.003929821 | 0.9772746 | 0.9794089 |
| 9 | 0.01201543 | 0.008973754 | 0.007297853 | 0.007516671 | 0.9550657 | 0.9592418 |
| 10 | 0.02402007 | 0.01793434 | 0.0145969 | 0.01500547 | 0.9121505 | 0.9201449 |
| 11 | 0.04802721 | 0.03583549 | 0.02928349 | 0.03002471 | 0.8320185 | 0.8466667 |
| 12 | 0.09602252 | 0.071615 | 0.05942141 | 0.06094531 | 0.6922548 | 0.7168444 |

**全13roundがPOTENTIALLY_FEASIBLE。** 最大m12でPF+RTE=0.1554439221 < sin(π/12)=0.2588190451、残phase-room0.1033751230。登録十分条件のPASSであり、未知の合成/law誤差や最終CAを認定しない。

λ増加に伴いattenuationは全13roundで低下し、m12は旧0.716844438→新0.692254786。一方、補正RTE operator残差は**全13roundで旧候補より小さく**、m12は0.060945309→0.059421406。λだけから全RTE誤差の大小は決まらず、変わったtail operatorの構造も関係する。PF誤差は全13roundで大きく、m12は0.071614998→0.096022516。残差の方向とattenuationの変化を分けて記録した。

各roundのraw complex mean signal・phase-room・旧値は[13セルCSV](../../artifacts/research_c_df_structural_pr/2026-10-28/PF_RTE_rounds13.csv)、[評価原本](../../artifacts/research_c_df_structural_pr/2026-10-28/run01/reference/)に保存した。

## 条件付きRPE shotsとenergy（Q3）

| Lane | 新総shots | 保存M4総shots | 新W=Σ2N_m2^m | 判定 |
|---|---:|---:|---:|---|
| ideal_exact_native（主） | 16,568 | 16,056 | 18,303,568 | CONDITIONAL_CA_PASS |
| assumed_raw_signal_budget | 16,638 | 16,120 | 18,387,660 | CONDITIONAL_CA_PASS |
| evidence | null | 比較しない | null | UNQUALIFIED_ROUNDS |

主laneのshotsは約3.19%増、coherent反復加重Wは約10.05%増。m10..12のshotsは5,570→6,054、W寄与は15,499,264→17,160,192。m12だけでshots2,644→2,990となった。高mの差が加重費用へ強く効く。X/Yそれぞれ同じN_mを算定し、[3lane×13round×2axis=78行](../../artifacts/research_c_df_structural_pr/2026-10-28/shots_round_axis78.csv)へ保存した。

二つのlaneは保存M4と同じ感度仮定である。主laneはexact native/ideal finite-RTE IID、第二laneはsynthesisとimplemented-lawのraw-signal誤差を各.001A_m以内と**仮定**する。仮定を測定値・証拠に昇格させていない。

既存energy評価の両仮定laneは数値十分条件0.0012783280386782687Ha/失敗率上限.01でCA以内と返した。これはown-DF oracle・declared deterministic evidence等に条件付きであり、actual CA/厳密CAや保証shotsを認定しない。Evidence laneの未知の全law合成/implemented-law誤差はnull、shots/総費用もnullのまま。

## 未知kernel価格の損益分岐とStage B GO（Q4）

主laneの6指標は`C_new,k(μ)=18,303,568 μ + S_k`。RZのshell項は57,988、head/readoutはinterrogationごとに一回だけ計上した。各occurrenceは独立trajectory、全round長回路の未観測global fusionは加えていない。

| 指標 | 旧M4平均 | 新平均 | 新記述SE | 新/旧平均 | 主laneのM4損益分岐 |
|---|---:|---:|---:|---:|---:|
| rz_count | 14076.437500 | 15047.468750 | 252.697767 | 1.068983 | 12790.824726 |
| rz_depth | 4342.281250 | 4768.375000 | 74.223653 | 1.098127 | 3945.697015 |
| cx_count | 5579.437500 | 5838.500000 | 78.565824 | 1.046432 | 5069.862860 |
| cx_depth | 3424.875000 | 3613.000000 | 38.561874 | 1.054929 | 3112.078334 |
| total_depth | 9574.531250 | 10486.500000 | 162.021361 | 1.095249 | 8700.081212 |
| circuit_size | 28014.437500 | 29972.156250 | 511.250677 | 1.069882 | 25455.855577 |

現行PR-aware r128を下回るには、主laneで新complete h-step kernelの平均が**12,790.82472592229 RZ未満**である必要があった。第二laneのRZ閾値は12,788.767108172002。旧kernel平均14,076.4375より低い閾値でも、それをnative中止理由にしなかった。

SOURCE/input/JU/general-K/law correctness、ground/alias、全13round、主shots/energy、finite positive RZ threshold、固定済みcohort/compiler/allowanceの登録条件が成立したため、追加承認を求めず固定32へ進んだ。[native_GO](../../artifacts/research_c_df_structural_pr/2026-10-28/run01/native_GO.json)、[4対照×2lane×6指標の閾値48行](../../artifacts/research_c_df_structural_pr/2026-10-28/breakeven48.csv)。

## Native32・正しさ・観測価格（Q5）

32件すべて128 signed K0 events。Master2026102806/cohort`tail_P4_s1_r128_native_fixed32_v1`、PCG64、SOURCE digest＋新JU identity＋role/domain/indexでseedを分離し、32 unique seedを保存した。旧M4と非paired、pool/適応停止/追加sampleなし。PCG64 substreamの使用をideal IID性の数学的証明とは扱わない。

Whole h=.1 complete controlled kernel、relative Gaussian融合はhead内、tailは隣接同basis共有、cross-head/tail一般relative融合なし。Qiskit1.3.0/P0/opt1/seed17/rz,sx,x,cx/approximation1/qubits_initially_zero=false、MPFR160 declared phase、phase fittingなし。両候補へ同じ合法融合・compiler条件を与えた。新basis/headは新J/Uから構成し、保存ancilla-only head/X/Y shell6件だけをfull構造・metrics照合の上で再利用した。

全32 raw/compiled full512/reference、unitarity、raw/compiled actual adjoint/return、complex X/YがPASS。Control0 identity/control1 reference・scalar/identity相対位相・全512入力出力を確認し、有限state probeやsector射影で代用していない。

| 固定32件での観測最大残差 | 値 |
|---|---:|
| raw_reference | 3.797145e-14 |
| raw_unitarity | 7.593925e-14 |
| compiled_reference | 2.742205e-12 |
| compiled_raw | 2.740736e-12 |
| compiled_unitarity | 1.543210e-13 |
| raw_adjoint_reference | 3.630619e-14 |
| compiled_adjoint_reference | 2.740513e-12 |
| raw_return | 7.205347e-14 |
| compiled_return | 1.349153e-13 |
| raw_XY | 4.135511e-14 |
| compiled_XY | 1.182690e-13 |
| raw_XY_norm | 3.474998e-14 |
| compiled_XY_norm | 1.123546e-13 |

Matrix allowance1e-10、complex signal2e-10、rtol0を実行前から維持した。以上は固定32回路のbinary64数値検証であり、未観測全lawに対する合成上界にはしない。[32正しさCSV](../../artifacts/research_c_df_structural_pr/2026-10-28/correctness32.csv)。

上のkernel表の通り、**全6指標の標本平均が旧Proposedより増加**した。新RZ平均15,047.46875/記述SE252.69777、旧14,076.4375比1.068983。RZ観測範囲11,843〜17,368、sample variance2,043,397.1603。CX平均は5,838.5（旧5,579.4375）。[metrics32](../../artifacts/research_c_df_structural_pr/2026-10-28/native_metrics32.csv)、[全6指標mean/variance/SE/extrema](../../artifacts/research_c_df_structural_pr/2026-10-28/run01/native/result.json)。

144 atomに減ってもr128のevent数とfull-frame Gaussian6種は残る。新U・係数分布・compiler・独立cohortは全て価格に関与し得るが、今回それらの個別因果効果を切り分ける追加計算はしていない。価格増の原因を単一要素へ断定しない。

## 条件付き総資源（Q6/Q7）

同じ32標本平均を全roundで再利用し、`C_k=W μ_k+S_k`、記述的な校正SEは`W×SE_kernel,k`とした。sqrt13で縮めず、32校正標本を量子shotsへ加えていない。Depthは既存serial staged modelであり、長回路を再compileした実depthではない。

| 主laneの指標 | 新総費用 | 旧M4総費用 | 比 | 新校正SEのみ |
|---|---:|---:|---:|---:|
| rz_count | 275,422,425,481.500 | 234,117,788,135.000 | 1.176427 | 4,625,270,754.925 |
| rz_depth | 87,278,334,050.000 | 72,220,391,610.500 | 1.208500 | 1,358,557,684.947 |
| cx_count | 106,865,381,768.000 | 92,796,579,603.000 | 1.151609 | 1,438,034,903.094 |
| cx_depth | 66,130,791,184.000 | 56,962,137,414.000 | 1.160960 | 705,819,887.421 |
| total_depth | 191,940,473,524.000 | 159,242,635,766.500 | 1.205333 | 2,965,568,995.730 |
| circuit_size | 548,597,507,720.500 | 465,933,091,247.000 | 1.177417 | 9,357,711,522.920 |

主laneのRZ差は**+41,304,637,346.5**、比1.1764267366。新総RZの記述校正SEは4,625,270,754.93だが、これは新cohortのみのSEで、旧候補との差の有意性・保証区間ではない。第二laneは新276,687,797,468.625／旧235,155,559,637.25、比1.1766160149で、同じ方向だった。

主laneではλ・shots/W側の不利を価格改善で補うという仮説に対し、観測価格も増加した。**今回の登録P4案について、有限校正を用いる二つの条件付き点比較では相殺も総資源削減も観測しなかった。** 新旧DF誤差は別々に保存予算内であり、精度を完全一致させた分解ではない。

副比較の主lane総RZ比：旧PR r64に0.933069、旧PR s4/r16に0.509768、保存M2の旧Proposed J/U決定論s1に2.704836。後二つを含めPR政策や表現条件が異なる参考比較である。M2は新P4自身の同J/U対照ではなく、PR固有効果の純粋な分離ではない。通常CDFγ0のmatching s1/r128校正も未取得で、値を移植していない。[2lane×6指標の総費用・SE・全対照比12行](../../artifacts/research_c_df_structural_pr/2026-10-28/conditional_resources12.csv)。

## Call・資源・保全

63 reference＋48 accuracy＋12 affine/GO＋325 native＋1 final=**449 attempts/449 COMPLETE/0 FAIL**。New fit/optimizer/DF norm/rank search/retry/resume/実量子shotsは0。新ground1回、PF/RTE13round、native32/compile32/actual adjoint64を固定範囲で完了した。QPY128件・CAR reference32件を含む[raw392件/175,427,613 bytes](../../artifacts/research_c_df_structural_pr/2026-10-28/run01/)を保全した。

準備の人工・回帰40 unique PASS（full512実人工testを含む）。科学SOURCE公開後の変更0。実行前32 affinity CPU/利用可能RAM約52GiB/低負荷を確認し、sector13roundとnative32を既存plannerで並列化、BLAS1 thread/workerでoversubscriptionを避けた。Qiskit build/compile/actual-adjoint-buildは既存共有mutexを維持し、CAR/Operator/matrixは並列に処理した。Task-local venv/cache以外の共有環境を変更していない。

[資源記録](../../artifacts/research_c_df_structural_pr/2026-10-28/process_time.txt)：wall17分18.59秒、user12,558.90秒＋system203.66秒、最大RSS3,197,856KiB、swap0、exit0。途中確認のavailable RAM約51GiB、memory pressure avg10/60/300は0。CPU/RAM/worker/wall人工capは追加していない。Qiskit lock待ち4,775.45秒はthreadごとの累積値でwall時間ではない。全64 build/compileを完了。未実測speedupは主張しない。

[Postflight](../../artifacts/research_c_df_structural_pr/2026-10-28/execution_postflight.json)でSOURCE93/input25/旧保護1054件、実行SOURCE、契約、call内訳、unique seeds、QPY/reference数、全raw hashを照合した。旧Stage A/B1/B2/C1/M1〜M4/M4-native/P4-21の原本・SOURCE・STOPを変更していない。CSVは固定raw記録の表示用抽出で、科学engine・sampling・compileを再実行していない。

## 科学的な問いと次の判断（Q8/Q9）

- **Q1**：λ増加に対しattenuationは悪化、補正RTE残差は全13roundで改善した。両者を同一の誤差と扱わない。
- **Q2**：PF誤差は全13roundで悪化した。新DF自身のexact evolutionとの比較である。
- **Q3**：全13適格、主shots16,568。Evidence laneは未資格でCA達成を認定しない。
- **Q4**：主kernelのRZ損益分岐12,790.82、実標本平均15,047.47で未達。
- **Q5**：KernelはRZ約6.90%/CX約4.64%増。Depth等も標本平均では増加した。
- **Q6/Q7**：Shots/W増加を価格側で補えず、条件付き総RZ約17.64%増。登録した固定P4案の性能改善は観測されなかった。
- **Q8**：「tail係数の疎化だけで安くなる」という今回の具体案には否定的な点比較となった。一方、λ最小化の普遍的十分性・必要性や、他supportの不可能性を証明しない。今回λも変わったので、同λで価格だけが異なる純粋ablationでもない。既存のtail-mass-only proxyを資源性能GOへ直結させない根拠として、PF・W・実compiled価格を別々に測る必要が明確になった。
- **Q9（提案のみ）**：別maskの盲目的探索より、既存のbasis関係とwhole-kernel価格を結ぶ機構整理を先に行い、cost-aware目的の妥当なsurrogate候補を絞ることを推奨する。固定U・凍結event価格weighted mass等の既存案と重複確認し、単event価格の加算をwhole-P0へ等置しない。今回の結果だけで新目的・solver・探索条件を採択しない。

Support制約とprojected startが異なるため、比較は保存された二つの最終DF候補の結果であり、疎構造化だけの純粋因果効果ではない。Finite32 SE/extremaを全law cost support、ideal-law母平均優位、厳密CAへ昇格させない。全law synthesis/implemented-law error、cost support、actual CA、実量子shotsは未資格/null。

**研究C／ユーザー**：この負の性能結果と限界を確認し、機構整理・cost-aware設計のどの最小検証を次に行うか判断する。**Codex**：後続依頼の範囲内で既存資産を使う技術設計を行い、新規再解析・fit・sampling・回路計算は別途許可された固定条件に限る。新実験や重要GPT詳細科学レビューは自動開始しない。

結果をcommit/pushし、remote SHA/tree・SOURCE・入力・旧原本・新原本の一致確認後、**mandatory STOP**。
