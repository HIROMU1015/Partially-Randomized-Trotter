# 研究C・C4：Q13限定科学実行（2026-10-24）

## 実行前固定と新しい承認

公開準備SOURCE `5038427e034c5e2cca6674998d28715cd93ec551` の
[限定契約](research_c_rustiq_limited_contract_20261024.json)を引き継ぐ。
そのSHA256は `e948a2e7fdf47e20b8af8cd7debae0631dabbd5e6ad77f419a9a9e2c668319a1`。
ユーザーが「次の作業に進んで大丈夫」「意図しないところで中断された場合は修正を行って再実行してよい」と明示承認した。
旧contractの承認待ち／retry0の記録は書き換えず、今回の承認と技術的中断時の再実行許可を別にbindする。

科学条件は4case `SS_motif_2,SS_motif_3,SS_motif_4,PP_motif_0` とB0/B1/R1のまま。
保存F1の4 mode・2粒子・occupied(0,1)、density-square合成fixture、L_D=1、残差2fragment、
T=1/5、q2、δ1/10の一点、r2、h1/20、K2。分子geometry／AO basis／分子DF rankは非該当。
Hamiltonian・DF・law・finite target・native basis/profile・許容差・gate順序を変えない。
各fresh attemptの上限は旧契約の28 reference、局所20／full12 build、adjoint12、wrapper24、
Operator68、Statevector48、cold24／warm24 request・実transpile24、Q13合成20のまま。
技術的中断時は失敗原本・STOP・SOURCEを保存し、必要な技術修正を新commitへ固定してfresh attemptを実施できる。
科学的FAILを隠すための条件変更・許容差緩和・case置換・追加探索は許可しない。

追加SOURCEは
[`rustiq_execution.py`](../../src/trottertracks/pr_dictionary_validation/rustiq_execution.py)、
入口は[`run_research_c_rustiq_limited.py`](../../scripts/run_research_c_rustiq_limited.py)、
管理testは[`test_research_c_rustiq_execution.py`](../../tests/management/test_research_c_rustiq_execution.py)。
既存adapter／builder／reference／compilerは変更しない。
`controlled_gate`のreport-only APIで同じOperatorを再評価せず保存するため、
実行入口ではOperatorを一度だけ取り、同じ既存`matrix_gate`で絶対誤差とunitarityを検査する。
比較するdiag(I,U)、tol1e-10／rtol0、位相fit禁止は同一である。

preflightで37 SOURCE hashと7保存入力hash、Python3.11.9／Qiskit1.3.0とAPIの一致を確認した。
24件のadapter＋実行管理unit/mock testがPASS。まだ実回路のPASSを意味しない。
共有環境への導入・変更なし。保存F1専用venvを読み、cache/tempは本実行専用領域へ置く。
起動時観測はCPU affinity32、load約1、MemAvailable約56 GB、CPU/memory pressureほぼ0。
独立wrapper task24に対応する最大24 process、BLAS/OpenMP/Numba/Qiskit各1 thread、
Rayonはavailable CPUs÷worker capacityとしてoversubscriptionを避ける。
gate間の依存はbarrierで保ち、各stage内の独立taskを並列化する。人工的CPU/RAM/wall上限なし。
GPUは既存driverに接続できず、今回のQ13／compiler／32次元検査はCPU経路を使う。

approval packetは固定実行commit、追加実行SOURCE hash、元contract hash、fresh outputへbindし、
最初の科学callより前に保存する。旧F1/F2 grantを再利用しない。
結果の保存先は`artifacts/research_c_rustiq_limited/2026-10-24/<fresh-run-id>/`。
実行後に結果・科学的限界・全attempt履歴を以下へ追記し、GitHub公開・照合後mandatory STOPする。

## 実行結果：登録された全gateがPASS

状態：**C4_LIMITED_CORRECTNESS_PASS_RESOURCES_RECORDED_MANDATORY_STOP**。
固定実行SOURCEは `b546c4e4f5d6645fd198b9ec3f85e3cd644ac4e9`。
実行中のSOURCE変更なし、attempt01が一回で完了し、技術的中断・retryともに0件。
前段のadapterは準備commitのままで、今回科学的な修正は必要なかった。
これはcommitへ固定するローカル数値検証であり、CIや外部独立再現の証拠ではない。

結果packetは
[`attempt01`](../../artifacts/research_c_rustiq_limited/2026-10-24/attempt01/)、
[実行summary](../../artifacts/research_c_rustiq_limited/2026-10-24/attempt01/summary.json)、
[費用比較](../../artifacts/research_c_rustiq_limited/2026-10-24/attempt01/comparison.json)、
[24 wrapper CSV](../../artifacts/research_c_rustiq_limited/2026-10-24/attempt01/metrics.csv)、
[provenance](../../artifacts/research_c_rustiq_limited/2026-10-24/attempt01/provenance.json)。
7つの保存inputは元hashのまま`attempt01/saved_inputs/`へ同梱し、50packageのlock/version一致と
Qiskit API/native binaryのhashも記録した。旧input、raw、grant、marker、STOPの原本は変更していない。

| gate | 完了数 | 最大絶対誤差 | 固定許容差 |
|---|---:|---:|---:|
| 局所controlled operator | 20 | 2.22045e-16 | 1e-10 |
| Full controlled `diag(I,U)` | 12 | 1.33229e-15 | 1e-10 |
| Actual circuit adjoint | 12 | 1.11031e-15 | 1e-10 |
| Raw X/Y複素信号 | 12 pair／24 evaluation | 1.55435e-15 | 2e-10 |
| Compiled full-wrapper operator | 24 | 1.12226e-14 | 1e-10 |
| Compiled X/Y複素信号 | 12 pair／24 evaluation | 5.03662e-15 | 2e-10 |

local/fullのunitarity検査もtol1e-10でPASS。
高精度referenceは90／128桁の28 bundleで、matrix agreement34件とsignal agreement4件が
1e-60を通過した。rtol0、global phase fitなし、許容差変更なし。
対象は前節の固定F1合成モデル、L_D=1、δ1/10の一点と4 motifだけである。

保存motif復元4 request、局所build20、full evolution12、actual adjoint12、wrapper24、
Operator68、Statevector48、cold／warm compiler各24、signal-pair gate24、資源metric24が完了。
624行のjournalは全開始／完了が対応し、FAILEDなし。
独立case workerで保存lawをrestoreし、対象armはSS／PPの2種類。law再生成とsamplingは0件。
Q13合成20回・入力triple60、実transpile24回、cold miss24／warm hit24、warm追加transpile0。
QPY92 file、operator NPY68 file、90/128 reference28 gzip JSONを保存した。

## Compiled 1-shot資源：SSの3caseで局所改善、PPでは同値

以下は**各axisの測定付きfull wrapper**の値で、X/Yで同じだった。
B0/B1の8対応pairで全8指標が一致したため、表ではB0/B1としてまとめる。
固定basis `rz,sx,x,cx`・opt1・seed17、状態準備／readout／ancilla測定を含む。

| case | RZ：B0/B1→R1 | CX：B0/B1→R1 | total depth：B0/B1→R1 | full size：B0/B1→R1 |
|---|---:|---:|---:|---:|
| `SS_motif_2` | 139→135 | 66→62 | 172→164 | 286→278 |
| `SS_motif_3` | 115→113 | 51→49 | 136→132 | 223→219 |
| `SS_motif_4` | 87→83 | 48→44 | 108→100 | 176→168 |
| `PP_motif_0` | 59→59 | 36→36 | 78→78 | 108→108 |

| 設計した4case×2axisの無重み合計 | B0/B1 | R1 | 変化 |
|---|---:|---:|---:|
| RZ count | 800 | 780 | −2.5000% |
| CX count | 402 | 382 | −4.9751% |
| RZ depth | 446 | 426 | −4.4843% |
| CX depth | 354 | 334 | −5.6497% |
| total depth | 988 | 948 | −4.0486% |
| quantum-only depth | 980 | 940 | −4.0816% |
| full-wrapper size | 1586 | 1546 | −2.5221% |
| measurement | 8 | 8 | 0% |

合計は設計した小さな検査集合の記述量で、実行すべき単一回路のdepthやlaw重み付き期待値ではない。
T count、認証shot、全round総資源、分子系や別δ窓の改善率へは外挿しない。
同じsaved law・同じ物理条件の1-shot回路合成経路だけを比較した結果である。

## 古典費用とcache

pipeline計時はwall21.3084秒、parent CPU4.4815秒＋children CPU121.9707秒。
24 worker capacity、各library thread1、workerの最大観測peak RSSは369,584 KiB。
これは並列化なしとの比較ではなく、speedup倍率を示さない。
process起動・大きな既存moduleのimportも含むpipeline時間と、以下のkernel区間を区別する。

| 保存された区間のwall秒合計 | B0 | B1 | R1 |
|---|---:|---:|---:|
| full evolution build（各4 request） | 0.53227 | 0.45757 | 0.46738 |
| adapter変換（builder内） | 対象外 | 0.003116 | 0.003099 |
| Q13合成（full builder内10 region） | 対象外 | 対象外 | 0.003436 |
| compiler cold（各8 request） | 0.44520 | 0.45386 | 0.47504 |
| compiler warm（各8 exact cache hit） | 0.01055 | 0.00850 | 0.00905 |

各区間はtask時間の和であり、並列jobのwall時間ではない。
adapter／synthesisはbuilderの部分区間なので再加算しない。
1回だけの短い計時で古典費用の優劣は認定しない。局所検査後のfull Q13はprocess-warm、
synthesis-cache-disabled。warm compilerは同じIRのcache hitで追加transpileを行っていない。
[route別wall/CPU内訳](../../artifacts/research_c_rustiq_limited/2026-10-24/attempt01/classical_costs_by_route.json)と
元journalを残した。pure-stdlibの
[`summarize_saved_results.py --check`](../../artifacts/research_c_rustiq_limited/2026-10-24/attempt01/summarize_saved_results.py)
で保存costを再照合でき、QuantumCircuit／Operator／Statevectorを新たに生成しない。

## 到達点・限界・次の担当

この固定scopeで、既存basis/run、scalar／identity／Taylor／product phase、非可換な異basis間順序、
full control、actual adjoint、raw／compiled X/Yを保ったQ13接続を数値的に確認した。
非可換な演算子を一つのRustiq regionへ渡す試験ではない。PPの非可換pairは既存fenceで分離する。
SS異frameのcommutatorも評価していない。

改善したregionには同一Pauli/supportの重複rotationが含まれる。
単純な同一Pauli角度結合baselineとの差は今回未検証であり、
Rustiq固有の新規手法やPR固有の科学的優位、広いlawの期待資源改善を認定しない。
今回確認したのは限定接続の正しさと、この設計検査集合の資源差である。

次の提案は、ユーザー／統括側が結果を確認し、
**同一Pauliの単純な角度結合を対照へ加える設計をCodexへ依頼するか判断すること**。
追加の実装・科学case・law重み付き評価は未承認。元workspace、旧SOURCE／結果／grant／STOPを保全し、
本結果のcommit・push・remote照合後mandatory STOP。C1、外部Rustiq、QuCLEAR、Gaussian singletonへ進まない。
