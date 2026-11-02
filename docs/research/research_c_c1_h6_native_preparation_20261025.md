# 研究C・C1：H6 native回路・合成誤差・費用評価の技術準備

2026-10-25。起点 `554f5fa93231c561aaffce1b50335cbb3ce35b92`。
Branch `codex/research-c-c1-h6-native-preparation-20261025`。

**技術準備完了。新規30＋既存87 tests、29 subtests PASS。実H6の数値decode・build/compile・科学再解析は0。4候補×literal/sharedのnative正しさ数値pilotとkernel費用を取得する、最大298 logical unitsの未承認契約を準備した。有限probeから全operator誤差上界は認定せず、13 roundの合成誤差は未資格を保持する。公開照合後mandatory STOP。**

[未承認科学契約](research_c_c1_h6_native_contract_unapproved_20261025.json)／[SOURCE・JUnit・人工証拠・環境](../../artifacts/research_c_c1_h6_native/2026-10-25/technical/)／[前段のPF52セル](research_c_c1_h6_saved_pf_execution_20261025.md)。

## 接続範囲と再利用

新 [c1_h6_native.py](../../src/trottertracks/c1_h6_native.py)、[小gate state-action helper](../../src/trottertracks/c1_h6_state_action.py)、[CLI](../../scripts/run_research_c_c1_h6_native.py)、人工test2 filesを追加した。旧H2/H4/H6 SOURCE・raw結果・grant・ledger・marker・STOPは変更しない。

再利用する主な資産は `NativeBuilder`、`native_gaussian`／Qiskit Nature Givens、一般K I/Z/ZZ adapter、`SectorCAR`、`readout_segments`、`actual_adjoint`、P0 `compiled_wrapper`、QPY、`circuit_metrics`、旧`numerical_segment`／`round_synthesis`、`Calls`／`resource_plan`。新fitter、Gaussian合成器、CAR、PF、数値シミュレータは作っていない。

保存CDF最終rank14とXDF selected rank14のJ/Uを、両方とも同じ一般K number emitterと `prepare_cdf`／`NativeBuilder` へ接続し、XDFのlabelだけを変更する。片方だけにsquare-root再分解や新しい最適化を適用しない。元g補正κ、scalar、元frame順、全frame ordered S2を維持する。4候補はCDF s1/s4、XDF s1/s4、δ0.1、m0..12。own-DF ground oracle／状態準備除外、CA0.00159 Ha／DF reserve CA/8を保持する。

前段の保存DF誤差・guard、ground、PF kernels・52セルは再利用する。新fit、optimizer、rank探索、DF誤差・PF52セルの再計算はない。旧runnerも再起動しない。

## 密unitaryを作らない正しさ検証

既存のQiskit Statevector経路とGPU helperを調査した。GPU helperはGPU backendと固有のnative gate対応を含むため今回の固定CPU環境へそのまま流用せず、既存環境にあるAer0.15.1 CPU double statevectorを使う小さい接続を追加した。新しい依存installやvenv変更はない。

state-action helperは、Qiskitの公式gate definitionをたどり、最大3 qubitのgateだけを既存 `Operator` で最大8×8行列へ変換する。その局所行列をAerへ渡す。大きいcompositeは小gateへ再帰展開し、8192×8192／4096×4096のunitaryは作らない。これはsimulation専用入力で、費用は変換前のP0回路から測る。

全state-actionは8192成分のfull controlled状態を保持する。sector referenceを期待出力へ埋め込むためのbit順変換は既存`bit_reverse_indices`を使うが、**個々のGaussian gateをsectorへ射影しない**。期待sector外の出力も比較対象に含むため、probeで生じたleakageを捨てない。

### 検証順序とcoverage

1. **局所30因子**：各armのone-body＋14 frameを、共通τ=1/20でraw controlled回路にする。元h/g/cとJ/Uから独立に作ったCAR sector400の局所expmを参照とし、seed固定の複素sector状態・control+の全出力を比較する。block0のκ・scalar、各frameの対角位相、Gaussian向き・逆変換を検査する。全局所時間や全入力の検証ではない。
2. **共有shell3件**：head H、X readout H＋measure、Y readout Sdg→H＋measureを既存関数で作り、P0 compileする。system操作がないことを検査した上で、ancillaだけの2×2行列でraw／compiled／独立理想とunitarityを確認する。
3. **kernel8経路**：4物理候補のliteral/sharedをそれぞれh=1/(10s)のS2 kernelへ接続する。5固定入力は、control0のfull-Fock複素random、control1のsector先頭basis・複素random・own-ground、control+のown-ground。raw対保存PF reference、compiled対raw、compiled対referenceを位相fitなしで比較する。
4. **actual adjoint**：実際のraw／compiled回路のinverseを、その回路が出したcontrol+ ground状態へ作用させ、元入力への復帰を検査する。
5. **X/Y**：外部oracleとして渡すown-groundに、実head、kernelのs回反復、実X/Y readoutを作用させる。raw／compiledを、保存kernelをs回作用させた参照signal、および互いに比較する。これはm0・total time0.1のsegmented Hadamard検証であり、全13 roundの測定検証ではない。

κ、scalar、frame順、時間係数、control0 identity／control1 evolution、相対位相は上記で検査する。raw control0はbuilderのB⁻¹→controlled diagonal→Bという構造でもidentityを支持するが、有限state-actionのPASSからcompiled全operatorの厳密同値性を主張しない。

### 位相処理の人工修正

人工試験で、Aerの `set_statevector` が先に適用された回路global phaseを上書きする挙動を確認した。既存 `simulate_statevector` と同じ方針で、**回路が宣言するglobal_phaseを分離し、state evolution後にその値を戻す**task-local処理へ修正した。出力・参照から位相を推定したりfitしたりしない。

独立Qiskit参照試験では、nested definitionのphase処理が入力viewを変更し得るため、参照helperに入力copyを渡すよう試験を修正した。インストール済みpackageや旧helperは変更していない。初回5 fail／参照入力試験2 failのログも保持し、最終30件はすべてPASS。これは人工技術試験で、科学retryではない。

## 合成誤差：取得する量と未資格の量

固定allowanceはstate max-component／norm drift1e-10、complex X/Y signal2e-10。既存P0数値pilotで使う閾値を適用し、正規化された状態の振幅・signalのgateとして使う。これをHamiltonian誤差Haや全operator norm上界へ読み替えない。

raw/reference・compiled/rawの各probe残差、probe別L2、norm drift、局所因子・adjoint・m0 X/Y誤差を保存する。head／readoutは2×2全operatorでの数値pilotだが、reference／norm roundingの厳密enclosureはない。

8経路×13 roundのcoherent反復数 `n=s*2^m` を旧`round_synthesis`へ接続する。ただしkernelのglobal operator distanceは **None** とする。有限probe最大値をn倍して上界と扱わず、104行すべてで `UNRESOLVED_FINITE_PROBES`、complex/rigorous signal boundはNoneを保持する。未知誤差の0補完はない。

後で妥当なglobal／必要な入力部分空間上のoperator上界が得られた場合は、unitary telescopingとhead・X/Yの誤差を接続できる。今回その上界を保証したことにはしない。必要なら全sector basis columnのstreaming actionや別の妥当な構造／enclosure根拠を独立した将来範囲として準備する。最初の科学実行にはその重い全列検証を含めない。

## 費用比較と公平性

P0はQiskit1.3.0、opt1、seed17、basis rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false、backend／layout／couplingなしを維持する。過去のQiskit borrowing不具合を避けるため、既存`CompileSerialCalls`で**compileだけ**を排他化する。

既存の合法的共有は、隣接する回転行列が完全一致するrunのbasis共有と、同basis内の既存`math.fsum`角度結合／exact-zero規則のみ。literalは元run境界を保つ。一般の相対Gaussian融合は現NativeBuilderに実装されておらず、今回追加・採択しない。別基底境界や反復kernel間に未観測の削減を仮定しない。

費用は**h-stepのcontrolled S2 kernel1個**についてRZ数・RZ depth・CX数・CX depth・total depth・size等を測る。4つの同s／同routeのCDF-XDF比較と、4候補それぞれのliteral-shared比較を保存する。head/readoutの費用も別保存する。s4のmacro全体・全round・shots加重費用を今回計算しない。

将来は保存DF誤差、PF誤差、資格に応じた合成誤差、RPE統計条件を共通CAへ接続した後、各axisの必要shotsと費用を集計する。今回の契約にshots・最終CA・interrogation総RZを含める必要はないと判断し、明示的に除外した。異種誤差を根拠なく単純加算せず、同CA総資源優位は未認定。

## 固定した未承認契約と規模

Scope `C1_H6_RANK14_NATIVE_PROBE_COST_V1`。
SOURCE manifest SHA256 `783a280916b7febbcd4c71628305cee94a7dafa7ec460393ab30dbd4f9a7fd27`。
契約SHA256 `516bd17b89dfe74d25bdbd59b14d128cdb5ada493504586190389774d6df2040`。

SOURCEは既存37＋新規5＝42 files、保存入力／PF来歴25 filesを固定した。Aerは既存evaluation lockにもある0.15.1を追加version照合し、fit環境は不要。future outputは `artifacts/research_c_c1_h6_native/2026-10-25/run01`。新fit・optimizer・rank探索・DF/PF再計算・retry/resumeは0。

| 科学call | 最大 |
|---|---:|
| 保存bundle読込・schema／保存予算gate | 1 |
| native model | 2 |
| sector CAR context／generators | 1＋2 |
| cached Gaussian basis要求 | 30 |
| 独立局所CAR reference／局所state-action | 30＋30 |
| raw回路build（局所30、kernel8、shell3） | 41 |
| P0 compile／shell検査 | 11＋3 |
| kernel／adjoint／X-Y action bundles | 8＋8＋8 |
| compiled metrics | 11 |
| 13-round未資格合成誤差profile | 104 |
| kernel費用比較 | 8 |
| **合計logical units** | **298** |

Gaussianは30回のmaterialization要求で、cacheによる実構築数はそれ以下。QPYは局所raw30＋kernel raw/compiled16＋shell raw/compiled6＝52件。statevector作用は最大214入力伝播／150 Aer batches（最大batch5）を想定する。Aerのshots=1指定は、測定のない決定論save-state処理の内部指定であり、RPE shots算定・量子samplingではない。

いずれかのidentity・schema・非有限値・phase・Gaussian・frame・compiler等のcorrectness FAILで依存callを停止し、開始済み作業だけをdrainしてpartial evidenceを保存する。未知合成誤差は未資格としてnative probe／費用処理を完了できるが、CA・shots・総費用へ進まない。正常完了もmandatory STOP。

## 人工検証・資源・残る事項

| 試験 | 結果 |
|---|---:|
| 新state-action：nested phase、det±・π/branch、既存Qiskit照合、13-qubit人工compositeで局所matrix限定、leakage・測定scope | 15 PASS |
| 新native：人工CAR/P0 end-to-end、phase/Gaussian向き/frame順/compile/adjoint不一致STOP、未資格誤差・入力不一致・passive preflight | 15 PASS |
| 旧sector/PF回帰 | 35 PASS |
| 旧Stage2 native回帰（独立process） | 27 PASS |
| 旧一般K回帰（独立process） | 13 tests＋29 subtests PASS |
| 旧決定論精度接続回帰（独立process） | 12 PASS |
| **合計** | **117 tests＋29 subtests PASS** |

実H6 NPZのdecodeを遮断したSOURCE42／input25／environment preflightとdefault CLIもPASS。人工の13-qubit回路は人工gateのみで、実H6回路ではない。旧tracked SOURCE・結果・STOPは保全した。前段PF52セルの値を再解析・再計算していない。

1状態は128 KiB、5状態batchは640 KiB、局所CAR referenceは最大400×400（complex128で2.56 MB）。大きい全unitaryより小さく、実際の主負荷はnative回路処理・compile・state actionsになる。利用可能CPU/RAMを確認した上で、共有basisの準備／buildは逐次、重い独立局所expm/actionと8 candidate actionsは既存resource_planで並列化する。Aer threadsはaffinity／quotaとworkersから決め、BLASを調整し過剰並列を避ける。恣意的なCPU/RAM/worker/wall capはなく、共有設定・他ユーザーには変更しない。

**最初のnative pilotを実行するための技術blockerは残っていない。** 実H6でのprobe／compiler PASSと費用値は未取得。有限probeの外のoperator同値性、global kernel合成誤差上界、13 roundのCA・shots・総資源は未資格／未評価のままである。

次はユーザー／研究Cが、この固定298-call native probe・kernel費用契約の限定科学実行を許可するか判断する。Codexは別途明示許可後にfresh run01で登録範囲だけ実行する。**今回は公開照合後mandatory STOPし、実H6の新build/compileや科学評価へ自動進行しない。**
