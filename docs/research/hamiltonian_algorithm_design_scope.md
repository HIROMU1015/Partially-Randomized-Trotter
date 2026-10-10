# N1/N2構成・N3係数下界の限定feasibility：結果前scope

2026-10-11 JST。利用者の設計採用指示を受け、
[提供設計 §8](hamiltonian_construction_inputs/hamiltonian_algorithm_design_2026-10-11.md)を一体の実装・検証作業にする。
基点は結果commit `f98050e9402e8c65bb0f3dd27c7bd30d12fe7069`。
独立branch/worktreeは `hamiltonian-construction-feasibility-20261011`。
前回reviewのA-core/B′停止、旧B/C保留、他Trackの契約・source・結果・STOPを保持する。
未commitの旧review受領worktreeは変更せず、今回の設計を独立入力として保存する。

## 固定する構成と問い

N1はJW 4-mode、ν=2、sector dimension6、signed exact synthetic DF rank2。
H=H1+.7 dΓ(g0)²−.4 dΓ(g1)²+.11I、H1=diag(.08,−.04,.03,−.02)。
geometry・化学basis・分子fittingなし。L_DはDF-prefix PRに該当せず、全fragmentを決定論S2で実装する。
model error予算εH=.02、T=.6、q=1/2/4/8、delta=.6/.3/.15/.075。
近縮退η0=(1,1.002,−.4,−.397)、η1=(.7,.703,−.2,−.196)。
三contextは局所gauge植込み、同じspectrumで群間のdense frame、離れたspectrum。
離れたη0=(1,1.25,−.4,−.1)、η1=(.7,.95,−.2,.1)。全frame角度はmoduleのfixtureに結果前固定する。
非可換fragmentのcommutatorも評価し、1粒子／1空孔だけのGaussian還元を性能根拠にしない。

constructorはn×n固有分解、全contiguous partitionsと平均中心、群projectorからの座標Gram-Schmidt、
column assignmentと位相規約からframeを返す。zero cutoffは全subset、shifted cutoffは全subsetを一つの平均へ置換する有限対照。
これらも同じgauge・compilerを使う。既知shifted/cutoff方法全体の最適化ではなく、強い有限対照の実装である。
特にSCDFの完全なtensor fitting・symmetry-shift最適化を再現したとは呼ばない。
factorごとのsector平方boundを加算し、元Hに対するモデル誤差を計上する。binary64評価でinterval certificateではない。

有限組合せからbasis-native RZ/CX別のpoint最小を選ぶ。selectorはfactor情報とbasis compile費用だけを使う。
選択の後でfull wrapperを測り、basis proxyの最小を全回路の最小と呼ばない。
exact native、N1のRZ/CX選択、同一近似ungauged、shiftedのRZ/CX選択、zero cutoff、
元H whole-Pauli、N1近似H whole-Pauliの最大9比較を全qで評価する。
近似S2とwhole-Pauli S2へ同じcompiler改善を適用する。

N2はJ-only constructor、JW occupation4、全ij二次形式、k<=2、S∈{−1,0,1}、row d<=2。
全40 signed atoms、1/2列の820候補をleast squaresでfitし、係数絶対値上界<=.0021の候補のみ保持する。
加算作用数proxy、workspace、k、boundで選ぶ有限小入力探索で、指数候補数を持つ。大系用のpolynomial constructorではない。
fixed contextsはsigned overlap、E03=E30=.001摂動、uniform Hamming、dense-real rank2、集団圧縮できない疎J。
元のSをconstructorへ渡さない。T=.7、直接densityは可換なのでPF誤差0、近似modeの元J誤差を数える。
generated charge、同じ近似のdirect、元J directを比較し、uniformには既知unsigned HWPを追加する。
HWP baselineは同じincrement回路のunsigned版。最適adder/QROM等を実装した包括的比較ではない。
signed two's-complement register、compute–controlled phase–uncompute、workspace真空復帰を
全system計算基底と両ancilla入力で検査する。workspace消去は真空入力上の保証で任意workspace初期値へ広げない。

N3は6-mode、ν=2、4 active＋2外部のsynthetic模型。activeは非対角でsector dimension6。
one-body e=(−.2,−.1,.1,.25,6,8)または末尾(.4,.6)。density .3 n0n1、
hopping(1,2)=.2、(0,3)=.1、(2,4)=.08、(3,5)=.04、(4,5)=.05。
真のground/gapをconstructorに渡さず、試行occupation01からU、係数norm、first-external classesのsortingから下界を得る。
Q-class間の全hopping係数normを保守的に差し引き、coupling別fixed pointを上側bracketでbinary64評価する。
energy error<=.004で返すか、最大couplingの外部orbitalをactiveへ戻して再計算する。
保証不能と削減不能を区別し、dense ground truthは選択後の評価だけに使う。
N3に時間発展誤差・回路費用・エネルギーsolver/QPE成功率の保証を付加しない。

## 精度・資源会計

N1/N2は固定時間complex Hadamard信号、εcomplex=.04、二軸合計失敗率.05、各軸ε/√2。
準備X0X1、ordinary control diag(I,U)、X/Y readoutを含む。測定1はmetadataでgate IR外。
deterministic evolutionなのでnormalization=1。shot方策はceil[2 log80/(ε/√2−b)²]、b>=ε/√2なら不適格。
N1のbはTεH＋近似Hへのexact小sector PF norm診断。元Hへ直接計算したbiasも別保存し、後者へ置換して勝敗を作らない。
N2のbはT×density coefficient bound。丸め・合成・hardware errorを厳密に認証したとは呼ばない。
RZ/CX/sx/x/size/depthとsystem/workspaceを別々に表示し、任意重みで合算しない。RZはlogical Tではない。
RTE残差回復、sampling law変更、quantum shots、分子load/fitting、GPU・ground-state分子solveは0。

## 実行・性能・証拠・STOP

実装・tests・scope・入力と保存verifierをcommit固定してから一回実行。
32 available CPUs/約52 GiB available memoryを読み取り確認。GPU driverは利用不能。
前stageでは1558 compileにwall約24秒。新batchの主負担もcompile/絶対位相検査と見込み、独立9 contextをspawn 2 workersへ配分する。
BLAS/OpenMP/Numba/Rayon各1、Qiskit parallel FALSE・transpile num_processes1。task/fixture/compile seedはworkerに依存しない。
worker AS4GiB、合計8GiB budget、per-file64MiB、aggregate compile768、per-context384、13qubits、native20000 gates/circuit。
新しい計算にはphase/total wall-timeとCPU-time capを設けない。tool polling timeoutは科学実行上限ではない。
大系full Operatorを作らず、N2はclean embeddingの全列への作用を検査し、native行列作用をvectorizeする。
初期referenceはQiskit Operator（小系）/Statevector（workspace系）で別検査する。
速度向上率は未測定。古いcompiler/サイズの時間からその倍率を主張しない。

失敗はfailure.json・完了済みcheckpoint・auditを保持する。通常の実装ミスを修正する場合は別source/別runとして履歴を残し、silent retryしない。
新設計の付録Python/JSONは独立input。再実行は式の再現で、科学batch・分子改善の実証と分ける。
NumPy/SciPyだけの保存verifierがexplicit JW・native tensor action・phase/workspace・shots・N3係数下界を照合する。
result/source/IR/auditを公開し、別Git取得で確認した後はGPTの研究判断へ戻す。
全結果でmandatory STOP、next_stage_authorized=false、central_hypothesis_adopted=null。
同じ条件へseed・角度・サイズを追加して利益を作らず、中心テーマ・新規性・分子優位を自動採択しない。
