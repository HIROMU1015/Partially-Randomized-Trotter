# A寄与分解・B′物理pair限定batch：結果前scope（2026-10-10）

利用者が採用した[科学review §9](representation_attribution_pair_inputs/representation_construction_comparison_scientific_review_2026-10-10.md)の一体作業。
base c24dcede10ed9726e1e0b1030eca749ba7bc5cba、独立branch/worktree `representation-attribution-pair-20261010`。
旧source/result/他Trackのscience・契約・STOP、元root dirty状態を保護する。

## 固定条件と候補

Aは前batchと同じ3-mode JW全Fock8、正平方synthetic DF rank2、geometry/化学basisなし。
g0=[[.8,.09,0],[.09,-.35,0],[0,0,.15]]、g1=[[-.2,0,0],[0,.6,.07],[0,.07,-.45]]。
H=ΣdΓ(g)²、独立onebody/constant correctionなし。native L_D=0/1/2を保持し、係数収集whole-H all-Pauli、
従来計算基底のdiagonal-square core、全占有対角coreを比較する。frame-angle探索なし。
全対角coreの追加項はΣi<j|gij|²(ni+nj−2ninj)。dense diag抽出はconstructorに使わない。
T=.4、q=1/2/4、delta=.4/.2/.1。準備X0は1粒子だがbias適格性は全Fockで評価する。
この状態上のGaussian等価性から相関化学への利得を推論しない。

Bは同じphysical3+aux1 JW、V4=右積G01(π/8)G23(π/6)G12(π/10)、u=Vの上3行。
density edges01=.7、12=.4、23=.25、03=−.2。従来反射、係数代数physical-Pauli、物理pair Z/Z/ZZ、物理pair Qを比較。
新directは係数代数で生成し、旧dense-reference directの保存証拠を変更しない。両者の再構成差を明記する。
cx=Σxi aiの場合QR列の共役をGaussian orbital frameとし、Δ=Σi<j|xi yj−xj yi|²を使う。
Δ=0のみskipし、nearzeroはretain。任意悪条件のbinary64安定性は主張しない。
ω=wΔ、P=n_b1 n_b2、Q=I−2P、scalar Σω/2、tail係数−ω/2。
productはGaussian-conjugated controlled CZ、rotationはancilla位相−θとcontrolled pair位相2θ。
T=.2、q=1/2/4、delta=.2/.1/.05、L_D0、r1、K2。X0X1準備・aux真空。
比較taskは物理第一moment複素信号で、reset/channel protectionの評価ではない。分子compression fittingなし。

## 精度・費用・次数別推定

全候補q=1/2/4のmean/normalization/full-physical-Fock biasを照合する。
actual compiled costは結果前固定q=1のみ。q全体最適性やglobal winnerを主張しない。
epsilon_complex=.05/.02、axis epsilon=epsilon/√2、二軸失敗率合計.05、
N=ceil[2Γ²/(epsilon/√2−b)² log80]。bはexact小系operator診断で厳密証明ではない。
ordinary controlled diag(I,U)、準備とX/Y readoutを含む、測定1のmetadataはgate IR外。
rz/sx/x/cx、Qiskit1.3.0、opt1、seed20261010、coupling mapなし。6費用軸を混ぜた重みを発明しない。

canonical finite-RTE lawは不変。cost推定に限り既存event factoryから次数条件付きeventを作る。
次数0は全componentを確率重み付き全列挙。次数2はイベント空間≤96なら全列挙、その他は96 IID条件付きdraw。
共通uniform seed20264110、96×3 uniformを全候補のinverse CDFへ渡す。重複列は同一候補・次数内だけcompile reuse。
期待費用Σp_order E[C|order]、SEは非全列挙stratumだけp_order sd/√96。
X/Y和と候補差は同じdrawの共分散を保持する。exact stratumは候補差drawに定数で寄与。
SEはengineering sampling診断で、厳密CI、compiler移送幅、量子shotノイズではない。
basis_calls/operations、reflection Z actionsはsource構造診断でnative RZ/CXから独立の資源単位ではない。

## 実行・保全・STOP

constructorだけのpreflightで最大1866 compile（重複cache前）を確認。
source/tests/scope/inputsをcommit固定してからbatchを一回実行。
CPU1/BLAS1、CPU600秒、wall900秒、AS4 GiB、file64 MiB、compile2048、全回路5qubit以下、native10000 gates/circuit。
失敗は別failure.json/auditを保持し、routine実装修正は説明する。silent再実行/結果上書きなし。
全native IRとsource SHA/実行envを保存。別実装NumPy/SciPy/exterior-powerの保存監査で絶対位相を再照合。
旧reviewのinline algebra/JSONをinputとして別保存し、新batch resultと区別する。

C追加run=0。THC/fermionic fragment/LCUの一次資料との限定照合だけを行い、QR/projectorの代数を新規性にしない。
小batchが終わったら全結果でmandatory STOP、next_stage_authorized=false、central_hypothesis_adopted=null。
資源差、既知辞書への還元、B′basis費用、次の中心taskの判断をGPTへ返す。大分子・大系・新しい条件scanは行わない。
