# N1/N3限定再設計：実行前の固定と数学確認

2026-10-11 JST。利用者の[実行指示](n13_redesign_inputs/execution_authorization_20261011.txt)により、
[元scope v1](n13_redesign_limited_scope_v1_20261011.md)の一回の限定batchを承認済み。
元scopeおよび準備記録の「未認可」は当時の状態として変更しない。
[実行plan](n13_redesign_execution_plan_20261011.json)が今回の認可・source/input hash・資源上限を結ぶ。
sourceは独立branchで実行前にcommitする。過去の科学結果の再レビューは行わない。

## 数学・数値解釈

N1の共通固有基底での平方誤差は、ν=2の6部分集合を有理数で列挙する。
全44 factor候補、178 pair、gauged/ungauged、q=1/2/4/8を維持する。
区間Gaussian一体作用はRZ/P/XXPlusYYの理想gateパラメータから作り、
元gと参照Qの差、およびgauged構成と共通Qによる近似との差をentry-L1で囲う。
固定小系のsubset列挙は一般fillingでのpolynomial算法ではない。

PFは[既知のS2上界](https://arxiv.org/html/1912.08854v2#S5.SS1)の12/24係数を使用する。
128bit区間Pauli係数を2^-128 gridの整数端点へ外向きに丸め、中心の交換子をGaussian整数で完全にまとめる。
中心からの係数L1半径をdA,dB、中心norm上界をa,bとすると、
二重交換子[A,[A,B]]の丸め追加上界は4{(a+dA)^2(b+dB)-a^2 b}。
これは三重線形展開と||[X,Y]||≤2||X||||Y||から得る。中心の相殺と丸め余裕を区別する。
identityを交換子から除くが、controlled phase費用と位相角の丸め会計に残す。

保存gには最大約1.1e-16の非対称成分がある。入力を変更せず、
γ≥||anti(H_original)||を係数entry-L1から取得する。
||exp(-itH_original)||≤exp(tγ)なので、Duhamel上界を保守的にT εH exp(Tγ)とする。
元scopeのT εHに追加するT εH{exp(Tγ)-1}を独立fieldへ保存する。
これは保証を厳しくする丸め補正であり、候補・精度・比較対象を変更しない。
元H whole-Pauli対照はHermitian部分の係数と区間midpointを実装するので、γとmidpoint差をモデル余裕に加える。
mpmathの[区間算術の説明](https://mpmath.readthedocs.io/en/latest/contexts.html#arbitrary-precision-interval-arithmetic-iv)に沿い、
使用する基本演算とsin/cos/exp/log/sqrtを限定する。1.3.0の複素conjugate/constructorの問題を通しtestで検出し、
新module内で明示的な実部・虚部処理へ修正した。旧source・libraryは変更しない。

N3の新方向保証は元scopeのSchur証明をそのまま適用する。
係数は旧入力と同じbinary64 literalを正確な有理数へ変換する。
LDL正定値、κ、固定λgrid、trial U、c、δgoal判定は全て有理数演算。
Gershgorin rはLDLがrで特異でも非厳密一体下界として使用できる。
真のa/E0/Cやその固有vectorはselector記録のhashを閉じた後の診断に限る。

旧scalar保証も有効：最初に占有した外部軌道でdiscarded空間を直交classへ分割する。
classのdiagonal最小値から全hopping・density絶対和を差し引くとclass下界を得る。
class間block normは外部に触れるhopping絶対和以下なので、block row下界として
各class下界から(class数−1)×offを差し引く。
activeからclass jへのblock normはその外部軌道とactiveのhopping絶対和βj以下。
U≥aと全dj>Uのとき、weighted Schur条件δ≥Σβj²/(dj−U+δ)は有効な幅保証となる。
粗い下界を真のgapへ交換しない。

## 実装成立・資源・保存

新module、runner、synthetic tests、独立保存verifierは元scopeに指定したpath。
最小検証はCAR ν0/1/2、Gaussian向き、Pauli/JW、交換子、非対角active、SPD失敗、
負density拒否、zero couplingと外部低エネルギー、δ境界、oracle拒否、
scalar ordinary control、native vector作用、weighted-L1重み、合成通し検証を含む18件。
[test履歴](../../artifacts/n13_redesign/2026-10-11/preexecution/test_history.json)に実行前の修正と失敗を保存する。
科学条件を使うN1/N3/固定Sの評価はsource固定後の一回だけ。

CPU32、GPU deviceなし、使用可能memory約51GiB。最大2 spawn workers、BLAS/OpenMP/Numba/Rayon各1、
Qiskit parallel FALSE、transpile num_processes1、optimization1、seed20261011、native rz/sx/x/cx、
qubits_initially_zero=False。主負担はcompile/全列作用検査。完全な発行回路fingerprintが一致したIRのみ再利用する。
旧速度との同条件比較をしていないのでspeedup率は主張しない。
worker AS4GiB、aggregate8GiB、shard32MiB、output512MiB、log16MiB、native20000gates。
wall/CPU time capはない。IRは個別lossless gzip JSONとhashで保存する。
異常時の完成IRとperiodic rows checkpointを保持し、科学retryはしない。

compiled回路は5qubitsの全32列でordinary control、scalar phase、任意system/control入力を検査する。
N1にworkspaceはない。compilerのscalar discrepancyのみ絶対作用確認後にphase補正を記録する。
binary64全列検査はcompiled回路の厳密機械certificateではない。
報告範囲はideal_gate_certificate_with_numerical_compile_validation。

独立保存verifierはproject/Qiskitをimportせず、NumPyとstdlibでnative全列、
別のsymplectic Pauli整数演算、LDL・Schur条件、LP primal/dualを照合する。
科学再実行・compile・LP solveを行わず、immutable CIや外部再現とは区別する。
結果は新run1へ保存し、GitHub公開・独立remote blob取得を確認した後mandatory STOP。
中心テーマnull、次の科学段階falseを維持する。
