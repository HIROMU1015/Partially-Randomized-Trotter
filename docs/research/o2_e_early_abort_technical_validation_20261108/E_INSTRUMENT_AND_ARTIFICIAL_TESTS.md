# 案E：2-label instrumentと人工correctness

2026-11-08 JST／SERVER1 GPUA100-gleap-01。設計SOURCEは 081959402f54f06048adb038f97911950091ae1c。本結果は**発明した1/2 system qubit回路・状態の人工検証**であり、分子geometry/basis/DF rank/L_D/δ窓は非該当。実H4・保存N1・旧rawを入力にしていない。

## 実装と再利用

- module：[e_label_erasure.py](../../../src/trottertracks/algorithm_codesign/e_label_erasure.py)
- test：[test_e_label_erasure.py](../../../tests/tracks/algorithm_codesign/test_e_label_erasure.py)
- runner：[run_e_label_erasure_artificial.py](../../../scripts/tracks/algorithm_codesign/run_e_label_erasure_artificial.py)
- 証拠：[SOURCE](../../../artifacts/o2_e_label_erasure/2026-11-08/artificial_run01/SOURCE.json)、[結果・全分岐](../../../artifacts/o2_e_label_erasure/2026-11-08/artificial_run01/TEST_RESULTS.json)、[log](../../../artifacts/o2_e_label_erasure/2026-11-08/artificial_run01/RUN.txt)、[STOP](../../../artifacts/o2_e_label_erasure/2026-11-08/artificial_run01/STOP.json)。

A2の _check_system_circuit とphaseを保持する to_gate().control() の接続、既存Hadamardのbit対応0→+1/1→−1を再利用した。A2のβ² decoder・fine−coarse信号・herald補正は使用しない。M人工testの非正規化branch列挙方式を、Clbit条件とsuccess block内の測定へ限定的に拡張した。既存 matrix_gate を位相fitなしの独立比較に使用し、runnerはMのfresh output＋SOURCE/results/log/STOP保存方式を再利用した。

A1のlaw/stream/CV facadeとRTE sourceも確認したが、実law・RNG・統計campaignがないため実行に接続しない。旧A2/M/A1/RTE/測定helperのSOURCE変更0。一般LCU framework・compiler・科学runnerは作らない。

## 一つの具体的なgate列

入力はsystem-only明示unitary A,B,W0,W1、exact rational 0<p<1、axis X/Yだけ。状態・成功率・path信号はadapterの引数にない。幅は人工1/2 system qubitに限定する。

system wires 0…n−1、signal a=n、pool k=n+1。ancillaは初期0。状態の準備は外部に置き、全attemptでfresh stateを消費する。

1. H(a)、a=1 controlled B。
2. kへ RY(2 atan2(√(1−p),√p))。
3. controls [k,a] の状態2=(0,1)でW0、状態3=(1,1)でW1。
4. kへ逆RY。scalar/identity/event phaseは各unitaryのglobal phaseごと制御する。
5. kをclassical bit0へ測定。
6. bit0=0の if_test 内だけでcontrolled A、XはH、YはSdg→H、aをbit1へ測定。
7. flag1はaxis bit未取得のまま0。flag0はbit1の±1。**全attemptを分母**にし、success数で割らず、自動retry0。

early_abort=False は同じcoherent prefixの後で全attemptにsuffixを実行し、最後にflagを読む比較経路。early失敗にはelse blockがなく、AもX/Y readoutも起動しない。Qiskit1.3の[if_test](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/qiskit.circuit.QuantumCircuit#if_test)を使用したが、hardware feed-forward対応は未確認。

## 独立参照と理想式

P=[[√p,−√(1−p)],[√(1−p),√p]] とすると

\[
S=K_0=pW_0+(1-p)W_1,\quad
K_1=\sqrt{p(1-p)}(W_1-W_0),\quad \sum_rK_r^\dagger K_r=I.
\]

prefixの出力は

\[
\frac{|0_a0_k\rangle|\psi\rangle+
 |1_a0_k\rangle SB|\psi\rangle+
 |1_a1_k\rangle K_1B|\psi\rangle}{\sqrt2}.
\]

失敗k=1にa=0成分がなく、Aを作用してもX/Y offdiagonal寄与は0。そのため失敗を捨てず0で記録し、suffixを省略できる。

\[
Z=\langle\psi|ASB|\psi\rangle,\quad
p_s=(1+\|SB\psi\|^2)/2,\quad
E[Y_X]+iE[Y_Y]=Z,\quad E[Y_a^2]=p_s.
\]

参照はB→W_j→Aを別々に合成した通常controlled Hadamardのlabel加重平均と、独立system行列積から作る。実行側のdynamic measure/if_elseをtest側で全分岐列挙し、分岐stateを正規化せずBorn質量を保持する。Statevector/Operatorはtestにだけあり、adapterへ参照値を渡さない。RNG、backend shots、transpile/native synthesisは0。

通常mixtureではV_raw=1−μ²、案EではV_E=p_s−μ²。μ²≤||SBψ||²=2p_s−1より V_raw/2≤V_E≤V_raw。testで両軸を照合した。factor2は通常mixtureとの比較で、最適層別化・直接測定への保証ではない。

## 人工結果

**新規14件＋関連A2回帰2件＝16件PASS、failure/error0。** Qiskit1.3.0、NumPy1.26.4、Python3.12.3。suite約0.536秒、process peak RSS542,804 KiB。最大4量子bitの小さい全分岐計算で、process並列化やGPUを使う負荷ではない。CPU/RAM/workerの人工上限・共有設定変更なし。

| 確認 | 入力と結果 |
|---|---|
| 全入力Kraus block | 1/2 system qubit、p=2/5・3/7。非可換A/B、非可換W0/W1。prefixの全system入力列と独立block行列が一致 |
| complex平均・M2 | 上記×X/Yの8条件でlabel別Hadamard・system行列・dynamic全分岐が一致。非正規化branch確率和も1 |
| 失敗zero・suffix省略 | a=0失敗blockは0。Aを実行した失敗参照でもX/Y寄与0。early failureはaxis bit=None、A/readoutなし |
| phase・sign・ordering | signed W1、A/B/Wのscalar phase、relative controlled phase、2-system wire順、Sdg→H、bit対応PASS |
| 退化・normalization | W0=W1でp_s=M2=1、失敗質量0、M2利益なし。全attemptの[+1,0,−1,0]はM2=1/2 |
| 標準LCU対照 | 独立positive LCU＋early flagと全branch state/bit、論理block実行数が一致 |
| 小角度対照 | Z/X回転、p=2/5、角0.2→0.02で失敗確率約0.009473→0.00009599。M2利益・省略率はともに縮小 |
| 不正入力 | 非unitary、未束縛角、偽gate名/未知oracle、幅不一致・8-system幅の人工sentinel、非有理p、未測定success bit等を拒否 |

主な非可換例はW1に符号があるがW0と同じunitaryの反対符号ではない。自明な反対符号pairを主要改善例にしない。許容差4×10^-12はbinary64参照のテスト閾値であり、一様engine誤差の証明ではない。

## PREP精度と未保証事項

理想PREPでは元pのtargetを保存する。実効pを2/5→9/20へ変更した人工例ではX/Y平均bias≈−0.0216543/−0.00514429で、(q−p)(μ0−μ1)と一致した。label消去後に元labelのimportance weightで自由補正できるとはしない。

逆PREPだけに角0.12の不整合を加えた対照では、失敗branchのa=0成分norm≈0.042401、X/Y平均bias≈−0.0130372/−0.00312434。失敗zeroの前提自体が壊れ得る例で、許容精度の採択や一般誤差上界ではない。

PREP/逆PREP精度、二重制御/phase、norm・Born・測定bit-grid、decoder、失敗省略biasの全support一様保証は**UNKNOWN**。旧A1 exact engineの上界は継承しない。人工PASSを実H4適格性・費用利益・hardware対応へ外挿しない。
