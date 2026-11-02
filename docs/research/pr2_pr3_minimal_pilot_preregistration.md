# PR-2／PR-3最小pilot事前登録

作成日：2026-09-27  
status：`PR23_MINIMAL_PILOTS_PREREGISTERED_NO_RESULTS`  
親方針：`PR_DIRECTION_STAGE_A_ADOPTED_NO_PILOT_YET`  
前段契約：[PR-1＋PR-3 Stage A選定契約](pr1_pr3_stage_a_contract.md)

本書は、PR-2とPR-3を各1回だけ実行し、その直後に必ず停止して主題を再選択する条件を結果前に
固定する。H12、長RPE、系サイズ展開、geometry展開、precision/split sweep、全候補compileを許可しない。

## 1. 文献で固定するPR-1--PR-4の位置付け

### PR-1：共通評価枠

- **既知**：deterministic/randomized partition、Pauli/DF実装、qDRIFT/RTE、compiled-cost評価。
- **残す差分**：同じestimand、物理時間、control、shot、state-preparation scopeで、deterministic workと
  random-tail workを欠落なく比較するrecord schema。
- **最小着地点**：PR-2/PR-3の両pilotが同じ`correctness / bias / variance / one-shot work /
  total work / scope`字段を返す。
- **ここでしないこと**：PR-1を独立の大規模benchmarkへしない。

### PR-2：圧縮＋random residual

- **最も近い先行研究**：[SPRINT/GRADE](https://arxiv.org/abs/2606.30741)はfactorization residualを
  Pauli項として表し、qDRIFT/RTE等で処理する構図を明示する。
  [RC-DF](https://quantum-journal.org/papers/q-2024-06-13-1371/)は圧縮DFと資源削減を既に扱う。
- **既知**：圧縮法、残差のrandom処理、圧縮率を資源へ使うこと自体。
- **新たな問い**：既存H4 DF表現で、粗いrankを止めたときのdeterministic compiled work削減が、
  明示残差のsampling one-normと1-sample回路費用を戻しても残るか。
- **最小着地点**：rank 3/6/9の構造的trade-offが存在するかを一度だけ判定する。
- **最大不確かさ**：小さいDF truncation/energy biasが、小さいsampling $\ell_1$ normを意味しないこと。
- **最小計算**：rank-12 H4 anchorから3 compression pointを作り、exact residual再構成、energy bias、
  compiled deterministic RZ、residual component分布、qDRIFT screening workを計算する。

### PR-3：tail-only extrapolation

- **最も近い先行研究**：[qFLO](https://arxiv.org/abs/2411.04240)はfull qDRIFT observableの
  $1/N$展開とRichardson外挿を与える。Composite/PRはpartitionとcoherent signalを既に扱う。
- **既知**：qDRIFT、Richardson weights、partial partition、Hadamard Re/Im信号。
- **新たな問い**：tailだけをrefineしたとき、外挿weightのshot増幅と各levelのbackbone再実行を戻しても
  通常partial-qDRIFTに対する余地があるか。
- **最小着地点**：固定2-qubit非可換系の$N,2N,4N$ exact meanと有限shot costで一度だけ判定する。
- **最大不確かさ**：bias cancellationがvariance/backbone費用を上回るか。
- **最小計算**：linear Re/Im estimand、tail-exact参照、通常PR、partial-tail Richardson、
  full-random、deterministic endpointを同じprimitive-rotation scopeで比較する。

### PR-4：partial-MLMC

- **最も近い先行研究**：[MLMC-qDRIFT](https://arxiv.org/abs/2604.26865v2)はindex-sharing coupling、
  augmented difference state、scaled observableを与え、partial deterministic/randomized partitionとの
  組合せを将来方向として明記する。
- **既知**：full-qDRIFTでのlevel couplingとvariance decay。
- **新たな問い**：共通$H_D$を含む物理的difference estimatorの全費用とsplit依存性。
- **最大不確かさ**：独立測定または単純controlled superpositionではshot varianceが$O(1)$のままで、
  augmented updateは一般に非ユニタリである。dilation、postselection、normalization、再実行費用が未固定。
- **判定**：`HOLD_PR4_ESTIMATOR_COST_UNFIXED`。今回は数値pilotを作らない。PR-2/3がともにSTOPした
  場合だけ、物理回路と成功確率を先に固定する別監査へ戻す。

## 2. 共通評価record

両pilotは次を必須fieldとして返す。

| 区分 | 必須内容 |
|---|---|
| input | Hamiltonian/model、basis、geometry、rank/split、物理時間、state、hash |
| correctness | Hermiticity/unitarity、exact residual reconstruction、probability normalization、有限値 |
| estimand | 元のexact target、同じouter methodのtail-exact target、Re/Imまたはenergy |
| error | approximation/discard bias、random-tail bias、outer biasを分離 |
| statistics | level mean、single-shot variance、weights、shot allocation |
| work | deterministic work、1 random sample、1 shot、全shot。異なる単位を足さない |
| scope | controlled/uncontrolled、state preparation、ancilla wrapper、compile条件、理論boundか実測か |
| decision | 固定gate、GO/CONDITIONAL/STOP、禁止する外挿 |

## 3. PR-2固定条件

### 3.1 入力と3点

- H4 linear chain、距離1.0 Å、STO-3G、8 qubit。
- OpenFermion low-rank decompositionのrank-12表現をpilot内の元Hamiltonian $H_{12}$とする。
  exact molecular Hamiltonianではなく、既存anchorと同じrank-12参照である。
- 圧縮点はdecomposition orderの先頭rank $r\in\{3,6,9\}$。rank-12をdeterministic endpointとする。
- constantとone-body correctionは全点で同一に保ち、
  $H_{12}=\widetilde H_r+\Delta H_r$をrank-12の省略DF blockからexactに再構成する。
- ground-energyは同じphysical sectorで評価する。

### 3.2 work scope

- deterministic work：second-order PF 1 step、time $T=0.1$、Qiskit basis
  `rz,cx,sx,x`、optimization level 0の`total_ref_rz_count`。
- residual representation：identityをrandom distributionから外し位相へ移す。
  coefficient thresholdは0。`exact_rte_lambda_r`、component count、support、basisを保存する。
- 1 qDRIFT sample：$Ue^{-i\theta Z/ZZ}U^\dagger$を同じbasis/optimizationでcompileしたRZ count。
  component probabilityで重み付けした期待RZ countを用いる。
- tail bias budgetは$\epsilon_{\rm tail}=10^{-2}$。screening step数は保守的な

  $$
  N_r=\max\left(1,\left\lceil\frac{2\lambda_r^2T^2}{\epsilon_{\rm tail}}\right\rceil\right).
  $$

- hybrid screening workは
  $G_r=G_D(r)+N_r\,\mathbb E[G_{R,1}(r)]$。
  full deterministic対照は$G_D(12)$。これは1 outer stepのscreeningでありRPE総costではない。

### 3.3 correctness gate

1. rank-12と各compressed表現のconstant/one-bodyが一致する。
2. rank-$r$ blockがrank-12 decompositionのprefixと最大絶対差$10^{-10}$以内で一致する。
3. dense small-systemで
   $\|H_{12}-\widetilde H_r-\Delta H_r\|_2/\max(1,\|H_{12}\|_2)\leq10^{-10}$。
4. sampling probability和の誤差が$10^{-12}$以下で、全costが有限・非負。

一つでも失敗すれば科学的STOPではなく`IMPLEMENTATION_INVALID`とし、結果選択へ使わない。

### 3.4 固定判定

eligible pointはcorrectness通過かつresidual非空の点とする。

- `GO_PR2`：いずれかの点でdiscard energy bias $>10^{-3}$ Ha、deterministic RZ saving $\geq20\%$、
  かつ$G_r/G_D(12)<1$。
- `CONDITIONAL_PR2`：上のbias/savingを満たし、$1\leq G_r/G_D(12)<2$。
- `STOP_PR2_NO_COMPETITIVE_TRADEOFF`：上記を満たす点がない、または全eligible点でratio $\geq2$。

qDRIFT boundとRZ countだけのscreeningなので、GOでも資源優位の最終結論とはしない。

## 4. PR-3固定条件

### 4.1 model、state、estimand

Pauliのqubit indexはlittle-endian matrix conventionで固定する。

$$
H_D=0.9Z_0+0.7X_0X_1,
\qquad
H_R=0.31X_0-0.27Z_0Z_1+0.19Y_1.
$$

$|00\rangle$へqubit 0の$R_y(0.73)$、qubit 1の$R_x(-0.41)$、CNOT$(0\to1)$を順に適用した
正規化stateを$|\psi\rangle$とする。物理時間は$T=0.8$。

主estimandは
$z=\langle\psi|U|\psi\rangle$のRe/Imである。参照を次のように分離する。

- $z_{\rm exact}$：$e^{-i(H_D+H_R)T}$。
- $z_{\rm tail\text{-}exact}$：同じpartial-$S_2$ backboneで中央tailだけ$e^{-iH_RT}$。
- $z_N$：中央tailを$N$ qDRIFT drawの平均演算子へ置換。

### 4.2 levelとestimator

- base $N=4$、levels $4,8,16$だけを使う。
- qDRIFT確率は$|h_j|/\lambda_R$、各draw angleは$\lambda_RT/N$、符号をPauliへ吸収する。
- exact meanは独立drawの平均演算子$M_N^N$から計算し、Monte Carlo path平均で代用しない。
- 最小外挿は

  $$
  z_{\rm ext}=2z_8-z_4
  $$

  とし、$z_{16}$は通常PRのfine対照に使う。位相は外挿しない。
- full-random対照は全5項をqDRIFT化し、同じlevelsとRe/Im estimatorを使う。
- deterministic endpointは5項のsecond-order Pauli PF 1 stepとする。

### 4.3 shotとwork

- Hadamard各軸のsingle-shot varianceは$1-\mu^2$。
- target complex-signal RMSEは$\epsilon_z=0.05$。systematic complex bias $b$が
  $|b|\geq\epsilon_z$ならそのmethodはineligible。
- 残りvariance budgetをRe/Imへ等分する。線形weights $w_i$の各軸shotは、整数丸め前に
  $n_i\propto|w_i|\sqrt{v_i/c_i}$となるcost-minimizing allocationを用い、各level最低1 shotへ切り上げる。
- 1-shot primitive-rotation workはpartial levelで$2L_D+N=4+N$、full-randomで$N$、
  deterministic endpointで$2L-1=9$。state preparationとancilla H/S/measurementは全method共通として
  主層から除外する。
- 通常PR対照はlevels 4/8/16のうちeligibleなtotal work最小点。

### 4.4 correctness gate

1. Pauli matricesがHermitian involutionで、state norm誤差とunitary defectが$10^{-12}$以下。
2. qDRIFT確率和誤差が$10^{-14}$以下。
3. $N$増加で$|z_N-z_{\rm tail\text{-}exact}|$が少なくとも4→8または8→16の一方で減少する。
4. 全mean、variance、shot、workが有限・非負。

失敗時は`IMPLEMENTATION_INVALID`または`MODEL_NOT_IN_ASYMPTOTIC_WINDOW`とし、外挿の勝敗に使わない。

### 4.5 固定判定

$b_8=|z_8-z_{\rm tail\text{-}exact}|$、
$b_{\rm ext}=|z_{\rm ext}-z_{\rm tail\text{-}exact}|$、
$R_G=G_{\rm ext}/G_{\rm ordinary,best}$とする。

- `GO_PR3`：$b_{\rm ext}\leq0.75b_8$かつ$R_G<1$。
- `CONDITIONAL_PR3`：bias条件を満たし、$1\leq R_G<2$。
- `STOP_PR3_VARIANCE_BACKBONE_DOMINATES`：bias条件を満たさない、eligible通常対照がない、または$R_G\geq2$。

outer-PF biasは勝敗に含めず別記する。fixed-time signalからfull RPEまたはenergy総costを主張しない。

## 5. pilot後の強制停止と主題選択

両pilot artifactを書いた直後に`STOP_AFTER_PR2_PR3_MINIMAL_PILOTS`を発動し、追加点を計算しない。
次の5項目を各0--2点で記録する。

1. 先行研究との差を限定して説明できる。
2. partial randomizationが結果のtrade-offに本質的である。
3. 中心結果を一文で表せる。
4. negative resultでも再利用可能な設計知見が残る。
5. 独立条件を含む完成までの追加作業が現実的である。

選択規則は次の順で適用する。

1. 一方だけGO/CONDITIONALならその案を主題候補にする。
2. 両方がGO/CONDITIONALなら合計点が2点以上高い案を選ぶ。
3. 差が2点未満なら主題を確定せず、より大きい不確かさを一件だけ文書監査する。追加数値は禁止。
4. 両方STOPならPR-4の物理difference estimator cost監査へ戻る。PR-5/6を自動実行しない。

どの判定でも、本pilot結果から別分子、未使用geometry、H12、precision/split sweep、長RPE、
compiled total costへ自動的に進まない。
