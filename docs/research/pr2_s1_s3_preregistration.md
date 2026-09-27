# PR-2 S1--S3 結果前事前登録・実装監査

日付: 2026-09-27  
状態: `PR2_S1_S3_PREREG_V1_FROZEN_FOR_EXTERNAL_REVIEW_EXECUTION_NOT_AUTHORIZED`  
親契約: [PR-2主研究契約](pr2_primary_research_contract.md)  
追加数値: 0（分子計算、signal評価、compile、trajectory sampling、量子shotを実行していない）

## 1. 目的と停止位置

本書は、PR-2のS1 controlled one-step、S2 fixed-time development、S3 independent transferを
**結果を見る前に**固定する。現時点では実装監査とdry-runだけを行い、S1を開始しない。

次の順序を変更しない。

1. 本書とdry-run manifestを外部の批判的レビューへ渡す。
2. 指摘を採用する場合は、実行前amendmentとして旧hashと変更理由を保存する。
3. S0の入力・環境・identity・実装test gateを通す。
4. S1だけを実行して停止する。
5. S1が全correctness gateを通過した場合だけS2を実行して停止する。
6. S2の固定判定後に明示的に継続を決めた場合だけS3を開封・実行する。

PR-3、PR-4--6、別分子、別basis、追加geometry、H12、長RPE、noise/backend、precision sweep、
最終total costへ自動的に進まない。

## 2. 実装監査の結論

| 能力 | 既存実装 | 判定 | S1前に必要なこと |
|---|---|---|---|
| exact DF tail・identity phase・sampling確率 | `df_rte_tail.py`、`rte.py` | 再利用可 | rankごとの再構成・確率和・normalization test |
| weight-ranked partial-$S_2$ one step | `df_partial_s2.py` | 再利用可 | B2-WとB3に使用 |
| repeated controlled partial-$S_2$ | `df_partial_s2_repeated.py` | 再利用可 | $q=1,8$のoperator probe |
| Re/Im Hadamard wrapper | `rpe_hadamard_interrogation.py` | $q\leq4$本経路は再利用可 | Re/Im符号とmeasurement mapping test |
| $q=8$ wrapper | `rpe_hadamard_compiled_cost_benchmark.py` | 検証用builderを再利用可 | PR-2 runnerからの接続test。通常wrapperのdomainを拡張したとは書かない |
| full-wrapper compiled cost | `research_direction_full_scope*.py`、`df_partial_s2_repeated_cost.py` | 構造を再利用可 | 新snapshot/rank/$\delta$で直接再compile。旧proxy係数は移送しない |
| finite-RTE平均signal・attenuation | `finite_rte_signal_validation.py`、`rte.py` | 数式部を再利用可 | PR-2固有のstate、partition、$q=8$ recordへ接続 |
| shot accounting | `rpe_resource_accounting.py` | そのままは不適合 | unit-radius eigenphase仮定を使わず、本書第7節の固定complex-signal式を実装 |
| pilot generation-prefix | `pr2_pr3_minimal_pilot.py` | 入力生成は確認済み | B2-G用partition adapterとB2-W identity gate |

既存full-scope artifactはH4 1.0 Åの別snapshot、主に$L_D=3$、別の$\delta$条件である。回路構築方針と
compiler contextは再利用できるが、cost値、affine係数、shot数をPR-2へ転記しない。

## 3. S0: 入力と実装のfreeze gate

### 3.1 共通分子入力

- model: linear H4、4 electrons、8 spin orbitals/qubits、singlet、charge 0
- geometry: `H` at $(0,0,d(i-1.5))$, $i=0,1,2,3$
- basis: STO-3G
- generation call: `build_df_h_d_from_molecule(4, distance=d, basis="sto-3g", df_rank=12)`
- development: $d=1.00$ Å
- independent transfer: $d=1.30$ Å
- target: 生成したrank-12 DF Hamiltonian。exact molecular Hamiltonianではない
- state: 同じrank-12 targetの4-electron singlet sector ground state。global phaseを固定し、state hashを保存

pilot artifactが記録するdevelopment Hamiltonian hashは
`d8b4aaf21afcc3935d5b5aa4d0805b358c5ec670d8104d25807c7cd0620a3dc3`である。一方、既存の
`13e4b10d...a3b` snapshotは別provenanceであり、pilot入力の代用品にしない。pilot実行は入力配列を
snapshot保存していないため、S0で同じ生成recipeを再実行して次を適用する。

- hashがpilot hashと一致: development入力としてfreezeする。
- hashが不一致: S1を開始せず`STOP_INPUT_REPRODUCTION_MISMATCH`。旧pilot結果と新入力を混ぜない。

1.30 Å入力もS1結果を見る前に生成し、Hamiltonian、sector、state、fragment order、package versionのhashを
freezeする。ただしS3までsignal、cost、候補順位を計算・開封しない。

### 3.2 prefix identity gate

rank $r\in\{3,6,9\}$ごとに次を保存する。

- B2-G: generation indices $(0,\ldots,r-1)$
- B2-W: `rank_df_fragments(..., weight_rule="lambda_frobenius_squared")`の先頭$r$ indices
- ordered indices、unordered set、各fragment hash、weight、tail hash

全rankでordered indicesまで一致すればB2-G/B2-WをB2へ統合する。不一致なら両方を残し、B2-G用の
明示partition adapterとtestが完成するまでS1を開始しない。集合が一致して順序だけ違う場合も、回路costへ
影響し得るので自動統合しない。

### 3.3 environment gate

pilot環境を固定基準とする。

- Python 3.11.0rc1、NumPy 1.26.4、SciPy 1.14.1、Qiskit 1.3.0
- OpenFermion、OpenFermion-PySCF、PySCFはS0で実versionとsource hashを記録
- compiler: basis `rz,sx,x,cx`、optimization level 1、seed 17、coupling map/backend/layout/routingなし

versionを合わせられない場合は黙って再生成せず`STOP_ENVIRONMENT_MISMATCH_NO_EXECUTION`とする。

### 3.4 correctness gate

S1開始前に自動testで次を全て通す。

1. Hamiltonian hash、sector basis、state hash、fragment orderがfreeze recordと一致する。
2. constant、one-body、全fragmentを含むexact residual再構成のrelative spectral errorが$10^{-10}$以下。
3. identityをsample distributionから除きphaseへ移す規約が全候補で同じ。
4. sampling確率和の誤差が$10^{-12}$以下、全確率・normalization・attenuation・costが有限かつ非負。
5. controlled circuitが$\mathrm{diag}(I,U)$とglobal phaseを除き$10^{-10}$以内で一致する。
6. ancilla $X/Y$ expectationがそれぞれsignalのRe/Imと$10^{-10}$以内で一致する。
7. $q=1,8$ repeated circuitが明示的なstep積と$10^{-10}$以内で一致する。
8. B0--B3でtarget、state、physical time、wrapper、compiler、cost scopeが一致する。

一つでも失敗したら`STOP_CORRECTNESS_OR_ESTIMAND_FAILURE`とし、threshold、rank、state、taskを変更して
同じS1を救済しない。

## 4. 固定baselineとrank

| ID | 内容 | rank運用 |
|---|---|---|
| B0 | residualを捨てたgeneration-prefix deterministic $S_2$ | rank 6主、3/9 control |
| B1 | rank-12 deterministic $S_2$ | endpoint |
| B2-G | generation-prefix deterministic backbone＋exact residual finite RTE | rank 6主、3/9 control |
| B2-W | weight-ranked通常prefix PR | rank 6主、3/9 control。B2-Gとidentityなら統合 |
| B3 | full-random endpoint、$L_D=0$ | strong endpoint baseline |

B4 non-prefix compressed residualは今回実行しない。rank 6をdevelopment anchorから変更しない。rank 3/9は
rank 6で選ばれたfinite-RTE設定をそのまま適用するcontrolであり、個別にretuneしない。

## 5. 共通estimandとexact reference

各stageの物理時間$T$に対し

$$
z_{12}(T)=\langle\psi_{12}|e^{-iH_{12}T}|\psi_{12}\rangle
$$

をtargetとする。$\psi_{12}$は同じ$H_{12}$のsector ground stateである。candidate $j$の有限RTEを含む
平均signalを$\mu_j(T)$とし、有限分布のnormalizationで事後補正せず、実際のsampled algorithmの平均を
そのまま比較する。従ってattenuationとtruncationは$\mu_j-z_{12}$のbiasへ含める。

次を別fieldで保存する。

- exact-target対candidate meanのRe/Im bias
- deterministic PF bias、discard bias、finite-RTE truncation bias、attenuation
- finite distribution normalization、$\lambda_R$、component count、sample support
- state preparationを除くfull Hadamard wrapper cost

## 6. finite-RTE候補と選択規則

B2-G、B2-W、B3の候補集合は結果前に

$$
r\in\{1,2,4,8,16,32\},\qquad K\in\{2,4\}
$$

と固定する。$K$は非負偶数のfinite paired-Taylor cutoffである。coefficient thresholdは0、identity policyは
`extract_identity_phase`、一つのpartial-$S_2$ stepにつきrandom tail occurrenceは1回とする。

各methodで第7節のaccuracyを満たす候補のうちshot込みexpected compiled RZが最小のものを選ぶ。同値なら
順に小さい$r$、小さい$K$を選ぶ。rank 6のB2-G/B2-WとB3は別々に選ぶ。rank 3/9 controlにはrank 6の
B2-G設定を固定移送する。S3にはS2で選ばれた設定をそのまま移し、1.30 Åで再選択しない。

## 7. 共通signal精度とshot accounting

PR-3最小pilotと同じcomplex-signal尺度を保ち、全method・stageで

$$
\epsilon_{\mathbb C}=0.05,\qquad
\epsilon_{\rm axis}=\epsilon_{\mathbb C}/\sqrt2,
\qquad \alpha_{\rm total}=0.05,
\qquad \alpha_{\rm axis}=0.025
$$

を固定する。precision sweepは行わない。axis $a\in\{\operatorname{Re},\operatorname{Im}\}$について
$b_{j,a}=|\mu_{j,a}-z_{12,a}|$とする。

- $b_{j,a}\ge\epsilon_{\rm axis}$ならそのcandidateはineligible。
- それ以外は$\epsilon^{\rm stat}_{j,a}=\epsilon_{\rm axis}-b_{j,a}$。
- bounded $\pm1$ Hadamard outcomeに対し

  $$
  N_{j,a}=\left\lceil\frac{2}{(\epsilon^{\rm stat}_{j,a})^2}
  \log\frac{2}{\alpha_{\rm axis}}\right\rceil.
  $$

union boundと三角不等式により、両axisが成功すればcomplex $\ell_2$ errorは0.05以下となる。B2/B3は
各Hadamard shotでfresh IID trajectoryを引く。trajectoryをshot間で固定再利用する結果は採用しない。
shotは解析的に数えるだけで実行しない。既存RPEのunit-radius/phase-budget shot式は用いない。

## 8. compiled-cost規約

primary metricは`compiled_rz_count`、secondaryはCX、size、depthとする。各axisについて、状態準備を除き
ancilla準備、ordinary controlled evolution、basis rotation、measurementを含むfull wrapperを数える。

random methodの1-shot期待costは32 trajectoryのMonte Carloで推定する。master seedはS1
`20260927101`、S2 `20260927102`、S3 `20260927130`とし、cell seedはstage、method、rank、$r$、$K$、
axisのcanonical JSON SHA-256先頭64 bitから決定する。Re/Imは同じtrajectory列を共有する。

primary RZ meanのrelative standard errorが2%を超えたcellだけ、事前に固定した別seed列で合計128 trajectoryへ
一度だけ拡張する。128でも2%を超えれば区間を保持し、追加sampleを行わない。旧full-scope affine proxyは
入力hashとcellが異なるためprimary値に使わない。

total workは

$$
G_j=\sum_{a\in\{\mathrm{Re},\mathrm{Im}\}}N_{j,a}
\,\mathbb E[C_{j,a}^{\rm full\ wrapper,no\ prep}]
$$

とする。状態準備、backend実行、noise、error mitigation、full RPE reconstructionは含めない。

## 9. stage別実行範囲

### S1: controlled one-step

- development 1.00 Åだけ、$T=\delta=0.1$、$q=1$
- B0、B1、B2-G、B2-W、B3のoperator/wrapper correctness
- rank 6で全$(r,K)$、rank 3/9はS1 correctness control
- exact/mean signal、normalization、attenuation、full-wrapper compiled costを記録

S1はcorrectness接続であり、positive resource conclusionを出さない。S1後に必ず停止する。

### S2: fixed-time development

- development 1.00 Åだけ、$T=0.8$、$\delta=0.1$、$q=8$
- rank 6の全固定候補とB0/B1/B3を比較
- 第6節の規則でmethod別candidateを選び、rank 3/9へB2-G設定を固定移送
- signal bias、shots、1-shot cost、total work、component breakdownを返す

S2後に親契約の10% materiality ruleでdevelopment判定を行い、必ず停止する。

### S3: independent transfer

- blind-frozen H4 1.30 Å、$T=0.8$、$\delta=0.1$、$q=8$
- S2のrank policy、候補、accuracy、compiler、seed rule、cost metric、判定を無変更で移送
- 1.30 Å向けのrank、$r$、$K$、threshold、shot予算、compiler policyの再調整を禁止

S3後は親契約のterminal decisionを一つ選び、正の結果でも追加系へ進まない。

## 10. decisionと報告規則

- S0/S1不成立: `STOP_CORRECTNESS_OR_ESTIMAND_FAILURE`
- developmentでB2-G rank 6がaccuracyを満たさない、またはB0/B1/B2-W/B3に10%以上支配される:
  `COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN`。S3で救済しない
- S2では残るがS3で移送しない、または区間が10%差を跨ぐ: `COMPLETE_CONDITIONAL_RESOURCE_MAP`
- S2/S3双方でaccuracyを満たし、B1比10%以上、B0にaccuracy/costで非劣、B2-W/B3を含むfrontier上:
  `COMPLETE_POSITIVE_RESOURCE_CROSSOVER`

B2-GとB2-Wが異なる場合、B2-Wを上回らない結果を「新手法の失敗」とは呼ばない。generation-order policyを
含むresource mapとして報告する。positive/negativeを問わず、deterministic saving、$\lambda_R$、sample cost、
finite bias、attenuation、shot inflation、wrapper overheadへ分解する。

## 11. 現在のblockerと外部レビュー項目

dry-run時点で次が未完了なので、**実行許可はfalse**である。

1. pilot hashと一致する1.00 Å snapshotが保存されていない。
2. 1.30 Å snapshot/hash/stateが未生成である。
3. B2-G/B2-W identity gateがpilot入力上で未実施である。
4. B2-GがB2-Wと異なる場合のexplicit-partition adapter/testがない。
5. PR-2固有のcomplex-signal shot式と$q=8$ runner/testが未実装である。
6. 本書に対する外部の批判的レビューが未実施である。

外部レビューでは、少なくとも「新規性の過大評価」「strong baseline不足」「blind transferの独立性」
「signal accuracyとshot式」「negative resultの完成性」「S1からS2/S3へ自動進行しないこと」を確認する。

## 12. 凍結時の結論

既存実装は、exact tail、controlled partial-$S_2$、Re/Im wrapper、$q=8$検証builder、compiled-cost skeletonを
再利用できる。一方、入力snapshot、prefix identity、PR-2固有shot式、runner/testは未完成である。
従って本書は**外部レビュー用v1**として凍結し、数値実行は認めない。
