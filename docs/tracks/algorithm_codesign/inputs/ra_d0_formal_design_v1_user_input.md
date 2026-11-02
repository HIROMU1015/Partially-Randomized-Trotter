# RA-D0 formal design v1

## 1. 目的と研究上の位置づけ

RA-D0の主RQを次に固定する。

> **同じ有限Taylor平均、同じ保存済みnative implementation table、同じfinite-confidence条件の下で、Taylor degreeごとの係数配分を自由化するRA-RTEは、precision最適化および既存ensemble全体の混合だけでは作れないresource pointを生成するか。**

RA-D0は新しいscience runではない。R1で既に取得したsynthesis sequence・event cost・strict errorのみを再利用するdevelopment optimizationである。

RA-D0では次を行わない。

- 新しいangleの生成
- pygridsynthの再実行
- 新しいsynthesis precision
- 新しいx、degree、Hamiltonian、geometry
- IS/PAI最適化
- CTSの新規取得
- DF・分子・GPU・trajectory
- R1の科学分類変更

入力の正本は以下とする。

- R1 result commit  
  `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b`
- R1.5 attribution  
  `af3d014d0a0cfcbbd25bb544f6544652fec92942`
- RA-RTE mathematical audit  
  `8a04c148a66d23dbc1f045086a95a5e19a6372dc`

数学modelは監査で確認された修正版single-block modelを採用する。

---

# 2. 固定task

RA-D0のprimary taskはR1と同じものに限定する。

\[
M=P_3(-i\sigma x\widehat R),
\qquad
\widehat R=\frac34Q_0+\frac14Q_1.
\]

primary implementation context：

\[
Q_0=Z_0,
\qquad
Q_1=V^\dagger Z_1V,
\qquad
V=e^{-i\pi X_0X_1/16}.
\]

使用する時間は

\[
x\in\{1/8,1/4\}.
\]

primary optimizationでは \(\sigma=+1\) を用いる。

R1.5でresource座標は二符号で一致しているが、RA-D0 table作成時に \(\sigma=-1\) の保存値との一致を再確認する。

cost/error tableが符号間で一致しなければ、符号を統合せず

`INCONCLUSIVE_SIGN_TABLE_MISMATCH`

としてSTOPする。

\(\sigma=-1\) を独立replicationとして数えない。

---

# 3. RA-D0 candidate table

## 3.1 ideal column

degree \(k\) の候補column \(j\) を

\[
D_{\cdot j}
\]

で表す。

非terminal columnでは、

\[
D_{k,j}=\cos\phi_j,
\qquad
D_{k+1,j}=\sin\phi_j.
\]

terminal pure-word columnでは対応する一つのdegree成分だけを1とする。

target coefficient vectorは

\[
t=
\left(
1,\,
x,\,
\frac{x^2}{2},\,
\frac{x^3}{6}
\right)^T.
\]

finite mean保存条件は

\[
Dw=t,\qquad w\ge0.
\]

---

## 3.2 R1から利用するideal prototype

各 \(x\) について、次の7 prototypeだけを使用する。

| ID | degree | ideal direction | 起源 |
|---|---:|---|---|
| O0 | 0→1 | \((1,x)/\sqrt{1+x^2}\) | ordinary / PTSC-K0 |
| O2 | 2→3 | \((1,x/3)/\sqrt{1+x^2/9}\) | ordinary |
| P2 | 2 | pure \(F_2\) | PTSC-K0 |
| P3 | 3 | pure \(F_3\) | PTSC-K0 |
| A0 | 0→1 | \((1,\rho)/\sqrt{1+\rho^2}\) | A |
| A1 | 1→2 | \((a_1,b_1)/\sqrt{a_1^2+b_1^2}\) | A odd/complement |
| A2 | 2→3 | \((1,\rho)/\sqrt{1+\rho^2}\) | A |

ここで

\[
\rho=
\frac{x+x^3/6}{1+x^2/2},
\]

\[
a_1=\frac{2x^3}{3(x^2+2)},
\qquad
b_1=\frac{2x^2}{x^2+6},
\]

なので

\[
\frac{b_1}{a_1}=\frac1\rho.
\]

A1のideal angleは大角度側であるが、実装cost/errorにはR1で使ったexact complement rewriteの保存値を用いる。

---

## 3.3 implementation variants

各prototypeについて、

\[
\epsilon_{\rm synth}
\in
\{10^{-3},10^{-4},10^{-6}\}
\]

の3 implementation variantsを用いる。

従って、重複を考慮しない最大column数は

\[
7\times3=21.
\]

新しいsynthesisは行わない。

同一ideal columnかつ同一conditional event lawについて、R1保存eventの

- native IR identity
- T/CX/1Q cost
- strict error
- phase semantics

が完全一致する場合のみduplicate implementationを統合する。

単にangleが同じという理由だけで統合しない。

---

# 4. column cost/errorの取得

R1 event recordsだけから各columnの条件付きresource tableを構成する。

column \(j\) 内のeventを \(\omega\)、conditional IID probabilityを

\[
\pi_{\omega|j}
\]

とする。

各resourceについて

\[
C_{j,Q}
=
\sum_\omega
\pi_{\omega|j}C_{\omega,Q},
\qquad
Q\in\{T,CX,1Q\}.
\]

controlled coherent task用bias coefficientは

\[
d_j
=
2\sum_\omega
\pi_{\omega|j}\delta_\omega,
\]

とする。

\(\delta_\omega\) はR1で保存されたstrict phase-preserving joint operator error upper。

R1の古いgroup coefficient rounding biasはRA-D0へ転記しない。

RA-D0では新しいweight \(w_j\) を使うため、

- implementation error → \(d_j\)
- LP / numerical mean residual → \(\xi\)

として分離する。

workspaceは期待値ではなく、

\[
W_j=\max_{\omega|j}W_\omega
\]

を用いる。

RA-D0のregistered contextでは

\[
W_{\max}=1
\]

を上限とする。

これを超えるcolumnがあれば、全baselineを含めcandidate tableから除外する。

---

# 5. baseline hierarchy

比較classは入れ子にする。

\[
B0\subseteq B1\subseteq B2\subseteq B3.
\]

このnesting自体を実装前semantic testにする。

## B0 — registered R1 profiles

R1で実行済みの

\[
\{\text{ordinary},\text{PTSC-K0},A\}
\times
\{10^{-3},10^{-4},10^{-6}\}
\]

そのもの。

各 \(x\) について9 profiles。

これはR1/R1.5の再現baseline。

---

## B1 — fixed representation + column-wise precision

representationのideal decompositionは固定する。

例えばordinaryならO0/O2のideal weightsは変更しない。

ただし各logical columnについて、3 precision variantsの間へweightを分配してよい。

representation \(r\) のgroup \(g\) の固定ideal coefficientを

\[
w^{(r)}_g
\]

とし、implementation variant \(p\) について

\[
w_{g,p}\ge0,
\qquad
\sum_p w_{g,p}=w^{(r)}_g.
\]

とする。

したがってB1は、

> representationを変えず、columnごとのprecisionだけをresource-awareに選ぶ

強いbaselineである。

単一precision選択はB1の特殊ケース。

---

## B2 — whole-ensemble mixture

ordinary、PTSC-K0、Aという完成済みrepresentationを丸ごと確率混合する。

representation shareを

\[
\theta_r\ge0,\qquad
\sum_r\theta_r=1
\]

とする。

各representation内部ではB1と同じcolumn-wise precision mixtureを許す。

従ってB2は、

> 「ordinaryとAを適切に混ぜただけで同じ改善が得られる」

という説明を最大限許したbaselineとなる。

正規化変数では

\[
z_r=y\theta_r
\]

を用い、

\[
\sum_p q_{r,g,p}=z_r w_g^{(r)},
\qquad
\sum_r z_r=y
\]

と書ける。

全制約は線形。

---

## B3 — degree-local RA-RTE

representation membershipを外す。

candidate tableにある全columnについて、

\[
Dq=yt,
\]

\[
\mathbf 1^Tq=1,
\]

\[
q\ge0,\qquad y>0
\]

だけをmean側の基本制約とする。

したがってdegree 0,1,2,3への寄与を、ordinary/A/PTSCの固定ratioから独立に再配分できる。

これがRA-D0で検証する追加自由度である。

---

# 6. finite-confidence contract

R1との比較可能性を保つため、

\[
\epsilon_{\rm axis}=\frac1{200},
\]

\[
\alpha_{\rm axis}=\frac1{5280}
\]

を変更しない。

新しいfamilywise allocationは作らない。

\[
\ell=
\log\frac{2}{\alpha_{\rm axis}}.
\]

実装では

\[
\ell_{\rm up}
\ge
\ell
\]

となる100 dps outward upperを使う。

固定axis shot数 \(n\) に対し、

\[
\kappa_n=
\frac{
\frac43\ell_{\rm up}
+
\sqrt{
(\frac43\ell_{\rm up})^2+
8n\ell_{\rm up}
}
}
{2n}
\]

のoutward upper

\[
\kappa_{n,\rm up}
\]

を使用する。

---

## numerical mean residual

solver / interval / probability quantizationによる残差を

\[
\xi
\]

として、

\[
e y-d^Tq-\xi
\ge
\kappa_{n,\rm up},
\qquad
e=\frac1{200}.
\]

mean matchingのinterval residualについて、

\[
\xi\ge
\|Dq-yt\|_1
\]

のcertified upperを保存する。

numerical residual budgetは

\[
\boxed{\delta_{\rm num}=10^{-12}}
\]

と固定し、

\[
\xi\le y\delta_{\rm num}
\]

を要求する。

これはmateriality thresholdではなくnumerical certification threshold。

R1で観測されたsynthesis biasより十分小さいnumerical bookkeeping条件としてのみ使用する。

---

# 7. certified sampler law

RA-D0は実際のshot samplingを行わないが、出力するdesign pointには実装可能なsampling lawを付与する。

nominal solver outputから、各 \(q_j\) を共通denominator

\[
2^{60}
\]

へlargest-remainder方式でquantizeする。

- \(q_j\ge0\)
- \(\sum_jq_j=1\)

をexact integer arithmeticで満たす。

\(y\) も \(2^{60}\) denominatorのrationalへ変換する。

その後、mean residual・confidence制約を再certificateする。

quantization後に条件を満たさなければ、

`UNCERTIFIED_NUMERICAL_POINT`

として棄却し、denominatorを結果後に増やさない。

内部involution sampling

\[
p=(3/4,1/4)
\]

はexact rational lawとして保持する。

---

# 8. shot grid

RA-D0ではhard total-resource capを使わない。

したがって数学監査で確認されたfactor-\(r\) grid保証を使用できる。

## 8.1 primary anchors

各 \(x\) について、R1のdistinct-basis controlled B0 profilesに保存された

\[
n_{\rm axis}
\]

の全unique値を

\[
\mathcal N_{\rm anchor}(x)
\]

とする。

これは結果前に既に登録・取得済みのshot countsであり、RA-D0のprimary evaluation pointsとする。

---

## 8.2 theoretical lower boundary

\[
T_\Sigma=
1+x+\frac{x^2}{2}+\frac{x^3}{6}.
\]

監査結果より、

\[
y\le
\frac{\sqrt2}{T_\Sigma}.
\]

\(d_j\ge0\) なので、

\[
h\le
e\frac{\sqrt2}{T_\Sigma}
=:h_{\max}.
\]

この値をBernstein式へ代入し、任意のRA-D0 feasible pointが必要とするshotsの理論的lower bound

\[
n_{\min}(x)
\]

をoutwardに計算する。

---

## 8.3 Pareto-relevant upper boundary

候補tableに対して、

\[
c_{T,\min}=\min_jC_{j,T},
\]

\[
c_{CX,\min}=\min_jC_{j,CX},
\]

\[
c_{1Q,\min}
=
\min_jC_{j,1Q}
+\frac52
\]

を求める。

R1 B0 profile \(b\) の保存総資源を

\[
G_T^{(b)},G_{CX}^{(b)},G_{1Q}^{(b)}
\]

とする。

そのprofileが、全candidateの任意混合を必ずdominateするshot数の下限は

\[
N_b=
\max_Q
\frac{G_Q^{(b)}}{2c_{Q,\min}}.
\]

そこで

\[
n_{\max}(x)
=
\left\lceil
\min_{b\in B0}N_b
\right\rceil.
\]

この値より十分大きい \(n\) では、少なくとも一つの保存B0 pointがcandidate tableから作れる任意のshot-proportional resource vectorをdominateする。

したがってRA-D0のPareto探索範囲を

\[
[n_{\min},n_{\max}]
\]

へ限定する。

もしいずれかの \(c_{Q,\min}=0\) なら、そのresourceをこのupper-bound証明から除外する。

全resourceでboundを構成できなければ、別途固定upper limitを発明せずGPT reviewへ戻す。

---

## 8.4 coverage grid

\[
r=1.005
\]

のgeometric gridを使う。

\[
n_{k+1}=
\left\lceil
1.005\,n_k
\right\rceil.
\]

\[
n_{\min}
\]

から

\[
n_{\max}
\]

まで生成し、

\[
\mathcal N_{\rm anchor}
\]

を全て追加してdeduplicateする。

primary evidenceはanchor points。

anchor外の点は

`COVERAGE_GRID`

として別tagにする。

gridによるfactor-\(r\) resource approximationは記録するが、hard-cap feasibility保存とは呼ばない。

---

# 9. resource coordinates

固定 \(n\) では、

\[
G_T=2n\,C_T^Tq,
\]

\[
G_{CX}=2n\,C_{CX}^Tq,
\]

\[
G_{1Q}
=
2n
\left(
C_{1Q}^Tq+\frac52
\right).
\]

primary resource spaceは

\[
\boxed{
(G_T,G_{CX},G_{1Q})
}
\]

とする。

secondaryとして、

\[
B=\frac1y,\qquad
B^2,\qquad
b_{\rm impl}=\frac{d^Tq}{y},
\]

\[
\xi/y,
\]

support size、peak workspace、solver/acquisition resourceを保存する。

T/CX/1Qにhardware換算係数を掛けた単一winner scoreは作らない。

---

# 10. Pareto / degree-local value query

RA-D0では「全3次元frontを完全に列挙した」とは主張しない。

代わりに、B2が作るregistered-budget regimeに対してB3がstrict improvementを持つかを、有限のepsilon-constraint queryで判定する。

## 10.1 budget set

各固定 \(n\) について、confidence-feasibleなB0 profilesを同じ \(n\) で評価する。

resource \(Q\) ごとに

\[
\mathcal K_Q(n)
\]

を、

- feasible B0 profilesの \(G_Q\)
- B2でのsingle-resource minimum

の集合として作る。

B3 resultを見てbudgetを追加しない。

---

## 10.2 epsilon-constraint queries

目的resourceを \(Q\)、残り二resourceを \(R,S\) とする。

全

\[
c_R\in\mathcal K_R(n),
\qquad
c_S\in\mathcal K_S(n)
\]

について、

\[
\min G_Q
\]

subject to

\[
G_R\le c_R,
\qquad
G_S\le c_S
\]

をB2とB3の両方で解く。

B2がinfeasibleなbudget pairはprimary comparisonから除く。

B3だけfeasibleの場合もdegree-local witnessとして保存する。

---

## 10.3 certified strict witness

同じ \(x,n,Q,c_R,c_S\) について、

\[
U^{B3}_{Q}
<
L^{B2}_{Q}
\]

がcertified intervalで成立した場合、

`STRICT_DEGREE_LOCAL_WITNESS`

とする。

ここで \(U,L\) はnumerical/sampler certificateを戻したobjective upper/lower。

point float差だけでwitnessにしない。

materiality thresholdは追加しない。

relative improvementも併記するが研究GOには使わない。

---

# 11. output classification

RA-D0は次の分類だけを返す。

### `D0_STRONG_DEGREE_LOCAL_SIGNAL`

両方の

\[
x=1/8,\;1/4
\]

について、少なくとも一つの**anchor shot point**で

`STRICT_DEGREE_LOCAL_WITNESS`

が存在する。

これは最も強いdevelopment result。

---

### `D0_LOCAL_DEGREE_LOCAL_SIGNAL`

strict witnessは存在するが、

- 一方のxだけ
- またはcoverage-gridのみ

である。

method delta候補は残るがtransferは弱い。

---

### `D0_NO_REGISTERED_WITNESS`

anchor / coverage gridの登録query全てでB3のstrict witnessなし。

これは連続angle・未登録precision・別task一般のno-goではない。

ただし、**R1 tableの自由度だけではRA-RTEを主methodへ進める根拠が得られなかった**というdevelopment STOP理由にする。

---

### `D0_TECHNICAL_INCONCLUSIVE`

- sign table mismatch
- solver/certificate不整合
- numerical quantization failure
- candidate table identity failure
- upper-bound construction不能
- source/provenance mismatch

等。

resultを見た再試行は行わない。

---

# 12. strong baselineの意味

RA-D0でB3がB2を改善した場合に初めて、

> degree-local coefficient allocationに、precision-only / whole-ensemble mixtureを超える追加価値がある

と言える。

それでもまだ、

- optimal importance sampling
- collected CTS
- 新しいnative implementation
- DF
- multi-block

への優位は言えない。

RA-D0がpositiveなら、次は同じ保存table上で**fixed-ensemble importance sampling baseline**を追加するRA-D1へ進む候補とする。

RA-D0でISを同時最適化しない。

---

# 13. 最新先行研究との境界

Resource-Optimal Importance Samplingは、固定されたrandomized quantum protocolについて、sampling probabilityを実行costとvarianceの両方から最適化する一般frameworkを与えている。したがってRA-D0の新規性を「cost-aware sampling」に置かない。RA-D0で変更しているのは、sampling distributionの前段にあるfinite-mean representationのdegree-local coefficient allocationである。

Structure-Aware Variance Reductionも、平均channelを変えずにrandomized Hamiltonian simulationのvarianceを削減する方向を扱っている。したがって「mean-preserving variance reduction」自体も新規性ではない。

2026年9月のMorisaki–Fujiiは、randomized Hamiltonian simulationでHamiltonian termのsampling probabilitiesをaverage-error criterionから最適化している。これもsampling-probability optimizationが独立した活発な方向であることを示す。

従ってRA-RTEで残すべきclaim候補は、

> **同じfinite Taylor meanを実現するunitary representationそのもののdegree-local freedomを、phase-preserving finite-precision implementationとfinite-confidence resource constraintの下で設計する**

という狭いものに限定する。

---

# 14. RA-D0後の分岐

`D0_STRONG_DEGREE_LOCAL_SIGNAL`：

RA-RTEは主method candidateとして残す。
ただし新synthesisへ行く前に、同じsaved tableでfixed-ensemble ISを強いbaselineとして追加する。

`D0_LOCAL_DEGREE_LOCAL_SIGNAL`：

IS baselineまで確認する価値はあるが、別x/degreeへの拡張はまだ行わない。

`D0_NO_REGISTERED_WITNESS`：

新angle探索やDFで救済しない。
RA-RTEのalgorithm routeは縮小し、

- restricted finite-mean family
- normalization theorem
- finite-confidence LP formulation

をtechnical/theory resultとして評価する。

全outcomeでmandatory STOPし、次の研究判断をGPTへ戻す。

---

# 15. RA-D0が主張しないもの

RA-D0は以下を主張しない。

- continuous angle spaceでのglobal optimum
- 任意LCUでのglobal optimum
- optimal samplingまで含むglobal optimum
- whole-circuit compile optimum
- hardware advantage
- DF advantage
- chemistry advantage
- multi-block optimum
- RPE total-cost optimum
- publication priority
- 世界初

RA-D0の結論は、**固定R1 implementation table内のdegree-local freedomに追加resource valueがあるか**だけに限定する。

---

# 16. execution boundary

次段階はCodexによる

1. candidate-table extractionの静的監査
2. B0⊂B1⊂B2⊂B3のsemantic verification
3. LP/certificate実装
4. result-prior query/grid artifact生成
5. source review

まで。

この時点ではRA-D0 optimizationをまだ実行しない。

source review後にだけsaved-table development one-shotを別authorizationで実施する。

新synthesis、science runner、DF、分子、GPU、trajectoryへのauthorizationは与えない。