# finite-RTE位相・信号半径分離契約（FR-0）

最終更新：2026-09-26 JST
状態：`FR1_COMPLETE_GO_FR2_MECHANISM_ONLY_FR2_NOT_STARTED`
基点：`all-r-coherent-opt2-reoptimization`、`4b28bc64fa99ef0d7b7aa1f01b47294d05a2d172`

本書は、有限Randomized Taylor Expansion（RTE）の打切り誤差を、Hadamard-testで観測する
複素信号の位相方向と半径方向へ分けて評価する研究案の数式・比較・主張範囲を固定する。
FR-0は解析契約の確認であり、新しい数値検証、回路compile、artifact生成は行っていない。

関連資料：

- [先行研究対応表](finite_rte_phase_amplitude_prior_art.md)
- [FR-1事前登録](finite_rte_phase_amplitude_fr1_preregistration.md)
- [研究再設計案](../../research_redesign_phase_amplitude_20260926.md)
- [FR-R0正scalar分離・構造比較契約](fr_revision_scalar_structure_contract.md)
- [既存finite-RTE検証](../finite_rte_signal_validation.md)
- [RTE規約](../rte_conventions.md)

## 1. 主RQと現在の判断

主RQは次である。

> finite-RTE打切り誤差について、通常のoperator-normだけの処理よりも、参照発展に対する
> 相対誤差のHermitian/anti-Hermitian成分を分けることで、Hadamard複素信号の位相誤差と
> 信号半径を安全かつ有用に認証できる条件は何か。

FR-0では、下記命題の代数と近接文献との差分が限定scopeで成立することを確認した。
続くFR-1は固定2×2 toyで完了し、G0/G1/G3/G4を通過したが、利用可能な
$\underline\rho=0.8$を使う有用性gate G2は不通過だった。真の$\rho$を使う場合だけ改善したため
`GO_FR2_MECHANISM_ONLY`とし、FR-2は開始しない。これは研究主題の採用、world-first、
H4/H12への移送、RPE総cost改善を意味しない。

## 2. finite-RTE演算子の定義

Hermitianな正規化Hamiltonian $h$、実時間$\tau$、偶数cutoff $K$に対し

$$
P_{K+1}(-i\tau h)=\sum_{n=0}^{K+1}\frac{(-i\tau h)^n}{n!},
\qquad
B_K(\tau)=
\sum_{\substack{n=0\\ n\;\mathrm{even}}}^{K}
\frac{|\tau|^n}{n!}
\sqrt{1+\frac{\tau^2}{(n+1)^2}}
$$

とする。既存実装のpaired Taylor RTEを独立sampleした一回の平均は

$$
\mathbb E[\widetilde U_K(\tau)]
=\frac{P_{K+1}(-i\tau h)}{B_K(\tau)}
$$

である。負時間でもTaylor係数には符号付き$\tau$を使い、sampling normalizationには
$|\tau|$を使う。

$N$個の有限RTE occurrenceが決定論的unitaryと交互に現れるとき、次の二つを区別する。

1. $A_{\rm corr}$：各occurrenceをTaylor numeratorで置換したnormalization-corrected演算子。
2. $A_{\rm mean}=A_{\rm corr}/\mathcal B$：実際のsampled-unitary列の平均演算子。

ここで

$$
\mathcal B=\prod_{j=1}^{N} B_{K_j}(\tau_j).
$$

参照unitary $U$と入力$|\psi\rangle$に対して

$$
z_0=\langle\psi|U|\psi\rangle,
\quad
z_{\rm corr}=\langle\psi|A_{\rm corr}|\psi\rangle,
\quad
z_{\rm obs}=\langle\psi|A_{\rm mean}|\psi\rangle
=\frac{z_{\rm corr}}{\mathcal B}
$$

とする。$\mathcal B>0$は実数なので、$z_{\rm obs}$と$z_{\rm corr}$の位相は同じで、
信号半径だけが$1/\mathcal B$倍される。

この積平均はoccurrenceごとにfresh independentなRTE trajectoryを使う場合の契約である。
同じ乱数列を複数occurrenceで再利用する相関samplingには、そのまま適用しない。

## 3. 一段相対誤差とscalar監査

一段のexact unitary $U_j$とcorrected近似$A_j$に対して

$$
D_j=U_j^\dagger A_j-I=F_j+iG_j,
$$

$$
F_j=\frac{D_j+D_j^\dagger}{2},
\qquad
G_j=\frac{D_j-D_j^\dagger}{2i}
$$

と置く。$F_j,G_j$はHermitianである。scalar eigenvalue $x$に対する

$$
d_K(x)=e^{ix}P_{K+1}(-ix)-1=F_K(x)+iG_K(x)
$$

の原点近傍は、FR-0の独立な有理数series監査で次となった。

| cutoff | radial $F_K(x)$ | tangential $G_K(x)$ |
|---:|---|---|
| 0 | $x^2/2-x^4/8+O(x^6)$ | $x^3/3-x^5/30+O(x^7)$ |
| 2 | $-x^4/24+x^6/72+O(x^8)$ | $-x^5/30+x^7/252+O(x^9)$ |
| 4 | $x^6/720-x^8/1920+O(x^{10})$ | $x^7/840-x^9/6480+O(x^{11})$ |

偶数$K$ではradial成分が$O(x^{K+2})$、tangential成分が$O(x^{K+3})$となる。
これは機構確認であり、非可換積における有用な改善を単独では証明しない。

## 4. 非可換積に対する命題

参照列とcorrected列を

$$
U=U_N\cdots U_1,
\qquad
A_{\rm corr}=A_N\cdots A_1,
\qquad
A_j=U_j(I+D_j)
$$

とする。各$D_j$を最終参照frameへunitary conjugationして得る$\widetilde D_j$は
$F_j,G_j$と同じoperator normを持つ。次を既知または直接計算可能な上界とする。

$$
\|F_j\|\le a_j,
\qquad
\|G_j\|\le b_j,
\qquad
\|D_j\|\le e_j,
$$

$$
a=\sum_j a_j,
\quad
b=\sum_j b_j,
\quad
s=\sum_j e_j,
\quad
R_2=\prod_j(1+e_j)-1-s.
$$

相対積を

$$
Q=U^\dagger A_{\rm corr}-I
=\sum_j\widetilde D_j+R
$$

と書くと、積展開とsubmultiplicativityから

$$
\|R\|\le R_2
$$

である。

### 命題

$\rho=|z_0|>0$、

$$
\kappa(\rho)=\frac{\sqrt{1-\rho^2}}{\rho}
$$

とする。さらに

$$
L=1-a-R_2-\kappa(\rho)(s+R_2),
$$

$$
Y=b+R_2+\kappa(\rho)(s+R_2)
$$

と定義する。$L>0$なら

$$
\operatorname{dist}_{\mathbb S^1}
\left(\arg z_{\rm corr},\arg z_0\right)
\le \arctan\frac{Y}{L},
$$

$$
|z_{\rm obs}|\ge\frac{\rho L}{\mathcal B}.
$$

位相距離はprincipal branch上の円周距離とする。

### 証明

$|\eta\rangle=(U^\dagger-z_0^*)|\psi\rangle$と置くと
$\langle\psi|\eta\rangle=0$、$\|\eta\|=\sqrt{1-\rho^2}$である。また

$$
\frac{z_{\rm corr}}{z_0}
=1+\langle\psi|Q|\psi\rangle
+\frac{\langle\eta|Q|\psi\rangle}{z_0}.
$$

$F_j,G_j$の期待値はそれぞれ実数、$i$倍された実数なので、第一項の実部低下を$a+R_2$、
虚部を$b+R_2$で抑えられる。直交成分はCauchy--Schwarzにより実部・虚部とも
$\kappa(\rho)(s+R_2)$以下である。従って$z_{\rm corr}/z_0$は実部$L$以上、虚部絶対値
$Y$以下の領域に入り、$L>0$なら上の位相境界を得る。絶対値は実部以上なので
$|z_{\rm corr}|\ge\rho L$、最後に$z_{\rm obs}=z_{\rm corr}/\mathcal B$を使う。

### 利用可能な半径下界だけを使う版

$0<\underline\rho\le\rho$しか利用しない場合、

$$
\overline\kappa
=\frac{\sqrt{1-\underline\rho^2}}{\underline\rho}
$$

で$\kappa$を置換した$L_{\rm av},Y_{\rm av}$を使う。$L_{\rm av}>0$なら

$$
\Delta\phi\le\arctan(Y_{\rm av}/L_{\rm av}),
\qquad
|z_{\rm obs}|\ge\underline\rho L_{\rm av}/\mathcal B.
$$

密行列から得た真の$\rho$を使う`PROPOSED-REF`と、事前に与えた$\underline\rho$だけを使う
`PROPOSED-AVAILABLE`を混同しない。

## 5. 比較baseline

同じ$U,A_{\rm corr},\mathcal B,\underline\rho$に対して次を比較する。

### OLD-NORM

$\eta_j=|\tau_j|\|h_j\|$としてTaylor remainder

$$
c_j=\sum_{n=K_j+2}^{\infty}\frac{\eta_j^n}{n!},
\qquad
E_{\rm old}=\prod_j(1+c_j)-1
$$

を使う。これは既存実装のconservativeなoperator-norm連鎖に対応する。

### STRONG-NORM

スペクトル区間だけを用いて

$$
e_j^{\rm spec}
=\sup_{|x|\le\eta_j}|e^{ix}P_{K_j+1}(-ix)-1|,
\qquad
E_{\rm strong}=\prod_j(1+e_j^{\rm spec})-1
$$

とする。有限gridの最大値だけをsupremumとして使わず、解析的な停留点評価または検証済み区間法を使う。

いずれも$E<\underline\rho$なら、共通のnorm-only変換として

$$
\Delta\phi\le\arcsin(E/\underline\rho),
\qquad
|z_{\rm obs}|\ge(\underline\rho-E)/\mathcal B.
$$

### 追加対照

- `SCALAR/EIGEN`：各stepが共通固有basisで可換な対照。高い位相次数のsanity check用。
- `PROPOSED-REF`：真の$\rho$を使う機構上限。実運用claimには使わない。
- `PROPOSED-AVAILABLE`：事前固定した$\underline\rho$だけを使う主候補。

## 6. 既存実装との対応

`src/trotterlib/rte.py`の`finite_taylor_operator`と
`finite_rte_corrected_operator`は$A_{\rm corr}$側、
`finite_rte_operator_moments`の`attenuated_event_mean_operator`は$A_{\rm mean}$側に対応する。
FR-1ではこの意味を変えず、新しい境界計算だけを追加した。

## 7. FR-0/FR-1の結論と主張境界

- FR-0数式契約：上記命題の範囲で成立。
- 先行研究：限定監査では同じ問い・同じ非可換state-conditioned境界を確認できないが、world-firstは主張しない。
- FR-1結果：G0/G1/G3/G4通過、G2不通過、`GO_FR2_MECHANISM_ONLY`。
- 現在の決定：旧FR-2を開始しない。後続のFR-R0で正scalar分離、情報層、強いbaselineを固定したが、FR-R1a/bは未事前登録・未開始である。
- 未実施：H4/H12、回路compile、RPE総cost、noise/backend、FR-2以降。
- 禁止する主張：一般にtight、既存norm boundより常に強い、資源優位性を実証した、研究主題を採用済み。

数値と判断は[FR-1結果](../finite_rte_phase_amplitude_validation.md)を参照する。実行時の本契約は
`artifacts/finite_rte_phase_amplitude/2026-09-26/fr0_contract_frozen.md`へ凍結したが、初回file patchの
行数不整合により本節とnorm-only式の末尾が欠けていた。FR-1の完全な入力・比較・gateは
`fr1_preregistration_frozen.md`に保存されており、実行条件と判定にはこの文書欠落の影響はない。
