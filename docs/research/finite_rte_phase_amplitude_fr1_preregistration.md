# FR-1 非可換toy機構試験 事前登録

最終更新：2026-09-26 JST
事前登録時状態：`PREREGISTERED_NOT_RUN`

実行後状態：`COMPLETED_GO_FR2_MECHANISM_ONLY_FR2_NOT_STARTED`
親契約：[finite-RTE位相・信号半径分離契約](finite_rte_phase_amplitude_contract.md)

本書はFR-1で実行する入力、比較、数値規約、判定gateを結果を見る前に固定する。
FR-1は2×2 dense matrixだけを使う機構試験であり、H4/H12、RPE、回路compile、Monte Carlo
shot実験は行わない。FR-1終了後に必ず停止し、FR-2へ自動的に進まない。

## 1. 検証する主張

次の二点だけを検証する。

1. [FR-0命題](finite_rte_phase_amplitude_contract.md#4-非可換積に対する命題)が、可換・非可換、
   正時間・負時間、K=0/2/4、反復で数値的に破れないこと。
2. 非可換条件で、真の信号半径をoracleとして使わない`PROPOSED-AVAILABLE`が、
   `STRONG-NORM`より位相認証に実質的な利益を持つ点が少なくとも一つ存在すること。

結果を見てgrid、閾値、状態、baselineを変更しない。探索追加が必要ならFR-1を閉じ、別契約にする。

## 2. 固定toy system

Pauli行列を$I,X,Z$とし、

$$
H_D=0.7Z,
\qquad
H_R(\theta)=h_\theta=\cos\theta\,Z+\sin\theta\,X,
\qquad
\|h_\theta\|=1
$$

とする。符号$\sigma\in\{+1,-1\}$、一outer stepの時間$\Delta$に対する対称参照を

$$
U_{\theta,\sigma}(\Delta)
=e^{-i\Delta H_D/2}
 e^{-i\sigma\Delta h_\theta}
 e^{-i\Delta H_D/2}
$$

とする。RTE short-step数$r$、cutoff $K$のcorrected近似は

$$
A_{\theta,\sigma}^{(K,r)}(\Delta)
=e^{-i\Delta H_D/2}
\left[P_{K+1}(-i\sigma\Delta h_\theta/r)\right]^r
e^{-i\Delta H_D/2}.
$$

outer反復$q$と固定総時間$T$について

$$
U=\left[U_{\theta,\sigma}(T/q)\right]^q,
\qquad
A_{\rm corr}=\left[A_{\theta,\sigma}^{(K,r)}(T/q)\right]^q,
$$

$$
\mathcal B=
\left[B_K(\sigma T/(qr))\right]^{rq},
\qquad
A_{\rm mean}=A_{\rm corr}/\mathcal B
$$

とする。各$r q$ occurrenceはfresh independent samplingを表す。

非対称対照では一stepだけ

$$
U^{\rm asym}=e^{-i\Delta H_D}e^{-i\sigma\Delta h_\theta},
\qquad
A_{\rm corr}^{\rm asym}=e^{-i\Delta H_D}
\left[P_{K+1}(-i\sigma\Delta h_\theta/r)\right]^r
$$

へ置換する。

## 3. 入力状態

各$U$を対角化し、固有位相を昇順で安定に並べ、固有vectorの最大絶対成分を実正にする。
次の三状態を使う。

1. `reference_eigenstate`：$U$の第一固有vector。構成上$\rho=|\langle\psi|U|\psi\rangle|=1$。
2. `analytic_mixture_state`：二つの固有vector$|u_0\rangle,|u_1\rangle$から
   $|\psi\rangle=\sqrt{0.9}|u_0\rangle+\sqrt{0.1}|u_1\rangle$とする。
   三角不等式から事前に$\underline\rho=0.8$を利用でき、これを主判定状態とする。
3. `physical_ground_state`：$H_D+\sigma H_R(\theta)$の基底状態。真の$\rho$を使う
   `PROPOSED-REF`の診断だけに使い、`PROPOSED-AVAILABLE`のGO根拠にはしない。

$\rho<10^{-12}$のrecordは位相未定義として失格にし、ゼロで割らない。

## 4. 事前固定grid

### 主grid

$$
\theta=\pi/3,\quad \sigma=+1,\quad T=0.8,\quad K=2,\quad r=1,
\quad q\in\{1,2,4,8,16\}.
$$

三状態を全て評価する。`analytic_mixture_state`＋$\underline\rho=0.8$が主判定行である。

### 対照grid

| 目的 | 固定値 |
|---|---|
| 可換対照 | $\theta=0$、その他は主gridと同じ全$q$ |
| 強非可換角度 | $\theta=\pi/2$、$q\in\{2,8\}$、その他は主gridと同じ |
| short-step依存 | $\theta=\pi/3,q=2,r\in\{1,2,4\},K=2,T=0.8$ |
| 負時間係数 | $\theta=\pi/3,q=2,r=1,K=2,T=0.8,\sigma=-1$ |
| 非対称配置 | $\theta=\pi/3,q=2,r=1,K=2,T=0.8,\sigma=+1$ |
| cutoff対照 | $\theta=\pi/3,q\in\{2,8\},r=1,K\in\{0,4\},T=0.8$ |
| 局所次数監査 | $q=r=1,\theta=\pi/3,\sigma=+1,K\in\{0,2,4\},\Delta\in\{0.05,0.1,0.2,0.4\}$ |

局所次数監査はlog-log slopeの参考診断であり、GO/STOPを単独で決めない。

## 5. identity・controlled位相semantic check

scalar phaseを無視しないことを別の一段検査で確認する。

$$
h_I=0.2I+0.4X+0.4Z,
\qquad \lambda=|0.2|+|0.4|+|0.4|=1.
$$

$K\in\{0,2\}$、$\tau\in\{-0.2,+0.2\}$について、全Taylor wordを明示列挙し、

- 通常blockの確率平均が$P_{K+1}(-i\tau h_I)/B_K(\tau)$と一致すること、
- control-$|0\rangle$ branchをidentity、control-$|1\rangle$ branchをsampled unitaryとした
  controlled blockで、identity termのglobal phaseがrelative phaseとして保持されること

を検査する。許容絶対誤差は$10^{-12}$とする。

## 6. 計算量と比較値

各recordでcomplex128により$U,A_{\rm corr},A_{\rm mean}$を直接構成し、

$$
z_0=\langle\psi|U|\psi\rangle,
\quad z_{\rm corr}=\langle\psi|A_{\rm corr}|\psi\rangle,
\quad z_{\rm obs}=z_{\rm corr}/\mathcal B
$$

を計算する。実位相誤差は

$$
\Delta\phi_{\rm actual}
=\left|\operatorname{Arg}(z_{\rm corr}\overline{z_0})\right|
$$

とし、信号半径は$|z_{\rm obs}|$とする。

各有限RTE occurrenceの

$$
D_j=e^{+i\sigma\Delta h_\theta/r}P_{K+1}(-i\sigma\Delta h_\theta/r)-I
$$

から$a_j=\|F_j\|_2$、$b_j=\|G_j\|_2$、$e_j=\|D_j\|_2$を直接計算し、
`PROPOSED-REF`と`PROPOSED-AVAILABLE`を評価する。比較は同じ演算子と状態について
`OLD-NORM`、検証済みspectral supremumを使う`STRONG-NORM`、可換`SCALAR/EIGEN`を用いる。

真値と境界の比較許容差は

$$
\tau_{\rm num}=512\,\epsilon_{\rm mach}
\max(1,N_{\rm factor})
\left(1+\max\{|z_0|,|z_{\rm corr}|,\mathcal B\}\right)
$$

とする。supremumは解析的停留点評価または外向き丸めを伴うinterval評価で確認し、
未検証の有限grid最大値を`STRONG-NORM`として使わない。

## 7. 固定gate

### G0：semantic consistency

identity・controlled位相検査が全件$10^{-12}$以内で一致する。

### G1：soundness

適用条件$L>0$または$E<\underline\rho$を満たす全recordで、実位相誤差が各位相上界を
$\tau_{\rm num}$より大きく超えず、実信号半径が各下界を$\tau_{\rm num}$より大きく下回らない。

### G2：非可換での有用性

主gridの`analytic_mixture_state`に、正の信号半径下界を保ったまま次のどちらかを満たす点が一つ以上ある。

1. `PROPOSED-AVAILABLE`位相上界が`STRONG-NORM`位相上界の50%以下。
2. `PROPOSED-AVAILABLE`が$10^{-3}$ rad以下を認証し、`STRONG-NORM`は認証しない。

### G3：conditioningと棄却

$L\le0$または入力半径下界が成立しない点を成功として数えず、`inconclusive`または`inapplicable`
として保存する。真の$\rho$を使う行だけの改善をavailable-informationの成功と呼ばない。

### G4：符号・次数移送

負時間、K=4、非対称配置の全適用recordでG1を満たす。K=0は既知対照として保存するが、
K=0が有用性gateを通ることは要求しない。

## 8. 終了時の分類

| 判定 | 条件 |
|---|---|
| `GO_FR2_AVAILABLE` | G0--G4を全て通過し、G2が$\underline\rho=0.8$だけで成立 |
| `GO_FR2_MECHANISM_ONLY` | G0/G1/G3/G4は通過するが、有用性が$\rho=1$または真の$\rho$を使う診断に限られる |
| `STOP_FR1_NO_NONCOMMUTING_GAIN` | soundnessは通過するがG2を満たさない |
| `STOP_FR1_BOUND_INVALID` | 数値誤差・実装誤りで説明できないG0/G1/G4違反がある |

`GO_FR2_AVAILABLE`でも研究主題を自動採用しない。FR-2のH4適用契約、新規性再監査、計算量と
利用可能入力の定義を先に固定する。`GO_FR2_MECHANISM_ONLY`ではFR-2を開始せず、oracle依存を
除けるか再設計する。

## 9. 成果物契約

実行する場合は次を同時に追加する。

- library module、runner、専用test。
- source hash付きexpected specification。
- 全recordとgate判定を含むmachine-readable JSON。
- validation文書と、各index、研究概要、研究ノートの更新。
- artifactをmanifestへ登録し、生成時のcommit・dirty状態・依存versionを記録。

FR-0時点ではこれらをまだ生成しておらず、既存の`VALIDATION_STATUS.md`と
`artifacts/validation_manifest.json`は変更しない。


## 10. 実行後追記（契約変更ではない）

2026-09-26に上記契約を変更せず実行した。G0/G1/G3/G4は通過、G2は不通過となり、
事前分類どおり`GO_FR2_MECHANISM_ONLY`とした。FR-2は開始していない。結果は
[FR-1検証文書](../finite_rte_phase_amplitude_validation.md)を正本とする。

実行時の本書は
`artifacts/finite_rte_phase_amplitude/2026-09-26/fr1_preregistration_frozen.md`へ凍結し、
SHA-256 `bc8066d7a31a3f46f476d2c591c7f0dad5e6f49023fc261dc518b61e2ec600a3`がartifact内の
source hashと一致する。本節は結果確認後のstatus注記であり、grid、閾値、状態、gateを変更しない。
