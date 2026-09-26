# FR-R1b：非一様spectrum 4×4最小判別 事前登録

最終更新：2026-09-26 JST  
状態：`PREREGISTERED_NOT_RUN`  
親契約：[FR-R0正scalar分離・構造比較契約](fr_revision_scalar_structure_contract.md)  
基点：`all-r-coherent-opt2-reoptimization`、`ecb7f4c007ab4dd98666a53035bf2eefff17f0bb`

本書は、正scalarを除いたfinite-RTE radial nonuniformityが、同じ利用可能情報を使う強い
norm対照を超えてHadamard信号位相の設定判断を変えるかを、4×4 toyで判別する条件を結果前に固定する。
FR-R1aは既知2×2データのposthoc説明であり、本書のgrid、予算、gateを変更する根拠に使わない。

## 1. 検証する主張

次の三点だけを検証する。

1. 正scalar再中心化後のnorm/FR境界が、非一様spectrum、非可換interleaving、負時間、K=4でもsoundか。
2. $\nu=0$または$1/2$で、同じI1情報を使う最適scalar norm対照を超える
   decision-relevantな認証差があるか。
3. その差または不成立を、radial spectral width、involution対照、state conditioningで説明できるか。

正scalar分離自体、dense 4×4計算、$\nu=1$だけの改善、上界比だけの改善は研究GOにしない。

## 2. 固定Hamiltonianとsequence

Pauli行列を$I,X,Z$とし、tailを

$$
h_\nu=\frac{1+\nu}{2}Z\otimes I
+\frac{1-\nu}{2}I\otimes Z,
\qquad \nu\in\{0,1/2,1\}
$$

とする。

$$
\operatorname{spec}(h_\nu)=\{1,\nu,-\nu,-1\},
\qquad \|h_\nu\|=1.
$$

決定論blockは次の二つを固定する。

$$
H_D^{\rm nc}=0.7X\otimes I+0.3Z\otimes X,
$$

$$
H_D^{\rm c}=0.7Z\otimes I+0.3I\otimes Z.
$$

共通物理時間$T=0.8$、outer step数$q\in\{2,4,8\}$、$\delta=T/q$とする。符号$\sigma$、
inner tail step数$r$に対して

$$
S_{\nu,D,\sigma}(\delta)
=e^{-iH_D\delta/2}e^{-i\sigma h_\nu\delta}e^{-iH_D\delta/2},
$$

$$
U_{\nu,D,\sigma,q}=S_{\nu,D,\sigma}(\delta)^q
$$

をexact-tail参照とする。cutoff $K$のcorrected近似は

$$
A_{\nu,D,\sigma,q}^{(K,r)}
=\left[
 e^{-iH_D\delta/2}
 \left\{P_{K+1}(-i\sigma h_\nu\delta/r)\right\}^{r}
 e^{-iH_D\delta/2}
\right]^q.
$$

一occurrenceの時間は$\tau=\sigma\delta/r$、発生回数は$N=qr$、normalizationは

$$
\mathcal B=B_K(\tau)^{N},
\qquad
A_{\rm mean}=A_{\rm corr}/\mathcal B
$$

である。$N$ occurrenceはfresh independent samplingを表す。積順序、符号付きTaylor係数、
$|\tau|$を使うnormalizationを結果後に変更しない。

## 3. 情報層

- **I0**：$\|h_\nu\|=1$、係数、$B_K$、Taylor remainder、連続区間
  $x\in[-|\tau|,|\tau|]$上の検証済みsupremum。
- **I1**：上式から代数的に既知の有限spectrum、involution、可換性、対称性。
- **I2**：dense $U,A$、真の$\rho$、actual phase、exact extrema、結果後最適化。

I2はsoundnessと機構診断にだけ使う。実用的な境界値、one-sided certification、GOには使わない。

## 4. 固定状態

同一$(\nu,H_D,\sigma)$について状態を一度だけ構成し、全$q$、K、rで同じvectorを再利用する。
各$q$の$U$固有basisから主状態を作り直さない。

### 4.1 canonical規約

Hermitian行列の固有値を昇順に並べる。絶対差$10^{-12}$以下の縮退clusterでは、
$Q=\operatorname{diag}(0,1,2,3)$をclusterへ射影して対角化し、その固有値順でbasisを固定する。
各vectorは最大絶対成分のうち最小indexの成分を実かつ非負にする。残る同率は辞書順で固定する。
unitaryの固有vectorはprincipal eigenphaseを区間(-π,π]へ写して昇順に並べ、位相差が10^{-12}以下の
clusterでは同じQ射影規則を使う。

### 4.2 標準三状態

各$(\nu,H_D,\sigma)$について次を作る。

1. `fixed_supplied_superposition`：$H_D+\sigma h_\nu$のcanonical eigenbasis
   $\{|v_k\rangle\}_{k=0}^3$から

   $$
   |\psi_{\rm sup}\rangle
   =\sqrt{0.9}|v_0\rangle
   +\sqrt{0.1/3}\sum_{k=1}^{3}|v_k\rangle
   $$

   とする。主比較には外部入力certificate $\underline\rho=0.8$だけを与える。denseで得た真の
   $\rho_q$はcertificate監査に使うが、境界計算へ戻さない。状態構成にdense toy情報を使うため、
   end-to-endなcheap preparationを示したとは主張しない。
2. `fixed_q8_reference_eigenstate`：$U_{\nu,D,\sigma,8}$のcanonical eigenvectorのうち
   eigenphaseが最小のもの。全$q$で同じvectorを使うI2機構診断。
3. `fixed_total_ground_state`：$H_D+\sigma h_\nu$のcanonical ground state。全$q$で同じvectorを使う
   physical-state診断。

主18条件の全てで`fixed_supplied_superposition`の真の$\rho_q$が$0.8-\tau_{\rm num}$以上でなければ、
入力certificateを無効としてGOを出さない。これはpromiseの監査であり、真の$\rho_q$を0.8より大きい値へ
置換する規則ではない。

### 4.3 signal-near-zero stress state

既存主条件$(\nu,H_D,q,\sigma,K,r)=(0,H_D^{\rm nc},4,+1,2,1)$の$U$について、固有位相差の
円周距離が最大となる固有vector pairを選び、同率はindex pairの辞書順で決める。その等重みsuperpositionを
一行だけ追加する。これはI2の適用不能・branch stressで、GOへ数えない。$|z_0|<10^{-12}$なら位相未定義、
それ以外でも$\underline\rho$を供給せず、available methodは`inapplicable`とする。

## 5. 固定gridとexpected count

### 5.1 primary

$$
\nu\in\{0,1/2,1\},\quad
q\in\{2,4,8\},\quad
H_D\in\{H_D^{\rm c},H_D^{\rm nc}\},
$$

$$
\sigma=+1,\quad K=2,\quad r=1,\quad T=0.8.
$$

3×3×2で18 matrix conditions、各条件3標準状態で54 state rowsである。

### 5.2 固定controls

| control | 固定条件 | state rows |
|---|---|---:|
| negative time | $\nu=1/2,H_D^{\rm nc},q=4,\sigma=-1,K=2,r=1$ | 標準3 |
| K=4 | $\nu=1/2,H_D^{\rm nc},q=4,\sigma=+1,K=4,r=1$ | 標準3 |
| near-zero | 主条件$\nu=0,H_D^{\rm nc},q=4$へ4.3のstateを追加 | 1 |

従ってexpected matrix conditionsは20、標準state rowsは60、special state rowは1、合計61である。

### 5.3 event-average semantic controls

$\nu=1/2,H_D=0,r=1,K=2,\tau\in\{-0.2,+0.2\}$の2条件で、全Taylor eventを列挙する。
ordinary平均が$P_3(-i\tau h_\nu)/B_2(\tau)$と一致し、controlled平均が
$\operatorname{diag}(I,P_3/B_2)$と一致することを絶対誤差$10^{-12}$以内で要求する。
positive scalarはcontrol-$|1\rangle$側の相対振幅に残し、global phaseとして除かない。

上記以外の$\nu,q,r,K,T,H_D$を追加しない。特に$q=16$、$r>1$、K=0、別角度は本実行に含めない。

## 6. 正scalarと比較法

一occurrenceの相対誤差を

$$
d_K(x)=e^{ix}P_{K+1}(-ix)-1,
\qquad x\in\tau\operatorname{spec}(h_\nu)
$$

とする。I1 spectrum上のHermitian extremaを$f_-,f_+$とし

$$
c_{\rm mid}=\frac{f_-+f_+}{2},\qquad
\gamma_{\rm mid}=1+c_{\rm mid}>0
$$

を共通scalarに使う。$\gamma_{\rm mid}\le0$なら適用不能である。

最適scalarは

$$
\gamma_*\in\mathop{\rm argmin}_{0.5\le\gamma\le1.5}
\max_{x\in\mathcal S}
\left|\frac{1+d_K(x)}{\gamma}-1\right|
$$

で定義する。I1では有限spectrum、I0では連続区間を使う。同率、精度、境界規則はFR-R1aと同じで、
境界到達は`boundary_inconclusive`とする。`OPT-SCALAR-FR-I1`は同じγ区間・同じ情報で
FR位相上界を最小化し、同じtie-breakと境界規則を使う。

各state rowで最低限次を評価する。

- `OLD-NORM-I0`
- `STRONG-NORM-I0`
- `STRONG-NORM-I1`
- `OLD-FR-I1`
- `SCALAR-NORM-COMMON-I1`
- `SCALAR-FR-COMMON-I1`
- `OPT-SCALAR-NORM-I1`
- `OPT-SCALAR-FR-I1`
- `INVOLUTION-POLAR-I1`（$\nu=1$だけ）
- `DENSE-ORACLE-I2`

共通$\gamma$比較でFR分解の増分を測り、最適化比較で強いscalar-only対照を超えるかを測る。
各法に同じ$U,A_{\rm corr},\mathcal B,\underline\rho$を使う。

再中心化後のnorm誤差を$E$、FR境界の実部下界を$L$とすると、物理半径下界はそれぞれ

$$
\rho_{{\rm obs},lb}^{\rm norm}
=\frac{\Gamma_c}{\mathcal B}(\underline\rho-E),
$$

$$
\rho_{{\rm obs},lb}^{\rm FR}
=\frac{\Gamma_c}{\mathcal B}\underline\rho L.
$$

$\Gamma_c$を落とした半径を設定判断に使わない。

## 7. 固定位相予算と数値規約

用途に結び付けないtoy感度予算として、結果を見る前に

$$
\beta\in\{10^{-2},10^{-3},10^{-4}\}\ {\rm rad}
$$

を固定する。methodが適用可能、位相上界が$\beta$以下、物理半径下界が0.2以上の場合だけ
`certified_at_beta=true`とする。結果後に中間予算を追加しない。

actual phaseは

$$
\Delta\phi_{\rm actual}
=|\operatorname{Arg}(z_{\rm corr}\overline{z_0})|
$$

とする。数値許容差は

$$
\tau_{\rm num}=512\epsilon_{\rm mach}\max(1,N_{\rm factor})
\left(1+\max\{|z_0|,|z_{\rm corr}|,\mathcal B,\Gamma_c\}\right).
$$

supremumと1次元最適化は解析解または検証済みinterval/branch-and-boundで認証し、有限grid最大を
厳密supremumと呼ばない。

## 8. 結果前の機構予測

- $\nu=1$では$h_\nu^2=I$なので、K=2のHermitian相対誤差は共通scalarであり、
  再中心化後のradial widthは数値許容差内で0になる。
- $\nu=0,1/2$では異なる$|x|$を持つため、K=2のleading radial term
  $-x^4/24$に非一様幅が残る。
- 従って「scalarだけで説明できる」対照は$\nu=1$、残差を調べる主条件は$\nu=0,1/2$である。
- 可換性はsoundnessとstate conditioningを分ける対照であり、非可換だけを都合よく選ばない。

予測と逆の結果も保存し、説明できなければGOにしない。

## 9. 固定gate

### R0：完全性・semantic

expected 20 matrix conditions、61 state rows、2 semantic controlsが重複・欠落なく完了し、
event-average residualが$10^{-12}$以内である。

### R1：入力certificate

主18条件の`fixed_supplied_superposition`で真の$\rho_q\ge0.8-\tau_{\rm num}$。真値は監査だけに使い、
境界入力は全て0.8のままとする。

### R2：soundness

全適用method recordで実位相が上界を超えず、実物理半径が下界を下回らない。許容は
$\tau_{\rm num}$だけとし、違反後に係数を膨らませない。

### R3：非一様機構

$\nu=1$の再中心化後radial widthが0、$\nu=0$と$1/2$の少なくとも一方で正となり、
leading-order予測と整合する。

### R4：同情報・共通scalar利益

$\nu=0$または$1/2$のprimary `fixed_supplied_superposition`で、同じI1情報と同じ
$\gamma_{\rm mid}$を使う`SCALAR-FR-COMMON-I1`が`SCALAR-NORM-COMMON-I1`より
$\tau_{\rm num}$を超えて小さい位相上界を持つ点が一つ以上ある。

### R5：decision relevance

$\nu=0$または$1/2$の同じprimary行・同じ固定$\beta$で、
`OPT-SCALAR-FR-I1`が`certified_at_beta=true`、`OPT-SCALAR-NORM-I1`がfalseとなる点が
一つ以上ある。両法の物理半径下界は0.2以上でなければならない。比の改善だけでは通過しない。

### R6：oracle非依存

R4/R5に真の$\rho$、dense spectrum、結果後scalar、near-zero stateを使わない。状態生成にdense toy情報を
使った事実は記録し、結論を「supplied $\rho$ certificate付き入力に条件付く」と限定する。

### R7：control移送

negative-timeとK=4の全適用recordでR2を満たし、involution polar対照とcontrolled scalar semanticが一致する。

## 10. 終了時の分類

優先順位順に分類する。

| status | 条件 |
|---|---|
| `STOP_FR_R_INPUT_OR_SOUNDNESS` | R0、R1、R2、R7のいずれか不通過 |
| `STOP_FR_R_INVOLUTION_OR_SCALAR_ONLY` | soundだがR3またはR4不通過、あるいは最適scalar normが全差を説明 |
| `MECHANISM_ONLY_NO_PRACTICAL_GO` | R3/R4は通るがR5またはR6不通過 |
| `GO_FR_R2_CANDIDATE_CONDITIONAL_ON_SUPPLIED_STATE` | R0--R7全通過 |

最後のstatusでもFR-R2やH4を自動開始しない。状態準備とcertificate取得costを含むend-to-end実用性を
示したとは呼ばない。

## 11. 実行・成果物契約

実行時には、library module、runner、専用test、expected specification、result JSON、結果文書を追加する。
expected specificationには本書SHA-256、commit、全condition ID、全state ID、method一覧、予算、threshold、
source hashを保存し、末尾を含む文書完全性を検査する。expectedを生成してから結果を見るまで、grid、状態、
予算、gateを変更しない。技術的失敗の修正は旧expectedを残し、新versionへ非上書きで行う。

保存項目は少なくとも、$U,A_{\rm corr},A_{\rm mean}$ fingerprint、state fingerprint、情報層、
$c,\gamma,\Gamma_c,\mathcal B$、radial width、全bound、actual phase/radius、適用不能理由、全gateを含む。

FR-R1aを先に実行しても本書を変更しない。FR-R1b終了後はstatusにかかわらず必ず停止し、同じtoyの
$\nu,q,\rho$追加、H4/H12、P-D S2、長RPE、full compiled総costへ進まない。
