# FR-R1a：既存FR-1の正scalar事後再解析計画

最終更新：2026-09-26 JST  
状態：`POSTHOC_PLAN_FROZEN_NOT_RUN`  
親契約：[FR-R0正scalar分離・構造比較契約](fr_revision_scalar_structure_contract.md)  
基点：`all-r-coherent-opt2-reoptimization`、`ecb7f4c007ab4dd98666a53035bf2eefff17f0bb`

本書は、完了済みFR-1の33条件・99状態を、正scalar処理と強いbaselineで再解析する
FR-R1aの事後計画を固定する。FR-R1aは既知データの説明監査であり、blind検証、旧G2の再判定、
新しい研究GOには使わない。

## 1. 固定入力

入力は次の完了済み証拠に限定する。

- result：`artifacts/finite_rte_phase_amplitude/2026-09-26/finite_rte_phase_amplitude_fr1_v1.json`
- content fingerprint：`d96b200163f3a432652656ae97c65c837e01fe81323cd69528373dff16ce6152`
- file SHA-256：`6a81a0ba6e39ba0f5d79ba026a2a45c3c65709e071e9c6fa5606b3e99a89b0f7`
- frozen preregistration SHA-256：`bc8066d7a31a3f46f476d2c591c7f0dad5e6f49023fc261dc518b61e2ec600a3`
- 条件数33、状態行99、適用method record 495

result、frozen preregistration、実装commitのいずれかが上記と一致しなければ停止する。入力JSONを
上書きしない。再解析用に2×2行列と状態を再構成する場合は、凍結条件だけから決定論的に再生成し、
既存`state_fingerprint`、信号、旧境界と一致することを先に確認する。不一致時に許容差や状態規約を
変更しない。

## 2. 保存する旧判断

次を変更しない。

- G0/G1/G3/G4通過、G2不通過。
- `GO_FR2_MECHANISM_ONLY`。
- 主比較の`analytic_mixture_state`は$\underline\rho=0.8$。
- 真の$\rho$を使う`PROPOSED-REF`はoracle診断。
- FR-2、H4/H12、compile、RPE総costは未開始。

FR-R1aの新しい表や分類を「修正版FR-1結果」または事前登録結果と呼ばない。

## 3. 解析対象

33条件・99状態を全て再計算対象に含める。主説明行は旧主gridの
`analytic_mixture_state`、

$$
T=0.8,\quad K=2,\quad r=1,\quad
q\in\{1,2,4,8,16\},\quad \underline\rho=0.8
$$

の5行である。可換、強非可換、short-step、負時間、非対称、K=0/4、局所次数は感度・semantic
監査として保持する。どの行も削除せず、都合のよい$q$だけを主結果へ昇格しない。

## 4. 正scalar再中心化

各有限tail occurrenceの相対誤差

$$
D=U_{\rm tail}^{\dagger}A_{\rm tail}-I=F+iG
$$

について、同じ許可情報から得た$F$の区間$[f_-,f_+]$を使い

$$
c_{\rm mid}=\frac{f_-+f_+}{2},\qquad
\gamma_{\rm mid}=1+c_{\rm mid}>0,
$$

$$
\widehat D=\frac{D-c_{\rm mid}I}{\gamma_{\rm mid}}
$$

とする。発生回数$N=qr$では

$$
\Gamma_c=\gamma_{\rm mid}^{N},\qquad
z_{\rm obs}=\frac{\Gamma_c}{\mathcal B}\widehat z_{\rm corr}
$$

である。$\Gamma_c$を位相上界からは除くが、物理半径下界から除かない。

### 最適scalar norm対照

同じspectral情報に対し

$$
\gamma_*\in\mathop{\rm argmin}_{0.5\le\gamma\le1.5}
\max_{x\in\mathcal S}
\left|\frac{1+d_K(x)}{\gamma}-1\right|
$$

を`OPT-SCALAR-NORM`に使う。$d_K(x)=e^{ix}P_{K+1}(-ix)-1$である。
同率なら$|\gamma-1|$が小さい方、それも同じなら小さい$\gamma$を選ぶ。I1では既知の有限spectrum、
I0では検証済み区間supremumを使う。global minimumを絶対誤差$10^{-12}$で認証し、有限grid最大だけを
supremumまたは最適解としない。最適値が0.5または1.5境界に達した行は`boundary_inconclusive`とし、
優位性判断へ使わない。

## 5. 固定比較法

各状態行について次を同じ$U,A_{\rm corr},\mathcal B,\underline\rho$で保存する。

1. 既存`OLD-NORM`。
2. 既存`STRONG-NORM`。
3. 既存`OLD-FR`。
4. `SCALAR-NORM-COMMON`：$\gamma_{\rm mid}$で再中心化したnorm-only境界。
5. `SCALAR-FR-COMMON`：同じ$\gamma_{\rm mid}$で再中心化したFR境界。
6. `OPT-SCALAR-NORM`：上記1次元最適化を使う強い対照。
7. `OPT-SCALAR-FR`：同じ許可情報内でFR位相上界を最適化する診断。
8. `INVOLUTION-POLAR`：$h^2=I$のexact polar対照。
9. `DENSE-ORACLE`：真の$\rho$、exact extremaを使う診断。

共通$\gamma$比較と各法最適化比較を混ぜない。I0/I1/I2を全recordへ保存し、I2を実用的改善へ数えない。

## 6. 固定位相予算と判定量

FR-R1bと独立に同時固定した診断用位相予算を

$$
\beta\in\{10^{-2},10^{-3},10^{-4}\}\ {\rm rad}
$$

とする。既存FR-1やFR-R1aの値の間へ後から予算を追加しない。methodが適用可能で、位相上界が
$\beta$以下、かつ物理半径下界が0.2以上のときだけ`certified_at_beta=true`とする。

各行で次を出力する。

- $c,\gamma,\Gamma_c,\mathcal B$。
- $F-cI$のspectral width、$\widehat a,\widehat b,\widehat e,\widehat R_2$。
- actual phase、corrected radius、physical observed radius。
- 全methodの位相上界、半径下界、適用不能理由。
- 共通$\gamma$でのFR/norm比。
- 最適化後FR/norm比。
- 各$\beta$でのone-sided certification。
- 情報層とoracle flag。

## 7. soundnessと説明分類

数値許容差は旧FR-1と同じ

$$
\tau_{\rm num}=512\epsilon_{\rm mach}\max(1,N_{\rm factor})
\left(1+\max\{|z_0|,|z_{\rm corr}|,\mathcal B,\Gamma_c\}\right)
$$

を使う。全適用recordで位相上界と物理半径下界の違反0を要求する。

FR-R1aの出力分類は次だけとする。

| status | 条件 |
|---|---|
| `POSTHOC_SCALAR_EXPLAINS_OLD_GAIN` | 主5行の改善が`OPT-SCALAR-NORM`で全て説明され、同情報one-sided差がない |
| `POSTHOC_RESIDUAL_FR_INCREMENT` | 共通$\gamma$でFRがnormより厳しい行があるが、研究GOとはしない |
| `POSTHOC_ORACLE_ONLY_INCREMENT` | 追加改善がI2または真の$\rho$だけに依存 |
| `POSTHOC_BOUND_OR_RECONSTRUCTION_FAILURE` | fingerprint、再構成、soundnessのいずれかが不一致 |

複数に該当する場合はfailureを最優先し、次にoracle-only、residual、scalar-onlyの順で記録する。
`POSTHOC_RESIDUAL_FR_INCREMENT`でもFR-R1bを合格扱いしない。

## 8. 成果物と停止規則

実行時に追加するものは、事後再解析module、runner、専用test、source-hash付きexpected JSON、
result JSON、結果文書である。元artifactは変更しない。resultには`posthoc=true`、旧decision、
旧artifact fingerprint、全method record、分類を保存する。

FR-R1a終了後に旧G2や旧decisionを変更しない。FR-R1bのgrid、位相予算、GO/STOPは
[FR-R1b事前登録](fr_revision_nonuniform_preregistration.md)ですでに独立固定されているため、
FR-R1aの結果を使って変更しない。FR-R1a単独からH4、FR-R2、追加toyへ進まない。
