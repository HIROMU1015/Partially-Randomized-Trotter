# P-D energy係数・random-tail負担Pareto監査 事前登録

固定日：2026-09-25 JST

## 目的

有限個の対称Product Formula（PF）について、固有位相energy biasだけで選ぶ公式と、
random tailの絶対時間

\[
\Gamma_R=\sum_j |b_j|
\]

およびstage構造を含めて選ぶ公式が食い違うかを判定する。これは高次PFを現行RPEへ
導入する本計算ではなく、P-Dを研究候補として残す独立pilotである。

## 探索済み条件とblind条件

- development：H4 linear chain、1.0 Å、STO-3G、8 qubit、4 electron sector、
  DF rank 12、`L_D=3`。候補gridの数値解像度を決めるため事前に挙動を確認済み。
- blind holdout：同一の固定Hamiltonian snapshot、`L_D=4`。expected-task artifactを
  固定する前にはP-D候補gridを計算しない。

従って`L_D=3`は発見用、`L_D=4`だけをselection-reversalのtransfer判定に使う。

## 固定候補

1. 二次Strang
2. 標準四次Yoshida composition
3. projectの四次`m=2`候補（同梱2026 PDFの8桁係数）
4. 八次Yoshida composition
5. Moralesらの八次`m=8`（同梱arXiv:2210.15817v1 Table II）

processor付き公式は含めない。係数列は`product_formula.py`と出典PDFのSHA-256へ固定する。
各公式は、係数和、対称性、非可換3次元toyでの局所演算子誤差次数、固定時間反復の
global誤差次数を通過した場合だけH4 Pareto候補に含める。固有位相biasのtoy slopeは
state-specific cancellationを含むため記録するが、このregistry gateには使わない。

## K02の区別

非可換な`H_D=A_1+A_2`と`H_R=B`を固定し、次を比較する。

- 外側Strang、`H_D`内部だけ四次
- `H_D/H_R`二block全体の四次
- `A_1/A_2/H_R`三term全体の四次

局所operator-error slopeについて、内部だけ四次の構成は3、全体四次は5を
許容幅0.35で満たし、両者の差が1.5以上であることをgateとする。

## H4二block参照

H4では`exp(-i H_D t)`と`exp(-i H_R t)`をsector dense行列から厳密に構成する。
これは外側係数の判別用toyであり、実際の多数fragmentからなる`H_D`内部誤差を含まない。

直接deltaは

\[
0.025,0.05,0.1,0.2,0.25,0.32,0.4
\]

に固定する。共通判断点は`delta=0.4`、許容energy biasは`1e-6 Ha`、target branch
weightは0.9995以上とする。

許容内候補について、energy-only選択はbias最小、tail-aware選択は`Gamma_R`最小、
同値ならstage数最小とする。次を全て満たす場合にselection reversalを通過とする。

- 許容候補が2個以上
- energy-onlyとtail-awareの公式が異なる
- tail-aware公式の`Gamma_R`が20%以上小さい
- tail-aware公式のfull exponential stage数が増えない
- 両候補が`(bias, Gamma_R, stage count)`のPareto集合にある

## finite-RTE診断

各tail occurrenceを独立RTEとみなし、1 outer step当たり総short-step数64、Taylor cutoff 2を
固定する。equal配分と`|b_j|`比例の整数配分について、`delta=0.02,0.4`で有限分布の
実`log B_K`、attenuation、打切り残差boundを計算する。`Gamma_R^2`は連続緩和の
診断であり、実測総costの下界とは呼ばない。

## 判断

registry、K02、finite-RTE数値整合、development reversal、blind reversal、unitarity、
target weightを全て通過した場合だけ
`advance_pd_as_conditional_candidate_pending_signed_time_and_inner_hd_validation`
とする。

通過しても次は、負時間RTEのoperator/control位相検証と、実際の`H_D`内部PF誤差を戻す
検証である。compiled回路、RPE総cost、H12、全PF family最適性、科学的優位性は主張しない。
