# P-D現実化 Go/No-Go gate 事前登録

固定日：2026-09-25 JST

## 1. 目的

P-D pilotで観測した

\[
\text{energy-only PF selection}
\ne
\text{random-tail-aware PF selection}
\]

が、exact `H_D/H_R`二blockという理想化だけで生じたものか、実際のpartial-randomized
PF設計にも残るかを判定する。本検証をP-DのGo/No-Go gateとし、終了後は結果にかかわらず
計算を一度停止して研究RQ、新規性、着地点、必要な本検証を再設計する。

本検証中はH12、長RPE、full総cost、backend/noise、多数のPF family、広い
`delta/r/L_D` gridへ進まない。

## 2. 固定入力と候補

- Hamiltonian：既存のH4 linear chain、1.0 Å、STO-3G、8 qubit、4 electron、DF rank 12
- sector dimension：70
- PF候補：`2nd`、`4th`、`4th(new_2)`、`8th(Yoshida)`、`8th(Morales)`
- 主判断delta：0.4
- 診断delta：0.2
- energy tolerance：`1e-6 Ha`
- minimum target branch weight：0.9995
- minimum tail-burden reduction：20%
- finite-RTE Taylor cutoff：2
- 1 outer step当たりのtail short-step総数：64
- tail short-step配分：`|b_j|`比例のlargest-remainder整数配分。各occurrenceへ最低1 step
- `H_D`内部substep数：各outer `H_D` occurrence当たり32。結果後に変更しない

`L_D=3`は既観測development、`L_D=4`はP-D pilotでouter exact結果を観測済みだが
fragment内部誤差は未観測のtransfer-confirmationとする。`L_D=5`は本検証の計算前に
energy/tail候補比較を行わないfresh holdoutとする。

## 3. D1：負時間finite-RTEとcontrolled relative phase

### 3.1 対象

固定5公式のtail係数列に現れる全ての負係数を対象にする。同じ絶対値の正時間を対照に置き、
主判断delta 0.4と固定short-step配分から得るsigned short-step timeを使う。

identity成分と2個以上の非可換非identity成分を持つ固定小行列tailを用いる。確率・有限分布の
normalizationは絶対時間で構成し、Taylor演算子の符号だけをsigned timeから保持する。

### 3.2 比較

各signed-time taskについて次を保存する。

1. finite-RTE eventを全列挙した平均ordinary operator
2. 同じ有限Taylor定義から直接構成したoracle operator
3. `M(-t)`と`M(t)^\dagger`の差
4. sampleごとのcontrolled operatorと`diag(I,U_sample)`の差
5. 平均controlled operatorと`diag(I,M(t))`の差
6. 抽出したidentity phaseと、通常/control branch間のrelative phase
7. fixed-seed sampled mean、標準誤差、exhaustive meanからの標準化残差

### 3.3 D1 gate

次を全taskで満たす場合だけD1を通過する。

- exhaustive ordinary residual spectral norm `<=1e-12`
- signed adjoint residual spectral norm `<=1e-12`
- samplewise controlled block residual spectral norm `<=1e-12`
- exhaustive controlled residual spectral norm `<=1e-12`
- identity/relative-phase residual `<=1e-12 rad`
- sampled meanは各実・虚matrix成分についてabsolute error `<=5e-3`または
  absolute standardized residual `<=4`の少なくとも一方を満たす
- 負時間で確率が負になる、normalizationが非正になる、符号情報が失われるtaskが0件

実装バグが原因の不通過は修正後に同じtask、seed、閾値で再実行できる。数理的または仕様上の
不成立、あるいは閾値を変えないと通らない場合はP-Dを停止する。

## 4. D2：実際のfragment内 `H_D` 誤差

### 4.1 fragment-level deterministic approximation

各`L_D`について累積dense deterministic Hamiltonian

\[
H_D(0),H_D(1),\ldots,H_D(L_D)
\]

を同一snapshotから構成し、

\[
A_0=H_D(0),\qquad A_k=H_D(k)-H_D(k-1)
\]

をone-body/correctionと順序付きDF fragmentのdense termとする。再構成誤差

\[
\left\|\sum_{k=0}^{L_D}A_k-H_D(L_D)\right\|_F
\]

を保存する。

各外側PFの`H_D` occurrence `exp(-i a_j H_D delta)`を、固定順
`A_0,A_1,...,A_LD`の二次対称PFを32 substep反復した演算子へ置き換える。tail occurrenceは
D2のenergy比較ではexact `exp(-i b_j H_R delta)`とし、負時間実装可能性は独立したD1で
gateする。exact-`H_D`外側公式とfragment-approximate-`H_D`外側公式を両方保存する。

### 4.2 D2 gate

`L_D=3,4`について次を全て満たす場合だけD2を通過する。

- fragment再構成Frobenius residual `<=1e-12`
- 全candidate/deltaのunitary defect `<=1e-10`
- target branch weight `>=0.9995`
- fragment内部誤差を含む候補が2個以上、主判断deltaで`1e-6 Ha`以内
- energy-only候補とtail-aware候補が異なる
- tail-aware候補のtail burdenがenergy-only候補より20%以上小さい
- tail-aware候補の実現stage proxyがenergy-only候補以下

tail burdenの主指標は、固定64 short-step・Taylor cutoff 2・`|b_j|`比例配分での
finite-RTE `log B_K`とする。`Gamma_R`、打切りbound、outer stage数、内部stage数を併記する。
tail-aware選択はfeasible候補内で

1. `log B_K`
2. summed truncation bound
3. `Gamma_R`
4. 実現stage proxy

のlexicographic最小とする。energy-only選択はabsolute energy bias最小とする。

## 5. D3：fresh holdoutでの選択差

`L_D=5`をfresh holdoutとし、D1とD2の実装・expected-task artifact・source hashを固定した後に
初めて候補比較を行う。D3は次を全て満たす場合に通過する。

- D1が通過済み
- `L_D=5`のfragment再構成、unitarity、target weight gateを通過
- energy tolerance内の候補が2個以上
- energy-onlyとtail-awareの選択が異なる
- tail burden reductionが20%以上
- tail-aware候補の実現stage proxyが増えない
- 両候補が`(energy bias, log B_K, realized stage proxy)`のPareto集合にある

fresh holdoutで選ばれる具体的な公式は固定しない。新4次以外がtail-aware候補になっても、
上記の選択差とgateを満たせばD3通過とする。

## 6. 総合判断

### Go

D1、D2、D3が全て通過した場合：

`advance_pd_to_formal_primary_candidate_then_stop_for_research_redesign`

とする。直後に計算を止め、次の順で研究方針を再設計する。

\[
\boxed{\text{研究RQ}\rightarrow\text{新規性監査}\rightarrow
\text{最小着地点}\rightarrow\text{必要な本検証}}
\]

### No-Go

- D1不通過：`stop_pd_signed_time_rte_not_validated`
- D1通過、D2不通過：`stop_or_narrow_pd_after_internal_hd_error`
- D1/D2通過、D3不通過：`stop_pd_selection_difference_did_not_transfer`

No-Go後にthreshold変更、同じH4 pathへの大量点追加、広いgrid探索でP-Dを救済しない。
P-Dを停止し、R3/R6/R8の問いを再定義する。

## 7. 証拠・実行規約

- D1 development diagnosticを除き、holdout数値を見る前にexpected-task artifactを生成する。
- expected-task artifactにはcandidate、delta、`L_D` role、substep、RTE cutoff/allocation、全gate、
  source/input hash、commit、dirty-worktree状態を保存する。
- taskごとの結果、失敗、再開状態を保存し、final artifactはexpected fingerprintを参照する。
- 結果を見た後にcandidate、threshold、holdout、substep、選択規則を変更しない。
- local dirty-worktree結果をimmutable CIまたは外部独立再現と呼ばない。
- 本検証の終了後はGo/No-Goにかかわらず新しい計算を自動開始しない。
