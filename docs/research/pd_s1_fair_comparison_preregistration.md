# P-D S1 公平再最適化の事前登録

日付: 2026-09-26  
状態: preregistered、結果未参照

## 1. 固定入力

- Hamiltonian snapshot:
  `artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz`
- `L_D=3`
- formulas: second / fourth / new fourth / Yoshida eighth / Morales eighth
- constructions: nested / native
- `T=0.8`
- `delta={0.1,0.2,0.4}`、対応する `q={8,4,2}`
- nested `m_D={8,16,32,64}`
- `R={16,32,64,128}` per outer step
- primary `K=2`
- `beta_total=8.0e-7 rad`
- `eta_decision=0.05`

## 2. 位相誤差の統一

基準 signal は exact ground state `psi0` に対する

`z_exact(T) = <psi0|exp(-i H T)|psi0>`

とする。

- nested outer error: exact `H_D/H_R` outer PF を `q` 回適用した signal と基準 signal の位相差
- nested inner error: internal-approximated `H_D` を含む outer PF と exact-block outer PF の位相差
- native deterministic error: fragment terms と `H_R` の native PF を `q` 回適用した signal と基準 signalの位相差
- finite RTE error: conservative signal-error bound を対応する deterministic reference signal の radius で割り、定義可能なら `asin` で位相へ写す

各成分は radian で別々に保存し、feasibility 判定にはその保守的和を用いる。energy-equivalent
値を表示する場合も `beta/T` として二次的に導出する。

## 3. cost proxy

- `C_stage`: outer PF の全 exponential stage 数を `q` 倍したもの
- `C_1shot`:
  - nested: deterministic `H_D` occurrence ごとの fragment action を `m_D` と stage 数で展開し、
    tail occurrence ごとの RTE short-step action を加えた解析 proxy
  - native: deterministic fragment occurrence と tail RTE short-step actionを加えた解析 proxy
  - B4 では有限分布が要求する sampled component action の期待数を各 short step に反映する。
    B2 は一 short step 一 action の leading proxy とし、この差も finite-model prediction error に含める
- B2/B4 objective: `G=C_1shot*shot_factor`

共通 base-shot factor は全候補に共通なので省略する。

## 4. B2

B2 は continuous proportional allocation と leading paired normalization を使う。

- total leading log-normalization:
  `ell_B2 = q * lambda_R^2 * delta^2 * Gamma_R^2 / R`
- `shot_factor_B2 = exp(2*ell_B2)`
- B2 feasibility: deterministic phase bound が `beta_total` 以下
- B2 objective: `C_1shot * shot_factor_B2`

B2 は finite truncation error を 0 と置く。この仮定により false acceptance が起きるかを
B4 に対して測る。

## 5. B4

B4 は各 outer step で `R` を tail occurrence へ deterministic largest-remainder 法で整数配分
する。各 occurrence には少なくとも 1 short step を割り当てる。配分不能な点は infeasible。

paired order weight と normalization は実装済み RTE 定義に一致させ、各 occurrence の exact
`log B` と有限 truncation bound を計算する。

- total exact log-normalization: outer-step 値を `q` 倍
- `shot_factor_B4 = exp(2*ell_B4)`
- total finite signal-error bound: occurrence bound の和を `q` 倍した保守的 bound
- finite phase bound: `asin(error_bound/reference_radius)` が定義できる場合のみ採用
- B4 feasibility: deterministic phase bound + finite phase bound が `beta_total` 以下
- B4 objective: `C_1shot * shot_factor_B4`

## 6. K=4 感度規則

K=2 の全 primary grid を評価した後、次の deterministic union に限って K=4 を評価する。

1. 各 construction で B2 が選んだ formula と deterministic setting
2. 各 construction で provisional B4(K=2) が選んだ formula と deterministic setting
3. K=2 では finite feasibility を一つも持たない formula/construction について、
   その formula 内で B2 が選ぶ deterministic setting
4. B2 または B4 の選択が `R=128` に達した formula/setting

trigger された deterministic setting については全 `R` grid を K=4 で再評価する。このため
K=4 の結論は限定感度解析であり、全 grid の K=4 exhaustive comparison とは呼ばない。

## 7. 境界規則

primary optimum が `m_D=64` または `R=128` に達し、case 分類または regret に影響し得る場合、
同じ formula/construction/delta/K に対して一段だけ `m_D=128` または `R=256` を評価する。

一段延長後も上限点が選択され、結論が変わり得る場合は `undetermined_boundary` として停止する。
下限点は物理的離散制約または登録 grid の下限として記録し、結果後に下限を拡張しない。

## 8. 選択と regret

各 baseline は、まず自身の feasibility 規則を満たす候補から objective を最小化する。tie は
`objective, deterministic_phase, C_1shot, formula_id, delta, m_D, R` の順で決める。

有限 model の reference optimum `G_ref` は評価済み B4 feasible set の最小値とする。各 baseline
が選んだ点を B4 で再評価し、

`regret = G_B4(selected)/G_ref - 1`

を計算する。B4 infeasible な点を選んだ場合は false acceptance とし regret を infinity とする。

## 9. 事前固定した分類規則

- A: B1/B2/B4 の選択が同一または B4 regret 5% 以下、false acceptance なし、model error が
  decision-relevant でなく、未解消境界なし
- B: B1 の B4 regret が 5% を超える一方、B2 regret は 5% 以下で false acceptance なし、
  K=4 sensitivity でも維持
- C: B2 regret が 5% を超える、B2 false acceptance、または K=2/K=4 の finite correction が
  decision-relevant な選択差を生む
- D: nested/native または `m_D` の最適配分が主差分であり、同一構成内の B2/B4 差より大きい

A–D の複数条件を満たす場合は `C > D > B > A` の優先順位で一次分類し、併存要因を別欄に残す。
未解消境界がある場合は一次分類に加えて `undetermined_boundary` を付け、GO 判定を出さない。

## 10. 強制停止

結果 artifact、検証文書、manifest、研究概要を更新した時点で S1 を終了する。S2、H12、長 RPE、
compiled total cost へは進まない。

