# P-D 主研究契約

日付: 2026-09-26  
状態: S0 固定済み、S1 実施前

## 1. 主研究課題

部分ランダム化 Hamiltonian simulation において、product formula (PF) ごとの
energy/phase error と randomized-tail burden を同じ物理時間・同じ位相誤差予算で
比較したとき、既知の簡易モデルで PF を十分に選択できる条件と、finite RTE・整数
配分・内部近似まで考慮しなければ選択を誤る条件を明らかにする。

研究目標は特定の PF を勝たせることではない。比較規則の十分性と破綻条件を同定する
ことを目標とする。

## 2. S1 の対象範囲

- 系: linear H4 chain、結合距離 1.0 Å、STO-3G
- sector: 8 qubits、4 electrons
- Hamiltonian 表現: 保存済み DF rank 12 snapshot
- split: 開発条件として `L_D=3`
- PF 候補:
  - second order
  - fourth order
  - new fourth order
  - Yoshida eighth order
  - Morales et al. eighth order
- 構成軸:
  - nested: `H_D` を内部 PF で近似し、外側で `H_D/H_R` PF を適用
  - native: fragment terms と `H_R` を一つの PF に直接入れる比較診断

native 構成は別の選択 baseline ではなく、構成上の直交軸として扱う。

## 3. 共通比較契約

- 共通物理時間: `T = 0.8 a.u.`
- 外側 step size: `delta in {0.1, 0.2, 0.4}`
- 外側 step 数: `q = T / delta in {8, 4, 2}`
- 共通総位相誤差予算: `beta_total = 8.0e-7 rad`
- nested の内部 substeps: `m_D in {8, 16, 32, 64}`
- outer step 当たりの RTE short-step 総数: `R in {16, 32, 64, 128}`
- primary finite cutoff: `K=2`
- decision-relevant sensitivity: `K=4`

energy error、operator/action error、signal phase error は直接加算しない。全候補を総時間
`T` における target signal の位相単位へ写し、外側 PF、内部 `H_D` 近似、finite RTE
の各成分を radian 単位で記録する。

## 4. 選択 baseline

- **B0**: deterministic phase bias 最小。tail cost は見ない。
- **B1a**: outer exponential-stage proxy 最小。
- **B1b**: realized component-action one-shot proxy 最小。
- **B2**: 既知の絶対 tail 時間 `Gamma_R`、連続配分、leading attenuation を含む。
- **B4**: finite cutoff、整数配分、exact paired normalization と有限 truncation bound を含む。

B2/B4 の比較目的関数は、共通の base-shot factor `N0` を除いた

`G_model = C_1shot * B_total^2`

とする。ここで `C_1shot` は B1b の解析 proxy、`B_total^2` は総時間 `T` に対する
shot inflation proxy である。これは compiled gate cost や実測 shot 数ではない。

## 5. S1 の判定量

PF 名の一致だけでなく、次を主判定量とする。

- B4 を参照した各 baseline の regret
- finite model で不適格な点を簡易 model が採用する false acceptance
- B2 の objective prediction error
- B2 の shot-factor prediction error
- 選択が探索上限・下限に達したか
- nested/native で結論が変わるか

decision-relevant regret の閾値は結果を見る前に `eta_decision = 0.05` と固定する。

## 6. S1 後の分類

- **Case A**: B1、B2、B4 が実質同じ判断を与え、regret が閾値以下、false
  acceptance がなく、未解消の境界依存もない。P-D は停止する。
- **Case B**: B1 と B2 は異なるが B2 と B4 は実質一致する。絶対 tail 時間 model
  で十分であり、新しい適用条件が得られない限り P-D を主題化しない。
- **Case C**: B2 と B4 が異なる、または有限補正により decision-relevant regret / feasibility
  error が生じる。finite-RTE-aware PF selection を主題候補とする。
- **Case D**: 主差分が internal `H_D` 精度または nested/native 構成にある。研究を
  inner/outer precision allocation または nested PF design に絞る。

分類不能な上限依存が残る場合は `undetermined_boundary` とし、A として停止しない。

## 7. S1 後の強制停止

S1 の結果を得た時点で必ず一度停止し、主 RQ、新規性、着地点、必要な本検証を再設計
する。S1 の結果だけを理由に S2 へ自動的に進まない。

この gate より前には H12、長時間 RPE、full compiled total cost、noise/backend、別分子、
大規模 `delta/r/L_D` grid を実施しない。

## 8. 主張しないこと

S1 は H4 の screening calculation であり、次を主張しない。

- H12 や一般分子への一般化
- 最終 RPE 総 cost の優位性
- compiled RZ/CX、wall time、実機 shot 数の優位性
- B4 bound の tightness
- native 構成の物理実装上の優位性

