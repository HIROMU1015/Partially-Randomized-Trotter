# m17の不一致：保存境界と丸め箇所の静的整理

2026-11-03 JST。**準備・SOURCE調査のみ。原因は未確定、現在の原因判定はINSUFFICIENT_EVIDENCE。**
実H4行列の再累乗、任意精度再評価、新しいmean・差・誤差上界の計算は行っていない。
以下の実H4の数値は元RESULTからの引用であり、再計算値ではない。

## 1. 固定された元結果

対象はlinear H4/隣接1.00 Å/STO-3G/保存legacy DF12、8 system qubit、4電子sector70、
Bのphysical ground state、Partial L_D3、単一δ=.02、m17/q131072/r32/K2。
coefficient_atol=1e-12、extract_identity_phase、元のN/α/β/epsilon・符号規約を保持する。

- 結果commit：`cd749066904d3f29b28b03c19ab69370372b157f`。
  [元報告](../rpe_m17_mean_validation_execution_20261103/REPORT.md)、
  [RESULT](../../../artifacts/rpe_m17_mean_validation/2026-11-03/run01/RESULT.json)。
- 科学実行SOURCE：`128e9a5bf2630cd0333ec5f69b81b25def78eb79`。
  [runner](../rpe_m17_mean_validation_preparation_20261103/run_m17_mean_validation.py)、
  [元固定計画](../rpe_m17_mean_validation_preparation_20261103/SCIENTIFIC_EXECUTION_PLAN.md)、
  [実装報告](../rpe_m17_mean_validation_preparation_20261103/IMPLEMENTATION_REPORT.md)、
  [人工13 tests](../rpe_m17_mean_validation_preparation_20261103/test_m17_mean_validation.py)、
  [検査記録](../rpe_m17_mean_validation_preparation_20261103/VALIDATION.json)。
- B state：commit `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`。
  V2：`78c49d0b5f1d71e63b5959e78813237923bbc648`。
  元library carrier：`4591b8d75a2a6a55c74a51225f0c61648023c8ee`。
  結果carrierと科学実行SOURCEを混同しない。

| 比較 | 保存complex残差 |
|---|---:|
| law対reference | 6.259420298065069e-10 |
| law対保存μ | 5.688626584284386e-10 |
| reference対保存μ | 8.21305022751065e-11 |

固定1e-10の数値診断はINCONSISTENT、実行はCOMPLETED_MANDATORY_STOP。
保存された非gating診断はA差4.795333763004963e-16、tail差5.983045574440441e-14（Frobenius）。
これだけから、最終差への寄与を定量化・特定していない。
既存の統計予算残余約8.60e-5と診断閾値1e-10は別用途である。

## 2. 保存NPZの実名と、まだ保存されていないもの

[operators.npz](../../../artifacts/rpe_m17_mean_validation/2026-11-03/run01/operators.npz)のbyte hashとNPY headerだけを確認した。
7個とも70x70、dtype `<c16`（complex128）。準備ではnumeric payloadを配列としてdecodeしていない。

| 指示中の略称 | 実際のarchive key | 意味 |
|---|---|---|
| A_prob | A_probability | 確率重みで構成した保持tail平均 |
| A_coeff | A_coefficient | 保持係数/lambda_Rの平均 |
| F | forward | deterministic forward half |
| R | reverse | deterministic reverse half |
| reference_tail | reference_tail | Taylor経路の、内側rを合成・attenuation適用済みのtail平均 |
| law_one_event | law_event_mean | literal-lawの一event平均 |
| law_tail | law_tail_mean | law_event_meanを内側r乗したtail平均 |

元のpublic keyやファイルをrenameしていない。
**S=R M_tail Fそのものは元NPZにない。** Task AでSを使うには、保存F/R/tailから一度組み立てる必要がある。
この新しいfloat64 Sを別の結果領域へ保存して固定すれば、その後のq乗と期待値の丸めを切り分けられる。
最初から高精度でR M_tail Fを組む場合とは、比較している入力境界が異なる。

またNPZには成分ごとの確率・係数・basis metadataの一覧はない。
RESULTにはorder確率・tau・B・lambda_R等があるが、各orderの実際のfloat64 rotation角やcos/sin値はない。
Task Bの全部を、これらの7行列だけから再現できるとは扱わない。

## 3. SOURCEで確認した演算順と丸め箇所

元の`grouped_averages`では、probability側は`math.fsum`した確率和で再正規化し、
coefficient側は抽出係数/lambda_Rを用いる。同じbasisを共有して行列へ共役する。
確率の表現、除算、diagonal総和、basis変換の行列積、共役の丸めが入る。
保存された確率和は.9999999999999999だが、その差の最終meanへの寄与は未評価。

`literal_law_mean`は、order確率pi_nを使い

    sum_n pi_n (-1)^(n/2) (cos(theta_n) I - i sin(theta_n) A_prob) A_prob^n

を構成する。theta_n=atan(tau/(n+1))、orderは0/2。scalarの除算・atan/cos/sin、
行列のn乗、積と加算で丸めが入り、その後にr32乗がある。

coefficient側は既存[finite_rte_operator_moments](../../../src/trotterlib/rte.py)を使う。
paired-Taylorはdegree K+1までで、corrected一eventのr乗を`exp(r*log(B))`で割る。
Taylor係数、積の順、r乗、log/exp、除算の丸めがliteral-law側と異なる。
数式上の同値性からfloat64でのbitwise一致は導けない。

両側とも最後は`step=reverse @ tail @ forward`（左結合）、
`np.linalg.matrix_power(step,q)`、stateへの作用、`np.vdot(psi,value)`を行う。
この後段でも丸めが入り、長qの影響を受け得る。原因・大きさは未計算である。

## 4. 切り分ける入力境界

| 検証 | 固定するもの | 高精度にする部分 | 分かる範囲 |
|---|---|---|---|
| A-step | 新しい基準計算で得たS64と保存psi64を正確なdyadic入力とする | q乗・作用・期待値 | S確定後の後段丸めの影響 |
| A-factor | 保存F64/R64/tail64/psi64を正確なdyadic入力とする | Sの組立てから後段まで | post-archive step組立ての寄与も含む |
| B-1 | 境界を定めたfloat64係数・角度等 | 平均tailの構成以後 | 固定scalar以降の構成丸め |
| B-2 | 元数式と、その定義上の入力 | 係数・確率・角度・normalizationも再評価 | float近似入力と数式定義の差。B-1と別実験 |

float64値を高精度へ持ち上げても、その値が生成される前に失われた桁は復元されない。
保存stateも再正規化せず、元と同じbinary64成分を使う。
S64固定とF/R/tail固定の違いを「外側q乗だけ」の効果にまとめない。

## 5. 現在の候補と限界

| 候補label | 静的な根拠・留保 |
|---|---|
| POWER_ROUNDOFF_PLAUSIBLE | 後段q乗・作用・期待値で誤差が入る経路がある。実H4で寄与は未評価 |
| CONSTRUCTION_ROUNDOFF_PLAUSIBLE | scalar・行列構成・内側r乗・S組立てに別の丸めがある。どの段階か未確定 |
| IMPLEMENTATION_OR_DEFINITION_DIFFERENCE_POSSIBLE | 精度を上げても固定入力間の差が残る可能性。丸めで生成された入力差との区別も未解決 |
| INSUFFICIENT_EVIDENCE | **現在の結論**。静的確認や小さい演算子差だけでは原因を確定できない |

数値実装の一致・収束は、真の周辺平均への数学的誤差上界ではない。
既存numeric_diagnostic_status=INCONSISTENT、u_num_bound=null、RPE_tolerance_certified=false、
scientific_status=UNDETERMINEDを維持する。
次は[Task Aの固定計画](PRECISION_ISOLATION_PLAN.md)。B-1/B-2へ無条件には進まない。
