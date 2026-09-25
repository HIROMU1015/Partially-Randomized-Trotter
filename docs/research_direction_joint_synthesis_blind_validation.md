# P-A v1 blind transfer validation

最終更新：2026-09-25 JST

## 結論

事前登録した2 stratumを完了し、両方で6 gateを全て通過した。固定P-A v1の`interval_union_dp`は、
現行`support_run_le_1`に対してpooled RZを

- 未使用H5 physical snapshotで17.076%削減
- 元H4 event streamのQiskit optimization level 2で6.598%削減

した。operator最大残差はそれぞれ$2.998\times10^{-15}$、$3.126\times10^{-15}$で、relative ancilla
phaseも一致した。従ってdecisionは

`advance_pa_v1_to_formal_primary_theme_candidate`

である。これはblind gateだけに基づく当時の判断である。後続の
[v1形式化・mechanism監査](research/pa_joint_synthesis_v1_formalization.md)では、全54 recordで
run内部分割が0件、holdoutの非零Taylor orderも0件と判明したため、現在は非退化mechanism検証待ちの
条件付き候補となった。さらに後続の
[非退化mechanism validation](research_direction_joint_synthesis_mechanism_validation.md)ではforced-support order-2の
30/30 taskで一区間baselineとの差がなく、固定規則どおりP-A interval DPを主候補から外してP-Cへ戻った。
ただし、blindの数値結果とgate通過は変わらない。これは世界初の手法、全Gaussian回路での最適性、
coupling/noise下の優位性、full partial-$S_2$／RPE総costまたは科学的優位性を確立しない。

## 事前登録とleakage管理

正本は
[P-A v1 blind transfer validation事前登録](research/pa_joint_synthesis_blind_validation_preregistration.md)
である。本計算前に48 holdout taskと6 operator probeを固定し、expected-task bundleのcontent fingerprintを
`40490bd435b1286b59213559bd0296649a79e190ffc99025450089080e335e05`とした。H4 holdout event streamは
元pilot digestと一致した。

固定したP-A v1 source、元pilot artifact、snapshot、seed、4 policy、primary metric、6 gateを変更していない。
実行中に次の2件のimplementation-only failureがあった。

1. dry-runと本実行のcommand文字列差をscientific plan差と誤認し、0 checkpointで停止した。
2. H5 24 compile後、operator関数の引数順誤りで停止した。operator resultとgate集計前だった。

いずれもevent、candidate、thresholdまたはsample数を変更せず修正し、fingerprint済みcheckpointから再開した。
部分checkpointの数値を使った方針変更は行っていない。

## 固定条件

### Stratum A：H5 physical transfer

- H5 linear chain、1.0 Å、STO-3G、10 qubit、charge +1、multiplicity 3
- DF rank 9、`L_D=4`、short-step time 0.025、finite Taylor cutoff 2
- 列長3、5、8を各8 trajectory、fresh master seed `2026092601`
- Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなし

### Stratum B：H4 compiler transfer

- H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、`L_D=3`
- `delta=0.02`、finite Taylor cutoff 2
- 元pilotと同じ列長3、5、8、各8 trajectoryおよび同じ3 operator probe
- Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 2、seed 17、coupling mapなし

両stratumとも`full_basis_shared`、`event_support_restricted`、`support_run_le_1`、
`interval_union_dp`を同じevent列で比較した。

## 結果

### Primary gate summary

| stratum | pooled RZ変化 | 最大trajectory悪化 | 4-policy oracle regret / full RZ | plan変更 | multi interval | union interval | operator最大残差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H5 physical | -17.076% | 0.000% | 0.000% | 23/24 | 23/24 | 24/24 | $2.998\times10^{-15}$ |
| H4 opt2 | -6.598% | +0.265% | 0.0116% | 21/24 | 22/24 | 24/24 | $3.126\times10^{-15}$ |

両stratumで、operator同値性、pooled RZ 2%以上改善、個別悪化5%以内、oracle regret 1%以内、
plan変更率20%以上、multi/support-union interval観測の6 gateを全て通過した。

### 列長別RZ

| stratum | 長さ3 | 長さ5 | 長さ8 |
|---|---:|---:|---:|
| H5 physical | -22.636% | -16.034% | -15.115% |
| H4 opt2 | -10.096% | -9.790% | -2.447% |

H5では全長で15%以上、H4 opt2でも全長で2%以上改善した。長い列ほど相対改善が小さくなるH4傾向は
元opt1 pilotと共通するため、今後の理論化ではrun長、support union、source-basis切替の分布を説明変数として
扱う必要がある。

### 補助metric

| stratum | RZ | CX | RZ depth | CX depth | total depth | circuit size |
|---|---:|---:|---:|---:|---:|---:|
| H5 physical | -17.076% | -13.024% | -1.539% | -0.838% | -1.923% | -16.693% |
| H4 opt2 | -6.598% | -5.546% | -0.314% | -0.061% | -0.304% | -6.901% |

RZとcircuit sizeの改善は明瞭だが、depth改善は小さい。従って「全回路指標を同程度に改善する」とは
主張しない。

## 完全性とprovenance

- expected／completed：holdout 48/48、operator probe 6/6
- missing／extra／duplicate checkpoint：0
- final content fingerprint：
  `78af3474898dbf989780ea5f2881cb5b61595609dd2698164b9846c1ce1c5919`
- final file SHA-256：
  `ff6a8f846795b3f56e3688c62eab3ad26c3ace6632063ba72ff97f4a12ac4ea3`
- Qiskit 1.3.0、NumPy 1.26.4、Python 3.11.0rc1
- 基点commit：`3336f030c56f4eb500b1bdf8f541619f9c445038`
- evidence status：local dirty worktree、外部再現／immutable CIなし

artifact：
`artifacts/research_direction_joint_synthesis_blind_validation/2026-09-25/pa_v1_h5_physical_h4_opt2_blind_v1.json`

## 検査

- blind validation専用test：`4 passed`
- 元P-A pilot・theme-selectionを含む関連test：`10 passed`
- library、runner、testの`py_compile`：通過
- `scripts/check_validation_manifest.py`：`Validation manifest OK`
- manifest JSON構文および`git diff --check`：通過

全suiteの再実行は行っていない。上記はlocal dirty worktreeでの関連回帰検査である。

## 研究判断と次工程

blind gate時点では、未使用physical snapshotとcompiler levelへの移送に基づきP-Aを正式候補へ進めた。
後続形式化により、実測改善はrun-level full/support-union選択までは支持するが、interval分割層を
one-segment baselineから区別しないと分かった。後続の事前登録済みmechanism判別でも差がなかったため、
P-Aのinterval claimを停止し、現在はP-Cを主研究候補とする。run-level policyの既存結果は保持する。

H12、長RPE総cost、追加$q>32$、coupling/noise、full wrapperはこの直後の必須作業ではない。
