# P-A：DF回路列のinterval-aware joint synthesis pilot

最終更新：2026-09-25 JST

## 結論

固定H4条件の未使用event列では、同じsource basisが続くrunを区間へ分け、区間ごとにfull basisまたは
support unionを保存するbasisを選ぶdynamic programming（DP）が、現行`support_run_le_1`より
pooled RZを7.194%減らした。未使用列長3、5、8の全てで平均RZが改善し、24 trajectory中21件で
現行policyと異なるbasis列を選んだ。事前固定した6 gateを全て通過したため、P-Aを主研究候補へ残す。

ただし、これはH4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、`L_D=3`、
`delta_time=0.02`、finite Taylor cutoff 2、topology-free Qiskit 1.3.0 optimization level 1だけの
local dirty-worktree結果である。広い回路合成文献に対する新規性、coupling map、backend/noise、
full partial-$S_2$ wrapper、RPE総costまたは科学的優位性は確立していない。

## 問いと比較契約

P-Aの問いは、既存のbasis共有・support限定・隣接basis cancellationを入れた後にも、event列全体の
support/run構造から追加のcompiled-cost削減を予測できるかである。全候補で隣接basis cancellationを
有効にし、次の4 policyを同じevent列で比較した。

1. `full_basis_shared`：元のfull Gaussian basisをrun内で共有する。
2. `event_support_restricted`：各eventを個別のsupport限定basisに置換する。
3. `support_run_le_1`：WP06-bで固定した現行policy。singleton runだけsupport限定にする。
4. `interval_union_dp`：各source-basis runを連続区間へ分割し、区間ごとにfull basisまたは
   区間内support unionを保存するcompletionを選ぶ。

DPの古典目的は、basis変換を前後に適用するoperation数の2倍を最小化し、同点なら区間数、
support union sizeの順に小さい案を選ぶ。compiled oracleは上記4候補のRZ最小値であり、全ての可能な
量子回路に対するoracleではない。

## 固定条件とholdout

- physical instance：H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12
- partial split：`L_D=3`
- RTE条件：`delta_time=0.02`、finite Taylor cutoff 2
- compiler：Qiskit 1.3.0、`rz/sx/x/cx`、optimization level 1、seed 17、coupling mapなし
- training diagnostic：列長2、4、6、各6 trajectory、seed 2026092501
- independent holdout：列長3、5、8、各8 trajectory、seed 2026092502
- operator probes：各holdout長1件、独立seed 2026092503

trainingは実装診断に使用したが、合否は24件の独立holdoutと3件のoperator probeで決めた。

## 事前固定したpilot gate

次を全て満たすときだけP-Aを研究候補へ残す。

1. operator最大残差が$10^{-10}$以下でrelative ancilla phaseが一致する。
2. pooled RZが現行policyより2%以上少ない。
3. 各trajectoryのRZ悪化が現行policy比5%以下である。
4. 4 policy内compiled oracleに対するpooled RZ regretがfull-basis RZの1%以下である。
5. holdout trajectoryの20%以上で現行policyとbasis選択が変わる。
6. multi-application intervalとsupport-union intervalが実際に選ばれる。

## 結果

| policy | 平均RZ | 現行比 | 平均CX | 平均total depth | 平均circuit size |
|---|---:|---:|---:|---:|---:|
| full basis共有 | 864.58 | +22.46% | 292.00 | 444.96 | 1,729.25 |
| eventごとsupport限定 | 962.67 | +36.35% | 343.92 | 748.92 | 1,937.75 |
| 現行`support_run_le_1` | 706.04 | 0% | 250.17 | 418.25 | 1,414.38 |
| interval-union DP | 655.25 | -7.19% | 237.00 | 416.38 | 1,302.83 |

列長別のinterval-union DPのRZ変化は、現行policy比で長さ3が-10.650%、長さ5が-11.255%、
長さ8が-2.522%だった。その他の主要診断は次のとおりである。

- 最大個別trajectory RZ悪化：+0.231%
- 4 policy内oracle regret / full-basis pooled RZ：0.00482%
- 現行policyからbasis列が変化：21/24 trajectory（87.5%）
- multi-application intervalを使用：22/24 trajectory
- support-union intervalを使用：24/24 trajectory
- operator最大残差：$3.126\times10^{-15}$

全6 gateが通過した。eventごとのsupport限定が現行比36.35%悪化した一方、run全体のsupport unionを
使うDPが改善したため、利益は「supportを小さくすれば常に良い」ことではなく、basis transition costと
run/support構造の共同選択に依存する。

## 判断と限界

P-Aは3 pilot中で最も強い直接効果を強いproject baselineに対して示し、未使用列長でも同じ方向を
再現した。このため暫定主題とする。ただし、現在のproject内baselineに対する差分と、広い量子回路合成・
fermionic Gaussian circuit最適化文献に対する新規性は別である。まず先行研究・新規性監査を行い、
通過した場合だけ未使用snapshotまたはcompiler contextのblind holdoutへ進む。

このpilotではproduction defaultを変更していない。H12、長RPE、追加$q>32$、総cost計算は次の
必須作業ではない。

## 証拠と再実行

- artifact：
  `artifacts/research_direction_joint_synthesis_pilot/2026-09-25/pa_h4_interval_union_joint_synthesis_v1.json`
- content fingerprint：
  `1a9840a4ee46daa6e3749272593acf3e8ae29be9fcecc1a05b0bd9ea817fbc37`
- implementation：`src/trotterlib/research_direction_joint_synthesis_pilot.py`
- runner：`scripts/run_research_direction_joint_synthesis_pilot.py`
- test：`tests/test_research_direction_joint_synthesis_pilot.py`
- 専用test：`3 passed`
- 全suite：`582 passed, 2 skipped, 4 warnings`、失敗0

artifactには入力snapshotとWP06-b artifactのSHA-256、seed、compiler条件、source hash、生成時commit、
dirty-worktree状態を記録する。
