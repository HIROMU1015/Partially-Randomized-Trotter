# P-D現実化 Go/No-Go gate

実行日：2026-09-25 JST

## 1. 結論

事前登録したD1、D2、D3は全て通過した。総合statusは

`advance_pd_to_formal_primary_candidate_then_stop_for_research_redesign`

である。P-Dを正式な主研究候補として研究RQ・新規性・最小着地点・必要な本検証の再設計へ進める。
ただし本検証の停止規則どおり、H12、長RPE、full総cost、backend/noise、大規模gridなどの追加計算は
自動的に開始しない。

## 2. 固定条件

- H4 linear chain、1.0 Å、STO-3G、8 qubit、4 electron、DF rank 12
- sector dimension：70
- PF候補：二次、標準4次、新4次、Yoshida 8次、Morales 8次
- 診断delta：0.2、主判断delta：0.4
- energy tolerance：`1e-6 Ha`
- minimum target branch weight：0.9995
- finite-RTE Taylor cutoff：2、tail short-step総数：64
- `H_D`内部二次対称PF：各outer `H_D` occurrence当たり32 substep
- development：`L_D=3`
- internal-`H_D` transfer：`L_D=4`
- fresh holdout：`L_D=5`

期待タスクはD1負時間14件とD2/D3候補15件の計29件。v2 expected fingerprintは
`8924d637e52b03900f32e2f167e63593cf9729b4b78d4dee3af87fca661183f4`である。

## 3. D1：負時間finite-RTEとcontrolled phase

固定5公式に含まれる負のtail occurrence 14件を全て評価した。identity項と非可換なX/Y/Z成分を持つ
2次元oracleで、signed timeを保持したfinite distributionを全列挙し、同じ有限Taylor演算子、
正時間側のadjoint、`diag(I,U)` controlled block、identity relative phase、fixed-seed sampled meanと照合した。

- completed / failed：14 / 0
- ordinary oracle residual最大：`3.3314e-16`
- signed adjoint residual最大：`0`
- controlled residual最大：`6.6613e-16`
- identity relative-phase residual最大：`3.4694e-18 rad`
- sampled mean最大絶対誤差：`5.1531e-4`
- sampled mean最大standardized residual：`2.4874`

固定閾値を全件で満たし、D1は通過した。これはdense小行列の意味論検証であり、Qiskitのcompiled
controlled回路検証ではない。

## 4. D2/D3：fragment内部H_D誤差と選択差

各`L_D`について累積`H_D(k)`の差からone-body/correctionと順序付きDF fragmentのdense termを作り、
outer PF内の各`exp(-i a_j H_D delta)`を固定32-substepの内部二次対称PFへ置き換えた。energy比較中の
各`H_R` occurrenceはexact exponentialとした。

主判断delta 0.4の結果は次のとおり。

| L_D | 役割 | energy-only | bias (Ha) | tail-aware | bias (Ha) | log B_K削減 | stage proxy | gate |
|---:|---|---|---:|---|---:|---:|---:|---|
| 3 | development | 8th(Morales) | 2.4554e-7 | 4th(new_2) | 6.0206e-7 | 81.634% | 4096 → 1408 | pass |
| 4 | internal-H_D transfer | 8th(Morales) | 2.7573e-7 | 4th(new_2) | 3.2495e-7 | 81.638% | 5248 → 1792 | pass |
| 5 | fresh holdout | 8th(Morales) | 2.7762e-7 | 2nd | 9.2982e-7 | 96.580% | 6400 → 768 | pass |

fragment再構成Frobenius residualは3 splitとも0。全候補・deltaの最大unitary defectは
`1.4382e-11`、最小target branch weightは`0.9999959245`で固定gate内だった。`L_D=3,4`では
新4次がtail-awareに残り、fresh `L_D=5`では二次へ変わったが、事前登録は具体的なtail-aware公式を
固定せず、energy-onlyとの選択差、20%以上の負担削減、stage非増加、Pareto非劣位をgateとしていた。
従ってD2とD3はいずれも通過する。

## 5. 判断と限界

現実化後も「energy biasだけでPFを選ぶ候補」と「random-tail負担を含めて選ぶ候補」は一致しなかった。
この結果は、P-Dの中心仮説を次の研究RQ候補へ進める根拠になる。

> 部分ランダム化Hamiltonian simulationでは、固有値精度だけでなくrandomized tailの絶対時間、
> normalization、sampling burdenを含むPF選択・設計指標が必要か。

一方、次は未評価である。

- D1のcompiled Qiskit controlled回路
- H4 full sampled finite-RTE operator
- `H_R` samplingと内部`H_D`誤差を同時に含むend-to-end operator
- compiled depth・hardware cost・状態準備・long RPE・最終総cost
- H12、別分子、backend/noise、全PF familyでのglobal optimality
- 科学的優位性または世界初性

`realized_stage_proxy`は固定したoperation-count proxyで、compiled depthや最終総costではない。

## 6. 実行・artifact監査

最初のv1実行は、旧P-D表に存在しないdelta 0.2を参照したため`KeyError`で結果artifact生成前に停止した。
タスク、seed、閾値を変更せず、finite-RTE burdenを固定式から直接評価するよう技術修正した。
v1 expected artifactは履歴として保持し、修正版source hashをv2 expected artifactへ非上書きで再固定した。

- preregistration commit：`014698f`
- technical-fix / v2-freeze commit：`087a984`
- full computation commit：`087a984c78e9d0e8389544b1f47a306374917a9a`
- elapsed：3.758668秒
- Python 3.11.0rc1、NumPy 1.26.4、SciPy 1.14.1
- expected artifact：`artifacts/research_direction_pd_realization/2026-09-25/pd_realization_expected_tasks_v2.json`
- expected fingerprint：`8924d637e52b03900f32e2f167e63593cf9729b4b78d4dee3af87fca661183f4`
- result artifact：`artifacts/research_direction_pd_realization/2026-09-25/pd_realization_go_no_go_v2.json`
- result fingerprint：`805a17f95497a4d61286748a126c01b1235fbe0d987528be86ea3938700b9ede`
- result file SHA-256：`59e8019805e429280a5f6338e38a1cbfd803ae8b51c30e87a965a2ef3dcb694d`
- 専用test：`4 passed`
- 全suite：`606 passed, 2 skipped, 4 warnings`、失敗0

計算開始前のworktree statusは空で、source/input hashはartifactへ保存した。ただしlocal resultであり、
immutable CIまたは外部独立再現とは呼ばない。

## 7. 次の停止点

ここで計算を止める。次は数値gridを広げることではなく、

1. 研究RQ
2. 既存のeigenvalue-optimized PF、near-integrable PF、randomized/partially randomized simulation、
   negative-time PF、cost-aware PF optimizationに対する新規性
3. 修士研究・論文としての最小着地点
4. その主張に本当に必要な最小検証

を再設計する。新規性が残った場合だけ、別`L_D`、別小系/geometry、別PF family等の最小一般化を
新しい事前登録の下で行う。
