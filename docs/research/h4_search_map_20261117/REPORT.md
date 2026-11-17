# H₄ 保存探索マップ

[閲覧用HTML（保存してブラウザで開く）](index.html) ／ [機械可読索引](INDEX.json) ／ [固定入力](SOURCES.json) ／ [運用・公開手順](HANDOFF.md)

Full PF・B0I・PR-Pauli・PR-DFについて、公開した探索対象と実評価を表示する。
結論の範囲は**収録した保存候補集合内**。ロードマップとは別のcoverage成果物であり、新しい科学資格・最安幅・winnerを採択しない。

|方式|Q3評価済み構成|評価配分（重複なし）|保存判定が適合|価格接続済み適合点の最安RZ|
|---|---:|---:|---:|---:|
|FULL_PF|4|256|33|480,258,064.00|
|TRUNCATED_PF|4|14|11|372,035,410.00|
|PR_PAULI|27|4592|1579|410,199,237.00|
|PR_DF|5|15|11|1,235,340,731.91|

公開controller summaryで評価された配分を数える。exact原本や価格のC側受入が不足する点は、
summary由来の表示と元SOURCEを残し、最安価格の選択には昇格しない。共通snapshot・比較条件はINDEX.comparison。

同一構成・配分・評価契約の再同期や別run再利用は累計で1件。新規表示は元NEW_Q3_JOBかcontroller自身の取得記録がある点だけ。
要求幅の厳密表現を保持し、binary64 0.2と厳密1/5を丸めて統合しない。unknownはunknownのまま。

|収録run|停止・進行状態|そのrunの新規評価|実行host（公開metadata）|固定結果|
|---|---|---:|---|---|
|H4_PR_PAULI_KAPPA_ABS_2_NEIGHBOURHOOD_R02|{"PR_PAULI": "STEP_LIMIT"}|40||[683241106](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/683241106ce00ea64c7a877281861654c0b5e7c7/docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round02/run01/ROUND_RESULT.json)|
|H4_PR_PAULI_DF_AUTOMATIC_PREPARATION_20261116_RUN02|{"PR_PAULI": "Q3_BUDGET_EXHAUSTED", "PR_DF": "Q3_BUDGET_EXHAUSTED"}|2||[0ef9668e9](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef9668e997b57214eeab2f944e4bcff967b7b4d/docs/research/d_pr_candidates_auto_20261116/run02/ROUND_RESULT.json)|
|H4_PR_BOTH_MULTIWIDTH_OPTIMIZATION_R06_20261117|"RUNNING_WITH_RECORDED_TECHNICAL_HOLDS"|0|GPUA100-gleap02|[b0257d780](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b0257d78026ade89bb3ba61dc88634570039dae8/docs/research/h4_pr_multiwidth_optimization_20261117/SCOPE_METADATA.json)|
|H4_PR_PAULI_KAPPA_ABS_2_NEIGHBOURHOOD_R03|{"PR_PAULI": "ONLY_SMALL_OR_DEFERRED_LEFT"}|2||[d7a976a62](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d7a976a629a6db68197b24f8f86734e109c8bf5d/docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round03/run01/ROUND_RESULT.json)|
|H4_PR_PAULI_KAPPA_ABS_2_SCHEDULE_R04|{"PR_PAULI": "ONLY_SMALL_OR_DEFERRED_LEFT"}|82||[e3b4c1114](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3b4c1114139258a2c6e5d3e14de8b73fc4f7cda/docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round04/run01/ROUND_RESULT.json)|
|D_FULL_PF_S2_2H_Q3_ONE_V1|{"FULL_PF": "Q3_BUDGET_EXHAUSTED"}|0||[55c174a3f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/55c174a3f04f560a57bd30abd19b1197b2edea67/docs/research/d_full_pf_s2_2h_q3_e2e_20261116/first_snapshot/ROUND_RESULT.json)|
|D_FULL_PF_S2_2H_Q3_PROPOSAL_ONLY|{"FULL_PF": "NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET"}|0||[90a52a211](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/90a52a211f6acc4d7b01e1017572c15224ac25de/docs/research/d_full_pf_s2_2h_q3_e2e_20261116/proposal_only/ROUND_RESULT.json)|
|D_FULL_PF_S2_2H_Q3_ONE_V1|{"FULL_PF": "Q3_BUDGET_EXHAUSTED"}|1||[feee120da](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/feee120daa61a68741e446535b7f34a35121522d/docs/research/d_full_pf_s2_2h_q3_e2e_20261116/run01/ROUND_RESULT.json)|
|D_PF_SAVED_REGISTRY_RETURN_02|{"FULL_PF": "NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET"}|0||[c94f290d5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c94f290d5fb1f14e244fb4f87903b58df6d27690/docs/research/d_pf_on_demand_evidence_20261116/optimizer_round02/ROUND_RESULT.json)|
|B_H4_FULL_PF_FINITE_DESCENT_20261116_RUN01|{"FULL_PF": "NO_IMPROVING_PROPOSAL_INPUTS_MISSING"}|244||[4b28d35da](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b28d35da197c7a8dcde0a8d60a6a0f9ed917d75/docs/research/h4_pf_optimizer_search_20261116/run01/FULL_PF/ROUND_RESULT.json)|

## 未取込・未確認の範囲

- STATUS_REPORTS_OUTCOMES_BUT_PER_ALLOCATION_CONTROLLER_RESULT_NOT_PUBLISHED：`{'commit': 'b0257d78026ade89bb3ba61dc88634570039dae8', 'path': 'docs/research/h4_pr_multiwidth_optimization_20261117/SCOPE_METADATA.json', 'selector': []}`

最新公開scopeの進捗件数だけからshot配分やQ3点を捏造しない。各マスで「対象・未評価」「評価済み適合0」「元記録未取込」を区別。
新規信号・sampling・compile・fit・Q3は全0件。既存の10分同期で更新する。検査と公開方法はHANDOFFを参照。
