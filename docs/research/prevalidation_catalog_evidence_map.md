# 事前検証カタログ：実施内容とGitHub証拠の対応表

最終更新：2026-09-25 JST

この文書は、[`partial_randomized_trotter_prevalidation_catalog.md`](../../partial_randomized_trotter_prevalidation_catalog.md)に載っている候補のうち、研究方向screeningとして実際に検証した範囲を、GitHub上の結果文書、machine-readable artifact、runner、testへ結び付ける索引である。カタログは144項目の計画を含むため、掲載されていること自体を「実施済み」と解釈しない。

## 1. 読み方

1. 現在の全体判断は[研究概要・現状](研究概要・現状.md)を読む。
2. 本書でカタログIDまたはwork packageから専用検証文書を探す。
3. 専用文書で固定条件、判定基準、数値、限界を確認する。
4. 対応artifactで数値とprovenanceを、対応testでschema、fingerprint、算術を確認する。
5. 再現性と利用可否は[`VALIDATION_STATUS.md`](../../VALIDATION_STATUS.md)と[`validation_manifest.json`](../../artifacts/validation_manifest.json)を優先する。

`completed`は代表H4条件または明記した再集計scope内でwork packageの終了条件を満たしたという意味であり、対応するカタログIDの全物理系・全精度・全compiler条件を完了したという意味ではない。

## 2. 共通範囲

- H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12
- 固定Hamiltonian/DF snapshot、主候補`L_D=3,12`
- 二次partial-S2、finite RTE、状態準備を除くRPE interrogation cost
- complete controlled Hadamard wrapperのcompiled RZ countを主指標とする
- 主なdelta候補は0.01と0.02
- optimization level 1と2を混ぜて最終区間を作らない

H4でのcoverageをH12、別分子、別geometry、別snapshot、backend/noiseへ外挿しない。

## 3. 実施内容と証拠

### 3.1 `screening_now`からGate S1

| 実施単位 | 対応ID | status・coverage | 文書 | artifact | runner / test |
|---|---|---|---|---|---|
| WP00 | A01–A03、B01、B05、B08、N03 | `completed_for_representative_H4_contract` | [screening](../research_direction_prevalidation.md) | [`wp00_comparison_contract_v1.json`](../../artifacts/research_direction_prevalidation/2026-09-21/wp00_comparison_contract_v1.json) | [`runner`](../../scripts/run_research_direction_prevalidation.py) / [`test`](../../tests/test_research_direction_prevalidation.py) |
| WP02 | G01–G03、G05、E02、F04、M01 | `completed_as_coverage_audit` | [screening](../research_direction_prevalidation.md) | [`wp02_round_horizon_coverage_v1.json`](../../artifacts/research_direction_prevalidation/2026-09-21/wp02_round_horizon_coverage_v1.json) | 同上 |
| WP01-S | C01–C03、E01、F02、H01、Q05 | `completed_model_conditional_undetermined`。screening用proxy | [screening](../research_direction_prevalidation.md) | [`wp01s_model_conditional_screening_v1.json`](../../artifacts/research_direction_prevalidation/2026-09-21/wp01s_model_conditional_screening_v1.json) | 同上 |
| WP04 | F03–F05、F08、G04、H02–H06 | `completed_model_conditional_ablation` | [WP04](../research_direction_ablation.md) | [`wp04_finite_rte_statistical_ablation_v1.json`](../../artifacts/research_direction_ablation/2026-09-21/wp04_finite_rte_statistical_ablation_v1.json) | [`runner`](../../scripts/run_research_direction_ablation.py) / [`test`](../../tests/test_research_direction_ablation.py) |
| WP03 | D01–D05、E01–E03、I01、Q02 | `completed_selection_invariant_intervals_overlap` | [WP03](../research_direction_pf_sensitivity.md) | [`artifact directory`](../../artifacts/research_direction_pf_sensitivity/2026-09-22/) | [`runner`](../../scripts/run_research_direction_pf_sensitivity.py) / [`test`](../../tests/test_research_direction_pf_sensitivity.py) |
| Gate S1 | 上記IDの判断統合 | `completed_direction_synthesis`。区間重なりを未判定とした | [Gate S1](../research_direction_gate_s1.md) | [`gate_s1_research_direction_decision_v1.json`](../../artifacts/research_direction_gate_s1/2026-09-22/gate_s1_research_direction_decision_v1.json) | [`runner`](../../scripts/run_research_direction_gate_s1.py) / [`test`](../../tests/test_research_direction_gate_s1.py) |

### 3.2 `decision_bridge`と候補比較

| 実施単位 | 対応ID | status・coverage | 文書 | artifact | runner / test |
|---|---|---|---|---|---|
| WP06-a | L02–L04、J04、J05 | `completed_triggered_focused_followup` | [構造pilot](../research_direction_structure_pilot.md) | [`wp06a`](../../artifacts/research_direction_structure_pilot/2026-09-22/wp06a_circuit_structure_pilot_v1.json) | [`runner`](../../scripts/run_research_direction_structure_pilot.py) / [`test`](../../tests/test_research_direction_structure_pilot.py) |
| WP06-b | 同上のfocused follow-up | `completed_sequence_policy_holdout` | [sequence policy](../research_direction_sequence_policy.md) | [`wp06b`](../../artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json) | [`runner`](../../scripts/run_research_direction_sequence_policy.py) / [`test`](../../tests/test_research_direction_sequence_policy.py) |
| WP05-a | L01、L06、L07、M01、M02、M05、N04 | `completed_full_scope_q4_holdout` | [full scope](../research_direction_full_scope.md) | [`wp05a`](../../artifacts/research_direction_full_scope/2026-09-22/wp05a_full_controlled_interrogation_connection_v1.json) | [`runner`](../../scripts/run_research_direction_full_scope.py) / [`test`](../../tests/test_research_direction_full_scope.py) |
| WP05-b/R | WP05と同じIDのq=8・delta拡張 | `completed_after_focused_replication` | [拡張・再検証](../research_direction_full_scope_extension.md) | [`WP05-b`](../../artifacts/research_direction_full_scope_extension/2026-09-22/wp05b_q8_delta_0p01_full_scope_extension_v1.json)、[`WP05-bR`](../../artifacts/research_direction_full_scope_replication/2026-09-22/wp05br_r32_32trajectory_replication_v1.json) | 対応する同語幹runner / test |
| WP01-D/C07 | C01–C03、C07、E01、F02、H01、Q05 | `completed_local_conditional_robust_undetermined` | [decision cost](../research_direction_decision_cost.md) | [`compute`](../../artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_full_scope_optimization_compute_v2.json)、[`synthesis`](../../artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_conditional_interval_synthesis_v1.json) | 対応する同語幹runner / test |
| G08 | G08 | `completed_round_dominance_audit` | [G08/M08](../research_direction_late_round_proxy.md) | [`g08`](../../artifacts/research_direction_round_dominance/2026-09-22/g08_round_cost_risk_proxy_dominance_v1.json) | [`runner`](../../scripts/run_research_direction_round_dominance.py) / [`test`](../../tests/test_research_direction_round_dominance.py) |
| M08 | M08 | `completed_q16_q32_holdout_and_reaggregation` | [G08/M08](../research_direction_late_round_proxy.md) | [`holdout`](../../artifacts/research_direction_proxy_precision/2026-09-22/m08_late_round_q16_q32_proxy_precision_v1.json)、[`reaggregation`](../../artifacts/research_direction_proxy_precision/2026-09-22/wp01d_c07_m08_measured_discrepancy_reaggregation_v1.json) | 対応する同語幹runner / test |
| M06/L08 | M06、L08 | `completed_focused_compiler_transfer_robust_undetermined` | [compiler transfer](../research_direction_compiler_transfer.md) | [`compute`](../../artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_same_trajectory_compute_v1.json)、[`analysis`](../../artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_focused_analysis_reaggregation_v1.json) | 対応する同語幹runner / test |
| N07/P03 | N07、P03 | `completed_uncertainty_ledger_break_even_robust_undetermined` | [uncertainty](../research_direction_uncertainty_break_even.md) | [`n07_p03`](../../artifacts/research_direction_uncertainty_break_even/2026-09-23/n07_p03_uncertainty_break_even_v1.json) | [`runner`](../../scripts/run_research_direction_uncertainty_break_even.py) / [`test`](../../tests/test_research_direction_uncertainty_break_even.py) |

### 3.3 判断統合とWP11 follow-up

| 実施単位 | 対応ID | status・coverage | 文書 | artifact | runner / test |
|---|---|---|---|---|---|
| WP11 | A04–A08、N08、P03、P08、Q08の判断統合 | `completed_scoped_direction_synthesis`。11 artifactをT1–T7へ統合 | [WP11](../research_direction_wp11_synthesis.md) | [`wp11`](../../artifacts/research_direction_wp11_synthesis/2026-09-23/wp11_scoped_direction_synthesis_v1.json) | [`runner`](../../scripts/run_research_direction_wp11_synthesis.py) / [`test`](../../tests/test_research_direction_wp11_synthesis.py) |
| M06-F | 主にM06/L08。G08/M08/N07/P03の判断入力を同一opt2 contextで更新 | `completed_all_r_coherent_opt2_reoptimization`。初期36＋fresh-32拡張15の51/51 task | [M06-F](../research_direction_full_opt2.md) | [`2026-09-24`](../../artifacts/research_direction_full_opt2/2026-09-24/)、[`2026-09-25`](../../artifacts/research_direction_full_opt2/2026-09-25/) | `scripts/run_research_direction_full_opt2_*.py` / `tests/test_research_direction_full_opt2*.py` |
| A0 | M06/M08のevidence-lineage補足 | `completed_selected_rz_q16_q32_reconciliation`。新規compileなし、単一H4 cell限定 | [M06-F A0](../research_direction_full_opt2.md#a0-proxy-lineage-reconciliation) | [`a0`](../../artifacts/research_direction_full_opt2/2026-09-25/m06f_a0_proxy_lineage_reconciliation_v1.json) | [`runner`](../../scripts/run_research_direction_proxy_lineage_reconciliation.py) / [`test`](../../tests/test_research_direction_proxy_lineage_reconciliation.py) |

`M06-F`と`A0`はカタログの独立した145番目以降のIDではなく、WP11後の追加検証名である。

## 4. 現在の結論

- Gate S1では候補区間が重なり、`undetermined_not_tied`とした。主な共通改善要因はbeta、次いでalpha再配分で、PF係数選択はshortlistを変えなかった。
- WP06-a/bとWP05-a/b/Rにより回路policyを固定し、complete controlled Hadamard wrapperまでscopeを拡張した。
- G08/M08で後半round支配を特定し、q=16,32直接holdoutを通過した。ただし直接domainはq<=32である。
- M06/L08ではcompiler変更後の区間が重なり、compiler-invariantな優位性は確立しなかった。
- N07/P03は状態準備costをパラメータ評価したが、準備回路自体は測定していない。
- M06-Fは51/51 taskと両gateを完了した。状態準備なし点推定は`L_D=3/12`で`1.263314e12/1.327822e12` compiled RZ、`L_D=3`が4.858%低いが、全不確かさ区間が重なるため頑健な優位性は主張しない。
- A0は最新fresh較正から旧`q=16,32`を再照合し、単一cellのselected RZを最大4.911%で通過させた。full-basis `q=32`は5.598%で、全metric・他cell・`q>32`へは一般化しない。

T4/T7を主軸、T1をH4条件付き成立限界へ範囲変更、T2/T5/T6を限定継続、T3を保留とする。

## 5. 未実施・延期

| 対象 | 扱い |
|---|---|
| WP07（K01–K03、K06） | work packageとして未実施。T3を保留 |
| WP08（R01、F07、N01） | work packageとして未実施 |
| WP09（I02–I06、J01–J03） | work packageとして未実施 |
| WP10（N03、N05、O01、O05等） | 未実施。外部instance pilotは将来候補 |
| H12 | 延期。H4/H6から外挿しない |
| q>32 opt2直接holdout | 未実施 |
| 状態準備回路、coupling/backend/noise | 未実施 |
| 最終総costと科学的優位性 | 未確定 |

上表に対応付けられていない残りのカタログIDは、この優先campaignで実施済みとは扱わない。

## 6. GitHubで確認できる範囲

GitHubには集計済みartifact、provenance、検証ロジック、runner、test、結果文書が含まれる。M06-Fの`checkpoints/`、`tasks/`、`worker_results/`はGit管理外である。clean checkoutではコミット済みartifactを検証できるが、生データからの完全再集計はserver evidenceがある環境だけで実行される。

clean-checkout相当testは`570 passed, 2 skipped, 4 warnings`、失敗0である。これはimmutable CIまたは外部独立再現を意味しない。

## 7. GPT向け確認事項

1. 各行の結果文書、artifact、testが存在するか確認する。
2. artifactの固定条件と2節の範囲が一致するか確認する。
3. `completed`を代表条件内の完了として扱い、カタログID全体へ一般化しない。
4. 点推定と区間、直接測定domainと外挿domainを分ける。
5. WP07–WP10、H12、状態準備、backend/noise、最終総costを完了扱いしない。
6. 数値の正本は専用文書とartifact、statusの正本は`VALIDATION_STATUS.md`とmanifestとする。
