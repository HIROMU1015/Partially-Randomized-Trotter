# 公開成果同期の状況

公開Git成果のみを読取。個別結果の取得・fit・価格予測・Q3は起動しない。
方式別の費用は[比較表](run01/REPORT.md)を参照。

|判断・接続待ちの固定成果|理由|
|---|---|
|`06956c643:docs/research/d_full_h_d7_holdout_20261115/HANDOFF_C.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`74cd9c262:docs/research/d_full_h_m0_reference_20261115/HANDOFF_C.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`64624ea07:docs/research/d_full_h_m0_zz_evidence_20261115/HANDOFF_C.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`4768ca3cd:docs/research/d_full_h_p2_calibration_resume_20261115/HANDOFF_C.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`e628c0d69:docs/research/d_full_h_p2_calibration_resume_20261115/HANDOFF_C.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`778118926:docs/research/d_pr_df_ld9_v0_results_20261116/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`e0b3cad2a:docs/research/d_pr_df_ld_split_results_20261115/C_HANDOFF.json`|ValueError: Unsupported saved index schema: e0b3cad2a024070389f757160f3a20d7da02c5ff:docs/research/d_pr_df_ld_split_results_20261115/SUMMARY.json|
|`e0b3cad2a:docs/research/d_pr_df_ld_split_results_20261115/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`326feadd0:docs/research/d_pr_df_signal_entry_20261116/C_HANDOFF.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`7cd8ea139:docs/research/d_pr_price_rule_applicability_20261116/C_HANDOFF.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`422db93fc:docs/research/full_h_kappa2_scale2_q3_results_20261115/C_RESULT_JOIN.json`|ValueError: Unsupported saved index schema: 422db93fc7bb599a7ea46c5b58e5dec0fda9ef08:docs/research/full_h_kappa2_scale2_q3_results_20261115/SUMMARY.json|
|`422db93fc:docs/research/full_h_kappa2_scale2_q3_results_20261115/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`6c5569c34:docs/research/full_h_kappa2_scale3o2_q3_results_20261115/C_RESULT_JOIN.json`|ValueError: Unsupported saved index schema: 6c5569c3479013b080b2ba53842585f1d670d3c9:docs/research/full_h_kappa2_scale3o2_q3_results_20261115/SUMMARY.json|
|`6c5569c34:docs/research/full_h_kappa2_scale3o2_q3_results_20261115/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`bec032fc2:docs/research/p7_s2_d01_same_shots_q3_results_20261115/C_RESULT_JOIN.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`bec032fc2:docs/research/p7_s2_d01_same_shots_q3_results_20261115/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`3b5cd49c8:docs/research/p7_s2_h_budget_q3_results_20261116/C_RESULT_JOIN.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`3b5cd49c8:docs/research/p7_s2_h_budget_q3_results_20261116/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`0b33b4afd:docs/research/p7_s2_local_shot_search_results_20261116/C_RESULT_JOIN.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`0b33b4afd:docs/research/p7_s2_local_shot_search_results_20261116/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`50be68b80:docs/research/pauli_q3_three_quarter_20261115/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`caf1417ef:docs/research/pauli_total_q3_accelerated_results_20261114/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`b72ea6b30:docs/research/pf_power_stability_h4_20261113/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`8dba096dc:docs/research/pf_shortlist_q3_fixed_results_20261115/C_RESULT_JOIN.json`|ValueError: Unsupported saved index schema: 8dba096dc39788b8948af9eafe394f03d386c520:docs/research/pf_shortlist_q3_fixed_results_20261115/SUMMARY.json|
|`8dba096dc:docs/research/pf_shortlist_q3_fixed_results_20261115/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`a82b0f0ec:docs/research/pr_df_fullh_q3_fixed_results_20261115/C_RESULT_JOIN.json`|ValueError: Unsupported saved index schema: a82b0f0ec7f0f11d9981ddb2343fcd5378a87503:docs/research/pr_df_fullh_q3_fixed_results_20261115/SUMMARY.json|
|`a82b0f0ec:docs/research/pr_df_fullh_q3_fixed_results_20261115/SUMMARY.json`|Unsupported saved schema; needs consumer adapter, not scientific rerun|
|`d7a976a62:docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round03/run01/ROUND_RESULT.json`|ROUND_RESULT selected allocation lacks matching public OUTCOME/hash/price binding; configuration-only match rejected. A later KAPPA_ABS_RESULTS/best_points selection is admitted separately; no job started|

未対応schema/結合不足は保存し、他の取込みを止めない。新科学計算の承認・採択・旧STOPは変更しない。
